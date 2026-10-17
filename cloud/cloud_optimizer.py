"""
cloud_optimizer.py — Cloud-based distributed strategy optimiser.

Implements the same interface as StrategyOptimizer (start, stop,
get_status, get_results) but distributes backtests across EC2 Spot
Instances via SQS/S3.

Usage:
    optimizer = CloudOptimizer(config)
    opt_id = optimizer.start("rsi", "EURUSD", "H1", ...)
    status = optimizer.get_status(opt_id)
    results = optimizer.get_results(opt_id)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import pandas as pd

from backtest_core import pip_value_per_lot as _pip_value_per_lot
from config import Config, TIMEFRAME_MAP
from optimizer import PARAM_GRIDS, generate_combos

from cloud.aws_infra import AWSInfra
from cloud.job_manager import (
    serialize_job_data, chunk_combos, build_sqs_messages,
    create_worker_package, BATCH_SIZE,
)
from cloud.result_collector import ResultCollector
from cloud.cost_tracker import CostTracker, estimate_job_cost
from cloud.setup import load_cloud_config

logger = logging.getLogger("cloud.optimizer")


# =====================================================================
#  Cloud Optimisation Result
# =====================================================================

@dataclass
class CloudOptimizationResult:
    """Tracks a cloud optimisation run."""
    opt_id: str = ""
    strategy_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    date_from: str = ""
    date_to: str = ""
    mode: str = "normal"
    optimize_sltp: bool = False
    target_metric: str = "profit_factor"
    initial_balance: float = 10000.0
    lot_size: float = 0.1
    min_trades: int = 3
    max_open_trades: int = 5

    status: str = "initializing"  # initializing | uploading | running | completed | failed | stopped
    error_msg: str = ""
    progress: float = 0.0
    total_combos: int = 0
    total_batches: int = 0
    completed_batches: int = 0
    start_time: float = 0.0
    elapsed_secs: float = 0.0
    eta_secs: float = 0.0

    # Cloud-specific
    num_workers: int = 0
    instance_ids: list[str] = field(default_factory=list)
    queue_url: str = ""
    dlq_url: str = ""

    # Results
    results: list[dict] = field(default_factory=list)
    best_params: dict = field(default_factory=dict)
    best_metric_value: float = 0.0

    # Cost
    cost_info: dict = field(default_factory=dict)

    def to_summary(self) -> dict:
        return {
            "opt_id": self.opt_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "mode": self.mode,
            "optimize_sltp": self.optimize_sltp,
            "target_metric": self.target_metric,
            "initial_balance": self.initial_balance,
            "lot_size": self.lot_size,
            "min_trades": self.min_trades,
            "max_open_trades": self.max_open_trades,
            "status": self.status,
            "error_msg": self.error_msg,
            "progress": round(self.progress, 2),
            "total_combos": self.total_combos,
            "tested_combos": self.completed_batches * BATCH_SIZE,
            "total_batches": self.total_batches,
            "completed_batches": self.completed_batches,
            "elapsed_secs": round(self.elapsed_secs, 1),
            "eta_secs": round(self.eta_secs, 1),
            "best_params": self.best_params,
            "best_metric_value": round(self.best_metric_value, 2),
            "results_count": len(self.results),
            "num_workers": self.num_workers,
            "cloud": True,
            "cost": self.cost_info,
        }


# =====================================================================
#  Cloud Optimizer
# =====================================================================

class CloudOptimizer:
    """Distributed optimiser using AWS Spot Instances.

    Same interface as StrategyOptimizer for dashboard compatibility.
    """

    _active_runs: dict[str, CloudOptimizationResult] = {}
    _stop_events: dict[str, threading.Event] = {}
    _collectors: dict[str, ResultCollector] = {}
    _cost_trackers: dict[str, CostTracker] = {}
    _lock = threading.Lock()

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        cloud_cfg = load_cloud_config()
        if not cloud_cfg or not cloud_cfg.get("setup_complete"):
            raise RuntimeError(
                "Cloud optimisation not set up. Run cloud setup first "
                "(POST /api/optimize/cloud/setup or python -m cloud.setup)."
            )
        self._cloud_config = cloud_cfg
        self._infra = AWSInfra(region=cloud_cfg["region"])
        self._bucket = cloud_cfg["bucket"]

    def start(
        self,
        strategy_name: str,
        symbol: str,
        timeframe_key: str,
        date_from: str,
        date_to: str,
        mode: str = "normal",
        optimize_sltp: bool = False,
        target_metric: str = "profit_factor",
        initial_balance: float = 10000.0,
        lot_size: float = 0.1,
        min_trades: int = 3,
        max_open_trades: int = 5,
    ) -> str:
        """Start a cloud optimisation. Returns opt_id."""
        if strategy_name not in PARAM_GRIDS:
            raise ValueError(
                f"Strategy '{strategy_name}' not supported. "
                f"Supported: {list(PARAM_GRIDS.keys())}"
            )

        opt_id = "cloud-opt-" + str(uuid.uuid4())[:8]
        result = CloudOptimizationResult(
            opt_id=opt_id,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe_key,
            date_from=date_from,
            date_to=date_to,
            mode=mode,
            optimize_sltp=optimize_sltp,
            target_metric=target_metric,
            initial_balance=initial_balance,
            lot_size=lot_size,
            min_trades=min_trades,
            max_open_trades=max_open_trades,
        )

        stop_event = threading.Event()
        with self._lock:
            self._active_runs[opt_id] = result
            self._stop_events[opt_id] = stop_event

        thread = threading.Thread(
            target=self._run_orchestrator,
            args=(result, stop_event),
            name=f"CloudOptimizer-{opt_id}",
            daemon=True,
        )
        thread.start()
        return opt_id

    def stop(self, opt_id: str) -> bool:
        """Stop a cloud optimisation. Terminates workers, collects partial results."""
        with self._lock:
            event = self._stop_events.get(opt_id)
            if event:
                event.set()
                return True
        return False

    def get_status(self, opt_id: str) -> dict | None:
        with self._lock:
            if opt_id in self._active_runs:
                return self._active_runs[opt_id].to_summary()
        return None

    def get_results(self, opt_id: str) -> dict | None:
        with self._lock:
            if opt_id in self._active_runs:
                r = self._active_runs[opt_id]
                out = r.to_summary()
                out["results"] = r.results[:100]
                return out
        return None

    # ── Orchestrator ──────────────────────────────────────────────────

    def _run_orchestrator(self, result: CloudOptimizationResult,
                          stop_event: threading.Event) -> None:
        result.start_time = time.time()
        try:
            self._execute(result, stop_event)
        except Exception as exc:
            logger.error("Cloud optimisation %s failed: %s",
                         result.opt_id, exc, exc_info=True)
            result.status = "failed"
            result.error_msg = str(exc)[:500]
            self._cleanup(result)
        finally:
            result.elapsed_secs = time.time() - result.start_time
            with self._lock:
                self._stop_events.pop(result.opt_id, None)
                self._collectors.pop(result.opt_id, None)
                self._cost_trackers.pop(result.opt_id, None)

    def _execute(self, result: CloudOptimizationResult,
                 stop_event: threading.Event) -> None:
        opt_id = result.opt_id
        logger.info("Cloud optimisation START: %s %s on %s %s [%s -> %s] mode=%s",
                     opt_id, result.strategy_name, result.symbol,
                     result.timeframe, result.date_from, result.date_to, result.mode)

        # ── 1. Fetch historical data from MT5 ────────────────────────
        result.status = "initializing"
        mt5_tf = TIMEFRAME_MAP.get(result.timeframe)
        if mt5_tf is None:
            raise ValueError(f"Unknown timeframe: {result.timeframe}")

        dt_from = datetime.fromisoformat(result.date_from).replace(tzinfo=timezone.utc)
        dt_to = datetime.fromisoformat(result.date_to).replace(tzinfo=timezone.utc)

        rates = mt5.copy_rates_range(result.symbol, mt5_tf, dt_from, dt_to)
        if rates is None or len(rates) == 0:
            raise ValueError(
                f"No historical data for {result.symbol} {result.timeframe} "
                f"from {result.date_from} to {result.date_to}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        logger.info("Fetched %d bars for cloud optimisation.", len(df))

        # ── 2. Get symbol info ───────────────────────────────────────
        sym_info = mt5.symbol_info(result.symbol)
        if sym_info is None:
            raise ValueError(f"Cannot get symbol info for {result.symbol}")
        point = sym_info.point
        digits = sym_info.digits
        # One shared pip definition — this used to special-case JPY with
        # its own formula, which disagreed with the live engine and with
        # the local sweep it is meant to mirror.
        pip_value_per_lot = _pip_value_per_lot(
            result.symbol, point,
            contract_size=float(
                getattr(sym_info, "trade_contract_size", 0) or 0
            ),
            tick_value=getattr(sym_info, "trade_tick_value", 0) or 0,
            tick_size=getattr(sym_info, "trade_tick_size", 0) or 0,
        )

        # ── 3. Generate parameter combos ─────────────────────────────
        combos = generate_combos(
            result.strategy_name, result.mode, result.optimize_sltp,
        )
        result.total_combos = len(combos)
        logger.info("Generated %d parameter combinations.", len(combos))

        if len(combos) == 0:
            raise ValueError("No valid parameter combinations generated.")

        # ── 4. Serialize and upload to S3 ────────────────────────────
        result.status = "uploading"
        parquet_bytes, meta_bytes = serialize_job_data(
            df, result.symbol, point, digits, pip_value_per_lot,
        )

        self._infra.upload_bytes(
            self._bucket, f"jobs/{opt_id}/data.parquet", parquet_bytes,
        )
        self._infra.upload_bytes(
            self._bucket, f"jobs/{opt_id}/meta.json", meta_bytes,
            content_type="application/json",
        )

        # Upload worker package
        worker_pkg = create_worker_package()
        self._infra.upload_bytes(
            self._bucket, f"jobs/{opt_id}/worker-package.tar.gz", worker_pkg,
        )

        if stop_event.is_set():
            result.status = "stopped"
            self._cleanup(result)
            return

        # ── 5. Create SQS queue and enqueue batches ──────────────────
        queue_url, dlq_url = self._infra.create_job_queue(opt_id)
        result.queue_url = queue_url
        result.dlq_url = dlq_url

        chunks = chunk_combos(combos)
        result.total_batches = len(chunks)

        backtest_config = {
            "lot_size": result.lot_size,
            "initial_balance": result.initial_balance,
            "default_sl": self.config.default_sl_pips,
            "default_tp": self.config.default_tp_pips,
            "max_open_trades": result.max_open_trades,
            "min_trades": result.min_trades,
        }

        messages = build_sqs_messages(
            opt_id, result.strategy_name, chunks, backtest_config,
        )
        self._infra.send_messages(queue_url, messages)

        if stop_event.is_set():
            result.status = "stopped"
            self._cleanup(result)
            return

        # ── 6. Provision EC2 Spot fleet ──────────────────────────────
        result.status = "running"
        instance_type = getattr(self.config, "aws_cloud_instance_type", "c6a.large")
        max_workers = getattr(self.config, "aws_cloud_max_workers", 10)
        max_runtime = getattr(self.config, "aws_cloud_max_runtime_mins", 120)
        max_cost = getattr(self.config, "aws_cloud_max_cost_per_job", 10.0)

        num_workers = min(
            max(1, result.total_combos // 200),
            max_workers,
            20,
        )
        result.num_workers = num_workers

        # Build user-data script from template
        bootstrap_template = (
            Path(__file__).parent / "worker" / "bootstrap.sh"
        ).read_text()
        user_data = bootstrap_template.replace(
            "{{SQS_QUEUE_URL}}", queue_url,
        ).replace(
            "{{S3_BUCKET}}", self._bucket,
        ).replace(
            "{{JOB_ID}}", opt_id,
        ).replace(
            "{{AWS_REGION}}", self._infra.region,
        )

        spot_price = self._infra.get_spot_price(instance_type)

        instance_ids = self._infra.launch_spot_fleet(
            num_instances=num_workers,
            instance_type=instance_type,
            user_data=user_data,
            job_id=opt_id,
            max_runtime_mins=max_runtime,
        )
        result.instance_ids = instance_ids

        # Set up result collector and cost tracker
        collector = ResultCollector(
            self._infra, self._bucket, opt_id,
            total_batches=result.total_batches,
            target_metric=result.target_metric,
        )
        cost_tracker = CostTracker(spot_price, max_cost=max_cost)
        cost_tracker.register_instances(instance_ids)

        with self._lock:
            self._collectors[opt_id] = collector
            self._cost_trackers[opt_id] = cost_tracker

        # ── 7. Poll for results ──────────────────────────────────────
        logger.info("Polling for results: %d batches across %d workers...",
                     result.total_batches, num_workers)

        while not stop_event.is_set():
            collector.poll()

            result.completed_batches = collector.completed_batches
            result.progress = collector.progress
            result.elapsed_secs = time.time() - result.start_time
            result.cost_info = cost_tracker.to_dict()

            # Update ETA
            if result.progress > 0.01:
                result.eta_secs = max(0,
                    (result.elapsed_secs / result.progress) - result.elapsed_secs)

            # Update best results incrementally
            ranked = collector.get_ranked_results(100)
            result.results = ranked
            if ranked:
                best_params, best_val = collector.get_best()
                result.best_params = best_params
                result.best_metric_value = best_val

            # Check completion
            if collector.is_complete:
                logger.info("All %d batches collected.", result.total_batches)
                break

            # Check budget
            if cost_tracker.budget_exceeded:
                logger.warning("Budget exceeded ($%.2f / $%.2f). Stopping.",
                               cost_tracker.estimated_cost, max_cost)
                result.error_msg = "Budget limit reached."
                break

            # Check if all workers have died
            alive = self._infra.get_job_instances(opt_id)
            queue_depth = self._infra.get_queue_depth(queue_url)
            if not alive and queue_depth > 0 and not collector.is_complete:
                logger.warning("All workers terminated but %d messages remain. "
                               "Stopping with partial results.", queue_depth)
                result.error_msg = "All workers terminated before completion."
                break

            time.sleep(5)

        # ── 8. Finalize ──────────────────────────────────────────────
        # Final poll to catch any last results
        collector.poll()
        result.completed_batches = collector.completed_batches
        result.progress = collector.progress
        result.results = collector.get_ranked_results(100)
        if result.results:
            best_params, best_val = collector.get_best()
            result.best_params = best_params
            result.best_metric_value = best_val

        was_stopped = stop_event.is_set()
        result.status = "stopped" if was_stopped else "completed"
        result.elapsed_secs = time.time() - result.start_time
        result.cost_info = cost_tracker.to_dict()

        logger.info(
            "Cloud optimisation %s: %s | %d/%d batches | %d results | "
            "Best %s: %.2f | Cost: $%.4f",
            result.status.upper(), opt_id, result.completed_batches,
            result.total_batches, len(result.results), result.target_metric,
            result.best_metric_value, cost_tracker.estimated_cost,
        )

        self._cleanup(result)

    def _cleanup(self, result: CloudOptimizationResult) -> None:
        """Terminate workers and clean up cloud resources."""
        opt_id = result.opt_id

        # Mark cost tracker instances as terminated
        tracker = self._cost_trackers.get(opt_id)
        if tracker and result.instance_ids:
            tracker.mark_terminated(result.instance_ids)
            result.cost_info = tracker.to_dict()

        # Terminate instances
        if result.instance_ids:
            self._infra.terminate_instances(result.instance_ids)

        # Delete SQS queues
        if result.queue_url:
            self._infra.delete_queue(result.queue_url)
        if result.dlq_url:
            self._infra.delete_queue(result.dlq_url)

        # Delete job data from S3 (keep results for a bit via lifecycle)
        try:
            self._infra.delete_prefix(self._bucket, f"jobs/{opt_id}/data.parquet")
            self._infra.delete_prefix(self._bucket, f"jobs/{opt_id}/meta.json")
            self._infra.delete_prefix(self._bucket, f"jobs/{opt_id}/worker-package.tar.gz")
        except Exception:
            pass

        logger.info("Cleaned up cloud resources for %s.", opt_id)

    # ── Cost Estimation (static) ─────────────────────────────────────

    def estimate(
        self,
        strategy_name: str,
        mode: str = "normal",
        optimize_sltp: bool = False,
    ) -> dict:
        """Estimate cost and time for a cloud optimisation job."""
        combos = generate_combos(strategy_name, mode, optimize_sltp)
        total_combos = len(combos)

        instance_type = getattr(self.config, "aws_cloud_instance_type", "c6a.large")
        max_workers = getattr(self.config, "aws_cloud_max_workers", 10)
        num_workers = min(max(1, total_combos // 200), max_workers, 20)
        spot_price = self._infra.get_spot_price(instance_type)

        return estimate_job_cost(
            total_combos=total_combos,
            spot_price_per_hour=spot_price,
            num_workers=num_workers,
        )
