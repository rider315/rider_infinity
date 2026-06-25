"""
optimizer.py — Strategy Auto-Optimizer for the Rider Infinity Trading Platform.

Performs grid-search optimisation over strategy parameters by running
multiple backtests against the same historical data and ranking the
results by a user-chosen metric (profit factor, win rate, Sharpe, etc.).

Key optimisation: historical candle data is fetched **once** and reused
for every parameter combination.
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Any

import MetaTrader5 as mt5
import pandas as pd

from config import Config, TIMEFRAME_MAP
from strategies import get_strategy
from strategies.base import SignalAction
from utils import pips_to_price, price_to_pips, setup_logger
from backtester import BacktestDB, BacktestTrade

logger = setup_logger("optimizer")


# =====================================================================
#  Parameter Grid Definitions
# =====================================================================

# Each strategy maps to a dict of {param_name: {quick, normal, deep}}
# where each mode is a list of values to test.

PARAM_GRIDS: dict[str, dict[str, dict[str, list]]] = {
    "ema": {
        "fast_period": {
            "quick":  [5, 9, 15, 21],
            "normal": [5, 7, 9, 12, 15, 18, 21],
            "deep":   list(range(3, 30, 2)),
        },
        "slow_period": {
            "quick":  [21, 30, 50, 100],
            "normal": [15, 21, 30, 40, 50, 75, 100],
            "deep":   list(range(10, 120, 5)),
        },
    },
    "rsi": {
        "period": {
            "quick":  [7, 14, 21],
            "normal": [5, 7, 10, 14, 21, 28],
            "deep":   list(range(5, 35, 2)),
        },
        "overbought": {
            "quick":  [65, 70, 75, 80],
            "normal": [60, 65, 70, 75, 80, 85],
            "deep":   list(range(60, 91, 3)),
        },
        "oversold": {
            "quick":  [20, 25, 30, 35],
            "normal": [15, 20, 25, 30, 35, 40],
            "deep":   list(range(10, 41, 3)),
        },
    },
    "bollinger": {
        "period": {
            "quick":  [10, 15, 20, 25, 30],
            "normal": [8, 10, 14, 18, 20, 25, 30, 40],
            "deep":   list(range(5, 50, 3)),
        },
        "std_dev": {
            "quick":  [1.5, 2.0, 2.5, 3.0],
            "normal": [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
            "deep":   [x / 10 for x in range(10, 36, 2)],
        },
    },
    "macd": {
        "fast": {
            "quick":  [8, 12, 16],
            "normal": [6, 8, 10, 12, 14, 16, 18],
            "deep":   list(range(5, 21, 1)),
        },
        "slow": {
            "quick":  [20, 26, 34],
            "normal": [18, 22, 26, 30, 34, 40],
            "deep":   list(range(15, 51, 2)),
        },
        "signal": {
            "quick":  [5, 9, 13],
            "normal": [3, 5, 7, 9, 11, 13],
            "deep":   list(range(3, 18, 2)),
        },
    },
    "grid": {
        "grid_size_pips": {
            "quick":  [10, 20, 30, 50],
            "normal": [5, 10, 15, 20, 30, 40, 50, 75],
            "deep":   [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100],
        },
        "num_levels": {
            "quick":  [3, 5, 7],
            "normal": [2, 3, 5, 7, 10],
            "deep":   list(range(2, 15)),
        },
        "max_open_orders": {
            "quick":  [5, 10],
            "normal": [3, 5, 8, 10, 15],
            "deep":   [3, 5, 8, 10, 12, 15, 20],
        },
    },
    "dca": {
        "dip_threshold_pct": {
            "quick":  [0.5, 1.0, 2.0],
            "normal": [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
            "deep":   [x / 10 for x in range(2, 40, 3)],
        },
        "max_positions": {
            "quick":  [3, 5, 7],
            "normal": [2, 3, 5, 7, 10],
            "deep":   list(range(2, 12)),
        },
        "take_profit_pct": {
            "quick":  [1.0, 2.0, 3.0],
            "normal": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            "deep":   [x / 10 for x in range(3, 60, 4)],
        },
        "reference_period": {
            "quick":  [20, 50, 100],
            "normal": [10, 20, 30, 50, 75, 100],
            "deep":   list(range(10, 120, 10)),
        },
    },
    # Mean reversion: the geometry knobs behind "cutting winners short".
    # sl_pips/tp_pips are swept here directly (reversion-appropriate ranges,
    # tighter than the trend-tuned global SLTP_GRIDS) — so optimise WITHOUT
    # the "optimise SL/TP" toggle for this strategy.
    "meanrev": {
        "z_threshold": {
            "quick":  [2.0, 2.5],
            "normal": [1.8, 2.0, 2.3, 2.6],
            "deep":   [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0],
        },
        "z_max": {
            "quick":  [3.5],
            "normal": [3.0, 3.5, 4.0],
            "deep":   [2.8, 3.0, 3.5, 4.0, 5.0],
        },
        "max_hold_bars": {
            "quick":  [16, 32],
            "normal": [12, 16, 24, 32, 48],
            "deep":   [8, 12, 16, 24, 32, 48, 64, 96],
        },
        "sl_pips": {
            "quick":  [30, 50],
            "normal": [20, 30, 40, 50],
            "deep":   [15, 20, 25, 30, 40, 50, 70],
        },
        "tp_pips": {
            "quick":  [15, 25],
            "normal": [10, 15, 20, 25, 30],
            "deep":   [8, 10, 12, 15, 20, 25, 30, 40],
        },
    },
    # The four below follow the standard pattern: structural params here,
    # SL/TP via the "optimise SL/TP" toggle (the trend-tuned global ranges
    # suit trend/breakout/reversal, unlike mean reversion).
    "donchian": {
        "channel_period": {
            "quick":  [34, 55, 89],
            "normal": [21, 34, 55, 89, 144],
            "deep":   list(range(20, 160, 8)),
        },
        "trend_period": {
            "quick":  [50, 100, 200],
            "normal": [50, 100, 150, 200],
            "deep":   list(range(50, 260, 25)),
        },
        "min_adx": {
            "quick":  [15, 20, 25],
            "normal": [10, 15, 20, 25, 30],
            "deep":   list(range(8, 35, 3)),
        },
    },
    "orb": {
        "or_minutes": {
            "quick":  [15, 30, 45],
            "normal": [15, 30, 45, 60],
            "deep":   [15, 30, 45, 60, 90],
        },
        "buffer_atr": {
            "quick":  [0.1, 0.25, 0.5],
            "normal": [0.0, 0.1, 0.25, 0.4, 0.6],
            "deep":   [x / 100 for x in range(0, 81, 10)],
        },
        "squeeze_factor": {
            "quick":  [1.0, 1.5],
            "normal": [0.8, 1.0, 1.2, 1.5, 2.0],
            "deep":   [x / 10 for x in range(6, 26, 2)],
        },
    },
    "sweep": {
        "swing_lookback": {
            "quick":  [30, 50, 80],
            "normal": [20, 30, 50, 80, 100],
            "deep":   list(range(20, 120, 10)),
        },
        "wick_ratio": {
            "quick":  [0.5, 0.6, 0.7],
            "normal": [0.45, 0.5, 0.55, 0.6, 0.7],
            "deep":   [x / 100 for x in range(40, 81, 5)],
        },
        "range_mult": {
            "quick":  [1.0, 1.3, 1.6],
            "normal": [1.0, 1.2, 1.4, 1.6, 2.0],
            "deep":   [x / 10 for x in range(10, 26, 2)],
        },
        "max_adx": {
            "quick":  [20, 25, 30],
            "normal": [15, 20, 25, 30],
            "deep":   list(range(15, 36, 3)),
        },
    },
    "regime": {
        "fast_ema": {
            "quick":  [10, 20],
            "normal": [10, 15, 20, 30],
            "deep":   list(range(8, 35, 4)),
        },
        "slow_ema": {
            "quick":  [50, 100],
            "normal": [40, 50, 75, 100],
            "deep":   list(range(40, 160, 15)),
        },
        "trend_adx": {
            "quick":  [22, 25, 28],
            "normal": [20, 22, 25, 28, 30],
            "deep":   list(range(20, 33, 2)),
        },
        "range_adx": {
            "quick":  [15, 18],
            "normal": [12, 15, 18, 20],
            "deep":   list(range(10, 22, 2)),
        },
        "z_threshold": {
            "quick":  [2.0, 2.5],
            "normal": [1.8, 2.0, 2.3, 2.6],
            "deep":   [1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
        },
    },
}

# SL/TP optimisation ranges (applied on top of indicator params)
SLTP_GRIDS: dict[str, dict[str, list]] = {
    "sl_pips": {
        "quick":  [30, 50, 80],
        "normal": [20, 30, 50, 70, 100],
        "deep":   [15, 25, 35, 50, 70, 100, 150],
    },
    "tp_pips": {
        "quick":  [50, 100, 150],
        "normal": [40, 60, 80, 100, 150, 200],
        "deep":   [30, 50, 75, 100, 130, 170, 200, 300],
    },
}

# Constraint validators per strategy
def _validate_ema(params: dict) -> bool:
    return params.get("fast_period", 9) < params.get("slow_period", 21)

def _validate_rsi(params: dict) -> bool:
    return params.get("oversold", 30) < params.get("overbought", 70)

def _validate_bollinger(params: dict) -> bool:
    return params.get("period", 20) >= 2 and params.get("std_dev", 2.0) > 0

def _validate_macd(params: dict) -> bool:
    return params.get("fast", 12) < params.get("slow", 26)

def _validate_grid(params: dict) -> bool:
    return params.get("grid_size_pips", 20) >= 2 and params.get("num_levels", 5) >= 1

def _validate_dca(params: dict) -> bool:
    return params.get("dip_threshold_pct", 1.0) > 0 and params.get("reference_period", 50) >= 5

def _validate_meanrev(params: dict) -> bool:
    return (
        params.get("z_threshold", 2.0) < params.get("z_max", 3.5)
        and params.get("max_hold_bars", 16) >= 1
        and params.get("sl_pips", 40) > 0
        and params.get("tp_pips", 30) > 0
    )

def _validate_regime(params: dict) -> bool:
    return (
        params.get("fast_ema", 20) < params.get("slow_ema", 50)
        and params.get("range_adx", 18) < params.get("trend_adx", 25)
    )

VALIDATORS = {
    "ema": _validate_ema,
    "rsi": _validate_rsi,
    "bollinger": _validate_bollinger,
    "macd": _validate_macd,
    "grid": _validate_grid,
    "dca": _validate_dca,
    "meanrev": _validate_meanrev,
    "regime": _validate_regime,
}
# donchian / orb / sweep have no cross-parameter constraints → default
# validator (always-valid) via VALIDATORS.get(name, lambda p: True).


# =====================================================================
#  Optimisation Result
# =====================================================================

@dataclass
class OptimizationResult:
    """Tracks an optimisation run."""
    opt_id: str = ""
    strategy_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    date_from: str = ""
    date_to: str = ""
    mode: str = "normal"         # quick | normal | deep
    optimize_sltp: bool = False
    target_metric: str = "profit_factor"
    initial_balance: float = 10000.0
    lot_size: float = 0.1
    min_trades: int = 3
    max_open_trades: int = 5

    status: str = "running"      # running | completed | failed | stopped
    error_msg: str = ""
    progress: float = 0.0
    total_combos: int = 0
    tested_combos: int = 0
    start_time: float = 0.0      # time.time() at start
    elapsed_secs: float = 0.0
    eta_secs: float = 0.0

    # Results — list of dicts sorted by target metric
    results: list[dict] = field(default_factory=list)
    best_params: dict = field(default_factory=dict)
    best_metric_value: float = 0.0

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
            "tested_combos": self.tested_combos,
            "elapsed_secs": round(self.elapsed_secs, 1),
            "eta_secs": round(self.eta_secs, 1),
            "best_params": self.best_params,
            "best_metric_value": round(self.best_metric_value, 2),
            "results_count": len(self.results),
        }


# =====================================================================
#  Strategy Optimizer
# =====================================================================

class StrategyOptimizer:
    """Grid-search optimiser for trading strategies.

    Fetches historical data once, then runs each parameter combination
    through a lightweight backtest loop, ranks results by the chosen
    metric, and stores top results.
    """

    _active_runs: dict[str, OptimizationResult] = {}
    _stop_events: dict[str, threading.Event] = {}
    _lock = threading.Lock()

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.db = BacktestDB()  # reuse the backtest DB for storing results

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
        """Start an optimisation in a background thread. Returns opt_id."""
        if strategy_name not in PARAM_GRIDS:
            raise ValueError(
                f"Strategy '{strategy_name}' not supported for optimisation. "
                f"Supported: {list(PARAM_GRIDS.keys())}"
            )
        if mode not in ("quick", "normal", "deep"):
            raise ValueError(f"Invalid mode: {mode}. Use quick/normal/deep.")

        opt_id = "opt-" + str(uuid.uuid4())[:8]
        result = OptimizationResult(
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
            target=self._run_worker,
            args=(result, stop_event),
            name=f"Optimizer-{opt_id}",
            daemon=True,
        )
        thread.start()
        return opt_id

    def stop(self, opt_id: str) -> bool:
        """Signal an optimisation to stop. Returns partial results."""
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
                out["results"] = r.results[:100]  # top 100
                return out
        return None

    # ── Worker ────────────────────────────────────────────────────────

    def _run_worker(self, result: OptimizationResult, stop_event: threading.Event) -> None:
        import time as _time
        result.start_time = _time.time()
        try:
            self._execute(result, stop_event)
        except Exception as exc:
            logger.error("Optimisation %s failed: %s", result.opt_id, exc, exc_info=True)
            result.status = "failed"
            result.error_msg = str(exc)[:500]
        finally:
            result.elapsed_secs = _time.time() - result.start_time
            with self._lock:
                self._stop_events.pop(result.opt_id, None)

    def _execute(self, result: OptimizationResult, stop_event: threading.Event) -> None:
        import time as _time
        logger.info(
            "⚡ Optimisation START: %s %s on %s %s [%s → %s] mode=%s",
            result.opt_id, result.strategy_name, result.symbol,
            result.timeframe, result.date_from, result.date_to, result.mode,
        )

        # ── 1. Fetch historical data ONCE ────────────────────────────
        mt5_tf = TIMEFRAME_MAP.get(result.timeframe)
        if mt5_tf is None:
            raise ValueError(f"Unknown timeframe: {result.timeframe}")

        date_from = datetime.fromisoformat(result.date_from).replace(tzinfo=timezone.utc)
        date_to = datetime.fromisoformat(result.date_to).replace(tzinfo=timezone.utc)

        rates = mt5.copy_rates_range(result.symbol, mt5_tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            raise ValueError(
                f"No historical data for {result.symbol} {result.timeframe} "
                f"from {result.date_from} to {result.date_to}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        logger.info("Fetched %d bars for optimisation.", len(df))

        # ── 2. Get symbol info ───────────────────────────────────────
        sym_info = mt5.symbol_info(result.symbol)
        if sym_info is None:
            raise ValueError(f"Cannot get symbol info for {result.symbol}")
        point = sym_info.point
        digits = sym_info.digits
        pip_value_per_lot = 10.0
        if "JPY" in result.symbol.upper():
            pip_value_per_lot = 1000.0 * point * 100

        # ── 3. Generate parameter combinations ───────────────────────
        combos = self._generate_combos(
            result.strategy_name, result.mode, result.optimize_sltp,
        )
        result.total_combos = len(combos)
        logger.info("Generated %d parameter combinations.", len(combos))

        if len(combos) == 0:
            raise ValueError("No valid parameter combinations generated.")

        # ── 4. Run backtest for each combination ─────────────────────
        all_results: list[dict] = []
        was_stopped = False
        for idx, params in enumerate(combos):
            # ── Check for stop request ────────────────────────────────
            if stop_event.is_set():
                was_stopped = True
                logger.info("⛔ Optimisation %s stopped by user at combo %d/%d.",
                            result.opt_id, idx, len(combos))
                break

            try:
                metrics = self._run_single(
                    strategy_name=result.strategy_name,
                    params=params,
                    df=df,
                    symbol=result.symbol,
                    point=point,
                    digits=digits,
                    pip_value_per_lot=pip_value_per_lot,
                    lot_size=result.lot_size,
                    initial_balance=result.initial_balance,
                    default_sl=params.get("sl_pips", self.config.default_sl_pips),
                    default_tp=params.get("tp_pips", self.config.default_tp_pips),
                    max_open_trades=result.max_open_trades,
                )
                # Filter combos with too few trades
                if metrics["total_trades"] >= result.min_trades:
                    metrics["params"] = params
                    all_results.append(metrics)
            except Exception as exc:
                logger.debug("Combo %d failed: %s", idx, exc)

            result.tested_combos = idx + 1
            result.progress = (idx + 1) / len(combos)

            # ── ETA calculation ───────────────────────────────────────
            elapsed = _time.time() - result.start_time
            result.elapsed_secs = elapsed
            if result.progress > 0:
                result.eta_secs = max(0, (elapsed / result.progress) - elapsed)

        # ── 5. Rank by target metric ─────────────────────────────────
        metric_key = result.target_metric
        # Higher is better for most metrics; for drawdown, lower is better
        reverse = metric_key not in ("max_drawdown", "max_drawdown_pct", "avg_loss_pips")

        all_results.sort(
            key=lambda r: r.get(metric_key, 0) or 0,
            reverse=reverse,
        )

        result.results = all_results[:100]  # keep top 100
        if all_results:
            best = all_results[0]
            result.best_params = best["params"]
            result.best_metric_value = best.get(metric_key, 0)

        result.status = "stopped" if was_stopped else "completed"
        result.progress = 1.0 if not was_stopped else result.progress

        logger.info(
            "⚡ Optimisation DONE: %s | %d combos tested | Best %s: %.2f | "
            "Best params: %s",
            result.opt_id, len(all_results), metric_key,
            result.best_metric_value, result.best_params,
        )

    # ── Combo Generator ──────────────────────────────────────────────

    def _generate_combos(
        self, strategy_name: str, mode: str, optimize_sltp: bool,
    ) -> list[dict]:
        """Generate all valid parameter combinations for a strategy."""
        grid = PARAM_GRIDS[strategy_name]
        validator = VALIDATORS.get(strategy_name, lambda p: True)

        # Build lists of (param_name, values) pairs
        param_names = []
        param_values = []
        for name, mode_dict in grid.items():
            values = mode_dict.get(mode, mode_dict["normal"])
            param_names.append(name)
            param_values.append(values)

        # Add SL/TP if requested (skip any the strategy grid already sweeps,
        # e.g. meanrev defines its own reversion-appropriate sl/tp ranges).
        if optimize_sltp:
            for sltp_name, mode_dict in SLTP_GRIDS.items():
                if sltp_name in param_names:
                    continue
                values = mode_dict.get(mode, mode_dict["normal"])
                param_names.append(sltp_name)
                param_values.append(values)

        # Generate cartesian product and filter
        combos = []
        for vals in product(*param_values):
            params = dict(zip(param_names, vals))
            if validator(params):
                combos.append(params)

        return combos

    # ── Single Backtest (lightweight) ────────────────────────────────

    def _run_single(
        self,
        strategy_name: str,
        params: dict,
        df: pd.DataFrame,
        symbol: str,
        point: float,
        digits: int,
        pip_value_per_lot: float,
        lot_size: float,
        initial_balance: float,
        default_sl: float,
        default_tp: float,
        max_open_trades: int = 5,
    ) -> dict:
        """Run a single backtest with given params. Returns metrics dict."""
        # Separate SL/TP from indicator params
        indicator_params = {k: v for k, v in params.items() if k not in ("sl_pips", "tp_pips")}
        sl_pips = params.get("sl_pips", default_sl)
        tp_pips = params.get("tp_pips", default_tp)

        # Instantiate strategy with these params
        strategy = get_strategy(strategy_name, sl_pips=sl_pips, tp_pips=tp_pips, **indicator_params)
        # Honour an optional time-stop (e.g. mean reversion's max_hold_bars) so
        # that parameter is actually swept rather than ignored.
        max_hold_bars = int(getattr(strategy, "max_hold_bars", 0) or 0)

        total_bars = len(df)
        WARMUP = min(200, total_bars // 3)
        equity = initial_balance
        open_trades: list[dict] = []
        closed_trades: list[BacktestTrade] = []
        pending_signal = None
        trade_counter = 0

        for i in range(WARMUP, total_bars):
            bar = df.iloc[i]
            bar_time = bar["time"].isoformat()

            # Execute pending signal at this bar's open
            if pending_signal is not None:
                sig = pending_signal
                pending_signal = None

                if len(open_trades) < max_open_trades:
                    entry_price = bar["open"]
                    trade_counter += 1

                    if sig.action == SignalAction.BUY:
                        sl_p = round(entry_price - pips_to_price(sl_pips, symbol, point), digits)
                        tp_p = round(entry_price + pips_to_price(tp_pips, symbol, point), digits)
                        direction = "BUY"
                    else:
                        sl_p = round(entry_price + pips_to_price(sl_pips, symbol, point), digits)
                        tp_p = round(entry_price - pips_to_price(tp_pips, symbol, point), digits)
                        direction = "SELL"

                    open_trades.append({
                        "direction": direction,
                        "entry_price": entry_price,
                        "entry_time": bar_time,
                        "entry_index": i,
                        "sl_price": sl_p, "tp_price": tp_p,
                    })

            # Check SL/TP for open trades
            still_open = []
            for trade in open_trades:
                hit = self._check_sl_tp(trade, bar)
                if not hit and max_hold_bars and (i - trade["entry_index"]) >= max_hold_bars:
                    hit = (bar["close"], "TIME")
                if hit:
                    exit_price, exit_reason = hit
                    pnl_pips = self._calc_pnl_pips(
                        trade["direction"], trade["entry_price"],
                        exit_price, symbol, point,
                    )
                    pnl_money = pnl_pips * pip_value_per_lot * lot_size
                    equity += pnl_money
                    closed_trades.append(BacktestTrade(
                        pnl_pips=round(pnl_pips, 1),
                        pnl_money=round(pnl_money, 2),
                        exit_reason=exit_reason,
                        entry_time=trade["entry_time"],
                        exit_time=bar_time,
                    ))
                else:
                    still_open.append(trade)
            open_trades = still_open

            # Run strategy
            window = df.iloc[max(0, i - 499):i + 1].copy().reset_index(drop=True)
            try:
                signal = strategy.analyze(window, symbol)
            except Exception:
                signal = None

            if signal and signal.action in (SignalAction.BUY, SignalAction.SELL):
                pending_signal = signal

        # Close remaining
        if open_trades:
            final = df.iloc[-1]
            for trade in open_trades:
                pnl_pips = self._calc_pnl_pips(
                    trade["direction"], trade["entry_price"],
                    final["close"], symbol, point,
                )
                pnl_money = pnl_pips * pip_value_per_lot * lot_size
                equity += pnl_money
                closed_trades.append(BacktestTrade(
                    pnl_pips=round(pnl_pips, 1),
                    pnl_money=round(pnl_money, 2),
                    exit_reason="END",
                    entry_time=trade["entry_time"],
                    exit_time=final["time"].isoformat(),
                ))

        # Compute metrics
        return self._compute_metrics(closed_trades, initial_balance, equity)

    @staticmethod
    def _check_sl_tp(trade: dict, bar: pd.Series) -> tuple[float, str] | None:
        sl, tp = trade["sl_price"], trade["tp_price"]
        high, low = bar["high"], bar["low"]
        if trade["direction"] == "BUY":
            sl_hit = low <= sl
            tp_hit = high >= tp
            if sl_hit and tp_hit:
                return (sl, "SL")
            if sl_hit:
                return (sl, "SL")
            if tp_hit:
                return (tp, "TP")
        else:
            sl_hit = high >= sl
            tp_hit = low <= tp
            if sl_hit and tp_hit:
                return (sl, "SL")
            if sl_hit:
                return (sl, "SL")
            if tp_hit:
                return (tp, "TP")
        return None

    @staticmethod
    def _calc_pnl_pips(direction, entry, exit_p, symbol, point):
        delta = (exit_p - entry) if direction == "BUY" else (entry - exit_p)
        return price_to_pips(delta, symbol, point)

    @staticmethod
    def _compute_metrics(
        trades: list[BacktestTrade],
        initial_balance: float,
        final_equity: float,
    ) -> dict:
        """Compute performance metrics from a list of trades."""
        total = len(trades)
        if total == 0:
            return {
                "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl_pips": 0, "total_pnl_money": 0,
                "profit_factor": 0, "max_drawdown_pct": 0,
                "sharpe_ratio": 0, "avg_win_pips": 0, "avg_loss_pips": 0,
            }

        wins = sum(1 for t in trades if t.pnl_pips > 0)
        losses = total - wins
        win_rate = wins / total * 100

        total_pnl_pips = sum(t.pnl_pips for t in trades)
        total_pnl_money = round(final_equity - initial_balance, 2)

        gross_profit = sum(t.pnl_money for t in trades if t.pnl_money > 0)
        gross_loss = abs(sum(t.pnl_money for t in trades if t.pnl_money < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            999.99 if gross_profit > 0 else 0
        )

        # Average win/loss
        w_pips = [t.pnl_pips for t in trades if t.pnl_pips > 0]
        l_pips = [t.pnl_pips for t in trades if t.pnl_pips <= 0]
        avg_win = (sum(w_pips) / len(w_pips)) if w_pips else 0
        avg_loss = (sum(l_pips) / len(l_pips)) if l_pips else 0

        # Max drawdown
        equity = initial_balance
        peak = equity
        max_dd_pct = 0.0
        for t in trades:
            equity += t.pnl_money
            if equity > peak:
                peak = equity
            dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        # Sharpe
        sharpe = 0.0
        if len(trades) > 1:
            rets = [t.pnl_money for t in trades]
            avg_r = sum(rets) / len(rets)
            var = sum((r - avg_r) ** 2 for r in rets) / (len(rets) - 1)
            std = math.sqrt(var) if var > 0 else 0
            sharpe = (avg_r / std * math.sqrt(252)) if std > 0 else 0

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl_pips": round(total_pnl_pips, 1),
            "total_pnl_money": round(total_pnl_money, 2),
            "profit_factor": round(min(profit_factor, 999.99), 2),
            "max_drawdown_pct": round(max_dd_pct, 1),
            "sharpe_ratio": round(sharpe, 2),
            "avg_win_pips": round(avg_win, 1),
            "avg_loss_pips": round(avg_loss, 1),
        }


# ======================================================================
#  ADVANCED OPTIMISATION (OctoBot-inspired)
# ======================================================================

import copy
import json
import multiprocessing
import random
from enum import Enum
from pathlib import Path


class OptimizationMode(Enum):
    """Available optimisation strategies."""
    BRUTE_FORCE = "brute_force"        # Existing exhaustive search
    GENETIC = "genetic"                # Genetic algorithm
    PARALLEL_BRUTE = "parallel_brute"  # Brute force + multiprocessing


@dataclass
class GeneticConfig:
    """
    Hyper-parameters for the genetic optimiser.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation (default 50).
    generations : int
        Maximum generations to evolve (default 30).
    mutation_rate : float
        Probability of mutation per gene (default 0.15).
    crossover_rate : float
        Probability of crossover (default 0.70).
    elite_pct : float
        Top fraction that survive unchanged (default 0.10).
    tournament_size : int
        Tournament selection participants (default 3).
    target_fitness : float | None
        Stop early if fitness exceeds this (default None).
    """
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.15
    crossover_rate: float = 0.70
    elite_pct: float = 0.10
    tournament_size: int = 3
    target_fitness: float | None = None


class GeneticOptimizer:
    """
    Genetic algorithm optimiser for strategy parameters.

    Evolves a population of parameter sets using selection, crossover,
    and mutation to find high-fitness configurations without exhaustive
    search.

    Parameters
    ----------
    config : GeneticConfig
        GA hyper-parameters.
    param_ranges : dict
        ``{param_name: [value1, value2, ...]}`` — discrete search space.
    backtest_fn : callable
        ``fn(params_dict) -> dict`` that runs a backtest and returns
        metrics (must include ``target_metric``).
    target_metric : str
        Key in the backtest result to maximise (default ``'profit_factor'``).
    """

    def __init__(
        self,
        config: GeneticConfig,
        param_ranges: dict[str, list],
        backtest_fn,
        target_metric: str = "profit_factor",
    ) -> None:
        self.config = config
        self.param_ranges = param_ranges
        self.backtest_fn = backtest_fn
        self.target_metric = target_metric
        self._logger = logging.getLogger("GeneticOptimizer")

    def run(self, callback=None) -> list[dict]:
        """
        Run the genetic algorithm.

        Parameters
        ----------
        callback : callable, optional
            ``callback(progress_pct, best_fitness, generation)``

        Returns
        -------
        list[dict]
            Top results sorted by fitness (descending).
        """
        population = self._init_population()
        best_ever: dict | None = None
        best_fitness_ever = float("-inf")
        stagnation = 0

        for gen in range(self.config.generations):
            # Evaluate fitness
            fitness = []
            for individual in population:
                try:
                    result = self.backtest_fn(individual)
                    f = result.get(self.target_metric, 0.0)
                except Exception:
                    f = 0.0
                fitness.append(f)

            # Track best
            gen_best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
            gen_best_fitness = fitness[gen_best_idx]

            if gen_best_fitness > best_fitness_ever:
                best_fitness_ever = gen_best_fitness
                best_ever = copy.deepcopy(population[gen_best_idx])
                stagnation = 0
            else:
                stagnation += 1

            self._logger.info(
                "Gen %d/%d: best=%.4f (ever=%.4f)",
                gen + 1, self.config.generations,
                gen_best_fitness, best_fitness_ever,
            )

            if callback:
                pct = (gen + 1) / self.config.generations * 100
                callback(pct, best_fitness_ever, gen + 1)

            # Early stopping
            if (
                self.config.target_fitness is not None
                and best_fitness_ever >= self.config.target_fitness
            ):
                self._logger.info("Target fitness reached. Stopping.")
                break

            if stagnation >= 5:
                self._logger.info("No improvement for 5 gens. Stopping.")
                break

            # Evolve next generation
            elite = self._get_elite(population, fitness)
            new_pop = list(elite)

            while len(new_pop) < self.config.population_size:
                p1 = self._tournament_select(population, fitness)
                p2 = self._tournament_select(population, fitness)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)

            population = new_pop

        # Final evaluation
        results = []
        for individual in population:
            try:
                result = self.backtest_fn(individual)
                result["params"] = individual
                results.append(result)
            except Exception:
                pass

        results.sort(key=lambda r: r.get(self.target_metric, 0), reverse=True)
        return results[:20]

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------

    def _init_population(self) -> list[dict]:
        """Create random initial population."""
        pop = []
        for _ in range(self.config.population_size):
            individual = {
                param: random.choice(values)
                for param, values in self.param_ranges.items()
            }
            pop.append(individual)
        return pop

    def _tournament_select(
        self, population: list[dict], fitness: list[float],
    ) -> dict:
        """Tournament selection."""
        indices = random.sample(
            range(len(population)),
            min(self.config.tournament_size, len(population)),
        )
        best_idx = max(indices, key=lambda i: fitness[i])
        return copy.deepcopy(population[best_idx])

    def _crossover(self, p1: dict, p2: dict) -> dict:
        """Uniform crossover."""
        if random.random() > self.config.crossover_rate:
            return copy.deepcopy(p1)

        child = {}
        for param in self.param_ranges:
            child[param] = p1[param] if random.random() < 0.5 else p2[param]
        return child

    def _mutate(self, individual: dict) -> dict:
        """Random mutation of parameters."""
        for param, values in self.param_ranges.items():
            if random.random() < self.config.mutation_rate:
                individual[param] = random.choice(values)
        return individual

    def _get_elite(
        self, population: list[dict], fitness: list[float],
    ) -> list[dict]:
        """Preserve top performers."""
        n = max(1, int(len(population) * self.config.elite_pct))
        ranked = sorted(
            range(len(population)),
            key=lambda i: fitness[i],
            reverse=True,
        )
        return [copy.deepcopy(population[i]) for i in ranked[:n]]

