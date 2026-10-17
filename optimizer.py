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
# The simulation kernel lives in backtest_core (pure, no MT5, no cloud
# dependency). The sweep previously imported it from cloud.worker, so the
# local optimiser could not run without the cloud package installed.
from backtest_core import (
    calc_pnl_pips as _backtest_calc_pnl_pips,
    check_sl_tp as _backtest_check_sl_tp,
    compute_metrics as _backtest_compute_metrics,
    pip_value_per_lot as _pip_value_per_lot,
    run_single as _backtest_run_single,
)

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
    # Grids are kept deliberately COARSE: these strategies recompute several
    # indicators per bar, so each backtest is slow (~tens of seconds on 6mo
    # M15). Quick ≈ 8 combos, Normal ≈ 30-40, Deep ≈ 100-160. Prefer a coarse
    # sweep + walk-forward over a dense in-sample grid (which just overfits).
    "meanrev": {
        "z_threshold": {
            "quick":  [2.0, 2.5],
            "normal": [2.0, 2.5],
            "deep":   [1.8, 2.0, 2.3],
        },
        "z_max": {
            "quick":  [3.5],
            "normal": [3.5],
            "deep":   [3.0, 3.5],
        },
        "max_hold_bars": {
            "quick":  [16, 32],
            "normal": [16, 24, 32],
            "deep":   [16, 24, 32, 48],
        },
        "sl_pips": {
            "quick":  [40],
            "normal": [30, 50],
            "deep":   [30, 50],
        },
        "tp_pips": {
            "quick":  [15, 25],
            "normal": [15, 25, 35],
            "deep":   [12, 18, 25],
        },
    },
    # The four below follow the standard pattern: structural params here,
    # SL/TP via the "optimise SL/TP" toggle (the trend-tuned global ranges
    # suit trend/breakout/reversal, unlike mean reversion).
    "donchian": {
        "channel_period": {
            "quick":  [34, 55],
            "normal": [34, 55, 89],
            "deep":   [21, 34, 55, 89, 144],
        },
        "trend_period": {
            "quick":  [100, 200],
            "normal": [50, 100, 200],
            "deep":   [50, 100, 150, 200],
        },
        "min_adx": {
            "quick":  [15, 25],
            "normal": [15, 20, 25],
            "deep":   [12, 18, 25],
        },
    },
    "orb": {
        "or_minutes": {
            "quick":  [15, 30],
            "normal": [15, 30, 45],
            "deep":   [15, 30, 45, 60],
        },
        "buffer_atr": {
            "quick":  [0.1, 0.25],
            "normal": [0.1, 0.25, 0.4],
            "deep":   [0.0, 0.1, 0.25, 0.4, 0.6],
        },
        "squeeze_factor": {
            "quick":  [1.0, 1.5],
            "normal": [1.0, 1.5, 2.0],
            "deep":   [0.8, 1.0, 1.5, 2.0],
        },
    },
    "sweep": {
        "swing_lookback": {
            "quick":  [30, 50],
            "normal": [30, 50, 80],
            "deep":   [20, 30, 50, 80, 100],
        },
        "wick_ratio": {
            "quick":  [0.5, 0.6],
            "normal": [0.5, 0.6],
            "deep":   [0.45, 0.55, 0.65],
        },
        "range_mult": {
            "quick":  [1.3],
            "normal": [1.2, 1.5],
            "deep":   [1.2, 1.5, 2.0],
        },
        "max_adx": {
            "quick":  [20, 25],
            "normal": [20, 25, 30],
            "deep":   [20, 25, 30],
        },
    },
    "regime": {
        "fast_ema": {
            "quick":  [10, 20],
            "normal": [10, 20],
            "deep":   [10, 15, 20],
        },
        "slow_ema": {
            "quick":  [50, 100],
            "normal": [50, 100],
            "deep":   [40, 75, 100],
        },
        "trend_adx": {
            "quick":  [25],
            "normal": [22, 28],
            "deep":   [22, 25, 28],
        },
        "range_adx": {
            "quick":  [18],
            "normal": [15, 18],
            "deep":   [12, 15, 18],
        },
        "z_threshold": {
            "quick":  [2.0, 2.5],
            "normal": [2.0, 2.5],
            "deep":   [2.0, 2.5],
        },
    },
    # Micro-Trend Scalper. Like meanrev, sl_pips/tp_pips are swept HERE
    # directly (scalping-tight ranges) — so optimise WITHOUT the global
    # "optimise SL/TP" toggle. ``tp_pips`` is the small profit target and
    # models the live Quick Exit (close at +N pips), so sweeping it finds the
    # best scalp-exit distance. Grids kept coarse (≈8 / 48 / 162 combos) —
    # the strategy recomputes EMA+ADX+ATR every bar, so each backtest is slow.
    "scalper": {
        "ema_fast": {
            "quick":  [5],
            "normal": [3, 5, 8],
            "deep":   [3, 5, 8],
        },
        "ema_slow": {
            "quick":  [20],
            "normal": [20],
            "deep":   [20, 30],
        },
        # Higher ADX floor = fewer but cleaner breakouts (better win rate).
        "adx_min": {
            "quick":  [18, 25],
            "normal": [20, 28],
            "deep":   [15, 22, 30],
        },
        "breakout_lookback": {
            "quick":  [8],
            "normal": [6, 10],
            "deep":   [6, 12],
        },
        # SL and TP BOTH swept so the optimiser can find a positive
        # reward:risk (the -2010-pip backtest used TP6<SL8, which needs a
        # 57% win rate to break even). tp_pips is also the live Quick-Exit
        # target. Note: backtest does NOT model spread, so favour combos with
        # a comfortable margin, not razor-thin edges.
        "sl_pips": {
            "quick":  [5, 8],
            "normal": [5, 8],
            "deep":   [4, 6, 8],
        },
        "tp_pips": {
            "quick":  [6, 10],
            "normal": [6, 10],
            "deep":   [6, 10, 15],
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

def _validate_scalper(params: dict) -> bool:
    return (
        params.get("ema_fast", 5) < params.get("ema_slow", 20)
        and params.get("breakout_lookback", 8) >= 2
        and params.get("sl_pips", 8) > 0
        and params.get("tp_pips", 12) > 0
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
    "scalper": _validate_scalper,
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
#  Combo Generator (module-level, used by both local & cloud optimiser)
# =====================================================================

def generate_combos(
    strategy_name: str, mode: str, optimize_sltp: bool,
) -> list[dict]:
    """Generate all valid parameter combinations for a strategy.

    Module-level so it can be imported by the cloud orchestrator without
    instantiating StrategyOptimizer.
    """
    grid = PARAM_GRIDS[strategy_name]
    validator = VALIDATORS.get(strategy_name, lambda p: True)

    param_names = []
    param_values = []
    for name, mode_dict in grid.items():
        values = mode_dict.get(mode, mode_dict["normal"])
        param_names.append(name)
        param_values.append(values)

    if optimize_sltp:
        for sltp_name, mode_dict in SLTP_GRIDS.items():
            if sltp_name in param_names:
                continue
            values = mode_dict.get(mode, mode_dict["normal"])
            param_names.append(sltp_name)
            param_values.append(values)

    combos = []
    for vals in product(*param_values):
        params = dict(zip(param_names, vals))
        if validator(params):
            combos.append(params)

    return combos


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
        # One shared pip definition — this used to special-case JPY with
        # its own formula, which disagreed with the live engine.
        pip_value_per_lot = _pip_value_per_lot(
            result.symbol, point,
            contract_size=float(
                getattr(sym_info, "trade_contract_size", 0) or 0
            ),
            tick_value=getattr(sym_info, "trade_tick_value", 0) or 0,
            tick_size=getattr(sym_info, "trade_tick_size", 0) or 0,
        )

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
        return generate_combos(strategy_name, mode, optimize_sltp)

    # ── Single Backtest — delegates to extracted pure function ───────

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
        return _backtest_run_single(
            strategy_name=strategy_name, params=params, df=df,
            symbol=symbol, point=point, digits=digits,
            pip_value_per_lot=pip_value_per_lot, lot_size=lot_size,
            initial_balance=initial_balance, default_sl=default_sl,
            default_tp=default_tp, max_open_trades=max_open_trades,
        )


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

