"""
validation.py — Runnable entry points for the robustness tools.

Three validation methods were fully implemented but unreachable:
:class:`optimizer.GeneticOptimizer`, :class:`backtester.WalkForwardAnalyzer`
and :class:`backtester.MonteCarloSimulator`.  Nothing called them.  The
only reference anywhere was ``bot.py``'s optimizer menu, which offered
them as options 1-3 and then *printed the import statements the user
would need to wire them up themselves*.

This module supplies the missing plumbing — fetch the bars, build the
``backtest_fn`` closure each tool expects, run it, summarise — so the
menu and the dashboard can offer them for real.

What each one answers
---------------------
**Genetic search** — the same question as the grid sweep, but it does not
enumerate the grid.  A deep grid on a three-parameter strategy is tens of
thousands of backtests; a genetic search samples a few hundred and
usually lands close to the optimum.  Use it when the grid is too large to
run.

**Walk-forward** — "do these parameters hold up out of sample?"  It
optimises on a window, validates on the window that follows, and slides
forward.  The grid sweep reports the best parameters *in hindsight* over
one period, which is the number most likely to be curve-fitted; the
robustness ratio here is how much of that performance survived on data
the optimiser never saw.

**Monte Carlo** — "how much of this equity curve was luck?"  It reshuffles
the realised trade sequence thousands of times.  The same set of trades
in a different order produces a different drawdown, and the 5th-percentile
outcome is a far more honest risk figure than the single historical path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import MetaTrader5 as mt5
import pandas as pd

from backtest_core import (
    pip_value_per_lot as _pip_value_per_lot,
    run_single as _run_backtest,
    spread_pips_from_symbol as _spread_pips_from_symbol,
)
from backtester import (
    MonteCarloResult,
    MonteCarloSimulator,
    WalkForwardAnalyzer,
    WalkForwardResult,
)
from config import Config, TIMEFRAME_MAP
from optimizer import (
    GeneticConfig,
    GeneticOptimizer,
    PARAM_GRIDS,
    generate_combos,
)

logger = logging.getLogger("validation")


# ---------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------

class MarketSlice:
    """Bars plus the symbol metadata a backtest needs, fetched once.

    Attributes
    ----------
    df : pandas.DataFrame
        OHLCV bars, columns ``time, open, high, low, close, volume``.
    symbol, point, digits, pip_value, spread_pips
        Symbol metadata resolved from the broker, including the
        per-trade spread cost every run here charges.
    """

    def __init__(self, df: pd.DataFrame, symbol: str, point: float,
                 digits: int, pip_value: float,
                 spread_pips: float = 0.0) -> None:
        self.df = df
        self.symbol = symbol
        self.point = point
        self.digits = digits
        self.pip_value = pip_value
        self.spread_pips = spread_pips

    def __len__(self) -> int:
        return len(self.df)


def fetch_slice(
    symbol: str, timeframe: str, date_from: str, date_to: str,
    connector: Any = None,
) -> MarketSlice:
    """Fetch bars and symbol metadata for a validation run.

    Parameters
    ----------
    symbol
        Canonical ticker; resolved to the broker's name when *connector*
        is supplied (brokers decorate symbols — ``GBPJPY`` → ``GBPJPY.r``).
    timeframe
        A :data:`config.TIMEFRAME_MAP` key such as ``"M5"``.
    date_from, date_to
        ISO dates bounding the sample.
    connector
        Optional :class:`mt5_connector.MT5Connector` for symbol resolution.

    Raises
    ------
    ValueError
        Unknown timeframe, unavailable symbol, or no bars in range.
    """
    mt5_tf = TIMEFRAME_MAP.get(timeframe)
    if mt5_tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    broker_symbol = symbol
    if connector is not None:
        resolved = connector.ensure_symbol(symbol)
        if resolved is None:
            raise ValueError(
                f"Symbol '{symbol}' is not available on the connected broker."
            )
        broker_symbol = resolved
    else:
        mt5.symbol_select(symbol, True)

    start = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)
    rates = mt5.copy_rates_range(broker_symbol, mt5_tf, start, end)
    if rates is None or len(rates) == 0:
        raise ValueError(
            f"No historical data for {broker_symbol} {timeframe} from "
            f"{date_from} to {date_to}. MT5 error: {mt5.last_error()}. "
            f"Open the symbol's chart on this timeframe and scroll back to "
            f"make the terminal download bars, then retry."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()

    sym_info = mt5.symbol_info(broker_symbol)
    if sym_info is None:
        raise ValueError(f"Cannot get symbol info for {broker_symbol}")

    pip_value = _pip_value_per_lot(
        broker_symbol, sym_info.point,
        contract_size=float(getattr(sym_info, "trade_contract_size", 0) or 0),
        tick_value=getattr(sym_info, "trade_tick_value", 0) or 0,
        tick_size=getattr(sym_info, "trade_tick_size", 0) or 0,
    )

    spread_pips = _spread_pips_from_symbol(
        getattr(sym_info, "spread", 0), broker_symbol, sym_info.point,
    )

    logger.info(
        "Fetched %d bars for %s %s (%s → %s); $/pip/lot=%.2f "
        "spread=%.2f pips.",
        len(df), broker_symbol, timeframe, date_from, date_to,
        pip_value, spread_pips,
    )
    return MarketSlice(
        df, broker_symbol, sym_info.point, sym_info.digits, pip_value,
        spread_pips=spread_pips,
    )


# ---------------------------------------------------------------------
# backtest_fn builders
# ---------------------------------------------------------------------

def make_backtest_fn(
    strategy_name: str, market: MarketSlice, config: Config,
    lot_size: float = 0.1, initial_balance: float = 10_000.0,
    max_open_trades: int = 5,
) -> Callable[[dict], dict]:
    """Return ``fn(params) -> metrics`` over this market slice.

    The shape :class:`optimizer.GeneticOptimizer` expects.
    """

    def _fn(params: dict) -> dict:
        return _run_backtest(
            strategy_name=strategy_name, params=params, df=market.df,
            symbol=market.symbol, point=market.point, digits=market.digits,
            pip_value_per_lot=market.pip_value, lot_size=lot_size,
            initial_balance=initial_balance,
            default_sl=config.default_sl_pips,
            default_tp=config.default_tp_pips,
            max_open_trades=max_open_trades,
            spread_pips=market.spread_pips,
        )

    return _fn


def make_windowed_backtest_fn(
    strategy_name: str, market: MarketSlice, config: Config,
    lot_size: float = 0.1, initial_balance: float = 10_000.0,
    max_open_trades: int = 5,
) -> Callable[[dict, pd.DataFrame], dict]:
    """Return ``fn(params, window_df) -> metrics``.

    The shape :meth:`backtester.WalkForwardAnalyzer.run` expects, which
    re-slices the data itself for each in/out-of-sample window.
    """

    def _fn(params: dict, window_df: pd.DataFrame) -> dict:
        return _run_backtest(
            strategy_name=strategy_name, params=params, df=window_df,
            symbol=market.symbol, point=market.point, digits=market.digits,
            pip_value_per_lot=market.pip_value, lot_size=lot_size,
            initial_balance=initial_balance,
            default_sl=config.default_sl_pips,
            default_tp=config.default_tp_pips,
            max_open_trades=max_open_trades,
            spread_pips=market.spread_pips,
        )

    return _fn


# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------

def run_genetic(
    strategy_name: str, symbol: str, timeframe: str,
    date_from: str, date_to: str, *,
    config: Config, connector: Any = None, mode: str = "normal",
    target_metric: str = "profit_factor",
    population_size: int = 50, generations: int = 30,
    optimize_sltp: bool = False,
    callback: Optional[Callable[[float, float, int], None]] = None,
    top_n: int = 10,
) -> dict:
    """Evolve strategy parameters instead of enumerating the grid.

    Returns
    -------
    dict
        ``{"method", "strategy", "symbol", "timeframe", "target_metric",
        "search_space", "evaluations", "results"}`` — ``results`` is the
        top *top_n* parameter sets, best first.

    Raises
    ------
    ValueError
        Unknown strategy, or no valid parameter combinations.
    """
    if strategy_name not in PARAM_GRIDS:
        raise ValueError(
            f"No parameter grid for '{strategy_name}'. "
            f"Available: {sorted(PARAM_GRIDS)}"
        )

    market = fetch_slice(symbol, timeframe, date_from, date_to, connector)

    # The GA samples one value per parameter, so it needs the per-parameter
    # value lists rather than the cartesian product.
    grid = PARAM_GRIDS[strategy_name]
    param_ranges = {
        name: mode_dict.get(mode, mode_dict["normal"])
        for name, mode_dict in grid.items()
    }
    if optimize_sltp:
        from optimizer import SLTP_GRIDS
        for name, mode_dict in SLTP_GRIDS.items():
            param_ranges.setdefault(
                name, mode_dict.get(mode, mode_dict["normal"]),
            )

    search_space = 1
    for values in param_ranges.values():
        search_space *= max(1, len(values))

    ga = GeneticOptimizer(
        config=GeneticConfig(
            population_size=population_size, generations=generations,
        ),
        param_ranges=param_ranges,
        backtest_fn=make_backtest_fn(strategy_name, market, config),
        target_metric=target_metric,
    )

    logger.info(
        "Genetic search: %s on %s %s — %d individuals x %d generations "
        "(%d evaluations) over a %d-combination space.",
        strategy_name, market.symbol, timeframe,
        population_size, generations, population_size * generations,
        search_space,
    )
    results = ga.run(callback=callback)

    return {
        "method": "genetic",
        "strategy": strategy_name,
        "symbol": market.symbol,
        "timeframe": timeframe,
        "target_metric": target_metric,
        "search_space": search_space,
        "evaluations": population_size * generations,
        "bars": len(market),
        "results": results[:top_n],
    }


def run_walk_forward(
    strategy_name: str, symbol: str, timeframe: str,
    date_from: str, date_to: str, *,
    config: Config, connector: Any = None, mode: str = "quick",
    target_metric: str = "profit_factor",
    in_sample_bars: int = 500, out_of_sample_bars: int = 200,
    step_bars: int = 200, optimize_sltp: bool = False,
    max_combos: int = 400,
) -> WalkForwardResult:
    """Optimise on a rolling window, validate on the window after it.

    A grid sweep reports the best parameters in hindsight over one
    period — the number most likely to be curve-fitted.  This reports
    how much of that performance survived on data the optimiser never
    saw: :attr:`WalkForwardResult.robustness_ratio` is out-of-sample
    divided by in-sample, and ``is_robust`` is that ratio above 0.5.

    Parameters
    ----------
    mode
        Grid density (``"quick"`` by default — the sweep re-runs for
        every window, so a deep grid here is very slow).
    max_combos
        Hard ceiling on the grid. Walk-forward cost is
        ``combos × windows``, which is how this becomes accidentally
        enormous; combinations beyond the ceiling are dropped.

    Raises
    ------
    ValueError
        Unknown strategy, no valid combinations, or too few bars for one
        in-sample plus out-of-sample window.
    """
    if strategy_name not in PARAM_GRIDS:
        raise ValueError(
            f"No parameter grid for '{strategy_name}'. "
            f"Available: {sorted(PARAM_GRIDS)}"
        )

    market = fetch_slice(symbol, timeframe, date_from, date_to, connector)
    needed = in_sample_bars + out_of_sample_bars
    if len(market) < needed:
        raise ValueError(
            f"Walk-forward needs at least {needed} bars "
            f"({in_sample_bars} in-sample + {out_of_sample_bars} "
            f"out-of-sample); the range returned {len(market)}. Widen the "
            f"dates or shrink the windows."
        )

    combos = generate_combos(strategy_name, mode, optimize_sltp)
    if not combos:
        raise ValueError(
            f"No valid parameter combinations for '{strategy_name}'."
        )
    if len(combos) > max_combos:
        logger.warning(
            "Grid has %d combinations; capping at %d for walk-forward "
            "(cost is combos x windows).", len(combos), max_combos,
        )
        combos = combos[:max_combos]

    analyzer = WalkForwardAnalyzer(
        in_sample_bars=in_sample_bars,
        out_of_sample_bars=out_of_sample_bars,
        step_bars=step_bars,
    )
    logger.info(
        "Walk-forward: %s on %s %s — %d combos, %d bars, "
        "IS=%d/OOS=%d/step=%d.",
        strategy_name, market.symbol, timeframe, len(combos),
        len(market), in_sample_bars, out_of_sample_bars, step_bars,
    )
    return analyzer.run(
        backtest_fn=make_windowed_backtest_fn(strategy_name, market, config),
        param_combos=combos,
        data_df=market.df,
        target_metric=target_metric,
    )


def run_monte_carlo(
    trades: list[dict], *, initial_balance: float = 10_000.0,
    num_simulations: int = 1000, confidence_level: float = 0.95,
) -> MonteCarloResult:
    """Reshuffle a realised trade sequence to separate skill from luck.

    The same trades in a different order produce a different drawdown, so
    the historical equity curve is one sample from a distribution.  This
    draws *num_simulations* of them and reports the percentiles — the
    5th-percentile return and worst-case drawdown are the honest risk
    figures to size from.

    Parameters
    ----------
    trades
        Closed trades; each needs a ``pnl_money`` key.  Rows from
        ``BacktestDB.get_trades`` or the live trade history both work.

    Raises
    ------
    ValueError
        Fewer than two trades carry a usable P&L — there is nothing to
        reshuffle.
    """
    usable = [
        {"pnl_money": float(t["pnl_money"])}
        for t in trades
        if t.get("pnl_money") is not None
    ]
    if len(usable) < 2:
        raise ValueError(
            f"Monte Carlo needs at least 2 closed trades with a P&L; "
            f"got {len(usable)}."
        )

    logger.info(
        "Monte Carlo: %d simulations over %d trades.",
        num_simulations, len(usable),
    )
    return MonteCarloSimulator(
        num_simulations=num_simulations,
        confidence_level=confidence_level,
    ).run(trades=usable, initial_balance=initial_balance)
