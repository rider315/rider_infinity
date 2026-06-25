"""
Strategy Registry — MT5 Forex Trading Bot
==========================================

Central registry for all trading strategies. Provides discovery,
instantiation, and listing of available strategies.

Usage:
    from strategies import get_strategy, list_strategies

    strategy = get_strategy("ema", fast_period=9, slow_period=21)
    available = list_strategies()
"""

from .ema_crossover import EMACrossoverStrategy
from .rsi_strategy import RSIStrategy
from .bollinger_strategy import BollingerStrategy
from .macd_strategy import MACDStrategy
from .news_strategy import NewsStrategy
from .grid_strategy import GridStrategy
from .dca_strategy import DCAStrategy
from .dsl_strategy import DSLStrategy
from .donchian_momentum_strategy import DonchianMomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .opening_range_strategy import OpeningRangeBreakoutStrategy
from .liquidity_sweep_strategy import LiquiditySweepStrategy
from .regime_switch_strategy import RegimeSwitchStrategy
from .scalper_strategy import ScalperStrategy

STRATEGY_REGISTRY: dict[str, tuple[type, str]] = {
    "ema": (EMACrossoverStrategy, "EMA Crossover — Trend Following"),
    "rsi": (RSIStrategy, "RSI — Mean Reversion"),
    "bollinger": (BollingerStrategy, "Bollinger Bands — Breakout/Squeeze"),
    "macd": (MACDStrategy, "MACD — Momentum"),
    "news": (NewsStrategy, "News Intelligence — AI Sentiment"),
    "grid": (GridStrategy, "Grid Trading — Volatility Harvesting"),
    "dca": (DCAStrategy, "DCA — Dollar Cost Averaging"),
    "dsl": (DSLStrategy, "DSL — Custom Rule Engine"),
    "donchian": (DonchianMomentumStrategy, "Donchian Momentum — CTA Trend Following"),
    "meanrev": (MeanReversionStrategy, "Statistical Mean Reversion — Session Fade"),
    "orb": (OpeningRangeBreakoutStrategy, "Opening Range Breakout — Session Volatility"),
    "sweep": (LiquiditySweepStrategy, "Liquidity Sweep — Stop-Run Reversal"),
    "regime": (RegimeSwitchStrategy, "Regime Switch — Adaptive Trend/Range"),
    "scalper": (ScalperStrategy, "Micro-Trend Scalper — Fast Breakout (pair with Quick Exit)"),
}



def get_strategy(name: str, **kwargs):
    """
    Instantiate a strategy by its registry key.

    Parameters
    ----------
    name : str
        Registry key (e.g. "ema", "rsi", "bollinger", "macd").
    **kwargs
        Forwarded to the strategy constructor.

    Returns
    -------
    BaseStrategy
        An initialised strategy instance.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    cls, _ = STRATEGY_REGISTRY[name]
    return cls(**kwargs)


def list_strategies() -> list[tuple[str, str]]:
    """
    Return a list of ``(key, description)`` tuples for every
    registered strategy.
    """
    return [(key, desc) for key, (_, desc) in STRATEGY_REGISTRY.items()]
