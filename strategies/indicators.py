"""
Shared Technical Indicators
============================

Pure, dependency-light indicator helpers shared by the newer strategies
(``donchian_momentum``, ``mean_reversion``, ``opening_range``,
``liquidity_sweep``, ``regime_switch``).

Design notes
------------
* Every function operates on the OHLCV ``DataFrame`` produced by
  :mod:`market_data` — columns ``time, open, high, low, close, volume``,
  ordered oldest → newest — or on a ``close`` price ``Series``.
* Functions return ``pandas`` objects (Series) unless documented otherwise,
  so callers can take ``.iloc[-1]`` for the latest value.
* All values are in **price units**, never pips.  Strategies deliberately
  do *not* convert price→pips internally (they have no ``symbol_info().point``);
  stop/target distances stay configurable and the central risk engine applies
  ATR/volatility sizing.  ATR here is used only for *relative* gating
  (e.g. "is this bar's range > 1.3× ATR?"), which needs no pip conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Moving averages ───────────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


# ── Oscillators ─────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (simple-average flavour, matches RSIStrategy)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ── Volatility / range ──────────────────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range. First bar falls back to (high − low)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ── Trend strength ──────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder).  >25 ≈ trending, <20 ≈ ranging."""
    high, low = df["high"], df["low"]
    up = high.diff()
    down = -low.diff()

    plus_dm = (((up > down) & (up > 0)) * up).fillna(0.0)
    minus_dm = (((down > up) & (down > 0)) * down).fillna(0.0)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0, np.nan)

    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


# ── Bollinger / dispersion ──────────────────────────────────────────────────

def bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Return (mid, upper, lower) Bollinger Bands."""
    mid = series.rolling(window=period).mean()
    sd = series.rolling(window=period).std()
    upper = mid + std_dev * sd
    lower = mid - std_dev * sd
    return mid, upper, lower


def bollinger_bandwidth(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """(upper − lower) / mid — a normalised squeeze/expansion measure."""
    mid, upper, lower = bollinger(series, period, std_dev)
    return (upper - lower) / mid.replace(0, np.nan)


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score of price vs its own mean (displacement measure)."""
    mean = series.rolling(window=period).mean()
    sd = series.rolling(window=period).std().replace(0, np.nan)
    return (series - mean) / sd


# ── Channels ────────────────────────────────────────────────────────────────

def donchian(df: pd.DataFrame, period: int = 55):
    """Return (upper, lower) Donchian channel Series over *period* bars."""
    upper = df["high"].rolling(window=period).max()
    lower = df["low"].rolling(window=period).min()
    return upper, lower


# ── Regime ──────────────────────────────────────────────────────────────────

def variance_ratio(series: pd.Series, lag: int = 5) -> float:
    """Lo–MacKinlay variance ratio of log-ish returns.

    VR > 1  → returns positively autocorrelated (trending).
    VR < 1  → returns mean-reverting.
    Returns 1.0 (neutral) when there is insufficient data.
    """
    returns = series.pct_change().dropna()
    if len(returns) < lag * 4:
        return 1.0
    var_1 = returns.var()
    if not var_1 or np.isnan(var_1):
        return 1.0
    var_k = returns.rolling(window=lag).sum().dropna().var()
    if np.isnan(var_k):
        return 1.0
    vr = var_k / (lag * var_1)
    return float(vr) if vr == vr else 1.0
