"""
market_data.py - Market data retrieval layer for the MT5 Forex Trading Bot.

Provides the ``MarketDataFeed`` class which wraps MetaTrader5 market-data
calls (candles, ticks, symbol info) and returns clean Python/Pandas objects
with comprehensive error handling and logging.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

logger = logging.getLogger("market_data")


class MarketDataFeed:
    """Fetches live market data from the MT5 terminal.

    All methods are stateless – the class simply provides a clean
    interface over the raw ``MetaTrader5`` functions, converting results
    into Pandas DataFrames or plain dictionaries and logging any errors.

    Usage::

        feed = MarketDataFeed()
        df   = feed.get_candles("EURUSD", mt5.TIMEFRAME_M5, count=200)
        tick = feed.get_tick("EURUSD")
    """

    def __init__(self) -> None:
        """Initialise the MarketDataFeed.

        No setup is required; the MT5 terminal must already be
        initialised via ``MT5Connector.connect()`` before calling any
        data methods.
        """
        pass

    # ── Candle / OHLCV data ──────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe: int,
        count: int = 200,
        drop_forming: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Retrieve historical OHLCV candles for a symbol.

        Args:
            symbol:    Instrument name (e.g. ``"EURUSD"``).
            timeframe: MT5 timeframe constant (e.g. ``mt5.TIMEFRAME_M5``).
            count:     Number of bars to fetch (default ``200``).
            drop_forming: When ``True``, drop the last (still-forming, not yet
                closed) candle so callers evaluate signals on CLOSED candles
                only. MT5's ``copy_rates_from_pos`` returns the in-progress
                candle at position 0, whose values mutate tick-by-tick; using
                it makes cross/threshold logic (e.g. RSI reversals) flicker and
                miss triggers. Live trading loops should pass ``True``.
                Defaults to ``False`` for backtesting / analysis callers.

        Returns:
            A ``pandas.DataFrame`` with columns
            ``[time, open, high, low, close, volume]`` where *time* is a
            timezone-aware UTC ``datetime``.  Returns ``None`` if the
            request fails.
        """
        # When dropping the forming candle, fetch one extra bar so the caller
        # still receives ``count`` *closed* candles.
        fetch_count = count + 1 if drop_forming else count
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, fetch_count)

        if rates is None or len(rates) == 0:
            error_code, error_msg = mt5.last_error()
            logger.error(
                "Failed to fetch candles for %s (tf=%s, count=%d)  [%s] %s",
                symbol,
                timeframe,
                count,
                error_code,
                error_msg,
            )
            return None

        df = pd.DataFrame(rates)

        # Keep only the columns the rest of the bot cares about.
        # MT5 returns: time, open, high, low, close, tick_volume,
        #              spread, real_volume
        df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Convert UNIX timestamps → UTC datetimes.
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # Drop the still-forming (last) candle so signals use closed bars only.
        if drop_forming and len(df) > 1:
            df = df.iloc[:-1].reset_index(drop=True)

        logger.debug(
            "Fetched %d candles for %s (tf=%s).", len(df), symbol, timeframe
        )
        return df

    # ── Tick data ────────────────────────────────────────────────────────

    def get_tick(self, symbol: str) -> Optional[dict]:
        """Get the latest tick for a symbol.

        Args:
            symbol: Instrument name (e.g. ``"EURUSD"``).

        Returns:
            A dictionary with keys ``bid``, ``ask``, ``time``, ``last``,
            ``volume``.  Returns ``None`` on failure.
        """
        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            error_code, error_msg = mt5.last_error()
            logger.error(
                "Failed to get tick for %s  [%s] %s",
                symbol,
                error_code,
                error_msg,
            )
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
            "last": tick.last,
            "volume": tick.volume,
        }

    # ── Symbol metadata ──────────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Retrieve static/trading-related properties of a symbol.

        Args:
            symbol: Instrument name (e.g. ``"EURUSD"``).

        Returns:
            A dictionary with keys::

                point, digits, trade_tick_size, trade_tick_value,
                volume_min, volume_max, volume_step, spread

            Returns ``None`` if the symbol cannot be found.
        """
        info = mt5.symbol_info(symbol)

        if info is None:
            error_code, error_msg = mt5.last_error()
            logger.error(
                "Failed to get symbol info for %s  [%s] %s",
                symbol,
                error_code,
                error_msg,
            )
            return None

        return {
            "point": info.point,
            "digits": info.digits,
            "trade_tick_size": info.trade_tick_size,
            "trade_tick_value": info.trade_tick_value,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "spread": info.spread,
        }

    # ── Convenience helpers ──────────────────────────────────────────────

    def get_spread_points(self, symbol: str) -> Optional[int]:
        """Return the current spread for *symbol* in points.

        Args:
            symbol: Instrument name (e.g. ``"EURUSD"``).

        Returns:
            The spread as an integer number of points, or ``None`` if the
            information is unavailable.
        """
        info = self.get_symbol_info(symbol)
        if info is None:
            return None

        spread: int = info["spread"]
        logger.debug("Spread for %s: %d points.", symbol, spread)
        return spread
