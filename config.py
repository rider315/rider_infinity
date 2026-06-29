"""
config.py - Centralised configuration for the Rider Infinity Trading Platform.

Loads settings from a `.env` file (via python-dotenv) and exposes them as
typed attributes on a `Config` instance.  Provides look-up tables for
MetaTrader 5 timeframe constants, symbol groups, and human-readable
timeframe options used throughout the bot.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Allow module to load for tests even without MT5 installed


# ── Load .env ────────────────────────────────────────────────────────────
# Walk up from this file's directory to locate the .env alongside it.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


# ── Timeframe mapping ────────────────────────────────────────────────────
# Maps human-readable timeframe strings to MT5 integer constants.
# If the MetaTrader5 package is not installed we fall back to the raw
# integer values so the rest of the code-base can still be imported.
TIMEFRAME_MAP: dict[str, int] = {
    "M1":  getattr(mt5, "TIMEFRAME_M1",  1),
    "M5":  getattr(mt5, "TIMEFRAME_M5",  5),
    "M15": getattr(mt5, "TIMEFRAME_M15", 15),
    "M30": getattr(mt5, "TIMEFRAME_M30", 30),
    "H1":  getattr(mt5, "TIMEFRAME_H1",  16385),
    "H4":  getattr(mt5, "TIMEFRAME_H4",  16388),
    "D1":  getattr(mt5, "TIMEFRAME_D1",  16408),
    "W1":  getattr(mt5, "TIMEFRAME_W1",  32769),
}

# Human-readable options for menus / prompts.
TIMEFRAME_OPTIONS: list[tuple[str, str]] = [
    ("M1",  "1 Minute"),
    ("M5",  "5 Minutes"),
    ("M15", "15 Minutes"),
    ("M30", "30 Minutes"),
    ("H1",  "1 Hour"),
    ("H4",  "4 Hours"),
    ("D1",  "1 Day"),
    ("W1",  "1 Week"),
]


# ── Symbol groups ────────────────────────────────────────────────────────
SYMBOL_GROUPS: dict[str, list[str]] = {
    "Majors":  ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
    "Crosses": ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPAUD"],
    "Metals":  ["XAUUSD", "XAGUSD"],
    "Exotics": ["USDTRY", "USDZAR", "USDMXN"],
}


# ── Config class ─────────────────────────────────────────────────────────
class Config:
    """Strongly-typed access to every setting the bot needs.

    Reads from environment variables (populated by `.env`) and applies
    sensible defaults where appropriate.  Raises ``SystemExit`` if
    mandatory MT5 credentials are missing so the bot fails fast.
    """

    def __init__(self) -> None:
        # --- MT5 connection (mandatory) ----------------------------------
        self.mt5_login: int = self._require_int("MT5_LOGIN")
        self.mt5_password: str = self._require("MT5_PASSWORD")
        self.mt5_server: str = self._require("MT5_SERVER")
        self.mt5_path: str = os.getenv(
            "MT5_PATH",
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
        )
        # Open the MT5 terminal as a visible window on connect (so it's
        # reachable on a server, e.g. to enable Algo Trading). Set false if you
        # manage the terminal yourself or run truly headless.
        self.mt5_auto_launch: bool = (
            os.getenv("MT5_AUTO_LAUNCH", "true").lower() in ("true", "1", "yes")
        )
        # Seconds to wait after launching the terminal before connecting.
        self.mt5_launch_wait_sec: float = float(os.getenv("MT5_LAUNCH_WAIT_SEC", "8"))

        # --- Trading defaults --------------------------------------------
        self.default_symbol: str = os.getenv("DEFAULT_SYMBOL", "EURUSD")
        self.default_timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "M5")
        self.default_lot_size: float = float(os.getenv("DEFAULT_LOT_SIZE", "0.01"))

        # --- Strategy defaults -------------------------------------------
        self.ema_fast: int = int(os.getenv("EMA_FAST", "9"))
        self.ema_slow: int = int(os.getenv("EMA_SLOW", "21"))
        self.rsi_period: int = int(os.getenv("RSI_PERIOD", "14"))
        self.rsi_overbought: int = int(os.getenv("RSI_OVERBOUGHT", "70"))
        self.rsi_oversold: int = int(os.getenv("RSI_OVERSOLD", "30"))
        self.bb_period: int = int(os.getenv("BB_PERIOD", "20"))
        self.bb_std_dev: float = float(os.getenv("BB_STD_DEV", "2.0"))
        self.macd_fast: int = int(os.getenv("MACD_FAST", "12"))
        self.macd_slow: int = int(os.getenv("MACD_SLOW", "26"))
        self.macd_signal: int = int(os.getenv("MACD_SIGNAL", "9"))

        # --- Risk management ---------------------------------------------
        self.max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", "1.0"))
        self.max_daily_drawdown: float = float(os.getenv("MAX_DAILY_DRAWDOWN", "3.0"))
        # Account-wide *portfolio* safety ceiling: the maximum number of
        # positions that may be open simultaneously across the WHOLE fleet
        # (every bot combined). This is a hard risk cap, NOT a per-bot
        # throttle — each bot is limited to its own positions separately via
        # ``max_positions_per_bot``. It must therefore be sized for the fleet:
        # with N bots each allowed 1 position, keep this >= N (plus headroom)
        # or the fleet starves itself. It only trips on a genuine runaway.
        self.max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "12"))
        # Max simultaneous positions a SINGLE bot may hold on its own symbol.
        # Keeps each bot independent; the fleet-wide cap above is the safety net.
        self.max_positions_per_bot: int = int(os.getenv("MAX_POSITIONS_PER_BOT", "1"))
        self.max_spread_points: int = int(os.getenv("MAX_SPREAD_POINTS", "30"))
        self.default_sl_pips: int = int(os.getenv("DEFAULT_SL_PIPS", "50"))
        self.default_tp_pips: int = int(os.getenv("DEFAULT_TP_PIPS", "100"))

        # --- Volatility-based stops & hard per-trade loss cap ------------
        # These exist to stop the "tiny win / huge loss" pattern: SL/TP are
        # sized from recent ATR (so gold & GBP get appropriately wider/narrower
        # stops), a minimum reward:risk ratio is enforced, and no single trade
        # may risk more than ``max_loss_per_trade_pct`` of equity.
        # NOTE: default OFF. A/B backtest showed ATR stops lower drawdown but
        # over-trade and erode the edge of mean-reversion strategies (RSI in
        # particular). Enable per preference for trend strategies / lower DD.
        self.use_atr_stops: bool = (
            os.getenv("USE_ATR_STOPS", "false").lower() in ("true", "1", "yes")
        )
        self.atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
        self.atr_sl_mult: float = float(os.getenv("ATR_SL_MULT", "1.5"))
        self.min_rr_ratio: float = float(os.getenv("MIN_RR_RATIO", "1.8"))
        self.min_sl_pips: float = float(os.getenv("MIN_SL_PIPS", "8"))
        self.max_sl_pips: float = float(os.getenv("MAX_SL_PIPS", "60"))
        self.max_loss_per_trade_pct: float = float(
            os.getenv("MAX_LOSS_PER_TRADE_PCT", "1.0")
        )

        # --- Trading-hours window (UTC) ---------------------------------
        # Only take new trades when the UTC hour is in [start, end).
        # Defaults 0..24 = trade all hours (no change). Backtests show the
        # Asian/off-hours session (≈21:00-07:00 UTC) is where most losing /
        # noise trades happen — set 7..21 to trade only London + New York.
        self.trade_hour_start: int = int(os.getenv("TRADE_HOUR_START", "0"))
        self.trade_hour_end: int = int(os.getenv("TRADE_HOUR_END", "24"))

        # --- Active trade management: breakeven + time-stop --------------
        # The export analysis showed the fatal pattern: winners get banked
        # early (Quick Exit / strategy flip) while losers run to the FULL
        # stop, so realised avg-loss (-$13) dwarfs avg-win (+$8) even at a
        # 55% win-rate. Breakeven fixes the asymmetry: once a trade is
        # ``breakeven_trigger_r`` of its risk (R) in profit, its stop is
        # pulled to entry (+ a tiny lock), turning would-be full-stop losers
        # into ~scratch trades. The time-stop force-closes positions that
        # have been held longer than ``max_trade_hours`` — this kills the
        # multi-day runaways (e.g. the −$76 AUDJPY held 6 days).
        self.breakeven_enabled: bool = (
            os.getenv("BREAKEVEN_ENABLED", "true").lower() in ("true", "1", "yes")
        )
        self.breakeven_trigger_r: float = float(os.getenv("BREAKEVEN_TRIGGER_R", "0.6"))
        self.breakeven_lock_pips: float = float(os.getenv("BREAKEVEN_LOCK_PIPS", "1.0"))
        self.max_trade_hours: float = float(os.getenv("MAX_TRADE_HOURS", "48"))

        # --- Bot settings ------------------------------------------------
        self.magic_number: int = int(os.getenv("MAGIC_NUMBER", "123456"))
        self.dry_run: bool = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

        # --- News Intelligence -------------------------------------------
        self.claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
        self.news_api_key: str = os.getenv("NEWS_API_KEY", "")
        self.finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
        self.news_analysis_enabled: bool = (
            os.getenv("NEWS_ANALYSIS_ENABLED", "true").lower() in ("true", "1", "yes")
        )
        self.news_polling_interval: int = int(os.getenv("NEWS_POLLING_INTERVAL", "60"))
        self.news_signal_weight: float = float(os.getenv("NEWS_SIGNAL_WEIGHT", "0.25"))
        self.news_lookback_hours: int = int(os.getenv("NEWS_LOOKBACK_HOURS", "4"))
        self.news_min_confidence: float = float(os.getenv("NEWS_MIN_CONFIDENCE", "0.6"))
        self.news_min_impact: float = float(os.getenv("NEWS_MIN_IMPACT", "0.5"))
        self.news_signal_threshold: float = float(os.getenv("NEWS_SIGNAL_THRESHOLD", "0.3"))
        self.news_cache_retention_days: int = int(os.getenv("NEWS_CACHE_RETENTION_DAYS", "30"))
        self.news_mode: str = os.getenv("NEWS_MODE", "combined")  # "combined" | "standalone" | "both"
        self.news_rss_feeds: list[str] = [
            feed.strip()
            for feed in os.getenv("NEWS_RSS_FEEDS", "").split(",")
            if feed.strip()
        ] or []  # Empty = use defaults

        # --- Cloud Optimisation (Optional) --------------------------------
        self.aws_cloud_enabled: bool = (
            os.getenv("AWS_CLOUD_ENABLED", "false").lower() in ("true", "1", "yes")
        )
        self.aws_region: str = os.getenv("AWS_REGION", "us-east-1")
        self.aws_cloud_s3_bucket: str = os.getenv("AWS_CLOUD_S3_BUCKET", "")
        self.aws_cloud_max_cost_per_job: float = float(
            os.getenv("AWS_CLOUD_MAX_COST_PER_JOB", "10.0")
        )
        self.aws_cloud_max_workers: int = int(
            os.getenv("AWS_CLOUD_MAX_WORKERS", "10")
        )
        self.aws_cloud_max_runtime_mins: int = int(
            os.getenv("AWS_CLOUD_MAX_RUNTIME_MINS", "120")
        )
        self.aws_cloud_instance_type: str = os.getenv(
            "AWS_CLOUD_INSTANCE_TYPE", "c6a.large"
        )

    # -- Convenience helpers -----------------------------------------------

    def get_mt5_timeframe(self, label: str | None = None) -> int:
        """Return the MT5 integer constant for a timeframe label.

        Args:
            label: One of the keys in ``TIMEFRAME_MAP`` (e.g. ``"M5"``).
                   Defaults to ``self.default_timeframe`` when *None*.

        Returns:
            MT5 timeframe constant (integer).

        Raises:
            ValueError: If the label is not recognised.
        """
        label = (label or self.default_timeframe).upper()
        if label not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unknown timeframe '{label}'. "
                f"Valid options: {', '.join(TIMEFRAME_MAP)}"
            )
        return TIMEFRAME_MAP[label]

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _require(var: str) -> str:
        """Return an env-var's value or exit with an error message."""
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            print(
                f"[CONFIG ERROR] '{var}' is missing or still set to a "
                f"placeholder.  Please update your .env file."
            )
            sys.exit(1)
        return value

    @staticmethod
    def _require_int(var: str) -> int:
        """Like ``_require`` but also casts to ``int``."""
        raw = Config._require(var)
        try:
            return int(raw)
        except ValueError:
            print(f"[CONFIG ERROR] '{var}' must be an integer, got '{raw}'.")
            sys.exit(1)

    # -- Representation ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Config login={self.mt5_login} server={self.mt5_server} "
            f"symbol={self.default_symbol} tf={self.default_timeframe} "
            f"dry_run={self.dry_run}>"
        )
