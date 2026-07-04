"""
multi_bot.py — Rider Infinity Multi-Bot Manager for running multiple trading bots simultaneously.

Each bot instance runs in its own thread with its own symbol, strategy,
timeframe, and lot configuration.  The manager provides a dashboard to
monitor, add, and remove bots in real-time.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import timedelta
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

try:
    from colorama import Fore, Style
except ImportError:
    class _Stub:
        def __getattr__(self, _name: str) -> str:
            return ""
    Fore = Style = _Stub()

import MetaTrader5 as mt5

from config import Config, TIMEFRAME_MAP, SYMBOL_GROUPS, TIMEFRAME_OPTIONS
from market_data import MarketDataFeed
from mt5_connector import MT5Connector
from order_manager import OrderManager
from risk_manager import RiskManager
from session_manager import SessionManager, is_crypto
from strategies import get_strategy, list_strategies
from strategies.base import SignalAction
from utils import compute_atr, pips_to_price, price_to_pips, save_trade_to_journal, setup_logger
from bot_history import BotHistoryDB
from logic_version import compute_logic_version

# How often (seconds) a Quick-Exit-armed bot re-checks its open position's
# floating profit while in a trade. Small enough to bank a few-pip move,
# large enough not to hammer the MT5 terminal.
QUICK_EXIT_POLL_SEC = 3


# ======================================================================
#  Bot Instance Status
# ======================================================================
class BotStatus(Enum):
    """Possible states of a bot instance."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    WAITING = "WAITING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# ======================================================================
#  Single Bot Instance — runs in its own thread
# ======================================================================
@dataclass
class BotInstance:
    """Represents a single trading bot running on one symbol/strategy."""

    bot_id: int
    symbol: str
    strategy_name: str
    timeframe_key: str
    lot_size: float
    lot_mode: str = "fixed"          # "fixed" | "auto"
    dry_run: bool = True
    strategy_params: dict = field(default_factory=dict)  # optimized indicator params

    # Quick Exit — independent, per-bot scalp-exit feature (UI toggle).
    # When enabled, the bot polls its open position frequently and closes it
    # the moment floating profit reaches ``quick_exit_pips`` pips. Lets any
    # strategy "thoda profit hote hi" bank it and free the slot for the next trade.
    quick_exit_enabled: bool = False
    quick_exit_pips: float = 5.0

    # Runtime state (not set by user)
    status: BotStatus = field(default=BotStatus.IDLE)
    last_signal: str = field(default="—")
    last_price: float = field(default=0.0)
    trades_taken: int = field(default=0)
    last_check: str = field(default="—")
    error_msg: str = field(default="")
    last_entry: float = field(default=0.0)
    last_sl: float = field(default=0.0)
    last_tp: float = field(default=0.0)
    last_ticket: int = field(default=0)
    active_tickets: list = field(default_factory=list)
    active_positions: list = field(default_factory=list)
    # Epoch until which new entries are blocked (set after a losing SL exit).
    cooldown_until: float = field(default=0.0)
    trigger_proximity: dict = field(default_factory=lambda: {
        "proximity_pct": 0.0, "direction": "NEUTRAL",
        "label": "—", "detail": "", "indicator_values": {},
    })
    blocked_reason: str = field(default="")
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def label(self) -> str:
        """Short human-readable label for this bot."""
        mode = "DRY" if self.dry_run else "LIVE"
        lot_display = "auto" if self.lot_mode == "auto" else f"{self.lot_size}"
        return f"Bot#{self.bot_id} | {self.symbol} | {self.strategy_name} | {self.timeframe_key} | {lot_display} | {mode}"


class MultiBotManager:
    """
    Manages multiple BotInstance objects, each running in its own thread.

    All bots share the same MT5 connection, config, and risk manager,
    but each has its own strategy instance, symbol, and timeframe.
    """

    def __init__(
        self,
        config: Config,
        connector: MT5Connector,
        market_data: MarketDataFeed,
        session_mgr: SessionManager,
        order_mgr: OrderManager,
        risk_mgr: RiskManager,
    ) -> None:
        self.config = config
        self.connector = connector
        self.market_data = market_data
        self.session_mgr = session_mgr
        self.order_mgr = order_mgr
        self.risk_mgr = risk_mgr
        self.logger = setup_logger("MultiBotManager")

        self._bots: Dict[int, BotInstance] = {}
        self._next_id: int = 1
        self._lock = threading.Lock()
        self._persist_path = Path(__file__).resolve().parent / "data" / "bot_configs.json"
        self._loading: bool = False  # suppress saves during _load_bots
        self.history_db = BotHistoryDB()

    # ------------------------------------------------------------------
    # Bot lifecycle
    # ------------------------------------------------------------------
    def add_bot(
        self,
        symbol: str,
        strategy_name: str,
        timeframe_key: str,
        lot_size: float = 0.01,
        lot_mode: str = "fixed",
        dry_run: bool = True,
        strategy_params: dict | None = None,
        quick_exit_enabled: bool = False,
        quick_exit_pips: float = 5.0,
    ) -> BotInstance:
        """Create and register a new bot instance (does NOT start it)."""
        with self._lock:
            bot = BotInstance(
                bot_id=self._next_id,
                symbol=symbol,
                strategy_name=strategy_name,
                timeframe_key=timeframe_key,
                lot_size=lot_size,
                lot_mode=lot_mode,
                dry_run=dry_run,
                strategy_params=strategy_params or {},
                quick_exit_enabled=bool(quick_exit_enabled),
                quick_exit_pips=float(quick_exit_pips),
            )
            self._bots[self._next_id] = bot
            self._next_id += 1
            self.logger.info("Added %s", bot.label)
            self._save_bots()
            if not self._loading:
                self.history_db.record_bot_created(
                    bot_id=bot.bot_id,
                    symbol=bot.symbol,
                    strategy_name=bot.strategy_name,
                    timeframe_key=bot.timeframe_key,
                    lot_size=bot.lot_size,
                    lot_mode=bot.lot_mode,
                    dry_run=bot.dry_run,
                    strategy_params=bot.strategy_params,
                )
            return bot

    def start_bot(self, bot_id: int) -> bool:
        """Start a bot's trading loop in a background thread."""
        bot = self._bots.get(bot_id)
        if bot is None:
            return False

        if bot.status == BotStatus.RUNNING:
            return True  # already running

        # Ensure symbol is available (resolves broker suffixes) and adopt
        # the broker's real name so ticks/orders target the right symbol.
        resolved = self.connector.ensure_symbol(bot.symbol)
        if not resolved:
            bot.status = BotStatus.ERROR
            bot.error_msg = f"Symbol {bot.symbol} not available"
            return False
        bot.symbol = resolved

        # Load strategy
        try:
            strategy = get_strategy(bot.strategy_name, **bot.strategy_params)
        except Exception as exc:
            bot.status = BotStatus.ERROR
            bot.error_msg = str(exc)
            return False

        bot._stop_event.clear()
        bot.status = BotStatus.RUNNING
        bot.error_msg = ""

        thread = threading.Thread(
            target=self._trading_loop,
            args=(bot, strategy),
            name=f"Bot-{bot.bot_id}-{bot.symbol}",
            daemon=True,
        )
        bot._thread = thread
        thread.start()
        self.logger.info("Started %s", bot.label)
        self._save_bots()
        if not self._loading:
            self.history_db.record_bot_started(bot.bot_id)
        return True

    def stop_bot(self, bot_id: int) -> bool:
        """Signal a running bot to stop gracefully."""
        bot = self._bots.get(bot_id)
        if bot is None:
            return False

        bot._stop_event.set()
        bot.status = BotStatus.STOPPED
        self.logger.info("Stopped %s", bot.label)
        self._save_bots()
        self.history_db.record_bot_stopped(bot.bot_id, bot.trades_taken)
        return True

    def remove_bot(self, bot_id: int) -> bool:
        """Stop and remove a bot entirely."""
        bot = self._bots.get(bot_id)
        if bot is None:
            return False

        bot._stop_event.set()
        # Wait briefly for thread to finish
        if bot._thread and bot._thread.is_alive():
            bot._thread.join(timeout=3)

        self.history_db.record_bot_removed(bot_id, bot.trades_taken)
        with self._lock:
            del self._bots[bot_id]
        self.logger.info("Removed Bot#%d", bot_id)
        self._save_bots()
        return True

    def start_all(self) -> int:
        """Start all idle/stopped bots. Returns count started."""
        count = 0
        for bot_id, bot in self._bots.items():
            if bot.status in (BotStatus.IDLE, BotStatus.STOPPED):
                if self.start_bot(bot_id):
                    count += 1
        return count

    def stop_all(self) -> int:
        """Stop all running bots. Returns count stopped."""
        count = 0
        for bot_id, bot in list(self._bots.items()):
            if bot.status == BotStatus.RUNNING:
                self.stop_bot(bot_id)
                count += 1
        return count

    def get_all_bots(self) -> List[BotInstance]:
        """Return a list of all registered bots."""
        return list(self._bots.values())

    def get_bot(self, bot_id: int) -> Optional[BotInstance]:
        """Return a specific bot by ID."""
        return self._bots.get(bot_id)

    @property
    def running_count(self) -> int:
        return sum(1 for b in self._bots.values() if b.status == BotStatus.RUNNING)

    @property
    def total_count(self) -> int:
        return len(self._bots)

    # ------------------------------------------------------------------
    # Persistence — survive server restarts
    # ------------------------------------------------------------------
    def _save_bots(self) -> None:
        """Persist all bot configurations to a JSON file."""
        if self._loading:
            return  # skip redundant saves during bulk restore
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            configs = []
            for bot in self._bots.values():
                configs.append({
                    "bot_id": bot.bot_id,
                    "symbol": bot.symbol,
                    "strategy_name": bot.strategy_name,
                    "timeframe_key": bot.timeframe_key,
                    "lot_size": bot.lot_size,
                    "lot_mode": bot.lot_mode,
                    "dry_run": bot.dry_run,
                    "strategy_params": bot.strategy_params,
                    "quick_exit_enabled": bot.quick_exit_enabled,
                    "quick_exit_pips": bot.quick_exit_pips,
                    "was_running": bot.status in (
                        BotStatus.RUNNING, BotStatus.WAITING,
                    ),
                    # Persist last trade info (important for DRY bots)
                    "last_entry": bot.last_entry,
                    "last_sl": bot.last_sl,
                    "last_tp": bot.last_tp,
                    "last_ticket": bot.last_ticket,
                })
            with open(self._persist_path, "w", encoding="utf-8") as fh:
                json.dump(configs, fh, indent=2)
            self.logger.debug(
                "Saved %d bot config(s) to %s", len(configs), self._persist_path,
            )
        except Exception as exc:
            self.logger.warning("Failed to save bot configs: %s", exc)

    def _load_bots(self) -> int:
        """Load bot configurations from disk and auto-start those that were running.

        Returns the number of bots restored.
        """
        if not self._persist_path.exists():
            self.logger.info("No saved bot configs found at %s", self._persist_path)
            return 0

        try:
            with open(self._persist_path, "r", encoding="utf-8") as fh:
                configs = json.load(fh)
        except Exception as exc:
            self.logger.warning("Failed to read bot configs: %s", exc)
            return 0

        if not isinstance(configs, list):
            self.logger.warning("Bot config file has invalid format.")
            return 0

        self._loading = True
        restored = 0
        for cfg in configs:
            try:
                bot = self.add_bot(
                    symbol=cfg["symbol"],
                    strategy_name=cfg["strategy_name"],
                    timeframe_key=cfg["timeframe_key"],
                    lot_size=float(cfg.get("lot_size", 0.01)),
                    lot_mode=cfg.get("lot_mode", "fixed"),
                    dry_run=cfg.get("dry_run", True),
                    strategy_params=cfg.get("strategy_params", {}),
                    quick_exit_enabled=cfg.get("quick_exit_enabled", False),
                    quick_exit_pips=cfg.get("quick_exit_pips", 5.0),
                )
                restored += 1

                # Restore trades_taken from history DB
                persisted_trades = self.history_db.get_total_trades(bot.bot_id)
                if persisted_trades > 0:
                    bot.trades_taken = persisted_trades

                # Restore last trade info (important for DRY bots)
                bot.last_entry = float(cfg.get("last_entry", 0.0))
                bot.last_sl = float(cfg.get("last_sl", 0.0))
                bot.last_tp = float(cfg.get("last_tp", 0.0))
                bot.last_ticket = int(cfg.get("last_ticket", 0))

                # Auto-start if it was running before the restart
                if cfg.get("was_running", False):
                    started = self.start_bot(bot.bot_id)
                    if started:
                        self.logger.info(
                            "Restored and started %s", bot.label,
                        )
                    else:
                        self.logger.warning(
                            "Restored %s but failed to start: %s",
                            bot.label, bot.error_msg,
                        )
                else:
                    self.logger.info("Restored %s (stopped)", bot.label)
            except Exception as exc:
                self.logger.warning(
                    "Failed to restore bot from config %s: %s", cfg, exc,
                )

        self._loading = False
        self._save_bots()  # single save with all restored bots

        self.logger.info(
            "Restored %d bot(s) from %s", restored, self._persist_path,
        )
        return restored


    # ------------------------------------------------------------------
    # Trading loop (runs inside a thread per bot)
    # ------------------------------------------------------------------
    def _trading_loop(self, bot: BotInstance, strategy) -> None:
        """Main trading loop for a single bot instance."""
        logger = logging.getLogger(f"Bot#{bot.bot_id}")

        try:
            mt5_tf = TIMEFRAME_MAP.get(bot.timeframe_key, 5)

            while not bot._stop_event.is_set():
                now = datetime.utcnow()
                bot.last_check = now.strftime("%H:%M:%S")

                # -- Market open check (crypto trades 24/7; forex respects the
                #    weekend/session calendar) --
                if not self.session_mgr.is_market_open(symbol=bot.symbol):
                    next_open = self.session_mgr.next_market_open()
                    bot.status = BotStatus.WAITING
                    bot.last_signal = f"MARKET CLOSED \u00b7 Opens {next_open['label']}"
                    bot.error_msg = f"Market closed. Next: {next_open['session']} in {next_open['label']}"
                    logger.info("Bot#%d waiting \u2013 market closed. Next open: %s in %s",
                                bot.bot_id, next_open['session'], next_open['label'])
                    bot._stop_event.wait(timeout=60)
                    continue

                # Market is open — ensure status is RUNNING
                if bot.status == BotStatus.WAITING:
                    bot.status = BotStatus.RUNNING
                    bot.error_msg = ""
                    logger.info("Bot#%d market opened \u2013 resuming trading.", bot.bot_id)

                # -- Connection check --
                if not self.connector.is_connected():
                    logger.warning("MT5 disconnected — waiting for reconnect …")
                    if not self.connector.reconnect():
                        bot._stop_event.wait(timeout=30)
                        continue

                # -- Fetch candles --
                # drop_forming=True → evaluate on CLOSED candles only so RSI
                # reversal/cross triggers don't flicker on the in-progress bar.
                df = self.market_data.get_candles(
                    bot.symbol, mt5_tf, count=500, drop_forming=True
                )
                if df is None or df.empty:
                    logger.warning("No data for %s — retrying.", bot.symbol)
                    bot._stop_event.wait(timeout=10)
                    continue

                current_price = df["close"].iloc[-1]
                bot.last_price = current_price

                # -- Run strategy --
                signal = strategy.analyze(df, bot.symbol)
                bot.last_signal = signal.action.value

                # -- Update trigger proximity --
                try:
                    prox = strategy.get_proximity(df, bot.symbol)
                    bot.trigger_proximity = {
                        "proximity_pct": prox.proximity_pct,
                        "direction": prox.direction,
                        "label": prox.label,
                        "detail": prox.detail,
                        "indicator_values": prox.indicator_values,
                    }
                except Exception as prox_exc:
                    logger.debug("Bot#%d proximity calc failed: %s", bot.bot_id, prox_exc)

                # -- Execute BUY / SELL --
                if signal.action in (SignalAction.BUY, SignalAction.SELL):
                    # Guard: post-loss cooldown. A breakout signal stays valid
                    # for several candles, so without this the bot re-enters
                    # the same move seconds after being stopped out (the
                    # loss-churn that dominated the losses in the export).
                    now_ts = time.time()
                    per_bot_cap = getattr(self.config, "max_positions_per_bot", 1)
                    if now_ts < bot.cooldown_until:
                        remaining = int(bot.cooldown_until - now_ts)
                        bot.blocked_reason = (
                            f"Post-loss cooldown ({remaining // 60}m"
                            f"{remaining % 60:02d}s left)"
                        )
                        logger.debug(
                            "Bot#%d signal=%s suppressed by post-loss cooldown "
                            "(%ds left).",
                            bot.bot_id, signal.action.value, remaining,
                        )
                    # Guard: per-bot position limit (keeps each bot independent;
                    # the fleet-wide ceiling lives in risk_mgr.can_trade).
                    elif len(bot.active_tickets) >= per_bot_cap:
                        bot.blocked_reason = (
                            f"Per-bot limit reached "
                            f"({len(bot.active_tickets)}/{per_bot_cap} open)"
                        )
                        logger.debug(
                            "Bot#%d signal=%s but at per-bot limit (%d/%d) — skipping.",
                            bot.bot_id, signal.action.value,
                            len(bot.active_tickets), per_bot_cap,
                        )
                    else:
                        bot.blocked_reason = ""  # clear on attempt
                        self._execute_signal(bot, signal, df, logger)
                else:
                    bot.blocked_reason = ""  # clear when no signal

                # -- Sync position state (detect externally closed trades) --
                self._sync_position_state(bot, logger)

                # -- Breakeven: once a trade is in profit, protect it so it
                #    can't run back to the full stop (fixes the asymmetric
                #    avg-loss > avg-win pattern that bled the account). --
                self._apply_breakeven(bot, logger)

                # -- Time-stop: force-close stale positions (kills multi-day
                #    runaways that never hit SL/TP). --
                self._apply_time_stop(bot, logger)

                # -- Quick Exit: bank small profit immediately (if enabled) --
                self._apply_quick_exit(bot, logger)

                # -- Sleep --
                # When Quick Exit is armed and a position is open, poll fast so
                # we can close near the target instead of waiting a whole candle.
                if (bot.quick_exit_enabled and bot.active_tickets
                        and not bot.dry_run):
                    sleep_sec = QUICK_EXIT_POLL_SEC
                else:
                    sleep_sec = max(5, self._seconds_until_next_candle(bot.timeframe_key))
                bot._stop_event.wait(timeout=sleep_sec)

        except Exception as exc:
            bot.status = BotStatus.ERROR
            bot.error_msg = str(exc)
            logger.error("Bot#%d crashed: %s", bot.bot_id, exc, exc_info=True)
            self.history_db.record_bot_error(bot.bot_id, str(exc))
            return

        bot.status = BotStatus.STOPPED

    def _resolve_sl_tp(self, signal, df, symbol: str, point: float) -> tuple[float, float]:
        """Return (sl_pips, tp_pips) sized for volatility, clamped, with RR floor.

        * If ``config.use_atr_stops`` is on and ATR is available, the stop is
          set to ``atr_sl_mult × ATR`` (converted to pips) — so volatile
          instruments like gold get wider stops and calm pairs get tighter ones.
        * The stop is clamped to ``[min_sl_pips, max_sl_pips]`` so it can never
          be absurdly wide (the root cause of the runaway losses).
        * The take-profit is forced to at least ``min_rr_ratio × sl_pips`` so a
          high win-rate can't be wrecked by tiny wins vs large losses.

        Crypto (BTC, ETH, …) is special-cased: the forex ``min/max_sl_pips``
        clamp is meaningless when 1 "pip" is a fraction of a dollar on a
        $60k instrument (a 60-pip stop would be a few dollars — an instant
        stop-out / broker "invalid stops" rejection). For crypto the stop is
        always ATR-driven and clamped to a **percentage-of-price** band so it
        scales with the instrument.
        """
        cfg = self.config
        rr = getattr(cfg, "min_rr_ratio", 1.8)
        # Baseline from the strategy's own suggestion (or global defaults).
        sl_pips = float(signal.sl_pips or cfg.default_sl_pips)
        tp_pips = float(signal.tp_pips or cfg.default_tp_pips)
        atr_based = False
        crypto = is_crypto(symbol)

        # Crypto always uses ATR stops (regardless of the global toggle); forex
        # only when use_atr_stops is on.
        if (crypto or getattr(cfg, "use_atr_stops", False)) and point > 0:
            atr_price = compute_atr(df, getattr(cfg, "atr_period", 14))
            if atr_price > 0:
                atr_pips = price_to_pips(atr_price * cfg.atr_sl_mult, symbol, point)
                if atr_pips > 0:
                    sl_pips = atr_pips
                    atr_based = True

        if crypto and point > 0:
            # Clamp to a fraction-of-price band (0.15%–1.5% of price) so the
            # stop is neither dust nor absurd. Fall back to ~0.4% of price if
            # ATR was unavailable.
            price = float(df["close"].iloc[-1])
            min_cx = price_to_pips(price * 0.0015, symbol, point)
            max_cx = price_to_pips(price * 0.015, symbol, point)
            if not atr_based:
                sl_pips = price_to_pips(price * 0.004, symbol, point)
            sl_pips = max(min_cx, min(max_cx, sl_pips))
            atr_based = True  # always scale TP proportionally for crypto
        else:
            # Forex/metals: clamp the stop distance to sane pip bounds.
            sl_pips = max(cfg.min_sl_pips, min(cfg.max_sl_pips, sl_pips))

        if atr_based:
            # Volatility-based: scale TP proportionally so it's actually
            # reachable (a fixed far-away TP is why trades rarely hit profit).
            tp_pips = sl_pips * rr
        else:
            # Fixed-pip fallback: keep the suggested TP but never below the RR floor.
            tp_pips = max(tp_pips, sl_pips * rr)
        return round(sl_pips, 1), round(tp_pips, 1)

    def _logic_version(self, bot: BotInstance) -> str:
        """Fingerprint of the code + settings shaping this bot's trades.

        Stamped on every trade row so exported history can be split into
        before/after any strategy or trade-management change. Covers the
        strategy source file, the engine files, the bot's strategy params,
        and every setting that alters trade behaviour without a code edit.
        """
        try:
            cfg = self.config
            settings = {
                "quick_exit_enabled": bot.quick_exit_enabled,
                "quick_exit_pips": bot.quick_exit_pips,
                "timeframe": bot.timeframe_key,
                "use_atr_stops": getattr(cfg, "use_atr_stops", False),
                "atr_sl_mult": getattr(cfg, "atr_sl_mult", None),
                "min_rr_ratio": getattr(cfg, "min_rr_ratio", None),
                "min_sl_pips": getattr(cfg, "min_sl_pips", None),
                "max_sl_pips": getattr(cfg, "max_sl_pips", None),
                "breakeven_enabled": getattr(cfg, "breakeven_enabled", None),
                "breakeven_trigger_r": getattr(cfg, "breakeven_trigger_r", None),
                "max_trade_hours": getattr(cfg, "max_trade_hours", None),
                "trade_hour_start": getattr(cfg, "trade_hour_start", None),
                "trade_hour_end": getattr(cfg, "trade_hour_end", None),
                "sl_cooldown_min": getattr(cfg, "sl_cooldown_min", None),
                "quick_exit_trail_pips": getattr(cfg, "quick_exit_trail_pips", None),
                "quick_exit_sl_cap_mult": getattr(cfg, "quick_exit_sl_cap_mult", None),
            }
            return compute_logic_version(
                bot.strategy_name, bot.strategy_params, settings
            )
        except Exception as exc:
            self.logger.debug("logic version compute failed: %s", exc)
            return ""

    def _execute_signal(self, bot: BotInstance, signal, df, logger: logging.Logger) -> None:
        """Place an order for a bot's signal."""
        symbol = signal.symbol or bot.symbol

        # Symbol metadata (needed for volatility-based stops & pip conversion).
        tick = self.market_data.get_tick(symbol)
        if tick is None:
            bot.blocked_reason = f"No live tick for {symbol} (symbol not in Market Watch / feed down)"
            logger.warning("Bot#%d trade skipped: %s", bot.bot_id, bot.blocked_reason)
            return
        sym_info = self.market_data.get_symbol_info(symbol)
        if sym_info is None:
            bot.blocked_reason = f"No symbol info for {symbol}"
            logger.warning("Bot#%d trade skipped: %s", bot.bot_id, bot.blocked_reason)
            return
        point = sym_info["point"]
        digits = sym_info["digits"]

        # ---- Resolve SL/TP: volatility-aware, capped, with an RR floor -----
        sl_pips, tp_pips = self._resolve_sl_tp(signal, df, symbol, point)

        # Quick-Exit bots bank a few pips per trade — their stop must be in
        # the same order of magnitude as the target. Without this cap a bot
        # can pair a 5-pip target with a strategy's 50-pip default stop
        # (risk 10× reward → needs >90% win-rate; the RSI bots' exact leak).
        if bot.quick_exit_enabled and bot.quick_exit_pips > 0:
            cap_mult = float(getattr(self.config, "quick_exit_sl_cap_mult", 0))
            if cap_mult > 0:
                cap = max(
                    bot.quick_exit_pips * cap_mult,
                    float(getattr(self.config, "min_sl_pips", 8)),
                )
                if sl_pips > cap:
                    logger.info(
                        "Bot#%d SL capped %.1f → %.1f pips (%.0f× quick-exit "
                        "target %.1f).",
                        bot.bot_id, sl_pips, cap, cap_mult, bot.quick_exit_pips,
                    )
                    sl_pips = round(cap, 1)

        # Lot size (risk-based when auto, otherwise the bot's fixed lot)
        if bot.lot_mode == "auto":
            lot = self.risk_mgr.calculate_lot_size(symbol, sl_pips)
        else:
            lot = bot.lot_size

        # Hard per-trade loss cap — shrinks (or rejects) the lot so a single
        # trade can never lose more than config.max_loss_per_trade_pct of equity.
        lot, cap_note = self.risk_mgr.cap_lot_for_max_loss(symbol, sl_pips, lot)
        if lot <= 0:
            bot.blocked_reason = cap_note
            logger.info("Bot#%d trade blocked: %s", bot.bot_id, cap_note)
            return

        # Risk check
        can, reason = self.risk_mgr.can_trade(symbol, lot)
        if not can:
            bot.blocked_reason = reason
            logger.info("Bot#%d trade blocked: %s", bot.bot_id, reason)
            return
        bot.blocked_reason = ""  # clear on successful attempt

        if signal.action == SignalAction.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            entry = tick["ask"]
            sl_price = round(entry - pips_to_price(sl_pips, symbol, point), digits)
            tp_price = round(entry + pips_to_price(tp_pips, symbol, point), digits)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            entry = tick["bid"]
            sl_price = round(entry + pips_to_price(sl_pips, symbol, point), digits)
            tp_price = round(entry - pips_to_price(tp_pips, symbol, point), digits)

        # Store last trade levels on the bot for dashboard display
        bot.last_entry = entry
        bot.last_sl = sl_price
        bot.last_tp = tp_price

        # Use a per-bot order manager with the bot's own dry_run setting
        # (the shared self.order_mgr may have dry_run=True from .env config)
        comment = f"Bot#{bot.bot_id}|{bot.strategy_name}"

        if bot.dry_run:
            logger.info(
                "[DRY] Bot#%d %s %s %.2f lots @ %.5f  SL=%.5f  TP=%.5f",
                bot.bot_id, signal.action.value, symbol, lot, entry, sl_price, tp_price,
            )
            bot.trades_taken += 1
            self.history_db.increment_trades(bot.bot_id)
            save_trade_to_journal({
                "time": datetime.utcnow().isoformat(),
                "bot_id": bot.bot_id,
                "symbol": symbol,
                "action": signal.action.value,
                "lot": lot,
                "entry": entry,
                "sl": sl_price,
                "tp": tp_price,
                "strategy": bot.strategy_name,
                "reason": signal.reason,
                "confidence": signal.confidence,
                "dry_run": True,
            })
            return

        # LIVE mode — shared order_mgr is always live; per-bot toggle controls dry/live
        result = self.order_mgr.open_position(
            symbol=symbol,
            order_type=order_type,
            lot=lot,
            sl_price=sl_price,
            tp_price=tp_price,
            comment=comment,
        )

        if result.get("retcode") == 10009:
            bot.trades_taken += 1
            self.history_db.increment_trades(bot.bot_id)
            # Track the MT5 position ticket so we can detect external closes
            ticket = result.get("order", 0)
            bot.last_ticket = ticket
            if ticket and ticket not in bot.active_tickets:
                bot.active_tickets.append(ticket)
            logger.info(
                "Bot#%d %s %s %.2f lots @ %.5f (ticket #%s)",
                bot.bot_id, signal.action.value, symbol, lot, entry,
                ticket,
            )
            save_trade_to_journal({
                "time": datetime.utcnow().isoformat(),
                "bot_id": bot.bot_id,
                "symbol": symbol,
                "action": signal.action.value,
                "lot": lot,
                "entry": entry,
                "sl": sl_price,
                "tp": tp_price,
                "strategy": bot.strategy_name,
                "reason": signal.reason,
                "confidence": signal.confidence,
                "dry_run": False,
            })
            # Record the open leg for per-trade performance tracking.
            # The matching close (exit price / profit / reason) is filled in
            # later by _sync_position_state when the ticket disappears.
            if ticket:
                self.history_db.record_trade_open(
                    bot_id=bot.bot_id,
                    ticket=ticket,
                    symbol=symbol,
                    strategy=bot.strategy_name,
                    action=signal.action.value,
                    lot=lot,
                    entry_price=entry,
                    sl=sl_price,
                    tp=tp_price,
                    logic_version=self._logic_version(bot),
                    reason=signal.reason or "",
                    confidence=signal.confidence,
                )
        else:
            retcode = result.get("retcode", -1)
            msg = OrderManager.get_retcode_message(retcode)
            # Surface the broker rejection on the dashboard instead of failing
            # silently — otherwise the bot shows a BUY/SELL signal with no trade
            # and no visible reason. Include the broker's own comment when present.
            broker_note = result.get("comment") or ""
            bot.blocked_reason = (
                f"Order rejected ({retcode}): {msg}"
                + (f" — {broker_note}" if broker_note and broker_note not in msg else "")
            )
            logger.error("Bot#%d order failed: %s", bot.bot_id, bot.blocked_reason)

    def _sync_position_state(self, bot: BotInstance, logger: logging.Logger) -> None:
        """Sync the bot's active positions from MT5.

        Discovers ALL open positions that belong to this bot by querying MT5
        for the bot's symbol and matching via comment pattern (``Bot#ID|...``)
        or magic number.  This ensures positions opened before the code update,
        or between restarts, are still tracked correctly.

        When a trade is closed from the MT5 terminal (externally), this method
        detects it and clears stale SL/TP/Entry values from the dashboard.
        """
        # Nothing to sync if bot is in dry-run mode
        if bot.dry_run:
            return

        try:
            # Query ALL open positions for this bot's symbol
            all_positions = mt5.positions_get(symbol=bot.symbol)
            if all_positions is None:
                all_positions = ()

            comment_prefix = f"Bot#{bot.bot_id}|"
            still_open = []
            active_pos_data = []

            for pos in all_positions:
                # Include ALL positions on this symbol UNLESS the comment
                # explicitly claims a DIFFERENT bot (e.g. "Bot#2|ema").
                # This catches positions opened by the bot, manually from
                # the terminal, or with broker-modified comments.
                comment = pos.comment or ""
                if (comment.startswith("Bot#")
                        and not comment.startswith(comment_prefix)):
                    continue  # belongs to a different RI bot

                still_open.append(pos.ticket)
                pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                active_pos_data.append({
                    "ticket": pos.ticket,
                    "type": pos_type,
                    "volume": pos.volume,
                    "entry": pos.price_open,
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "open_time": getattr(pos, "time", 0),
                })

            # Log closed positions (were tracked but no longer open)
            prev_tickets = set(bot.active_tickets)
            new_tickets = set(still_open)
            closed = prev_tickets - new_tickets
            if closed:
                logger.info(
                    "Bot#%d positions closed externally: %s — "
                    "clearing from dashboard.",
                    bot.bot_id, list(closed),
                )
                for ticket in closed:
                    self._record_closed_trade(bot, ticket, logger)

            # Update tracking
            bot.active_tickets = still_open
            bot.active_positions = active_pos_data

            if active_pos_data:
                # Update last_* fields from the most recent position
                latest = active_pos_data[-1]
                bot.last_entry = latest["entry"]
                bot.last_sl = latest["sl"]
                bot.last_tp = latest["tp"]
                bot.last_ticket = latest["ticket"]
            else:
                # All positions closed — clear stale values
                bot.last_entry = 0.0
                bot.last_sl = 0.0
                bot.last_tp = 0.0
                bot.last_ticket = 0

        except Exception as exc:
            logger.debug(
                "Bot#%d position sync check failed: %s", bot.bot_id, exc,
            )

    def _apply_breakeven(self, bot: BotInstance, logger: logging.Logger) -> None:
        """Pull a winning trade's stop to breakeven once it's far enough in profit.

        This is the single highest-impact fix from the trade-export analysis.
        The account bled because losers ran to the **full** stop while winners
        were banked early — so the realised average loss (≈-$13) was far larger
        than the average win (≈+$8), turning a 55% win-rate into a net loss.

        Logic: for each open position, compute its risk ``R = |entry - SL|``.
        Once floating profit reaches ``breakeven_trigger_r × R``, move the stop
        to entry plus a small ``breakeven_lock_pips`` buffer. A trade that first
        goes green and then reverses now exits at ≈breakeven instead of a full
        loss, which collapses the avg-loss and flips expectancy positive.

        Skipped in dry-run (no live position to modify) and once the stop is
        already at/beyond breakeven (so we don't spam ``modify`` requests).
        """
        if bot.dry_run or not getattr(self.config, "breakeven_enabled", True):
            return
        if not bot.active_positions:
            return

        trigger_r = float(getattr(self.config, "breakeven_trigger_r", 0.6))
        lock_pips = float(getattr(self.config, "breakeven_lock_pips", 1.0))
        if trigger_r <= 0:
            return

        for pos in list(bot.active_positions):
            entry = pos.get("entry", 0.0)
            sl = pos.get("sl", 0.0)
            current = pos.get("price_current", 0.0)
            ticket = pos.get("ticket")
            ptype = pos.get("type")
            if not entry or not current or not ticket or not sl:
                continue

            risk = abs(entry - sl)
            if risk <= 0:
                continue

            # Already locked at/beyond breakeven? Nothing to do.
            if ptype == "BUY" and sl >= entry:
                continue
            if ptype == "SELL" and sl <= entry:
                continue

            profit_price = (current - entry) if ptype == "BUY" else (entry - current)
            if profit_price >= trigger_r * risk:
                ok = self.order_mgr.set_breakeven(ticket, offset_pips=lock_pips)
                if ok:
                    logger.info(
                        "Bot#%d breakeven set for #%s at +%.2fR (lock +%.1f pips).",
                        bot.bot_id, ticket, profit_price / risk, lock_pips,
                    )

    def _apply_time_stop(self, bot: BotInstance, logger: logging.Logger) -> None:
        """Force-close positions held longer than ``config.max_trade_hours``.

        The export showed several trades held for days (e.g. a −$76 AUDJPY open
        for ~6 days, and a −$76 position open ~9,000 minutes). Those are stuck
        trades that will neither hit SL nor TP and just tie up risk/margin. A
        hard maximum holding time caps that tail. Disabled when
        ``max_trade_hours <= 0``; skipped in dry-run.
        """
        if bot.dry_run:
            return
        max_hours = float(getattr(self.config, "max_trade_hours", 0))
        if max_hours <= 0 or not bot.active_positions:
            return

        now = time.time()
        for pos in list(bot.active_positions):
            opened = pos.get("open_time", 0)
            ticket = pos.get("ticket")
            if not opened or not ticket:
                continue
            age_hours = (now - opened) / 3600.0
            if age_hours >= max_hours:
                logger.info(
                    "Bot#%d time-stop closing #%s (held %.1fh ≥ %.1fh limit).",
                    bot.bot_id, ticket, age_hours, max_hours,
                )
                ok = self.order_mgr.close_position(ticket)
                if not ok:
                    logger.warning(
                        "Bot#%d time-stop close failed for #%s — will retry.",
                        bot.bot_id, ticket,
                    )

    def _apply_quick_exit(self, bot: BotInstance, logger: logging.Logger) -> None:
        """Close any open position the instant it reaches the quick-profit target.

        Independent of the strategy: works on *any* bot once toggled on. The
        floating profit is measured in pips from fill price to current price so
        it is lot-size independent and predictable. The actual close happens via
        the live order manager; ``_sync_position_state`` then reconciles state
        and records the closed trade on the next loop tick.
        """
        if not bot.quick_exit_enabled or bot.dry_run:
            return
        if not bot.active_positions:
            return

        target = float(bot.quick_exit_pips)
        if target <= 0:
            return

        sym_info = self.market_data.get_symbol_info(bot.symbol)
        if sym_info is None or sym_info.get("point", 0) <= 0:
            return
        point = sym_info["point"]
        digits = sym_info.get("digits", 5)
        trail = float(getattr(self.config, "quick_exit_trail_pips", 0.0))

        for pos in list(bot.active_positions):
            entry = pos.get("entry", 0.0)
            current = pos.get("price_current", 0.0)
            if not entry or not current:
                continue

            # Profit in price units → pips (BUY closes at bid, SELL at ask;
            # price_current already reflects the closeable side per position).
            if pos.get("type") == "BUY":
                profit_price = current - entry
            else:
                profit_price = entry - current
            profit_pips = price_to_pips(profit_price, bot.symbol, point)

            if profit_pips < target:
                continue
            ticket = pos.get("ticket")

            # Trailing-lock mode: instead of banking exactly +target and
            # eating the full stop on losers (inverted R:R), ratchet the SL
            # to lock in (profit − trail) pips and let the winner run. The
            # 3s poll keeps raising the lock as profit grows; a reversal
            # exits via the (now winning) stop.
            if trail > 0:
                lock_pips = profit_pips - trail
                lock_dist = pips_to_price(lock_pips, bot.symbol, point)
                cur_sl = pos.get("sl", 0.0) or 0.0
                if pos.get("type") == "BUY":
                    new_sl = round(entry + lock_dist, digits)
                    improves = new_sl > cur_sl
                else:
                    new_sl = round(entry - lock_dist, digits)
                    improves = cur_sl == 0.0 or new_sl < cur_sl
                if not improves:
                    continue
                if self.order_mgr.modify_position(ticket, sl=new_sl):
                    logger.info(
                        "Bot#%d Quick Exit trail — #%s at +%.1f pips, "
                        "SL locked at +%.1f pips (%.5f).",
                        bot.bot_id, ticket, profit_pips, lock_pips, new_sl,
                    )
                    continue
                # Broker refused the tight stop (min stop distance etc.) —
                # fall back to the classic hard close so profit is banked.
                logger.warning(
                    "Bot#%d Quick Exit trail modify failed for #%s — "
                    "falling back to hard close.",
                    bot.bot_id, ticket,
                )

            logger.info(
                "Bot#%d Quick Exit — closing #%s at +%.1f pips "
                "(target %.1f).",
                bot.bot_id, ticket, profit_pips, target,
            )
            ok = self.order_mgr.close_position(ticket)
            if not ok:
                logger.warning(
                    "Bot#%d Quick Exit close failed for #%s — will retry.",
                    bot.bot_id, ticket,
                )

    # Map MT5 deal reason codes to human-readable exit reasons.
    _DEAL_REASON_MAP = {
        0: "MANUAL",      # DEAL_REASON_CLIENT
        1: "MANUAL",      # DEAL_REASON_MOBILE
        2: "MANUAL",      # DEAL_REASON_WEB
        3: "EXPERT",      # DEAL_REASON_EXPERT
        4: "SL",          # DEAL_REASON_SL
        5: "TP",          # DEAL_REASON_TP
        6: "STOP_OUT",    # DEAL_REASON_SO
        7: "ROLLOVER",    # DEAL_REASON_ROLLOVER
    }

    def _record_closed_trade(
        self, bot: "BotInstance", ticket: int, logger: logging.Logger
    ) -> None:
        """Fetch MT5 deal history for a closed position and persist its outcome.

        Sums the closing (``DEAL_ENTRY_OUT``) deals for the position to obtain
        net profit, swap and commission, derives the exit reason from the deal
        ``reason`` code, and updates the matching trade row.
        """
        try:
            deals = mt5.history_deals_get(position=ticket)
            if not deals:
                logger.debug("Bot#%d no deal history for closed #%s", bot.bot_id, ticket)
                return

            exit_price = 0.0
            profit = swap = commission = 0.0
            close_epoch = 0
            reason_code = None
            for d in deals:
                # DEAL_ENTRY_OUT == 1 (also OUT_BY == 2 for hedged closes)
                if getattr(d, "entry", 0) in (1, 2):
                    profit += d.profit
                    swap += getattr(d, "swap", 0.0)
                    commission += getattr(d, "commission", 0.0)
                    exit_price = d.price
                    close_epoch = max(close_epoch, getattr(d, "time", 0))
                    reason_code = getattr(d, "reason", None)

            exit_reason = self._DEAL_REASON_MAP.get(reason_code, "UNKNOWN")
            close_time = (
                datetime.utcfromtimestamp(close_epoch).isoformat()
                if close_epoch else None
            )

            # Post-loss cooldown: a stop-out at a loss blocks re-entry for a
            # while. Winning SL exits (e.g. a trailed Quick-Exit stop) don't
            # trigger it — those are banked winners, not failed entries.
            cooldown_min = float(getattr(self.config, "sl_cooldown_min", 0))
            if (
                cooldown_min > 0
                and exit_reason in ("SL", "STOP_OUT")
                and profit < 0
            ):
                bot.cooldown_until = time.time() + cooldown_min * 60.0
                logger.info(
                    "Bot#%d stop-loss at %.2f — entry cooldown for %.0f min.",
                    bot.bot_id, profit, cooldown_min,
                )

            # Pips relative to the stored entry, if we can resolve point size.
            pips = None
            try:
                trade_row = self.history_db.get_trade_by_ticket(ticket)
                sym_info = self.market_data.get_symbol_info(bot.symbol)
                if trade_row and sym_info and trade_row.get("entry_price"):
                    point = sym_info["point"]
                    pip_size = point * 10 if sym_info.get("digits", 5) in (3, 5) else point
                    diff = exit_price - trade_row["entry_price"]
                    # SELL trades profit when price falls
                    if trade_row["action"] == "SELL":
                        diff = -diff
                    pips = round(diff / pip_size, 1) if pip_size else None
            except Exception:
                pips = None

            updated = self.history_db.record_trade_close(
                ticket=ticket,
                exit_price=exit_price,
                profit=round(profit, 2),
                swap=round(swap, 2),
                commission=round(commission, 2),
                pips=pips,
                exit_reason=exit_reason,
                close_time=close_time,
            )
            if updated:
                logger.info(
                    "Bot#%d trade #%s closed: %s profit=%.2f reason=%s",
                    bot.bot_id, ticket, bot.symbol, profit, exit_reason,
                )
        except Exception as exc:
            logger.debug("Bot#%d _record_closed_trade(#%s) failed: %s",
                         bot.bot_id, ticket, exc)

    def backfill_trades_from_mt5(self, days: int = 90) -> dict:
        """Reconstruct past CLOSED live trades from MT5 deal history.

        Groups account deals by ``position_id``, pairs the entry (IN) deal with
        its close (OUT) deals, and imports each completed position into the
        ``trades`` table.  Idempotent — re-running skips already-imported
        tickets.  Bot id / strategy are parsed from the deal comment
        (``Bot#<id>|<strategy>``) when present.

        Returns a summary dict: ``{"imported", "positions", "skipped"}``.
        """
        logger = logging.getLogger("MultiBot")
        to_date = datetime.now() + timedelta(days=1)
        from_date = datetime.now() - timedelta(days=max(1, days))
        try:
            deals = mt5.history_deals_get(from_date, to_date)
        except Exception as exc:
            logger.error("backfill: history_deals_get failed: %s", exc)
            return {"imported": 0, "positions": 0, "skipped": 0, "error": str(exc)}
        if not deals:
            return {"imported": 0, "positions": 0, "skipped": 0}

        groups: dict[int, list] = {}
        for d in deals:
            pid = getattr(d, "position_id", 0)
            if pid:
                groups.setdefault(pid, []).append(d)

        imported = skipped = 0
        for pid, dl in groups.items():
            in_deals = [d for d in dl if getattr(d, "entry", -1) == 0]      # DEAL_ENTRY_IN
            out_deals = [d for d in dl if getattr(d, "entry", -1) in (1, 3)]  # OUT / OUT_BY
            if not in_deals or not out_deals:
                skipped += 1  # still open, or incomplete history
                continue

            ind = in_deals[0]
            symbol = ind.symbol
            action = "BUY" if ind.type == 0 else "SELL"  # DEAL_TYPE_BUY == 0
            entry_price = ind.price
            open_time = datetime.utcfromtimestamp(ind.time).isoformat()

            profit = sum(d.profit for d in out_deals)
            swap = sum(getattr(d, "swap", 0.0) for d in dl)
            commission = sum(getattr(d, "commission", 0.0) for d in dl)
            exit_price = out_deals[-1].price
            close_time = datetime.utcfromtimestamp(
                max(d.time for d in out_deals)
            ).isoformat()
            exit_reason = self._DEAL_REASON_MAP.get(
                getattr(out_deals[-1], "reason", None), "UNKNOWN"
            )

            # Attribute to a bot via the comment, if it follows our convention.
            comment = ind.comment or out_deals[-1].comment or ""
            bot_id = None
            strategy = ""
            m = re.match(r"Bot#(\d+)\|?(.*)", comment)
            if m:
                bot_id = int(m.group(1))
                strategy = m.group(2)

            pips = None
            try:
                si = mt5.symbol_info(symbol)
                if si:
                    pip_size = si.point * 10 if si.digits in (3, 5) else si.point
                    diff = exit_price - entry_price
                    if action == "SELL":
                        diff = -diff
                    pips = round(diff / pip_size, 1) if pip_size else None
            except Exception:
                pips = None

            if self.history_db.import_closed_trade(
                ticket=pid, symbol=symbol, action=action, lot=ind.volume,
                entry_price=entry_price, exit_price=exit_price,
                profit=round(profit, 2), open_time=open_time, close_time=close_time,
                swap=round(swap, 2), commission=round(commission, 2), pips=pips,
                exit_reason=exit_reason, bot_id=bot_id, strategy=strategy,
                reason=comment,
            ):
                imported += 1
            else:
                skipped += 1  # already imported

        logger.info("backfill: imported=%d skipped=%d positions=%d",
                    imported, skipped, len(groups))
        return {"imported": imported, "positions": len(groups), "skipped": skipped}

    @staticmethod
    def _seconds_until_next_candle(timeframe_key: str) -> int:
        """Estimate seconds remaining until the current candle closes."""
        tf_seconds = {
            "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
            "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800,
        }
        interval = tf_seconds.get(timeframe_key, 300)
        now_ts = int(datetime.utcnow().timestamp())
        elapsed = now_ts % interval
        return interval - elapsed


# ======================================================================
#  Interactive Multi-Bot Menu (used by bot.py)
# ======================================================================
def multi_bot_menu(manager: MultiBotManager, config: Config) -> None:
    """Interactive menu for managing multiple bots."""
    while True:
        _show_multi_dashboard(manager)
        choice = input(f"{Fore.GREEN}  ➤ Select option: {Style.RESET_ALL}").strip()
        print()

        if choice == "0":
            # Stop all before exiting
            running = manager.running_count
            if running > 0:
                confirm = input(
                    f"  {Fore.YELLOW}⚠ {running} bot(s) running. Stop all? (y/n): {Style.RESET_ALL}"
                ).strip().lower()
                if confirm in ("y", "yes"):
                    manager.stop_all()
                    print(f"  {Fore.GREEN}✔ All bots stopped.{Style.RESET_ALL}")
            break

        elif choice == "1":
            _add_bot_wizard(manager, config)

        elif choice == "2":
            _show_bot_status(manager)

        elif choice == "3":
            _start_bot_menu(manager)

        elif choice == "4":
            _stop_bot_menu(manager)

        elif choice == "5":
            _remove_bot_menu(manager)

        elif choice == "6":
            count = manager.start_all()
            print(f"  {Fore.GREEN}✔ Started {count} bot(s).{Style.RESET_ALL}")

        elif choice == "7":
            count = manager.stop_all()
            print(f"  {Fore.GREEN}✔ Stopped {count} bot(s).{Style.RESET_ALL}")

        elif choice == "8":
            _toggle_bot_dry_run(manager)

        else:
            print(f"  {Fore.RED}Invalid option.{Style.RESET_ALL}")

        input(f"\n{Fore.CYAN}  Press Enter to continue …{Style.RESET_ALL}")


def _show_multi_dashboard(manager: MultiBotManager) -> None:
    """Print the multi-bot management menu."""
    print(f"\n{Fore.CYAN}")
    print("  ╔══════════════════════════════════════════╗")
    print("  ║         MULTI-BOT CONTROL CENTER         ║")
    print("  ╠══════════════════════════════════════════╣")
    print("  ║  1. Add New Bot                          ║")
    print("  ║  2. View All Bots (Dashboard)            ║")
    print("  ║  3. Start a Bot                          ║")
    print("  ║  4. Stop a Bot                           ║")
    print("  ║  5. Remove a Bot                         ║")
    print("  ║  6. Start ALL Bots                       ║")
    print("  ║  7. Stop ALL Bots                        ║")
    print("  ║  8. Toggle Dry Run (per bot)             ║")
    print("  ║  0. Back to Main Menu                    ║")
    print("  ╚══════════════════════════════════════════╝")
    print(f"{Style.RESET_ALL}")

    # Quick status bar
    total = manager.total_count
    running = manager.running_count
    colour = Fore.GREEN if running > 0 else Fore.YELLOW
    print(
        f"  {Fore.WHITE}Bots: {colour}{running} running{Fore.WHITE}"
        f" / {total} total{Style.RESET_ALL}"
    )
    print()


def _show_bot_status(manager: MultiBotManager) -> None:
    """Print a dashboard table of all bots."""
    bots = manager.get_all_bots()
    if not bots:
        print(f"  {Fore.YELLOW}No bots configured. Add one first!{Style.RESET_ALL}")
        return

    print(f"  {Fore.CYAN}══════════════════════════════════════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(
        f"  {Fore.WHITE}{'ID':<5} {'Symbol':<10} {'Strategy':<12} {'TF':<5} "
        f"{'Lot':<8} {'Status':<10} {'Signal':<10} {'Price':<12} "
        f"{'Trades':<8} {'Last Check':<10} {'Mode':<5}{Style.RESET_ALL}"
    )
    print(f"  {Fore.CYAN}{'─'*90}{Style.RESET_ALL}")

    for bot in bots:
        # Status colour
        if bot.status == BotStatus.RUNNING:
            status_colour = Fore.GREEN
        elif bot.status == BotStatus.ERROR:
            status_colour = Fore.RED
        elif bot.status == BotStatus.STOPPED:
            status_colour = Fore.YELLOW
        else:
            status_colour = Fore.WHITE

        # Signal colour
        if bot.last_signal == "BUY":
            sig_colour = Fore.GREEN
        elif bot.last_signal == "SELL":
            sig_colour = Fore.RED
        else:
            sig_colour = Fore.WHITE

        lot_display = "auto" if bot.lot_mode == "auto" else f"{bot.lot_size:.2f}"
        mode = "DRY" if bot.dry_run else "LIVE"
        mode_colour = Fore.YELLOW if bot.dry_run else Fore.RED

        print(
            f"  {Fore.WHITE}{bot.bot_id:<5} "
            f"{bot.symbol:<10} "
            f"{bot.strategy_name:<12} "
            f"{bot.timeframe_key:<5} "
            f"{lot_display:<8} "
            f"{status_colour}{bot.status.value:<10}{Style.RESET_ALL} "
            f"{sig_colour}{bot.last_signal:<10}{Style.RESET_ALL} "
            f"{Fore.YELLOW}{bot.last_price:<12.5f}{Style.RESET_ALL} "
            f"{Fore.WHITE}{bot.trades_taken:<8} "
            f"{bot.last_check:<10} "
            f"{mode_colour}{mode:<5}{Style.RESET_ALL}"
        )

        if bot.status == BotStatus.ERROR and bot.error_msg:
            print(f"       {Fore.RED}↳ Error: {bot.error_msg}{Style.RESET_ALL}")

    print(f"  {Fore.CYAN}══════════════════════════════════════════════════════════════════════════════════════{Style.RESET_ALL}")


def _add_bot_wizard(manager: MultiBotManager, config: Config) -> None:
    """Step-by-step wizard to add a new bot."""
    print(f"  {Fore.CYAN}── Add New Bot ──{Style.RESET_ALL}\n")

    # 1. Symbol
    print(f"  {Fore.WHITE}Symbol Categories:{Style.RESET_ALL}")
    categories = list(SYMBOL_GROUPS.keys())
    for idx, cat in enumerate(categories, 1):
        symbols = ", ".join(SYMBOL_GROUPS[cat])
        print(f"    {idx}. {cat}: {Fore.YELLOW}{symbols}{Style.RESET_ALL}")
    print(f"    {len(categories)+1}. Custom symbol")

    cat_choice = input(f"\n  {Fore.GREEN}Category #: {Style.RESET_ALL}").strip()
    try:
        cat_idx = int(cat_choice) - 1
    except ValueError:
        print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
        return

    if cat_idx == len(categories):
        symbol = input(f"  {Fore.GREEN}Enter symbol: {Style.RESET_ALL}").strip().upper()
    elif 0 <= cat_idx < len(categories):
        cat_name = categories[cat_idx]
        symbols = SYMBOL_GROUPS[cat_name]
        print()
        for idx, sym in enumerate(symbols, 1):
            print(f"    {idx}. {sym}")
        sym_choice = input(f"\n  {Fore.GREEN}Symbol #: {Style.RESET_ALL}").strip()
        try:
            sym_idx = int(sym_choice) - 1
            symbol = symbols[sym_idx]
        except (ValueError, IndexError):
            print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
            return
    else:
        print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
        return

    # 2. Strategy
    strategies = list_strategies()
    print(f"\n  {Fore.WHITE}Strategies:{Style.RESET_ALL}")
    for idx, (key, desc) in enumerate(strategies, 1):
        print(f"    {idx}. {desc} [{key}]")
    strat_choice = input(f"\n  {Fore.GREEN}Strategy #: {Style.RESET_ALL}").strip()
    try:
        strat_idx = int(strat_choice) - 1
        strategy_name = strategies[strat_idx][0]
    except (ValueError, IndexError):
        print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
        return

    # 3. Timeframe
    print(f"\n  {Fore.WHITE}Timeframes:{Style.RESET_ALL}")
    for idx, (key, display) in enumerate(TIMEFRAME_OPTIONS, 1):
        print(f"    {idx}. {display} ({key})")
    tf_choice = input(f"\n  {Fore.GREEN}Timeframe #: {Style.RESET_ALL}").strip()
    try:
        tf_idx = int(tf_choice) - 1
        timeframe_key = TIMEFRAME_OPTIONS[tf_idx][0]
    except (ValueError, IndexError):
        print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
        return

    # 4. Lot size
    print(f"\n  {Fore.WHITE}Lot Size:{Style.RESET_ALL}")
    print(f"    1. Fixed lot")
    print(f"    2. Auto (risk-based)")
    lot_choice = input(f"\n  {Fore.GREEN}Option #: {Style.RESET_ALL}").strip()

    lot_mode = "fixed"
    lot_size = config.default_lot_size

    if lot_choice == "2":
        lot_mode = "auto"
    else:
        raw = input(f"  {Fore.GREEN}Lot size (e.g. 0.01): {Style.RESET_ALL}").strip()
        try:
            lot_size = float(raw)
        except ValueError:
            lot_size = config.default_lot_size

    # 5. Dry run
    dry_choice = input(
        f"\n  {Fore.GREEN}Dry run? (y/n) [y]: {Style.RESET_ALL}"
    ).strip().lower()
    dry_run = dry_choice not in ("n", "no")

    # Create the bot
    bot = manager.add_bot(
        symbol=symbol,
        strategy_name=strategy_name,
        timeframe_key=timeframe_key,
        lot_size=lot_size,
        lot_mode=lot_mode,
        dry_run=dry_run,
    )

    print(f"\n  {Fore.GREEN}✔ Created: {bot.label}{Style.RESET_ALL}")

    # Auto-start?
    start = input(
        f"  {Fore.GREEN}Start now? (y/n) [y]: {Style.RESET_ALL}"
    ).strip().lower()
    if start not in ("n", "no"):
        if manager.start_bot(bot.bot_id):
            print(f"  {Fore.GREEN}✔ Bot#{bot.bot_id} started!{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}✘ Failed to start: {bot.error_msg}{Style.RESET_ALL}")


def _start_bot_menu(manager: MultiBotManager) -> None:
    """Pick a bot to start."""
    bots = [b for b in manager.get_all_bots() if b.status != BotStatus.RUNNING]
    if not bots:
        print(f"  {Fore.YELLOW}No idle bots to start.{Style.RESET_ALL}")
        return

    print(f"  {Fore.CYAN}── Start a Bot ──{Style.RESET_ALL}")
    for b in bots:
        print(f"  {b.bot_id}. {b.label}  [{b.status.value}]")

    choice = input(f"\n  {Fore.GREEN}Bot ID to start: {Style.RESET_ALL}").strip()
    try:
        bot_id = int(choice)
    except ValueError:
        print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
        return

    if manager.start_bot(bot_id):
        print(f"  {Fore.GREEN}✔ Bot#{bot_id} started!{Style.RESET_ALL}")
    else:
        bot = manager.get_bot(bot_id)
        msg = bot.error_msg if bot else "Bot not found"
        print(f"  {Fore.RED}✘ Failed: {msg}{Style.RESET_ALL}")


def _stop_bot_menu(manager: MultiBotManager) -> None:
    """Pick a bot to stop."""
    bots = [b for b in manager.get_all_bots() if b.status == BotStatus.RUNNING]
    if not bots:
        print(f"  {Fore.YELLOW}No running bots to stop.{Style.RESET_ALL}")
        return

    print(f"  {Fore.CYAN}── Stop a Bot ──{Style.RESET_ALL}")
    for b in bots:
        print(
            f"  {b.bot_id}. {b.symbol} | {b.strategy_name} | {b.timeframe_key} "
            f"| Signal: {b.last_signal} | Trades: {b.trades_taken}"
        )

    choice = input(f"\n  {Fore.GREEN}Bot ID to stop: {Style.RESET_ALL}").strip()
    try:
        bot_id = int(choice)
    except ValueError:
        print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
        return

    if manager.stop_bot(bot_id):
        print(f"  {Fore.GREEN}✔ Bot#{bot_id} stopped.{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}✘ Bot not found.{Style.RESET_ALL}")


def _remove_bot_menu(manager: MultiBotManager) -> None:
    """Pick a bot to remove permanently."""
    bots = manager.get_all_bots()
    if not bots:
        print(f"  {Fore.YELLOW}No bots to remove.{Style.RESET_ALL}")
        return

    print(f"  {Fore.CYAN}── Remove a Bot ──{Style.RESET_ALL}")
    for b in bots:
        print(f"  {b.bot_id}. {b.label}  [{b.status.value}]")

    choice = input(f"\n  {Fore.GREEN}Bot ID to remove: {Style.RESET_ALL}").strip()
    try:
        bot_id = int(choice)
    except ValueError:
        print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
        return

    confirm = input(
        f"  {Fore.YELLOW}Remove Bot#{bot_id}? (y/n): {Style.RESET_ALL}"
    ).strip().lower()
    if confirm not in ("y", "yes"):
        print(f"  {Fore.CYAN}Cancelled.{Style.RESET_ALL}")
        return

    if manager.remove_bot(bot_id):
        print(f"  {Fore.GREEN}✔ Bot#{bot_id} removed.{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}✘ Bot not found.{Style.RESET_ALL}")


def _toggle_bot_dry_run(manager: MultiBotManager) -> None:
    """Toggle dry-run mode for a specific bot."""
    bots = manager.get_all_bots()
    if not bots:
        print(f"  {Fore.YELLOW}No bots configured.{Style.RESET_ALL}")
        return

    print(f"  {Fore.CYAN}── Toggle Dry Run ──{Style.RESET_ALL}")
    for b in bots:
        mode = f"{Fore.YELLOW}DRY RUN" if b.dry_run else f"{Fore.RED}⚠ LIVE"
        print(
            f"  {b.bot_id}. {b.symbol} | {b.strategy_name} | {b.timeframe_key} "
            f"| {mode}{Style.RESET_ALL}  [{b.status.value}]"
        )

    choice = input(f"\n  {Fore.GREEN}Bot ID to toggle: {Style.RESET_ALL}").strip()
    try:
        bot_id = int(choice)
    except ValueError:
        print(f"  {Fore.RED}Invalid.{Style.RESET_ALL}")
        return

    bot = manager.get_bot(bot_id)
    if bot is None:
        print(f"  {Fore.RED}Bot not found.{Style.RESET_ALL}")
        return

    # Must be stopped to toggle
    if bot.status == BotStatus.RUNNING:
        print(f"  {Fore.YELLOW}⚠ Stop the bot first before changing mode.{Style.RESET_ALL}")
        return

    if bot.dry_run:
        # Switching to LIVE — show warning
        print(f"\n  {Fore.RED}{'='*50}")
        print(f"  ⚠  WARNING: SWITCHING TO LIVE TRADING  ⚠")
        print(f"  This will place REAL orders with REAL money!")
        print(f"  {'='*50}{Style.RESET_ALL}")
        confirm = input(
            f"  {Fore.RED}Type 'LIVE' to confirm: {Style.RESET_ALL}"
        ).strip()
        if confirm != "LIVE":
            print(f"  {Fore.CYAN}Cancelled. Staying in dry-run.{Style.RESET_ALL}")
            return
        bot.dry_run = False
        print(f"  {Fore.RED}✔ Bot#{bot_id} is now in LIVE mode!{Style.RESET_ALL}")
    else:
        bot.dry_run = True
        print(f"  {Fore.GREEN}✔ Bot#{bot_id} is now in DRY RUN mode.{Style.RESET_ALL}")

    # Persist the change so it survives restarts
    manager._save_bots()
    manager.history_db.record_mode_changed(bot_id, bot.dry_run)
