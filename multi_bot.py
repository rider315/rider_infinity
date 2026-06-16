"""
multi_bot.py — Rider Infinity Multi-Bot Manager for running multiple trading bots simultaneously.

Each bot instance runs in its own thread with its own symbol, strategy,
timeframe, and lot configuration.  The manager provides a dashboard to
monitor, add, and remove bots in real-time.
"""

from __future__ import annotations

import json
import logging
import threading
import time
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
from session_manager import SessionManager
from strategies import get_strategy, list_strategies
from strategies.base import SignalAction
from utils import pips_to_price, save_trade_to_journal, setup_logger
from bot_history import BotHistoryDB


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

        # Ensure symbol is available
        if not self.connector.ensure_symbol(bot.symbol):
            bot.status = BotStatus.ERROR
            bot.error_msg = f"Symbol {bot.symbol} not available"
            return False

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

                # -- Market open check --
                if not self.session_mgr.is_market_open():
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
                df = self.market_data.get_candles(bot.symbol, mt5_tf, count=500)
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
                    # Guard: skip if this bot already has an open position
                    if bot.active_tickets:
                        bot.blocked_reason = f"Already has {len(bot.active_tickets)} open position(s)"
                        logger.debug(
                            "Bot#%d signal=%s but already has %d active position(s) — skipping.",
                            bot.bot_id, signal.action.value, len(bot.active_tickets),
                        )
                    else:
                        bot.blocked_reason = ""  # clear on attempt
                        self._execute_signal(bot, signal, logger)
                else:
                    bot.blocked_reason = ""  # clear when no signal

                # -- Sync position state (detect externally closed trades) --
                self._sync_position_state(bot, logger)

                # -- Sleep until next candle --
                sleep_sec = self._seconds_until_next_candle(bot.timeframe_key)
                sleep_sec = max(5, sleep_sec)
                bot._stop_event.wait(timeout=sleep_sec)

        except Exception as exc:
            bot.status = BotStatus.ERROR
            bot.error_msg = str(exc)
            logger.error("Bot#%d crashed: %s", bot.bot_id, exc, exc_info=True)
            self.history_db.record_bot_error(bot.bot_id, str(exc))
            return

        bot.status = BotStatus.STOPPED

    def _execute_signal(self, bot: BotInstance, signal, logger: logging.Logger) -> None:
        """Place an order for a bot's signal."""
        symbol = signal.symbol or bot.symbol
        sl_pips = signal.sl_pips or self.config.default_sl_pips
        tp_pips = signal.tp_pips or self.config.default_tp_pips

        # Lot size
        if bot.lot_mode == "auto":
            lot = self.risk_mgr.calculate_lot_size(symbol, sl_pips)
        else:
            lot = bot.lot_size

        # Risk check
        can, reason = self.risk_mgr.can_trade(symbol, lot)
        if not can:
            bot.blocked_reason = reason
            logger.info("Bot#%d trade blocked: %s", bot.bot_id, reason)
            return
        bot.blocked_reason = ""  # clear on successful attempt

        # SL / TP prices
        tick = self.market_data.get_tick(symbol)
        if tick is None:
            return
        sym_info = self.market_data.get_symbol_info(symbol)
        if sym_info is None:
            return

        point = sym_info["point"]
        digits = sym_info["digits"]

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
        else:
            msg = OrderManager.get_retcode_message(result.get("retcode", -1))
            logger.error("Bot#%d order failed: %s", bot.bot_id, msg)

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
