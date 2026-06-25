"""
bot.py — Main Orchestrator for the Rider Infinity Trading Platform.

Launch this file to start the interactive CLI trading system.

Usage
-----
Interactive mode::

    python bot.py

CLI / non-interactive mode::

    python bot.py --strategy ema --symbol EURUSD --timeframe M5 --lot 0.01 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
except ImportError:
    # Graceful fallback when colorama is not installed.
    class _Stub:
        """Provides empty-string attributes so colour references compile."""

        def __getattr__(self, _name: str) -> str:
            return ""

    Fore = Style = _Stub()  # type: ignore[assignment]

import MetaTrader5 as mt5

from config import Config, SYMBOL_GROUPS, TIMEFRAME_OPTIONS, TIMEFRAME_MAP
from mt5_connector import MT5Connector
from market_data import MarketDataFeed
from order_manager import OrderManager
from session_manager import SessionManager

from risk_manager import RiskManager
from multi_bot import MultiBotManager, multi_bot_menu
from strategies import get_strategy, list_strategies, STRATEGY_REGISTRY
from strategies.base import SignalAction
from evaluation_matrix import EvaluationMatrix, EvalScore, EvaluatorState, AggregationMode
from trailing_profiles import get_trailing_profile, TrailingState
from automation.engine import AutomationEngine
from profile_manager import ProfileManager, TradingProfile
from event_bus import EventBus, EventChannel, emit_sync
from utils import (
    clear_screen,
    format_account_info,
    pips_to_price,
    print_banner,
    save_trade_to_journal,
    setup_logger,
)


# ======================================================================
#  RiderInfinityBot — the main application class
# ======================================================================
class RiderInfinityBot:
    """Rider Infinity — Interactive trading platform with CLI menus."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.config = Config()
        self.logger = setup_logger("RiderInfinity")

        # Active selections (overridden interactively or via CLI args)
        self.symbol: str = self.config.default_symbol
        self.timeframe_key: str = self.config.default_timeframe
        self.strategy_name: Optional[str] = None
        self.lot_size: float = self.config.default_lot_size
        self.lot_mode: str = "fixed"  # "fixed" | "auto"
        self.dry_run: bool = self.config.dry_run

        # Components — initialised in run()
        self.connector: Optional[MT5Connector] = None
        self.market_data: Optional[MarketDataFeed] = None
        self.session_mgr: Optional[SessionManager] = None
        self.order_mgr: Optional[OrderManager] = None
        self.risk_mgr: Optional[RiskManager] = None
        self.multi_mgr: Optional[MultiBotManager] = None

        # --- OctoBot-inspired components ---
        self.automation_engine: Optional[AutomationEngine] = None
        self.profile_mgr: Optional[ProfileManager] = None
        self.trailing_states: dict = {}  # ticket -> TrailingState
        self.trailing_profile_name: str = "atr"  # default trailing profile

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, args: Optional[argparse.Namespace] = None) -> None:
        """
        Connect to MT5, optionally apply CLI arguments, and either start
        trading directly (CLI mode) or enter the interactive menu loop.
        """
        print_banner()

        # --- MT5 connection -------------------------------------------------
        self.connector = MT5Connector(self.config)
        if not self.connector.connect():
            print(f"{Fore.RED}✘ Failed to connect to MetaTrader 5.{Style.RESET_ALL}")
            print("  Check your .env / Config settings and make sure MT5 is running.")
            sys.exit(1)

        self.market_data = MarketDataFeed()
        self.session_mgr = SessionManager()
        self.order_mgr = OrderManager(self.config, dry_run=self.dry_run)
        self.risk_mgr = RiskManager(self.config, self.connector)
        self.risk_mgr.reset_daily_tracking()

        self._show_account_summary()

        # --- CLI (non-interactive) mode -------------------------------------
        if args and args.strategy:
            self._apply_cli_args(args)
            self.start_trading()
            return

        # --- Interactive menu loop ------------------------------------------
        try:
            self._menu_loop()
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted — shutting down …{Style.RESET_ALL}")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _shutdown(self) -> None:
        if self.connector:
            self.connector.disconnect()
        print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")

    # ------------------------------------------------------------------
    # CLI args
    # ------------------------------------------------------------------
    def _apply_cli_args(self, args: argparse.Namespace) -> None:
        """Apply command-line arguments, overriding interactive defaults."""
        if args.symbol:
            self.symbol = args.symbol.upper()
        if args.timeframe:
            self.timeframe_key = args.timeframe.upper()
        if args.strategy:
            self.strategy_name = args.strategy.lower()
        if args.lot:
            self.lot_size = args.lot
            self.lot_mode = "fixed"
        if args.auto_lot:
            self.lot_mode = "auto"
        if args.dry_run:
            self.dry_run = True
            if self.order_mgr:
                self.order_mgr.dry_run = True

        print(f"{Fore.CYAN}CLI mode — bypassing interactive menus.{Style.RESET_ALL}")

    # ------------------------------------------------------------------
    # Account summary
    # ------------------------------------------------------------------
    def _show_account_summary(self) -> None:
        assert self.connector is not None
        info = self.connector.get_account_info()
        if info is None:
            print(f"{Fore.YELLOW}⚠ Could not retrieve account info.{Style.RESET_ALL}")
            return
        # info is already a dict — format_account_info expects a dict
        print(format_account_info(info))
        print()

    # ===================================================================
    #  INTERACTIVE MENU SYSTEM
    # ===================================================================
    def _menu_loop(self) -> None:
        """Main interactive menu loop."""
        while True:
            self._show_main_menu()
            choice = input(f"{Fore.GREEN}  ➤ Select option: {Style.RESET_ALL}").strip()
            print()

            actions = {
                "1": self.start_trading,
                "2": self.select_symbol,
                "3": self.select_timeframe,
                "4": self.select_strategy,
                "5": self.set_lot_size,
                "6": self.view_account_info,
                "7": self.view_open_positions,
                "8": self.close_all_positions_menu,
                "9": self.toggle_dry_run,
                "10": self.show_session_info,
                "11": self.multi_bot_control,
                "12": self.profile_manager_menu,
                "13": self.automation_menu,
                "14": self.optimizer_menu,
                "15": self.risk_dashboard_menu,
                "0": None,  # exit sentinel
            }

            if choice == "0":
                break

            action = actions.get(choice)
            if action is None and choice != "0":
                print(f"{Fore.RED}  Invalid option.{Style.RESET_ALL}")
                continue
            if action:
                try:
                    action()
                except KeyboardInterrupt:
                    print(f"\n{Fore.YELLOW}  Returning to menu …{Style.RESET_ALL}")

            input(f"\n{Fore.CYAN}  Press Enter to continue …{Style.RESET_ALL}")

    # ------------------------------------------------------------------
    def _show_main_menu(self) -> None:
        clear_screen()
        dry_tag = f" {Fore.YELLOW}[DRY RUN]{Style.RESET_ALL}" if self.dry_run else ""
        strat_tag = self.strategy_name or "not selected"

        print(f"{Fore.YELLOW}")
        print("  +--------------------------------------+")
        print("  |       RIDER INFINITY (RI) v2.0       |")
        print("  +--------------------------------------+")
        print("  |  1. Start Trading (Single Bot)       |")
        print("  |  2. Select Symbol                    |")
        print("  |  3. Select Timeframe                 |")
        print("  |  4. Select Strategy                  |")
        print("  |  5. Set Lot Size                     |")
        print("  |  6. View Account Info                |")
        print("  |  7. View Open Positions              |")
        print("  |  8. Close All Positions              |")
        print("  |  9. Toggle Dry Run                   |")
        print("  | 10. Market Session Info              |")
        print("  | 11. Multi-Bot Manager                |")
        print("  +--------------------------------------+")
        print("  | 12. Profile Manager          [NEW]   |")
        print("  | 13. Automation Rules         [NEW]   |")
        print("  | 14. Strategy Optimizer       [NEW]   |")
        print("  | 15. Risk Dashboard           [NEW]   |")
        print("  |  0. Exit                             |")
        print("  +--------------------------------------+")
        print(f"{Style.RESET_ALL}")

        # Current settings strip
        print(
            f"  {Fore.WHITE}Symbol: {Fore.YELLOW}{self.symbol}  "
            f"{Fore.WHITE}TF: {Fore.YELLOW}{self.timeframe_key}  "
            f"{Fore.WHITE}Strategy: {Fore.YELLOW}{strat_tag}  "
            f"{Fore.WHITE}Lot: {Fore.YELLOW}"
            f"{'auto' if self.lot_mode == 'auto' else self.lot_size}"
            f"{dry_tag}{Style.RESET_ALL}"
        )
        print()

    # ===================================================================
    #  MENU ACTIONS
    # ===================================================================

    # ---- 2. Select Symbol ------------------------------------------------
    def select_symbol(self) -> None:
        """Interactive symbol picker grouped by category."""
        groups = SYMBOL_GROUPS
        categories = list(groups.keys())

        print(f"  {Fore.CYAN}── Symbol Categories ──{Style.RESET_ALL}")
        for idx, cat in enumerate(categories, 1):
            print(f"  {Fore.WHITE}{idx}. {cat}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}{len(categories)+1}. Custom symbol{Style.RESET_ALL}")

        cat_choice = input(f"\n  {Fore.GREEN}Category #: {Style.RESET_ALL}").strip()
        try:
            cat_idx = int(cat_choice) - 1
        except ValueError:
            print(f"  {Fore.RED}Invalid input.{Style.RESET_ALL}")
            return

        # Custom symbol
        if cat_idx == len(categories):
            custom = input(f"  {Fore.GREEN}Enter symbol (e.g. EURUSD): {Style.RESET_ALL}").strip().upper()
            if custom:
                self.symbol = custom
                print(f"  {Fore.GREEN}✔ Symbol set to {self.symbol}{Style.RESET_ALL}")
            return

        if not 0 <= cat_idx < len(categories):
            print(f"  {Fore.RED}Invalid category.{Style.RESET_ALL}")
            return

        cat_name = categories[cat_idx]
        symbols = groups[cat_name]

        print(f"\n  {Fore.CYAN}── {cat_name} ──{Style.RESET_ALL}")
        for idx, sym in enumerate(symbols, 1):
            print(f"  {Fore.WHITE}{idx}. {sym}{Style.RESET_ALL}")

        sym_choice = input(f"\n  {Fore.GREEN}Symbol #: {Style.RESET_ALL}").strip()
        try:
            sym_idx = int(sym_choice) - 1
        except ValueError:
            print(f"  {Fore.RED}Invalid input.{Style.RESET_ALL}")
            return

        if 0 <= sym_idx < len(symbols):
            self.symbol = symbols[sym_idx]
            print(f"  {Fore.GREEN}✔ Symbol set to {self.symbol}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}Invalid selection.{Style.RESET_ALL}")

    # ---- 3. Select Timeframe ----------------------------------------------
    def select_timeframe(self) -> None:
        """Interactive timeframe picker."""
        options = TIMEFRAME_OPTIONS  # list of (key, display_name)

        print(f"  {Fore.CYAN}── Timeframes ──{Style.RESET_ALL}")
        for idx, (key, display) in enumerate(options, 1):
            marker = " ◄" if key == self.timeframe_key else ""
            print(f"  {Fore.WHITE}{idx}. {display} ({key}){Fore.YELLOW}{marker}{Style.RESET_ALL}")

        choice = input(f"\n  {Fore.GREEN}Timeframe #: {Style.RESET_ALL}").strip()
        try:
            idx = int(choice) - 1
        except ValueError:
            print(f"  {Fore.RED}Invalid input.{Style.RESET_ALL}")
            return

        if 0 <= idx < len(options):
            self.timeframe_key = options[idx][0]
            print(f"  {Fore.GREEN}✔ Timeframe set to {self.timeframe_key}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}Invalid selection.{Style.RESET_ALL}")

    # ---- 4. Select Strategy -----------------------------------------------
    def select_strategy(self) -> None:
        """Interactive strategy picker."""
        strategies = list_strategies()  # [(key, description), ...]
        if not strategies:
            print(f"  {Fore.RED}No strategies registered.{Style.RESET_ALL}")
            return

        print(f"  {Fore.CYAN}── Available Strategies ──{Style.RESET_ALL}")
        for idx, (key, desc) in enumerate(strategies, 1):
            marker = " ◄" if key == self.strategy_name else ""
            print(f"  {Fore.WHITE}{idx}. {desc} [{key}]{Fore.YELLOW}{marker}{Style.RESET_ALL}")

        choice = input(f"\n  {Fore.GREEN}Strategy #: {Style.RESET_ALL}").strip()
        try:
            idx = int(choice) - 1
        except ValueError:
            print(f"  {Fore.RED}Invalid input.{Style.RESET_ALL}")
            return

        if 0 <= idx < len(strategies):
            self.strategy_name = strategies[idx][0]  # store the key, not the tuple
            print(f"  {Fore.GREEN}✔ Strategy set to {self.strategy_name}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}Invalid selection.{Style.RESET_ALL}")

    # ---- 5. Set Lot Size ---------------------------------------------------
    def set_lot_size(self) -> None:
        """Prompt user for a fixed lot size or auto (risk-based) mode."""
        print(f"  {Fore.CYAN}── Lot Size ──{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}1. Fixed lot size{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}2. Auto (risk-based sizing){Style.RESET_ALL}")

        choice = input(f"\n  {Fore.GREEN}Option #: {Style.RESET_ALL}").strip()

        if choice == "2":
            self.lot_mode = "auto"
            print(
                f"  {Fore.GREEN}✔ Lot mode set to AUTO "
                f"(risk = {self.config.max_risk_per_trade}% of equity){Style.RESET_ALL}"
            )
            return

        if choice != "1":
            print(f"  {Fore.RED}Invalid option.{Style.RESET_ALL}")
            return

        # Fixed lot
        raw = input(f"  {Fore.GREEN}Enter lot size (e.g. 0.01): {Style.RESET_ALL}").strip()
        try:
            lot = float(raw)
        except ValueError:
            print(f"  {Fore.RED}Invalid number.{Style.RESET_ALL}")
            return

        sym_info = mt5.symbol_info(self.symbol)
        if sym_info:
            if lot < sym_info.volume_min:
                print(
                    f"  {Fore.YELLOW}⚠ Below minimum ({sym_info.volume_min}). "
                    f"Using minimum.{Style.RESET_ALL}"
                )
                lot = sym_info.volume_min
            elif lot > sym_info.volume_max:
                print(
                    f"  {Fore.YELLOW}⚠ Above maximum ({sym_info.volume_max}). "
                    f"Using maximum.{Style.RESET_ALL}"
                )
                lot = sym_info.volume_max

            # Round to volume step
            step = sym_info.volume_step
            if step > 0:
                lot = round(round(lot / step) * step, 8)

        self.lot_size = lot
        self.lot_mode = "fixed"
        print(f"  {Fore.GREEN}✔ Lot size set to {self.lot_size}{Style.RESET_ALL}")

    # ---- 6. View Account Info ---------------------------------------------
    def view_account_info(self) -> None:
        assert self.connector is not None
        self._show_account_summary()

        if self.risk_mgr:
            pnl = self.risk_mgr.get_daily_pnl()
            colour = Fore.GREEN if pnl >= 0 else Fore.RED
            print(f"  {Fore.WHITE}Daily P&L: {colour}${pnl:+.2f}{Style.RESET_ALL}")

    # ---- 7. View Open Positions -------------------------------------------
    def view_open_positions(self) -> None:
        assert self.order_mgr is not None
        positions = self.order_mgr.get_open_positions()
        if not positions:
            print(f"  {Fore.YELLOW}No open positions.{Style.RESET_ALL}")
            return

        print(f"  {Fore.CYAN}── Open Positions ({len(positions)}) ──{Style.RESET_ALL}")
        print(
            f"  {'Ticket':<12} {'Symbol':<10} {'Type':<6} {'Volume':<8} "
            f"{'Open':<10} {'Current':<10} {'P&L':>10}"
        )
        print(f"  {'─'*12} {'─'*10} {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

        total_pnl = 0.0
        for p in positions:
            colour = Fore.GREEN if p["profit"] >= 0 else Fore.RED
            print(
                f"  {p['ticket']:<12} {p['symbol']:<10} {p['type']:<6} "
                f"{p['volume']:<8.2f} {p['price_open']:<10.5f} "
                f"{p['price_current']:<10.5f} "
                f"{colour}{p['profit']:>+10.2f}{Style.RESET_ALL}"
            )
            total_pnl += p["profit"]

        colour = Fore.GREEN if total_pnl >= 0 else Fore.RED
        print(f"\n  {Fore.WHITE}Total P&L: {colour}${total_pnl:+.2f}{Style.RESET_ALL}")

    # ---- 8. Close All Positions -------------------------------------------
    def close_all_positions_menu(self) -> None:
        assert self.order_mgr is not None
        positions = self.order_mgr.get_open_positions()
        if not positions:
            print(f"  {Fore.YELLOW}No open positions.{Style.RESET_ALL}")
            return

        print(
            f"  {Fore.YELLOW}⚠ This will close {len(positions)} position(s).{Style.RESET_ALL}"
        )
        confirm = input(f"  {Fore.RED}Type 'YES' to confirm: {Style.RESET_ALL}").strip()
        if confirm != "YES":
            print(f"  {Fore.CYAN}Cancelled.{Style.RESET_ALL}")
            return

        closed = self.order_mgr.close_all_positions()
        print(f"  {Fore.GREEN}✔ Closed {closed} position(s).{Style.RESET_ALL}")

    # ---- 9. Toggle Dry Run -----------------------------------------------
    def toggle_dry_run(self) -> None:
        self.dry_run = not self.dry_run
        if self.order_mgr:
            self.order_mgr.dry_run = self.dry_run
        state = "ON" if self.dry_run else "OFF"
        colour = Fore.YELLOW if self.dry_run else Fore.GREEN
        print(f"  {colour}✔ Dry-run mode: {state}{Style.RESET_ALL}")

    # ---- 10. Market Session Info ------------------------------------------
    def show_session_info(self) -> None:
        assert self.session_mgr is not None
        active = self.session_mgr.get_active_sessions()
        market_open = self.session_mgr.is_market_open()
        overlap = self.session_mgr.is_overlap()

        print(f"  {Fore.CYAN}── Market Sessions ──{Style.RESET_ALL}")
        print(
            f"  {Fore.WHITE}Market open : "
            f"{Fore.GREEN + 'Yes' if market_open else Fore.RED + 'No'}"
            f"{Style.RESET_ALL}"
        )
        print(
            f"  {Fore.WHITE}Active      : "
            f"{Fore.YELLOW}{', '.join(active) if active else 'None'}{Style.RESET_ALL}"
        )
        print(
            f"  {Fore.WHITE}Overlap     : "
            f"{Fore.GREEN + 'Yes' if overlap else Fore.WHITE + 'No'}"
            f"{Style.RESET_ALL}"
        )

        if active:
            for session in active:
                best = self.session_mgr.get_best_pairs_for_session(session)
                print(
                    f"  {Fore.WHITE}Best pairs ({session}): "
                    f"{Fore.YELLOW}{', '.join(best)}{Style.RESET_ALL}"
                )

    # ---- 11. Multi-Bot Manager --------------------------------------------
    def multi_bot_control(self) -> None:
        """Open the multi-bot control center dashboard."""
        # Lazy-init the multi-bot manager on first use
        if self.multi_mgr is None:
            assert self.connector is not None
            assert self.market_data is not None
            assert self.session_mgr is not None
            assert self.order_mgr is not None
            assert self.risk_mgr is not None

            self.multi_mgr = MultiBotManager(
                config=self.config,
                connector=self.connector,
                market_data=self.market_data,
                session_mgr=self.session_mgr,
                order_mgr=self.order_mgr,
                risk_mgr=self.risk_mgr,
            )

        multi_bot_menu(self.multi_mgr, self.config)

    # ===================================================================
    #  TRADING LOOP
    # ===================================================================
    def start_trading(self) -> None:
        """Enter the main automated trading loop."""
        # --- Pre-flight validation -----------------------------------------
        if not self.strategy_name:
            print(f"  {Fore.RED}✘ No strategy selected. Use menu option 4 first.{Style.RESET_ALL}")
            return

        if self.timeframe_key not in TIMEFRAME_MAP:
            print(f"  {Fore.RED}✘ Invalid timeframe: {self.timeframe_key}{Style.RESET_ALL}")
            return

        assert self.connector is not None
        if not self.connector.ensure_symbol(self.symbol):
            print(f"  {Fore.RED}✘ Symbol {self.symbol} not available.{Style.RESET_ALL}")
            return

        # --- Resolve the strategy instance ---------------------------------
        try:
            strategy = get_strategy(self.strategy_name)
        except Exception as exc:
            print(f"  {Fore.RED}✘ Cannot load strategy '{self.strategy_name}': {exc}{Style.RESET_ALL}")
            return

        mt5_tf = TIMEFRAME_MAP[self.timeframe_key]

        # --- Confirmation banner -------------------------------------------
        print(f"\n  {Fore.CYAN}═══ Trading Configuration ═══{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Symbol    : {Fore.YELLOW}{self.symbol}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Timeframe : {Fore.YELLOW}{self.timeframe_key}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Strategy  : {Fore.YELLOW}{self.strategy_name}{Style.RESET_ALL}")
        lot_display = "AUTO (risk-based)" if self.lot_mode == "auto" else str(self.lot_size)
        print(f"  {Fore.WHITE}Lot size  : {Fore.YELLOW}{lot_display}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Dry run   : {Fore.YELLOW}{self.dry_run}{Style.RESET_ALL}")
        print()

        confirm = input(
            f"  {Fore.GREEN}Start trading? (y/n): {Style.RESET_ALL}"
        ).strip().lower()
        if confirm not in ("y", "yes"):
            print(f"  {Fore.CYAN}Cancelled.{Style.RESET_ALL}")
            return

        print(f"\n  {Fore.GREEN}▶ Trading started. Press Ctrl+C to stop.{Style.RESET_ALL}\n")

        assert self.market_data is not None
        assert self.order_mgr is not None
        assert self.risk_mgr is not None
        assert self.session_mgr is not None

        # --- Initialise new components for this session --------------------
        if self.automation_engine is None:
            self.automation_engine = AutomationEngine()
            try:
                for rule in AutomationEngine.create_default_rules():
                    self.automation_engine.add_rule(rule)
                self.logger.info("Automation engine loaded with default rules.")
            except Exception as exc:
                self.logger.warning("Could not load default automation rules: %s", exc)

        trailing_profile = get_trailing_profile(self.trailing_profile_name)
        self.trailing_states = {}  # reset per session

        try:
            emit_sync(EventChannel.SYSTEM, "trading_started", {
                "symbol": self.symbol, "strategy": self.strategy_name,
                "timeframe": self.timeframe_key,
            }, source="TradingLoop")
        except Exception:
            pass

        # --- Main loop -----------------------------------------------------
        try:
            while True:
                now = datetime.utcnow()

                # -- Market open check --
                if not self.session_mgr.is_market_open():
                    self._print_status(now, 0.0, "MARKET CLOSED", 0)
                    time.sleep(60)
                    continue

                # -- Reconnect if needed --
                if not self.connector.is_connected():
                    self.logger.warning("MT5 disconnected — attempting reconnect …")
                    if not self.connector.reconnect():
                        self.logger.error("Reconnect failed. Retrying in 30s.")
                        time.sleep(30)
                        continue

                # -- Fetch candles (closed bars only; drop the forming one so
                # signal triggers don't flicker on the in-progress candle) --
                df = self.market_data.get_candles(
                    self.symbol, mt5_tf, count=500, drop_forming=True
                )
                if df is None or df.empty:
                    self.logger.warning("No candle data for %s.", self.symbol)
                    time.sleep(10)
                    continue

                current_price = df["close"].iloc[-1]

                # -- Update risk state each iteration --
                try:
                    self.risk_mgr.update_risk_state()
                except Exception:
                    self.logger.debug("Risk state update skipped.")

                # -- Automation engine check --
                try:
                    auto_context = {
                        "symbol": self.symbol,
                        "current_price": current_price,
                        "daily_pnl": self.risk_mgr.get_daily_pnl(),
                        "open_positions": len(
                            self.order_mgr.get_open_positions()
                        ),
                        "connector": self.connector,
                        "order_manager": self.order_mgr,
                    }
                    try:
                        metrics = self.risk_mgr.get_risk_metrics()
                        auto_context["drawdown_pct"] = metrics.get(
                            "drawdown_pct", 0
                        )
                    except Exception:
                        auto_context["drawdown_pct"] = 0

                    executed = self.automation_engine.evaluate(auto_context)
                    if executed:
                        self.logger.info(
                            "Automation rules fired: %s", executed
                        )
                except Exception as exc:
                    self.logger.debug("Automation check error: %s", exc)

                # -- Run strategy --
                signal = strategy.analyze(df, self.symbol)
                positions = self.order_mgr.get_open_positions(self.symbol)

                signal_display = signal.action.value  # "BUY"/"SELL"/"HOLD"

                self._print_status(
                    now,
                    current_price,
                    signal_display,
                    len(positions),
                )

                # -- Publish signal event --
                try:
                    emit_sync(EventChannel.SIGNAL, "signal_generated", {
                        "symbol": self.symbol,
                        "action": signal.action.value,
                        "confidence": signal.confidence,
                        "strategy": self.strategy_name,
                    }, source="TradingLoop")
                except Exception:
                    pass

                # -- Act on BUY / SELL signals --
                if signal.action in (SignalAction.BUY, SignalAction.SELL):
                    self._execute_signal(signal)

                # -- Update trailing stops on open positions --
                try:
                    self._update_trailing_stops(
                        trailing_profile, current_price, df
                    )
                except Exception as exc:
                    self.logger.debug("Trailing update error: %s", exc)

                # -- Sleep until next candle ---
                sleep_sec = self._seconds_until_next_candle(self.timeframe_key)
                sleep_sec = max(5, sleep_sec)  # floor at 5 s
                self.logger.debug("Sleeping %d s until next candle.", sleep_sec)
                time.sleep(sleep_sec)

        except KeyboardInterrupt:
            print(f"\n  {Fore.YELLOW}Trading stopped by user.{Style.RESET_ALL}")
            self.logger.info("Trading loop terminated by user.")

    # ------------------------------------------------------------------
    # Execute a signal
    # ------------------------------------------------------------------
    def _execute_signal(self, signal) -> None:
        """Validate risk, compute prices, and place an order for *signal*."""
        assert self.order_mgr is not None
        assert self.risk_mgr is not None
        assert self.market_data is not None

        symbol = signal.symbol or self.symbol
        sl_pips = signal.sl_pips or self.config.default_sl_pips
        tp_pips = signal.tp_pips or self.config.default_tp_pips

        # --- Lot size -------------------------------------------------------
        if self.lot_mode == "auto":
            lot = self.risk_mgr.calculate_lot_size(symbol, sl_pips)
        else:
            lot = self.lot_size

        # --- Risk gate check (enhanced with correlation + recovery) ---------
        direction = "BUY" if signal.action == SignalAction.BUY else "SELL"
        can, reason = self.risk_mgr.can_trade_enhanced(symbol, lot, direction)
        if not can:
            self.logger.info("Trade blocked by risk manager: %s", reason)
            print(
                f"  {Fore.YELLOW}Trade blocked: {reason}{Style.RESET_ALL}"
            )
            return

        # --- SL / TP prices -------------------------------------------------
        tick = self.market_data.get_tick(symbol)
        if tick is None:
            self.logger.warning("Cannot get tick for %s — skipping signal.", symbol)
            return

        sym_info = self.market_data.get_symbol_info(symbol)
        if sym_info is None:
            self.logger.warning("Cannot get symbol info for %s.", symbol)
            return

        # get_tick and get_symbol_info return dicts
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

        comment = f"{self.strategy_name}|{signal.reason[:20] if signal.reason else ''}"

        # --- Place order ----------------------------------------------------
        result = self.order_mgr.open_position(
            symbol=symbol,
            order_type=order_type,
            lot=lot,
            sl_price=sl_price,
            tp_price=tp_price,
            comment=comment,
        )

        if result.get("retcode") == 10009:
            action_str = "BUY" if signal.action == SignalAction.BUY else "SELL"
            print(
                f"  {Fore.GREEN}✔ {action_str} {symbol} {lot} lots @ "
                f"{entry:.5f}  SL={sl_price:.5f}  TP={tp_price:.5f}{Style.RESET_ALL}"
            )
            save_trade_to_journal(
                {
                    "time": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "action": action_str,
                    "lot": lot,
                    "entry": entry,
                    "sl": sl_price,
                    "tp": tp_price,
                    "strategy": self.strategy_name,
                    "reason": signal.reason,
                    "confidence": signal.confidence,
                    "dry_run": self.dry_run,
                }
            )

            # -- Set up trailing stop state --
            ticket = result.get("order")
            if ticket:
                self.trailing_states[ticket] = TrailingState(
                    ticket=ticket,
                    symbol=symbol,
                    direction=action_str,
                    entry_price=entry,
                    current_sl=sl_price,
                    highest_price=entry,
                )

            # -- Publish order event --
            try:
                emit_sync(EventChannel.ORDER, "order_placed", {
                    "symbol": symbol, "action": action_str,
                    "lot": lot, "entry": entry, "ticket": ticket,
                }, source="TradingLoop")
            except Exception:
                pass
        else:
            msg = OrderManager.get_retcode_message(result.get("retcode", -1))
            print(
                f"  {Fore.RED}✘ Order failed: {msg} "
                f"(retcode={result.get('retcode')}){Style.RESET_ALL}"
            )

    # ------------------------------------------------------------------
    # Status line
    # ------------------------------------------------------------------
    @staticmethod
    def _print_status(
        now: datetime, price: float, signal_str: str, open_count: int
    ) -> None:
        ts = now.strftime("%H:%M:%S")
        print(
            f"  {Fore.WHITE}[{ts}]  "
            f"Price: {Fore.YELLOW}{price:<10.5f}  "
            f"{Fore.WHITE}Signal: {Fore.CYAN}{signal_str:<14}  "
            f"{Fore.WHITE}Open: {Fore.YELLOW}{open_count}{Style.RESET_ALL}"
        )

    # ------------------------------------------------------------------
    # Candle timer helper
    # ------------------------------------------------------------------
    @staticmethod
    def _seconds_until_next_candle(timeframe_key: str) -> int:
        """
        Estimate seconds remaining until the current candle closes.

        Uses a simple wall-clock calculation based on the timeframe
        duration.  Returns the number of seconds to wait.
        """
        tf_seconds = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
            "W1": 604800,
            "MN1": 2592000,
        }
        interval = tf_seconds.get(timeframe_key, 300)  # default 5 min
        now_ts = int(datetime.utcnow().timestamp())
        elapsed = now_ts % interval
        remaining = interval - elapsed
        return remaining

    # ===================================================================
    #  TRAILING STOP MANAGEMENT
    # ===================================================================
    def _update_trailing_stops(
        self, profile, current_price: float, df,
    ) -> None:
        """Update trailing stops on all tracked positions."""
        if not self.trailing_states:
            return

        closed_tickets = []
        for ticket, state in self.trailing_states.items():
            # Check if position is still open
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                closed_tickets.append(ticket)
                continue

            # Update highest/lowest price
            if state.direction == "BUY":
                if current_price > state.highest_price:
                    state.highest_price = current_price
            else:
                if current_price < state.highest_price:
                    state.highest_price = current_price

            # Calculate current ATR for ATR-based trailing
            current_atr = None
            try:
                import pandas as pd
                high = df["high"]
                low = df["low"]
                close = df["close"]
                prev_close = close.shift(1)
                tr = pd.concat([
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ], axis=1).max(axis=1)
                current_atr = float(tr.rolling(14).mean().iloc[-1])
            except Exception:
                pass

            update = profile.update(
                state,
                current_price=current_price,
                current_atr=current_atr,
            )

            if update and update.new_sl is not None and update.new_sl != state.current_sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": update.new_sl,
                    "tp": pos[0].tp,
                }
                mod_result = mt5.order_send(request)
                if mod_result and mod_result.retcode == 10009:
                    state.current_sl = update.new_sl
                    self.logger.info(
                        "Trailing SL updated: ticket=%d, new_sl=%.5f",
                        ticket, update.new_sl,
                    )

        # Clean up closed positions
        for t in closed_tickets:
            del self.trailing_states[t]

    # ===================================================================
    #  12. PROFILE MANAGER MENU
    # ===================================================================
    def profile_manager_menu(self) -> None:
        """Interactive profile management."""
        if self.profile_mgr is None:
            self.profile_mgr = ProfileManager()

        while True:
            print(f"\n  {Fore.CYAN}== Profile Manager =={Style.RESET_ALL}")
            print(f"  {Fore.WHITE}1. List profiles{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}2. Create default profiles{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}3. Activate a profile{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}4. View active profile{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}5. Auto-select by session{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}0. Back{Style.RESET_ALL}")

            choice = input(f"  {Fore.GREEN}> Option: {Style.RESET_ALL}").strip()
            if choice == "0":
                break

            if choice == "1":
                profiles = self.profile_mgr.list_profiles()
                if not profiles:
                    print(f"  {Fore.YELLOW}No profiles found. Use option 2 to create defaults.{Style.RESET_ALL}")
                    continue
                print(f"\n  {Fore.CYAN}{'Name':<25} {'Strategy':<10} {'TF':<5} {'Active':<8}{Style.RESET_ALL}")
                for p in profiles:
                    active = f"{Fore.GREEN}< YES" if p.is_active else ""
                    print(f"  {Fore.WHITE}{p.name:<25} {p.strategy_name:<10} {p.timeframe:<5} {active}{Style.RESET_ALL}")

            elif choice == "2":
                created = self.profile_mgr.create_default_profiles()
                print(f"  {Fore.GREEN}Created {len(created)} default profiles.{Style.RESET_ALL}")

            elif choice == "3":
                profiles = self.profile_mgr.list_profiles()
                if not profiles:
                    print(f"  {Fore.YELLOW}No profiles.{Style.RESET_ALL}")
                    continue
                for idx, p in enumerate(profiles, 1):
                    tag = " <" if p.is_active else ""
                    print(f"  {Fore.WHITE}{idx}. {p.name}{Fore.YELLOW}{tag}{Style.RESET_ALL}")
                raw = input(f"  {Fore.GREEN}Profile #: {Style.RESET_ALL}").strip()
                try:
                    sel = int(raw) - 1
                    if 0 <= sel < len(profiles):
                        p = profiles[sel]
                        self.profile_mgr.activate_profile(p.profile_id)
                        self.strategy_name = p.strategy_name
                        self.timeframe_key = p.timeframe
                        self.lot_mode = p.lot_mode
                        self.lot_size = p.lot_size
                        if p.symbols:
                            self.symbol = p.symbols[0]
                        if p.trailing_profile:
                            self.trailing_profile_name = p.trailing_profile
                        print(f"  {Fore.GREEN}Profile '{p.name}' activated!{Style.RESET_ALL}")
                        print(f"    Strategy={p.strategy_name}, TF={p.timeframe}")
                        print(f"    Symbol={p.symbols[0] if p.symbols else 'N/A'}, Lot={p.lot_mode}")
                    else:
                        print(f"  {Fore.RED}Invalid selection.{Style.RESET_ALL}")
                except ValueError:
                    print(f"  {Fore.RED}Invalid input.{Style.RESET_ALL}")

            elif choice == "4":
                active = self.profile_mgr.get_active_profile()
                if active:
                    print(f"\n  {Fore.CYAN}Active Profile: {Fore.YELLOW}{active.name}{Style.RESET_ALL}")
                    print(f"  Strategy  : {active.strategy_name}")
                    print(f"  Timeframe : {active.timeframe}")
                    print(f"  Symbols   : {', '.join(active.symbols)}")
                    print(f"  Lot Mode  : {active.lot_mode} ({active.lot_size})")
                    print(f"  Trailing  : {active.trailing_profile or 'None'}")
                else:
                    print(f"  {Fore.YELLOW}No active profile.{Style.RESET_ALL}")

            elif choice == "5":
                if self.session_mgr:
                    active_sessions = self.session_mgr.get_active_sessions()
                    if active_sessions:
                        session = active_sessions[0]
                        profile = self.profile_mgr.get_profile_for_session(session)
                        if profile:
                            self.profile_mgr.activate_profile(profile.profile_id)
                            self.strategy_name = profile.strategy_name
                            self.timeframe_key = profile.timeframe
                            print(f"  {Fore.GREEN}Auto-selected '{profile.name}' for {session} session.{Style.RESET_ALL}")
                        else:
                            print(f"  {Fore.YELLOW}No profile for '{session}' session.{Style.RESET_ALL}")
                    else:
                        print(f"  {Fore.YELLOW}No active sessions right now.{Style.RESET_ALL}")

    # ===================================================================
    #  13. AUTOMATION RULES MENU
    # ===================================================================
    def automation_menu(self) -> None:
        """View and manage automation rules."""
        if self.automation_engine is None:
            self.automation_engine = AutomationEngine()

        while True:
            print(f"\n  {Fore.CYAN}== Automation Rules =={Style.RESET_ALL}")
            print(f"  {Fore.WHITE}1. View all rules{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}2. Load default safety rules{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}3. Enable/disable a rule{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}4. View execution log{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}0. Back{Style.RESET_ALL}")

            choice = input(f"  {Fore.GREEN}> Option: {Style.RESET_ALL}").strip()
            if choice == "0":
                break

            if choice == "1":
                rules = self.automation_engine.get_rules()
                if not rules:
                    print(f"  {Fore.YELLOW}No rules loaded. Use option 2.{Style.RESET_ALL}")
                    continue
                print(f"\n  {Fore.CYAN}{'#':<4} {'Name':<28} {'Enabled':<10}{Style.RESET_ALL}")
                for idx, rule in enumerate(rules, 1):
                    status = f"{Fore.GREEN}ON" if rule.enabled else f"{Fore.RED}OFF"
                    print(f"  {Fore.WHITE}{idx:<4} {rule.name:<28} {status}{Style.RESET_ALL}")

            elif choice == "2":
                try:
                    defaults = AutomationEngine.create_default_rules()
                    for rule in defaults:
                        self.automation_engine.add_rule(rule)
                    print(f"  {Fore.GREEN}Loaded {len(defaults)} default safety rules:{Style.RESET_ALL}")
                    for r in defaults:
                        print(f"    - {r.name}")
                except Exception as exc:
                    print(f"  {Fore.RED}Error loading rules: {exc}{Style.RESET_ALL}")

            elif choice == "3":
                rules = self.automation_engine.get_rules()
                if not rules:
                    print(f"  {Fore.YELLOW}No rules.{Style.RESET_ALL}")
                    continue
                for idx, rule in enumerate(rules, 1):
                    status = "ON" if rule.enabled else "OFF"
                    print(f"  {idx}. {rule.name} [{status}]")
                raw = input(f"  {Fore.GREEN}Toggle rule #: {Style.RESET_ALL}").strip()
                try:
                    sel = int(raw) - 1
                    if 0 <= sel < len(rules):
                        rule = rules[sel]
                        if rule.enabled:
                            self.automation_engine.disable_rule(rule.rule_id)
                            print(f"  {Fore.YELLOW}'{rule.name}' DISABLED.{Style.RESET_ALL}")
                        else:
                            self.automation_engine.enable_rule(rule.rule_id)
                            print(f"  {Fore.GREEN}'{rule.name}' ENABLED.{Style.RESET_ALL}")
                except (ValueError, IndexError):
                    print(f"  {Fore.RED}Invalid input.{Style.RESET_ALL}")

            elif choice == "4":
                log = self.automation_engine.get_execution_log()
                if not log:
                    print(f"  {Fore.YELLOW}No executions yet.{Style.RESET_ALL}")
                    continue
                print(f"\n  {Fore.CYAN}-- Recent Executions --{Style.RESET_ALL}")
                for entry in list(log)[-10:]:
                    ts = entry.get("timestamp", "")
                    name = entry.get("rule_name", "unknown")
                    print(f"  [{ts[:19]}] {Fore.YELLOW}{name}{Style.RESET_ALL}")

    # ===================================================================
    #  14. OPTIMIZER MENU
    # ===================================================================
    def optimizer_menu(self) -> None:
        """Run the genetic optimizer or walk-forward analysis."""
        print(f"\n  {Fore.CYAN}== Strategy Optimizer =={Style.RESET_ALL}")
        print(f"  {Fore.WHITE}1. Genetic Optimizer{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}2. Walk-Forward Analysis{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}3. Monte Carlo Simulation{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}0. Back{Style.RESET_ALL}")

        choice = input(f"  {Fore.GREEN}> Option: {Style.RESET_ALL}").strip()

        if choice == "1":
            if not self.strategy_name:
                print(f"  {Fore.RED}Select a strategy first (option 4).{Style.RESET_ALL}")
                return
            print(f"\n  {Fore.YELLOW}Genetic Optimizer for '{self.strategy_name}'{Style.RESET_ALL}")
            print(f"  Symbol: {self.symbol}, Timeframe: {self.timeframe_key}")
            print(f"\n  {Fore.WHITE}Use the optimizer module directly:{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  from optimizer import GeneticOptimizer, GeneticConfig{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  ga = GeneticOptimizer(config, param_ranges, backtest_fn){Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  results = ga.run(){Style.RESET_ALL}")

        elif choice == "2":
            print(f"\n  {Fore.YELLOW}Walk-Forward Analysis{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}Validates strategy robustness via sliding windows.{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  from backtester import WalkForwardAnalyzer{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  wfa = WalkForwardAnalyzer(in_sample_bars=500, out_of_sample_bars=200){Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  result = wfa.run(backtest_fn, param_combos, data_df){Style.RESET_ALL}")

        elif choice == "3":
            print(f"\n  {Fore.YELLOW}Monte Carlo Simulation{Style.RESET_ALL}")
            print(f"  {Fore.WHITE}Shuffles trade order 1000x to test equity curve.{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  from backtester import MonteCarloSimulator{Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  mc = MonteCarloSimulator(num_simulations=1000){Style.RESET_ALL}")
            print(f"  {Fore.CYAN}  result = mc.run(trades, initial_balance=10000){Style.RESET_ALL}")

    # ===================================================================
    #  15. RISK DASHBOARD
    # ===================================================================
    def risk_dashboard_menu(self) -> None:
        """Display comprehensive risk metrics."""
        assert self.risk_mgr is not None

        print(f"\n  {Fore.CYAN}== Risk Dashboard =={Style.RESET_ALL}")

        try:
            metrics = self.risk_mgr.get_risk_metrics()
        except Exception as exc:
            print(f"  {Fore.RED}Cannot load risk metrics: {exc}{Style.RESET_ALL}")
            return

        pnl = metrics.get("daily_pnl", 0)
        dd = metrics.get("drawdown_pct", 0)
        positions = metrics.get("open_positions_count", 0)
        exposure = metrics.get("total_exposure", 0)
        recovery = metrics.get("recovery_mode", False)
        recovery_mult = metrics.get("recovery_multiplier", 1.0)
        peak = metrics.get("peak_equity", 0)
        current = metrics.get("current_equity", 0)

        pnl_colour = Fore.GREEN if pnl >= 0 else Fore.RED
        dd_colour = Fore.GREEN if dd < 3 else (Fore.YELLOW if dd < 5 else Fore.RED)
        recovery_colour = Fore.RED if recovery else Fore.GREEN

        print(f"  {Fore.WHITE}+-------------------------------------+{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}| Daily P&L      : {pnl_colour}${pnl:>+10.2f}{Fore.WHITE}        |{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}| Drawdown       : {dd_colour}{dd:>8.2f}%{Fore.WHITE}          |{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}| Open Positions : {Fore.YELLOW}{positions:>8}{Fore.WHITE}           |{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}| Total Exposure : {Fore.YELLOW}{exposure:>8.2f}{Fore.WHITE} lots      |{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}| Peak Equity    : {Fore.CYAN}${peak:>10.2f}{Fore.WHITE}      |{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}| Current Equity : {Fore.CYAN}${current:>10.2f}{Fore.WHITE}      |{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}| Recovery Mode  : {recovery_colour}{'ACTIVE' if recovery else 'OFF':>8}{Fore.WHITE}           |{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}| Lot Multiplier : {Fore.YELLOW}{recovery_mult:>8.2f}x{Fore.WHITE}          |{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}+-------------------------------------+{Style.RESET_ALL}")

        # Currency exposure
        currency_exp = metrics.get("currency_exposure", {})
        if currency_exp:
            print(f"\n  {Fore.CYAN}-- Currency Exposure --{Style.RESET_ALL}")
            for ccy, exp in currency_exp.items():
                bar_len = int(abs(exp) * 20)
                direction = "+" if exp > 0 else "-"
                colour = Fore.GREEN if exp > 0 else Fore.RED
                bar = "#" * min(bar_len, 20)
                print(f"  {Fore.WHITE}{ccy}: {colour}{direction}{abs(exp):.2f} {bar}{Style.RESET_ALL}")

        # Correlation warnings
        warnings = metrics.get("correlation_warnings", [])
        if warnings:
            print(f"\n  {Fore.YELLOW}Correlation Warnings:{Style.RESET_ALL}")
            for w in warnings:
                print(f"    {Fore.YELLOW}* {w}{Style.RESET_ALL}")

        # Trailing stops status
        if self.trailing_states:
            print(f"\n  {Fore.CYAN}-- Trailing Stops ({len(self.trailing_states)}) --{Style.RESET_ALL}")
            for ticket, state in self.trailing_states.items():
                print(
                    f"  {Fore.WHITE}#{ticket} {state.symbol} {state.direction} "
                    f"entry={state.entry_price:.5f} SL={state.current_sl:.5f}{Style.RESET_ALL}"
                )


# ======================================================================
#  ARGUMENT PARSER
# ======================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rider Infinity — interactive & CLI modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bot.py                                           "
            "# Interactive menu\n"
            "  python bot.py --strategy ema --symbol EURUSD --timeframe M5 --lot 0.01\n"
            "  python bot.py --strategy rsi --symbol GBPUSD --timeframe H1 --auto-lot --dry-run\n"
        ),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy name (e.g. ema, rsi, macd). Triggers CLI mode.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (e.g. EURUSD).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Candle timeframe key (e.g. M1, M5, H1, D1).",
    )
    parser.add_argument(
        "--lot",
        type=float,
        default=None,
        help="Fixed lot size (e.g. 0.01).",
    )
    parser.add_argument(
        "--auto-lot",
        action="store_true",
        default=False,
        help="Use risk-based automatic lot sizing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate trades without sending real orders.",
    )
    return parser


# ======================================================================
#  ENTRY POINT
# ======================================================================
def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    bot = RiderInfinityBot()

    # If any strategy was given on CLI, pass args to skip menus.
    bot.run(args=args if args.strategy else None)


if __name__ == "__main__":
    main()
