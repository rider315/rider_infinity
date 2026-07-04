"""
dashboard.py — Flask Web Dashboard for the Rider Infinity Trading Platform.

Provides a REST API and serves a single-page dashboard UI for
monitoring and controlling multiple trading bots, viewing account
information, open positions, session status, and the trade journal.

Run standalone:
    python dashboard.py

The server binds to http://localhost:5000 by default.
"""

from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

import performance
from config import Config, SYMBOL_GROUPS, TIMEFRAME_OPTIONS, TIMEFRAME_MAP
from market_data import MarketDataFeed
from mt5_connector import MT5Connector
from multi_bot import BotInstance, BotStatus, MultiBotManager
from order_manager import OrderManager
from risk_manager import RiskManager
from session_manager import SessionManager
from strategies import list_strategies
from utils import setup_logger

# ── Module-level references (populated in ``create_app`` / ``main``) ──
_config: Config | None = None
_connector: MT5Connector | None = None
_market_data: MarketDataFeed | None = None
_session_mgr: SessionManager | None = None
_order_mgr: OrderManager | None = None
_risk_mgr: RiskManager | None = None
_bot_mgr: MultiBotManager | None = None
_news_engine: Any | None = None  # NewsIntelligenceEngine (lazy import)
_trailing_positions: dict = {}  # ticket -> {profile, params, enabled}

logger = setup_logger("dashboard")

# ── Paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_JOURNAL_PATH = _PROJECT_ROOT / "logs" / "trade_journal.csv"


# =====================================================================
#  Flask Application Factory
# =====================================================================

def create_app() -> Flask:
    """Build and return the configured Flask application."""
    app = Flask(
        __name__,
        template_folder=str(_PROJECT_ROOT / "templates"),
        static_folder=str(_PROJECT_ROOT / "static"),
    )

    # ── Page route ────────────────────────────────────────────────────

    @app.route("/")
    def index():
        """Serve the main dashboard HTML page."""
        try:
            return render_template("dashboard.html")
        except Exception as exc:
            logger.error("Failed to render dashboard: %s", exc)
            return jsonify({"error": "Template not found"}), 500

    # ==================================================================
    #  GET endpoints
    # ==================================================================

    @app.route("/api/account", methods=["GET"])
    def api_account():
        """Return account information from MT5."""
        try:
            info = _connector.get_account_info()
            if info is None:
                return jsonify({"error": "Unable to retrieve account info"}), 502
            return jsonify(info)
        except Exception as exc:
            logger.error("GET /api/account error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bots", methods=["GET"])
    def api_bots_list():
        """Return a list of all registered bots with full state."""
        try:
            bots = _bot_mgr.get_all_bots()
            return jsonify([_serialise_bot(b) for b in bots])
        except Exception as exc:
            logger.error("GET /api/bots error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bots/<int:bot_id>", methods=["GET"])
    def api_bot_detail(bot_id: int):
        """Return a single bot's info."""
        try:
            bot = _bot_mgr.get_bot(bot_id)
            if bot is None:
                return jsonify({"error": f"Bot #{bot_id} not found"}), 404
            return jsonify(_serialise_bot(bot))
        except Exception as exc:
            logger.error("GET /api/bots/%d error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/positions", methods=["GET"])
    def api_positions():
        """Return all open positions."""
        try:
            positions = _order_mgr.get_open_positions()
            return jsonify(positions)
        except Exception as exc:
            logger.error("GET /api/positions error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/session", methods=["GET"])
    def api_session():
        """Return current session information."""
        try:
            active = _session_mgr.get_active_sessions()
            best_pairs: dict[str, list[str]] = {}
            for session_name in active:
                try:
                    best_pairs[session_name] = (
                        _session_mgr.get_best_pairs_for_session(session_name)
                    )
                except ValueError:
                    best_pairs[session_name] = []

            return jsonify({
                "active_sessions": active,
                "is_market_open": _session_mgr.is_market_open(),
                "is_overlap": _session_mgr.is_overlap(),
                "best_pairs": best_pairs,
            })
        except Exception as exc:
            logger.error("GET /api/session error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/strategies", methods=["GET"])
    def api_strategies():
        """Return available trading strategies."""
        try:
            strategies = list_strategies()
            return jsonify(strategies)  # list of [key, desc] tuples
        except Exception as exc:
            logger.error("GET /api/strategies error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/symbols", methods=["GET"])
    def api_symbols():
        """Return the symbol groups dictionary."""
        try:
            return jsonify(SYMBOL_GROUPS)
        except Exception as exc:
            logger.error("GET /api/symbols error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/timeframes", methods=["GET"])
    def api_timeframes():
        """Return available timeframe options."""
        try:
            return jsonify(TIMEFRAME_OPTIONS)  # list of [key, display] tuples
        except Exception as exc:
            logger.error("GET /api/timeframes error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/journal", methods=["GET"])
    def api_journal():
        """Return the most recent 50 trades from the history DB.

        Uses bot_history.db (has full P&L data) instead of the
        append-only trade_journal.csv (entry-only).
        """
        try:
            trades = _bot_mgr.history_db.get_trades()
            # Most recent 50, newest first
            trades = trades[-50:]
            trades.reverse()
            return jsonify(trades)
        except Exception as exc:
            logger.error("GET /api/journal error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/daily-pnl", methods=["GET"])
    def api_daily_pnl():
        """Return today's profit/loss from the risk manager."""
        try:
            pnl = _risk_mgr.get_daily_pnl()
            return jsonify({"pnl": pnl})
        except Exception as exc:
            logger.error("GET /api/daily-pnl error: %s", exc)
            return jsonify({"error": str(exc)}), 500
    @app.route("/api/risk-dashboard", methods=["GET"])
    def api_risk_dashboard():
        """Return comprehensive risk metrics."""
        try:
            metrics = _risk_mgr.get_risk_metrics()
            return jsonify(metrics)
        except Exception as exc:
            logger.error("GET /api/risk-dashboard error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/automation/rules", methods=["GET"])
    def api_automation_rules():
        """Return automation rules status."""
        try:
            from automation.engine import AutomationEngine
            engine = AutomationEngine()
            try:
                for rule in AutomationEngine.create_default_rules():
                    engine.add_rule(rule)
            except Exception:
                pass
            rules = engine.get_rules()
            return jsonify([
                {
                    "name": r.name,
                    "enabled": r.enabled,
                    "one_shot": r.one_shot,
                    "execution_count": r.execution_count,
                }
                for r in rules
            ])
        except Exception as exc:
            logger.error("GET /api/automation/rules error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ==================================================================
    #  POST / DELETE endpoints
    # ==================================================================

    @app.route("/api/bots", methods=["POST"])
    def api_bot_create():
        """Create a new bot and auto-start it.

        Expected JSON body::
            {
                "symbol":   "EURUSD",
                "strategy": "ema",
                "timeframe": "M5",
                "lot_size": 0.01,
                "lot_mode": "fixed",
                "dry_run":  true
            }
        """
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Request body is required"}), 400

            symbol = data.get("symbol")
            strategy = data.get("strategy")
            timeframe = data.get("timeframe")

            if not all([symbol, strategy, timeframe]):
                return jsonify({
                    "error": "symbol, strategy, and timeframe are required"
                }), 400

            lot_size = float(data.get("lot_size", _config.default_lot_size))
            lot_mode = data.get("lot_mode", "fixed")
            dry_run = bool(data.get("dry_run", True))
            strategy_params = data.get("strategy_params", {})
            quick_exit_enabled = bool(data.get("quick_exit_enabled", False))
            try:
                quick_exit_pips = float(data.get("quick_exit_pips", 5.0))
            except (TypeError, ValueError):
                quick_exit_pips = 5.0
            if quick_exit_pips <= 0:
                quick_exit_pips = 5.0

            # Validate timeframe
            if timeframe not in TIMEFRAME_MAP:
                return jsonify({
                    "error": f"Invalid timeframe '{timeframe}'. "
                             f"Valid: {list(TIMEFRAME_MAP.keys())}"
                }), 400

            bot = _bot_mgr.add_bot(
                symbol=symbol,
                strategy_name=strategy,
                timeframe_key=timeframe,
                lot_size=lot_size,
                lot_mode=lot_mode,
                dry_run=dry_run,
                strategy_params=strategy_params,
                quick_exit_enabled=quick_exit_enabled,
                quick_exit_pips=quick_exit_pips,
            )

            # Auto-start the bot
            started = _bot_mgr.start_bot(bot.bot_id)
            if not started:
                logger.warning(
                    "Bot #%d created but failed to start: %s",
                    bot.bot_id, bot.error_msg,
                )

            return jsonify(_serialise_bot(bot)), 201

        except Exception as exc:
            logger.error("POST /api/bots error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bots/<int:bot_id>/start", methods=["POST"])
    def api_bot_start(bot_id: int):
        """Start a stopped bot."""
        try:
            bot = _bot_mgr.get_bot(bot_id)
            if bot is None:
                return jsonify({"error": f"Bot #{bot_id} not found"}), 404

            ok = _bot_mgr.start_bot(bot_id)
            if ok:
                return jsonify({"message": f"Bot #{bot_id} started", **_serialise_bot(bot)})
            return jsonify({
                "error": f"Failed to start Bot #{bot_id}",
                "detail": bot.error_msg,
            }), 400
        except Exception as exc:
            logger.error("POST /api/bots/%d/start error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bots/<int:bot_id>/stop", methods=["POST"])
    def api_bot_stop(bot_id: int):
        """Stop a running bot."""
        try:
            bot = _bot_mgr.get_bot(bot_id)
            if bot is None:
                return jsonify({"error": f"Bot #{bot_id} not found"}), 404

            ok = _bot_mgr.stop_bot(bot_id)
            if ok:
                return jsonify({"message": f"Bot #{bot_id} stopped", **_serialise_bot(bot)})
            return jsonify({"error": f"Failed to stop Bot #{bot_id}"}), 400
        except Exception as exc:
            logger.error("POST /api/bots/%d/stop error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bots/<int:bot_id>/toggle-dry-run", methods=["POST"])
    def api_bot_toggle_dry_run(bot_id: int):
        """Toggle dry_run mode for a bot. Bot must be stopped."""
        try:
            bot = _bot_mgr.get_bot(bot_id)
            if bot is None:
                return jsonify({"error": f"Bot #{bot_id} not found"}), 404

            if bot.status == BotStatus.RUNNING:
                return jsonify({
                    "error": "Bot must be stopped before toggling dry-run mode"
                }), 409

            bot.dry_run = not bot.dry_run
            mode = "DRY RUN" if bot.dry_run else "LIVE"
            logger.info("Bot #%d toggled to %s mode", bot_id, mode)
            _bot_mgr._save_bots()
            _bot_mgr.history_db.record_mode_changed(bot_id, bot.dry_run)

            return jsonify({
                "message": f"Bot #{bot_id} is now in {mode} mode",
                **_serialise_bot(bot),
            })
        except Exception as exc:
            logger.error("POST /api/bots/%d/toggle-dry-run error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bots/<int:bot_id>/quick-exit", methods=["POST"])
    def api_bot_quick_exit(bot_id: int):
        """Enable/disable Quick Exit (scalp exit) and set its pip target.

        Safe to change while the bot is running — the trading loop reads the
        flag each tick.  Body (all optional)::
            { "enabled": true, "pips": 5 }
        """
        try:
            bot = _bot_mgr.get_bot(bot_id)
            if bot is None:
                return jsonify({"error": f"Bot #{bot_id} not found"}), 404

            data = request.get_json(force=True, silent=True) or {}

            if "enabled" in data:
                bot.quick_exit_enabled = bool(data["enabled"])
            else:
                # No explicit value → treat as a toggle.
                bot.quick_exit_enabled = not bot.quick_exit_enabled

            if "pips" in data:
                try:
                    pips = float(data["pips"])
                    if pips > 0:
                        bot.quick_exit_pips = pips
                except (TypeError, ValueError):
                    return jsonify({"error": "pips must be a positive number"}), 400

            _bot_mgr._save_bots()
            state = "ON" if bot.quick_exit_enabled else "OFF"
            logger.info(
                "Bot #%d Quick Exit %s (target %.1f pips)",
                bot_id, state, bot.quick_exit_pips,
            )
            return jsonify({
                "message": f"Bot #{bot_id} Quick Exit {state} "
                           f"@ {bot.quick_exit_pips:g} pips",
                **_serialise_bot(bot),
            })
        except Exception as exc:
            logger.error("POST /api/bots/%d/quick-exit error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bots/<int:bot_id>", methods=["DELETE"])
    def api_bot_remove(bot_id: int):
        """Remove a bot entirely (stops it first if running)."""
        try:
            bot = _bot_mgr.get_bot(bot_id)
            if bot is None:
                return jsonify({"error": f"Bot #{bot_id} not found"}), 404

            ok = _bot_mgr.remove_bot(bot_id)
            if ok:
                return jsonify({"message": f"Bot #{bot_id} removed"})
            return jsonify({"error": f"Failed to remove Bot #{bot_id}"}), 400
        except Exception as exc:
            logger.error("DELETE /api/bots/%d error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    # ==================================================================
    #  Bot History endpoints
    # ==================================================================

    @app.route("/api/bot-history", methods=["GET"])
    def api_bot_history():
        """Return all bot records (active + removed) with event counts."""
        try:
            history = _bot_mgr.history_db.get_all_history()
            return jsonify(history)
        except Exception as exc:
            logger.error("GET /api/bot-history error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bot-history/<int:bot_id>/events", methods=["GET"])
    def api_bot_history_events(bot_id: int):
        """Return lifecycle events for a specific bot."""
        try:
            events = _bot_mgr.history_db.get_bot_events(bot_id)
            return jsonify(events)
        except Exception as exc:
            logger.error("GET /api/bot-history/%d/events error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bot-history/<int:bot_id>/trades", methods=["GET"])
    def api_bot_history_trades(bot_id: int):
        """Return trades from bot_history DB filtered by bot_id."""
        try:
            trades = _bot_mgr.history_db.get_trades(bot_id=bot_id)
            trades.reverse()  # newest first
            return jsonify(trades[:100])
        except Exception as exc:
            logger.error("GET /api/bot-history/%d/trades error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    # ==================================================================
    #  Performance export (per-trade outcomes + metrics)
    # ==================================================================
    def _perf_filters() -> dict:
        """Parse common performance query params: bot_id, start, end, slab."""
        bot_id = request.args.get("bot_id", type=int)
        start = request.args.get("start") or None  # ISO date e.g. 2026-06-01
        end = request.args.get("end") or None
        slab = request.args.get("slab", "daily")
        if slab not in ("daily", "weekly", "monthly"):
            slab = "daily"
        # Make an inclusive end-of-day for a bare date.
        if end and len(end) == 10:
            end = end + "T23:59:59"
        return {"bot_id": bot_id, "start": start, "end": end, "slab": slab}

    @app.route("/api/performance/trades", methods=["GET"])
    def api_performance_trades():
        """Return per-trade outcome rows as JSON (for the UI table)."""
        try:
            f = _perf_filters()
            trades = _bot_mgr.history_db.get_trades(
                bot_id=f["bot_id"], start=f["start"], end=f["end"],
            )
            return jsonify(trades)
        except Exception as exc:
            logger.error("GET /api/performance/trades error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/performance/trades.csv", methods=["GET"])
    def api_performance_trades_csv():
        """Download every trade (open + close detail) as a CSV file."""
        try:
            f = _perf_filters()
            trades = _bot_mgr.history_db.get_trades(
                bot_id=f["bot_id"], start=f["start"], end=f["end"],
            )
            csv_text = performance.trades_to_csv(trades)
            scope = f"bot{f['bot_id']}" if f["bot_id"] else "all"
            fname = f"trades_{scope}_{datetime.utcnow():%Y%m%d_%H%M%S}.csv"
            return Response(
                csv_text,
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename={fname}"},
            )
        except Exception as exc:
            logger.error("GET /api/performance/trades.csv error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/performance/summary", methods=["GET"])
    def api_performance_summary():
        """Return computed performance metrics (overall + time-slab breakdowns)."""
        try:
            f = _perf_filters()
            trades = _bot_mgr.history_db.get_trades(
                bot_id=f["bot_id"], start=f["start"], end=f["end"],
            )
            summary = performance.build_summary(
                trades, slab=f["slab"], bot_id=f["bot_id"],
                start=f["start"], end=f["end"],
            )
            # ?download=1 streams the same JSON as a file attachment.
            if request.args.get("download"):
                import json as _json
                scope = f"bot{f['bot_id']}" if f["bot_id"] else "all"
                fname = f"performance_{scope}_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
                return Response(
                    _json.dumps(summary, indent=2),
                    mimetype="application/json",
                    headers={"Content-Disposition": f"attachment; filename={fname}"},
                )
            return jsonify(summary)
        except Exception as exc:
            logger.error("GET /api/performance/summary error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/performance/logic-versions", methods=["GET"])
    def api_performance_logic_versions():
        """Return the logic-version registry: every code/settings fingerprint
        ever stamped on a trade, with first-seen time and settings snapshot."""
        try:
            from logic_version import load_registry
            return jsonify(load_registry())
        except Exception as exc:
            logger.error("GET /api/performance/logic-versions error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/performance/backfill", methods=["POST"])
    def api_performance_backfill():
        """Import past CLOSED live trades from MT5 deal history into the DB."""
        try:
            days = request.args.get("days", default=90, type=int)
            result = _bot_mgr.backfill_trades_from_mt5(days=days)
            return jsonify(result)
        except Exception as exc:
            logger.error("POST /api/performance/backfill error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/bot-history/<int:bot_id>/restore", methods=["POST"])
    def api_bot_history_restore(bot_id: int):
        """Re-create a previously removed bot with the same config."""
        try:
            record = _bot_mgr.history_db.get_bot_record(bot_id)
            if record is None:
                return jsonify({"error": f"No history record for Bot #{bot_id}"}), 404
            if record.get("removed_at") is None:
                return jsonify({"error": f"Bot #{bot_id} is still active, not removed"}), 409

            # Re-create bot with same config
            bot = _bot_mgr.add_bot(
                symbol=record["symbol"],
                strategy_name=record["strategy_name"],
                timeframe_key=record["timeframe_key"],
                lot_size=float(record.get("lot_size", 0.01)),
                lot_mode=record.get("lot_mode", "fixed"),
                dry_run=bool(record.get("dry_run", True)),
                strategy_params=record.get("strategy_params", {}),
            )

            # Auto-start the restored bot
            started = _bot_mgr.start_bot(bot.bot_id)
            msg = f"Bot #{bot_id} restored as Bot #{bot.bot_id}"
            if not started:
                msg += f" (created but failed to start: {bot.error_msg})"

            return jsonify({
                "message": msg,
                "old_bot_id": bot_id,
                "new_bot_id": bot.bot_id,
                **_serialise_bot(bot),
            }), 201
        except Exception as exc:
            logger.error("POST /api/bot-history/%d/restore error: %s", bot_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/positions/close-all", methods=["POST"])
    def api_close_all_positions():
        """Close every open position and report exactly how many actually closed."""
        try:
            before = _order_mgr.get_open_positions() or []
            total = len(before)
            closed = _order_mgr.close_all_positions()
            remaining = total - closed
            ok = remaining == 0
            if total == 0:
                message = "No open positions to close"
            elif ok:
                message = f"Closed {closed} position(s)"
            else:
                # Closes that didn't go through — surface the actual broker
                # reason so the user knows *why* positions remain open.
                err = getattr(_order_mgr, "last_close_error", None) or {}
                reason = err.get("message") or "broker rejected the close"
                code = err.get("retcode")
                broker = err.get("comment")
                # Actionable hints for the codes that block *all* closes — these
                # are almost always a terminal/account setting, not a bug.
                _hints = {
                    10027: "Enable the 'Algo Trading' button in the MT5 terminal "
                           "(Tools → Options → Expert Advisors → Allow algorithmic trading).",
                    10018: "Market is closed for this symbol right now.",
                    10017: "Trading is disabled for this account/symbol.",
                    10016: "Broker rejected the stops/price — try again.",
                    10026: "Algo trading is disabled on the broker's server.",
                }
                hint = _hints.get(code, "")
                # For the Algo-Trading block, name the EXACT terminal instance
                # the API is bound to — on a server it's often a different one
                # than the window the user toggled.
                if code == 10027 and _connector is not None:
                    try:
                        ts = _connector.terminal_status()
                        if not ts.get("trade_allowed", False):
                            dp = str(ts.get("data_path") or ts.get("path") or "")
                            if "systemprofile" in dp.lower() or "system32" in dp.lower():
                                hint += (
                                    " The dashboard is running as a Windows "
                                    "SERVICE/SYSTEM account, so its MT5 terminal "
                                    "has no clickable window. Run the dashboard "
                                    "from an interactive Administrator session "
                                    "(where you can enable Algo Trading in MT5)."
                                )
                            else:
                                hint += (
                                    f" NOTE: orders are sent through the terminal at "
                                    f"'{dp}'. Enable Algo Trading in THAT instance."
                                )
                    except Exception:
                        pass
                detail = reason
                if code is not None:
                    detail += f" [retcode {code}]"
                if broker:
                    detail += f" — '{broker}'"
                if hint:
                    detail += f". {hint}"
                message = (
                    f"Closed {closed} of {total}; {remaining} still open: {detail}"
                )
            return jsonify({
                "message": message,
                "ok": ok,
                "closed": closed,
                "total": total,
                "remaining": remaining,
                "error": (getattr(_order_mgr, "last_close_error", None) if not ok else None),
            }), (200 if ok else 207)
        except Exception as exc:
            logger.error("POST /api/positions/close-all error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/test-trade", methods=["POST"])
    def api_test_trade():
        """Place a minimal test trade to verify execution pipeline.

        Expected JSON body::
            {
                "symbol":   "EURUSD",       # required
                "action":   "BUY",          # BUY or SELL (default BUY)
                "lot":      0.01,           # default 0.01
                "dry_run":  true            # default true
            }
        """
        import MetaTrader5 as mt5
        from utils import pips_to_price

        try:
            data = request.get_json(force=True) or {}
            symbol = data.get("symbol", "EURUSD")
            action = data.get("action", "BUY").upper()
            lot = float(data.get("lot", 0.01))
            dry_run = bool(data.get("dry_run", True))

            # Ensure symbol is available (resolves broker suffixes,
            # e.g. EURUSD -> EURUSD.a / EURUSDm), then use the real name.
            resolved = _connector.ensure_symbol(symbol)
            if not resolved:
                return jsonify({"error": f"Symbol {symbol} not available"}), 400
            symbol = resolved

            # Get tick and symbol info
            tick = _market_data.get_tick(symbol)
            if tick is None:
                return jsonify({"error": "Cannot get tick data"}), 502

            sym_info = _market_data.get_symbol_info(symbol)
            if sym_info is None:
                return jsonify({"error": "Cannot get symbol info"}), 502

            point = sym_info["point"]
            digits = sym_info["digits"]
            sl_pips = _config.default_sl_pips
            tp_pips = _config.default_tp_pips

            if action == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                entry = tick["ask"]
                sl_price = round(entry - pips_to_price(sl_pips, symbol, point), digits)
                tp_price = round(entry + pips_to_price(tp_pips, symbol, point), digits)
            else:
                order_type = mt5.ORDER_TYPE_SELL
                entry = tick["bid"]
                sl_price = round(entry + pips_to_price(sl_pips, symbol, point), digits)
                tp_price = round(entry - pips_to_price(tp_pips, symbol, point), digits)

            result_info = {
                "symbol": symbol,
                "action": action,
                "lot": lot,
                "entry": entry,
                "sl": sl_price,
                "tp": tp_price,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
            }

            if dry_run:
                logger.info(
                    "[TEST-DRY] %s %s %.2f lots @ %.5f  SL=%.5f  TP=%.5f",
                    action, symbol, lot, entry, sl_price, tp_price,
                )
                return jsonify({
                    "message": f"[DRY RUN] Test {action} {symbol} {lot} lots @ {entry:.5f}",
                    "dry_run": True,
                    **result_info,
                })

            # LIVE test trade
            order_mgr_live = OrderManager(_config, dry_run=False)
            result = order_mgr_live.open_position(
                symbol=symbol,
                order_type=order_type,
                lot=lot,
                sl_price=sl_price,
                tp_price=tp_price,
                comment="TEST-TRADE",
            )

            retcode = result.get("retcode", -1)
            if retcode == 10009:
                logger.info(
                    "[TEST-LIVE] %s %s %.2f lots @ %.5f — SUCCESS",
                    action, symbol, lot, entry,
                )
                return jsonify({
                    "message": f"✅ Test {action} {symbol} {lot} lots @ {entry:.5f} — ORDER PLACED!",
                    "dry_run": False,
                    "retcode": retcode,
                    **result_info,
                })
            else:
                msg = OrderManager.get_retcode_message(retcode)
                # Show which terminal instance actually handled (and rejected)
                # this order, plus its live Algo-Trading state. This proves
                # whether the API is bound to the same terminal you're toggling.
                term = {}
                try:
                    term = _connector.terminal_status()
                except Exception:
                    pass
                err = f"Order failed: {msg}"
                if retcode == 10027:
                    dp = str(term.get("data_path") or term.get("path") or "?")
                    err += (
                        f" | API is bound to terminal data_path='{dp}', "
                        f"trade_allowed={term.get('trade_allowed')}. Enable Algo "
                        f"Trading in THIS exact terminal (not another window on "
                        f"the same account)."
                    )
                    if "systemprofile" in dp.lower() or "system32" in dp.lower():
                        err += (
                            " This path is the Windows SYSTEM account — a "
                            "headless service terminal with no window. Kill it "
                            "(taskkill /F /IM terminal64.exe), stop the service, "
                            "and run the dashboard interactively."
                        )
                return jsonify({
                    "error": err,
                    "retcode": retcode,
                    "terminal": term,
                    **result_info,
                }), 400

        except Exception as exc:
            logger.error("POST /api/test-trade error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/terminal-status", methods=["GET"])
    def api_terminal_status():
        """Report the MT5 terminal the API is bound to and its trade permission.

        Use this to confirm whether order_send will be accepted: ``trade_allowed``
        must be True, and ``data_path`` must be the terminal you actually control
        (not a headless SYSTEM-account instance).
        """
        try:
            status = _connector.terminal_status()
            return jsonify(status)
        except Exception as exc:
            logger.error("GET /api/terminal-status error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/limits", methods=["GET", "POST"])
    def api_limits():
        """Get or set the position limits that gate new trades.

        ``max_positions_per_bot`` — how many simultaneous positions a single
        bot may hold (the "Per-bot limit reached (n/N open)" block).
        ``max_open_positions`` — the account-wide safety ceiling across the
        whole fleet.

        Changes apply immediately to the live, shared Config object (no
        restart) and are persisted to ``.env`` so they survive a restart.

        GET  -> current values.
        POST body: ``{"max_positions_per_bot": int, "max_open_positions": int}``
        """
        if request.method == "GET":
            return jsonify({
                "max_positions_per_bot": _config.max_positions_per_bot,
                "max_open_positions": _config.max_open_positions,
            })

        # -- POST: validate --------------------------------------------------
        try:
            data = request.get_json(force=True) or {}
            per_bot = int(data.get("max_positions_per_bot",
                                   _config.max_positions_per_bot))
            fleet = int(data.get("max_open_positions",
                                 _config.max_open_positions))
        except (TypeError, ValueError):
            return jsonify({"error": "Limits must be whole numbers."}), 400

        if not 1 <= per_bot <= 100:
            return jsonify({"error": "Per-bot limit must be between 1 and 100."}), 400
        if not 1 <= fleet <= 500:
            return jsonify({"error": "Fleet limit must be between 1 and 500."}), 400

        # -- Apply live (shared Config -> manager + risk manager) ------------
        _config.max_positions_per_bot = per_bot
        _config.max_open_positions = fleet

        # -- Persist to .env so it survives restarts -------------------------
        persisted = True
        try:
            from dotenv import set_key
            env_path = str(Path(__file__).resolve().parent / ".env")
            set_key(env_path, "MAX_POSITIONS_PER_BOT", str(per_bot))
            set_key(env_path, "MAX_OPEN_POSITIONS", str(fleet))
        except Exception as exc:
            persisted = False
            logger.warning("Could not persist limits to .env: %s", exc)

        # -- Warn if the fleet ceiling can starve the running bots -----------
        warning = None
        try:
            n_bots = len(_bot_mgr.get_all_bots()) if _bot_mgr else 0
            if n_bots and fleet < n_bots * per_bot:
                warning = (
                    f"Fleet cap {fleet} is below {n_bots} bots × {per_bot} "
                    f"per-bot ({n_bots * per_bot}). Some bots may be starved — "
                    f"raise the fleet limit to at least {n_bots * per_bot}."
                )
        except Exception:
            pass

        logger.info(
            "Limits updated via UI: per_bot=%d fleet=%d (persisted=%s)",
            per_bot, fleet, persisted,
        )
        return jsonify({
            "message": "Limits saved." if persisted
            else "Applied live, but .env write failed (will reset on restart).",
            "max_positions_per_bot": per_bot,
            "max_open_positions": fleet,
            "persisted": persisted,
            "warning": warning,
        })

    # ==================================================================
    #  News Intelligence API endpoints
    # ==================================================================

    @app.route("/api/news/recent", methods=["GET"])
    def api_news_recent():
        """Return recent news articles with their AI analysis results."""
        try:
            if _news_engine is None:
                return jsonify({"error": "News engine not initialized"}), 503
            hours = request.args.get("hours", 24, type=int)
            limit = request.args.get("limit", 50, type=int)
            articles = _news_engine["cache"].get_recent_articles(
                limit=limit, hours=hours,
            )
            return jsonify(articles)
        except Exception as exc:
            logger.error("GET /api/news/recent error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/news/signals", methods=["GET"])
    def api_news_signals():
        """Return active news-based trading signals."""
        try:
            if _news_engine is None:
                return jsonify({"error": "News engine not initialized"}), 503
            symbol = request.args.get("symbol", None)
            # Get signals from the signal composer (live in-memory)
            composer = _news_engine["composer"]
            if symbol:
                sig = composer.get_signal(symbol)
                signals = {symbol: sig} if sig else {}
            else:
                signals = composer.get_all_signals()
            # Also get persisted signals from cache
            cached = _news_engine["cache"].get_recent_signals(
                symbol=symbol, limit=20,
            )
            # Provide metadata so the UI can distinguish states
            composer_summary = composer.get_summary()
            return jsonify({
                "live_signals": signals,
                "cached_signals": cached,
                "has_analyses": composer_summary.get("has_analyses", False),
                "analyzed_symbols": composer_summary.get("analyzed_symbols", 0),
                "signaling_symbols": composer_summary.get("signaling_symbols", 0),
                "total_analyses": composer_summary.get("total_analyses", 0),
            })
        except Exception as exc:
            logger.error("GET /api/news/signals error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/news/stats", methods=["GET"])
    def api_news_stats():
        """Return news engine statistics."""
        try:
            if _news_engine is None:
                return jsonify({
                    "enabled": False,
                    "reason": "News engine not initialized",
                })
            ingestion = _news_engine["ingestion"]
            analyzer = _news_engine["analyzer"]
            composer = _news_engine["composer"]
            cache = _news_engine["cache"]
            # Get cost from persistent DB, not in-memory
            usage = cache.get_total_usage() if hasattr(cache, 'get_total_usage') else (
                analyzer.get_usage_stats() if analyzer else {}
            )
            # Use cache (SQLite) for consistent article/analyzed counts
            cache_stats = cache.get_signal_stats()
            composer_summary = composer.get_summary()
            unanalyzed = cache.get_unanalyzed_articles(limit=1)
            return jsonify({
                "enabled": True,
                "analysis_paused": _news_engine.get("analysis_paused", False),
                "ingestion": {
                    "is_running": ingestion.is_running,
                    "article_count": ingestion.article_count,
                    "source_status": ingestion.source_status,
                },
                "analyzer": usage,
                "composer": composer_summary,
                "cache": cache_stats,
                "has_analyses": composer_summary.get("has_analyses", False),
                "unanalyzed_count": len(unanalyzed),
            })
        except Exception as exc:
            logger.error("GET /api/news/stats error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/news/timeline", methods=["GET"])
    def api_news_timeline():
        """Return sentiment timeline data for charts."""
        try:
            if _news_engine is None:
                return jsonify({"error": "News engine not initialized"}), 503
            hours = request.args.get("hours", 24, type=int)
            symbol = request.args.get("symbol", None)
            timeline = _news_engine["cache"].get_sentiment_timeline(
                hours=hours, symbol=symbol,
            )
            return jsonify(timeline)
        except Exception as exc:
            logger.error("GET /api/news/timeline error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/news/sources", methods=["GET"])
    def api_news_sources():
        """Return health status of each news source."""
        try:
            if _news_engine is None:
                return jsonify({"error": "News engine not initialized"}), 503
            return jsonify(_news_engine["cache"].get_source_health())
        except Exception as exc:
            logger.error("GET /api/news/sources error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/news/toggle", methods=["POST"])
    def api_news_toggle():
        """Toggle AI analysis ON/OFF. Ingestion continues regardless."""
        try:
            if _news_engine is None:
                return jsonify({"error": "News engine not initialized"}), 503
            _news_engine["analysis_paused"] = not _news_engine["analysis_paused"]
            paused = _news_engine["analysis_paused"]
            state = "OFF (paused)" if paused else "ON (active)"
            logger.info("News analysis toggled: %s", state)
            # Persist toggle state so it survives restarts
            _save_ai_toggle_state(paused)
            return jsonify({
                "analysis_paused": paused,
                "message": f"AI analysis is now {state}",
            })
        except Exception as exc:
            logger.error("POST /api/news/toggle error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/news/refresh", methods=["POST"])
    def api_news_refresh():
        """Force-refresh signals from the cache (re-compose from analyses)."""
        try:
            if _news_engine is None:
                return jsonify({"error": "News engine not initialized"}), 503
            cache = _news_engine["cache"]
            composer = _news_engine["composer"]
            # Reload ALL recent analyses into the composer (including NEUTRAL)
            recent = cache.get_recent_articles(limit=100, hours=6)
            reloaded = 0
            for art in recent:
                if art.get("sentiment"):
                    composer.add_analysis({
                        "sentiment": art["sentiment"],
                        "confidence": art.get("analysis_confidence", 0),
                        "impact_score": art.get("impact_score", 0),
                        "affected_symbols": art.get("affected_symbols", []),
                        "reasoning": art.get("analysis_reasoning", ""),
                        "suggested_action": art.get("suggested_action", "HOLD"),
                    }, article_id=art["id"])
                    reloaded += 1
            all_signals = composer.get_all_signals()
            actionable = {
                sym: sig for sym, sig in all_signals.items()
                if sig.get("action") in ("BUY", "SELL")
            }
            return jsonify({
                "reloaded_analyses": reloaded,
                "total_signals": len(all_signals),
                "actionable_signals": len(actionable),
                "signals": all_signals,
            })
        except Exception as exc:
            logger.error("POST /api/news/refresh error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ==================================================================
    #  Strategy Backtester endpoints
    # ==================================================================

    # Lazily initialised — created on first use
    _backtest_engine = None

    def _get_backtest_engine():
        nonlocal _backtest_engine
        if _backtest_engine is None:
            from backtester import BacktestEngine
            # Pass the live connector so the engine can resolve broker symbol
            # suffixes (GBPJPY → GBPJPY.r …) and enable the symbol in Market
            # Watch before pulling history — otherwise copy_rates_range fails
            # with (-2, 'Invalid params').
            _backtest_engine = BacktestEngine(_config, _connector)
        return _backtest_engine

    @app.route("/api/backtest/run", methods=["POST"])
    def api_backtest_run():
        """Start a new backtest in a background thread."""
        try:
            body = request.get_json(force=True)
            required = ["strategy", "symbol", "timeframe", "date_from", "date_to"]
            for key in required:
                if key not in body:
                    return jsonify({"error": f"Missing field: {key}"}), 400

            engine = _get_backtest_engine()
            run_id = engine.run_async(
                strategy_name=body["strategy"],
                symbol=body["symbol"],
                timeframe_key=body["timeframe"],
                date_from=body["date_from"],
                date_to=body["date_to"],
                initial_balance=float(body.get("initial_balance", 10000)),
                lot_size=float(body.get("lot_size", 0.1)),
                sl_pips=float(body.get("sl_pips", 0)),
                tp_pips=float(body.get("tp_pips", 0)),
                max_open_trades=int(body.get("max_open_trades", 5)),
                strategy_params=body.get("strategy_params"),
            )
            return jsonify({"run_id": run_id, "status": "running"})
        except Exception as exc:
            logger.error("POST /api/backtest/run error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/backtest/status/<run_id>", methods=["GET"])
    def api_backtest_status(run_id: str):
        """Poll the status/progress of a running backtest."""
        try:
            engine = _get_backtest_engine()
            status = engine.get_status(run_id)
            if status is None:
                return jsonify({"error": "Run not found"}), 404
            return jsonify(status)
        except Exception as exc:
            logger.error("GET /api/backtest/status error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/backtest/results/<run_id>", methods=["GET"])
    def api_backtest_results(run_id: str):
        """Return full results: metrics + trades + equity curve."""
        try:
            engine = _get_backtest_engine()
            results = engine.get_results(run_id)
            if results is None:
                return jsonify({"error": "Run not found"}), 404
            return jsonify(results)
        except Exception as exc:
            logger.error("GET /api/backtest/results error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/backtest/<run_id>/trades.csv", methods=["GET"])
    def api_backtest_trades_csv(run_id: str):
        """Download a backtest run's trades as CSV (with entry/exit time + session)."""
        try:
            engine = _get_backtest_engine()
            run = engine.db.get_run(run_id)
            if run is None:
                return jsonify({"error": "Run not found"}), 404
            trades = engine.db.get_trades(run_id)
            csv_text = performance.backtest_trades_to_csv(run, trades)
            fname = (f"backtest_{run.get('strategy_name','')}_{run.get('symbol','')}"
                     f"_{run.get('timeframe','')}_{run_id}.csv")
            return Response(
                csv_text, mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename={fname}"},
            )
        except Exception as exc:
            logger.error("GET /api/backtest/%s/trades.csv error: %s", run_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/backtest/<run_id>/summary", methods=["GET"])
    def api_backtest_summary(run_id: str):
        """Backtest performance summary (overall + time-of-day + breakdowns)."""
        try:
            engine = _get_backtest_engine()
            run = engine.db.get_run(run_id)
            if run is None:
                return jsonify({"error": "Run not found"}), 404
            trades = engine.db.get_trades(run_id)
            slab = request.args.get("slab", "daily")
            summary = performance.build_backtest_summary(run, trades, slab=slab)
            if request.args.get("download"):
                import json as _json
                fname = (f"backtest_summary_{run.get('strategy_name','')}_"
                         f"{run.get('symbol','')}_{run_id}.json")
                return Response(
                    _json.dumps(summary, indent=2), mimetype="application/json",
                    headers={"Content-Disposition": f"attachment; filename={fname}"},
                )
            return jsonify(summary)
        except Exception as exc:
            logger.error("GET /api/backtest/%s/summary error: %s", run_id, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/backtest/history", methods=["GET"])
    def api_backtest_history():
        """Return recent backtest run summaries."""
        try:
            engine = _get_backtest_engine()
            limit = request.args.get("limit", 50, type=int)
            history = engine.get_history(limit)
            return jsonify(history)
        except Exception as exc:
            logger.error("GET /api/backtest/history error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/backtest/<run_id>", methods=["DELETE"])
    def api_backtest_delete(run_id: str):
        """Delete a backtest run and its data."""
        try:
            engine = _get_backtest_engine()
            deleted = engine.delete_run(run_id)
            if not deleted:
                return jsonify({"error": "Run not found"}), 404
            return jsonify({"deleted": True, "run_id": run_id})
        except Exception as exc:
            logger.error("DELETE /api/backtest error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ==================================================================
    #  Strategy Optimizer endpoints
    # ==================================================================

    _optimizer = None
    _cloud_optimizer = None

    def _get_optimizer():
        nonlocal _optimizer
        if _optimizer is None:
            from optimizer import StrategyOptimizer
            _optimizer = StrategyOptimizer(_config)
        return _optimizer

    def _get_cloud_optimizer():
        nonlocal _cloud_optimizer
        if _cloud_optimizer is None:
            from cloud.cloud_optimizer import CloudOptimizer
            _cloud_optimizer = CloudOptimizer(_config)
        return _cloud_optimizer

    @app.route("/api/optimize/run", methods=["POST"])
    def api_optimize_run():
        """Start a strategy optimisation.

        Pass ``"cloud": true`` in the body to distribute across AWS
        Spot Instances instead of running locally.
        """
        try:
            body = request.get_json(force=True)
            required = ["strategy", "symbol", "timeframe", "date_from", "date_to"]
            for key in required:
                if key not in body:
                    return jsonify({"error": f"Missing field: {key}"}), 400

            use_cloud = body.get("cloud", False)

            if use_cloud:
                optimizer = _get_cloud_optimizer()
            else:
                optimizer = _get_optimizer()

            opt_id = optimizer.start(
                strategy_name=body["strategy"],
                symbol=body["symbol"],
                timeframe_key=body["timeframe"],
                date_from=body["date_from"],
                date_to=body["date_to"],
                mode=body.get("mode", "normal"),
                optimize_sltp=body.get("optimize_sltp", False),
                target_metric=body.get("target_metric", "profit_factor"),
                initial_balance=float(body.get("initial_balance", 10000)),
                lot_size=float(body.get("lot_size", 0.1)),
                min_trades=int(body.get("min_trades", 3)),
                max_open_trades=int(body.get("max_open_trades", 5)),
            )
            return jsonify({
                "opt_id": opt_id,
                "status": "running",
                "cloud": use_cloud,
            })
        except Exception as exc:
            logger.error("POST /api/optimize/run error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimize/stop/<opt_id>", methods=["POST"])
    def api_optimize_stop(opt_id: str):
        """Stop a running optimisation. Partial results are preserved."""
        try:
            # Check both local and cloud optimizers
            optimizer = _get_optimizer()
            stopped = optimizer.stop(opt_id)
            if not stopped and _cloud_optimizer is not None:
                stopped = _cloud_optimizer.stop(opt_id)
            if not stopped:
                return jsonify({"error": "Optimisation not found or already finished"}), 404
            return jsonify({"stopped": True, "opt_id": opt_id})
        except Exception as exc:
            logger.error("POST /api/optimize/stop error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimize/status/<opt_id>", methods=["GET"])
    def api_optimize_status(opt_id: str):
        """Poll optimisation progress (checks both local and cloud)."""
        try:
            optimizer = _get_optimizer()
            status = optimizer.get_status(opt_id)
            if status is None and _cloud_optimizer is not None:
                status = _cloud_optimizer.get_status(opt_id)
            if status is None:
                return jsonify({"error": "Optimisation not found"}), 404
            return jsonify(status)
        except Exception as exc:
            logger.error("GET /api/optimize/status error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimize/results/<opt_id>", methods=["GET"])
    def api_optimize_results(opt_id: str):
        """Return full optimisation results (checks both local and cloud)."""
        try:
            optimizer = _get_optimizer()
            results = optimizer.get_results(opt_id)
            if results is None and _cloud_optimizer is not None:
                results = _cloud_optimizer.get_results(opt_id)
            if results is None:
                return jsonify({"error": "Optimisation not found"}), 404
            return jsonify(results)
        except Exception as exc:
            logger.error("GET /api/optimize/results error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ── Cloud-specific endpoints ─────────────────────────────────────

    @app.route("/api/optimize/cloud/config", methods=["GET"])
    def api_cloud_config():
        """Check cloud optimisation availability and configuration."""
        try:
            from cloud.setup import verify_setup
            status = verify_setup()
            status["enabled"] = getattr(_config, "aws_cloud_enabled", False)
            return jsonify(status)
        except ImportError:
            return jsonify({"ready": False, "enabled": False,
                            "reason": "Cloud module not installed (pip install boto3 pyarrow)."})
        except Exception as exc:
            logger.error("GET /api/optimize/cloud/config error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimize/cloud/setup", methods=["POST"])
    def api_cloud_setup():
        """Run one-time AWS infrastructure setup."""
        try:
            from cloud.setup import run_setup
            region = getattr(_config, "aws_region", "us-east-1")
            bucket = getattr(_config, "aws_cloud_s3_bucket", "") or None
            result = run_setup(region=region, bucket_name=bucket)
            return jsonify(result)
        except Exception as exc:
            logger.error("POST /api/optimize/cloud/setup error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimize/cloud/estimate", methods=["POST"])
    def api_cloud_estimate():
        """Estimate cost and time for a cloud optimisation job."""
        try:
            body = request.get_json(force=True)
            cloud_opt = _get_cloud_optimizer()
            estimate = cloud_opt.estimate(
                strategy_name=body["strategy"],
                mode=body.get("mode", "normal"),
                optimize_sltp=body.get("optimize_sltp", False),
            )
            return jsonify(estimate)
        except Exception as exc:
            logger.error("POST /api/optimize/cloud/estimate error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimize/grids", methods=["GET"])
    def api_optimize_grids():
        """Return the parameter grid info (combo counts per strategy/mode)."""
        try:
            from optimizer import PARAM_GRIDS, SLTP_GRIDS, VALIDATORS
            from itertools import product as _product
            info = {}
            for strat, grid in PARAM_GRIDS.items():
                info[strat] = {}
                validator = VALIDATORS.get(strat, lambda p: True)
                for mode in ("quick", "normal", "deep"):
                    names = []
                    values = []
                    for pname, mode_dict in grid.items():
                        vals = mode_dict.get(mode, mode_dict["normal"])
                        names.append(pname)
                        values.append(vals)
                    count = sum(1 for v in _product(*values) if validator(dict(zip(names, v))))
                    # With SL/TP
                    sltp_count = 1
                    for _, md in SLTP_GRIDS.items():
                        sltp_count *= len(md.get(mode, md["normal"]))
                    info[strat][mode] = {
                        "combos": count,
                        "combos_with_sltp": count * sltp_count,
                        "params": list(grid.keys()),
                    }
            return jsonify(info)
        except Exception as exc:
            logger.error("GET /api/optimize/grids error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ==================================================================
    #  Optimized Strategy Repository endpoints
    # ==================================================================

    @app.route("/api/optimized/save", methods=["POST"])
    def api_optimized_save():
        """Save an optimized strategy to the repository."""
        try:
            body = request.get_json(force=True)
            required = ["strategy_name", "symbol", "timeframe", "params"]
            for key in required:
                if key not in body:
                    return jsonify({"error": f"Missing field: {key}"}), 400
            from backtester import BacktestDB
            db = BacktestDB()
            opt_id = db.save_optimized_strategy(body)
            return jsonify({"id": opt_id, "saved": True})
        except Exception as exc:
            logger.error("POST /api/optimized/save error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimized/lookup", methods=["GET"])
    def api_optimized_lookup():
        """Check if an optimized strategy exists for a strategy+symbol+timeframe combo."""
        try:
            strategy = request.args.get("strategy", "")
            symbol = request.args.get("symbol", "")
            timeframe = request.args.get("timeframe", "")
            if not strategy or not symbol or not timeframe:
                return jsonify({"error": "strategy, symbol, and timeframe are required"}), 400
            from backtester import BacktestDB
            db = BacktestDB()
            result = db.get_optimized_strategy(strategy, symbol, timeframe)
            return jsonify({"found": result is not None, "data": result})
        except Exception as exc:
            logger.error("GET /api/optimized/lookup error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimized/all", methods=["GET"])
    def api_optimized_all():
        """List all saved optimized strategies."""
        try:
            from backtester import BacktestDB
            db = BacktestDB()
            results = db.get_all_optimized()
            return jsonify({"strategies": results})
        except Exception as exc:
            logger.error("GET /api/optimized/all error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/optimized/<opt_id>", methods=["DELETE"])
    def api_optimized_delete(opt_id: str):
        """Delete a saved optimized strategy."""
        try:
            from backtester import BacktestDB
            db = BacktestDB()
            deleted = db.delete_optimized(opt_id)
            if not deleted:
                return jsonify({"error": "Not found"}), 404
            return jsonify({"deleted": True})
        except Exception as exc:
            logger.error("DELETE /api/optimized error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ==================================================================
    #  Order Management endpoints
    # ==================================================================

    @app.route("/api/positions/<int:ticket>/modify", methods=["POST"])
    def api_position_modify(ticket: int):
        """Modify SL/TP of an open position."""
        try:
            data = request.get_json(force=True) or {}
            sl = data.get("sl")
            tp = data.get("tp")
            if sl is not None:
                sl = float(sl)
            if tp is not None:
                tp = float(tp)
            ok = _order_mgr.modify_position(ticket, sl=sl, tp=tp)
            if ok:
                return jsonify({"message": f"Position #{ticket} modified", "ticket": ticket})
            return jsonify({"error": f"Failed to modify position #{ticket}"}), 400
        except Exception as exc:
            logger.error("POST /api/positions/%d/modify error: %s", ticket, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/positions/<int:ticket>/partial-close", methods=["POST"])
    def api_position_partial_close(ticket: int):
        """Partially close an open position."""
        try:
            data = request.get_json(force=True) or {}
            close_pct = float(data.get("close_pct", 50))
            ok = _order_mgr.partial_close_position(ticket, close_pct)
            if ok:
                return jsonify({"message": f"Partially closed {close_pct}% of #{ticket}", "ticket": ticket})
            return jsonify({"error": f"Failed to partially close #{ticket}"}), 400
        except Exception as exc:
            logger.error("POST /api/positions/%d/partial-close error: %s", ticket, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/positions/<int:ticket>/breakeven", methods=["POST"])
    def api_position_breakeven(ticket: int):
        """Move SL to entry price (breakeven)."""
        try:
            data = request.get_json(force=True) or {}
            offset_pips = float(data.get("offset_pips", 0))
            ok = _order_mgr.set_breakeven(ticket, offset_pips=offset_pips)
            if ok:
                return jsonify({"message": f"Breakeven set for #{ticket}", "ticket": ticket})
            return jsonify({"error": f"Failed to set breakeven for #{ticket}. Position may not be in profit."}), 400
        except Exception as exc:
            logger.error("POST /api/positions/%d/breakeven error: %s", ticket, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/positions/<int:ticket>/trailing", methods=["POST"])
    def api_position_trailing(ticket: int):
        """Enable/configure trailing stop for a position."""
        try:
            data = request.get_json(force=True) or {}
            enabled = data.get("enabled", True)
            profile_name = data.get("profile", "atr")
            params = data.get("params", {})

            if not enabled:
                # Remove trailing from tracking
                _trailing_positions.pop(ticket, None)
                return jsonify({"message": f"Trailing stop disabled for #{ticket}", "ticket": ticket, "trailing": False})

            # Validate profile exists
            from trailing_profiles import list_trailing_profiles
            available = [name for name, _ in list_trailing_profiles()]
            if profile_name not in available:
                return jsonify({"error": f"Unknown trailing profile '{profile_name}'. Available: {available}"}), 400

            # Store trailing config for this position
            _trailing_positions[ticket] = {
                "profile": profile_name,
                "params": params,
                "enabled": True,
            }
            return jsonify({
                "message": f"Trailing stop '{profile_name}' enabled for #{ticket}",
                "ticket": ticket,
                "trailing": True,
                "profile": profile_name,
            })
        except Exception as exc:
            logger.error("POST /api/positions/%d/trailing error: %s", ticket, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/positions/<int:ticket>/close", methods=["POST"])
    def api_position_close(ticket: int):
        """Close a single open position."""
        try:
            ok = _order_mgr.close_position(ticket)
            if ok:
                return jsonify({"message": f"Position #{ticket} closed", "ticket": ticket})
            return jsonify({"error": f"Failed to close position #{ticket}"}), 400
        except Exception as exc:
            logger.error("POST /api/positions/%d/close error: %s", ticket, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/pending", methods=["GET"])
    def api_pending_orders():
        """Return all pending orders."""
        try:
            orders = _order_mgr.get_pending_orders()
            # Attach trailing info if available
            return jsonify(orders)
        except Exception as exc:
            logger.error("GET /api/pending error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/pending/<int:ticket>/modify", methods=["POST"])
    def api_pending_modify(ticket: int):
        """Modify a pending order."""
        try:
            data = request.get_json(force=True) or {}
            price = data.get("price")
            sl = data.get("sl")
            tp = data.get("tp")
            exp_str = data.get("expiration")

            if price is not None:
                price = float(price)
            if sl is not None:
                sl = float(sl)
            if tp is not None:
                tp = float(tp)

            expiration = None
            if exp_str:
                from datetime import datetime
                expiration = datetime.fromisoformat(str(exp_str))

            ok = _order_mgr.modify_pending_order(
                ticket, price=price, sl=sl, tp=tp, expiration=expiration,
            )
            if ok:
                return jsonify({"message": f"Pending order #{ticket} modified", "ticket": ticket})
            return jsonify({"error": f"Failed to modify pending order #{ticket}"}), 400
        except Exception as exc:
            logger.error("POST /api/pending/%d/modify error: %s", ticket, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/pending/<int:ticket>/cancel", methods=["POST"])
    def api_pending_cancel(ticket: int):
        """Cancel a pending order."""
        try:
            ok = _order_mgr.cancel_pending_order(ticket)
            if ok:
                return jsonify({"message": f"Pending order #{ticket} cancelled", "ticket": ticket})
            return jsonify({"error": f"Failed to cancel pending order #{ticket}"}), 400
        except Exception as exc:
            logger.error("POST /api/pending/%d/cancel error: %s", ticket, exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/trailing-profiles", methods=["GET"])
    def api_trailing_profiles():
        """Return available trailing stop profiles."""
        try:
            from trailing_profiles import list_trailing_profiles
            profiles = list_trailing_profiles()
            return jsonify([{"name": name, "description": desc} for name, desc in profiles])
        except Exception as exc:
            logger.error("GET /api/trailing-profiles error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/positions/trailing-status", methods=["GET"])
    def api_trailing_status():
        """Return which positions have trailing stops enabled."""
        try:
            return jsonify(_trailing_positions)
        except Exception as exc:
            logger.error("GET /api/positions/trailing-status error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    return app




# =====================================================================
#  Helpers
# =====================================================================

_AI_TOGGLE_FILE = Path(__file__).resolve().parent / "data" / "ai_toggle_state.json"


def _load_ai_toggle_state() -> bool:
    """Load the persisted AI analysis paused state. Returns False (ON) if no file exists."""
    try:
        if _AI_TOGGLE_FILE.exists():
            import json
            data = json.loads(_AI_TOGGLE_FILE.read_text())
            return data.get("analysis_paused", False)
    except Exception:
        pass
    return False


def _save_ai_toggle_state(paused: bool) -> None:
    """Save the AI analysis paused state to disk."""
    try:
        import json
        _AI_TOGGLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _AI_TOGGLE_FILE.write_text(json.dumps({"analysis_paused": paused}))
    except Exception:
        pass

def _serialise_bot(bot: BotInstance) -> dict[str, Any]:
    """Convert a BotInstance to a JSON-safe dictionary."""
    return {
        "bot_id": bot.bot_id,
        "symbol": bot.symbol,
        "strategy_name": bot.strategy_name,
        "timeframe_key": bot.timeframe_key,
        "lot_size": bot.lot_size,
        "lot_mode": bot.lot_mode,
        "status": bot.status.value,
        "last_signal": bot.last_signal,
        "last_price": bot.last_price,
        "trades_taken": bot.trades_taken,
        "last_check": bot.last_check,
        "dry_run": bot.dry_run,
        "error_msg": bot.error_msg,
        "trigger_proximity": bot.trigger_proximity,
        "last_entry": bot.last_entry,
        "last_sl": bot.last_sl,
        "last_tp": bot.last_tp,
        "last_ticket": bot.last_ticket,
        "active_positions": bot.active_positions,
        "blocked_reason": bot.blocked_reason,
        "quick_exit_enabled": bot.quick_exit_enabled,
        "quick_exit_pips": bot.quick_exit_pips,
    }


# =====================================================================
#  Initialisation & Entry Point
# =====================================================================

def _init_subsystems() -> None:
    """Initialise all bot subsystems and store them at module level."""
    global _config, _connector, _market_data, _session_mgr
    global _order_mgr, _risk_mgr, _bot_mgr, _news_engine

    logger.info("Loading configuration …")
    _config = Config()
    logger.info("Config loaded: %s", _config)

    logger.info("Connecting to MT5 …")
    _connector = MT5Connector(_config)
    if not _connector.connect():
        logger.error("Failed to connect to MT5. Exiting.")
        sys.exit(1)
    logger.info("MT5 connected successfully.")

    _market_data = MarketDataFeed()
    _session_mgr = SessionManager()
    _order_mgr = OrderManager(_config, dry_run=False)  # always live; per-bot toggle controls dry/live
    _risk_mgr = RiskManager(_config, _connector)
    _bot_mgr = MultiBotManager(
        config=_config,
        connector=_connector,
        market_data=_market_data,
        session_mgr=_session_mgr,
        order_mgr=_order_mgr,
        risk_mgr=_risk_mgr,
    )
    logger.info("All core subsystems initialised.")

    # -- Restore saved bots from disk --------------------------------------
    restored = _bot_mgr._load_bots()
    if restored > 0:
        logger.info("Restored %d bot(s) from previous session.", restored)

    # -- News Intelligence Engine ------------------------------------------
    _news_engine = _init_news_engine(_config)

    # -- Cloud cleanup (best-effort) ---------------------------------------
    if getattr(_config, "aws_cloud_enabled", False):
        try:
            from cloud.cleanup import run_full_cleanup
            import threading as _thr
            _thr.Thread(target=run_full_cleanup, daemon=True,
                        name="cloud-cleanup").start()
            logger.info("Cloud cleanup started in background.")
        except Exception as exc:
            logger.debug("Cloud cleanup skipped: %s", exc)


def _init_news_engine(config: Config) -> dict | None:
    """Initialise the News Intelligence engine if configured.

    Returns a dict with all news subsystem references, or None if
    the engine could not be started (missing API keys, etc.).
    """
    if not config.news_analysis_enabled:
        logger.info("News Intelligence is disabled (NEWS_ANALYSIS_ENABLED=false).")
        return None

    if not config.finnhub_api_key:
        logger.warning(
            "News Intelligence: FINNHUB_API_KEY not set — "
            "news ingestion will use RSS feeds only."
        )

    if not config.claude_api_key:
        logger.warning(
            "News Intelligence: CLAUDE_API_KEY not set — "
            "news analysis will be disabled (ingestion only)."
        )

    try:
        from news_intelligence.news_ingestion import NewsIngestionEngine
        from news_intelligence.news_cache import NewsCache
        from news_intelligence.news_analyzer import ClaudeNewsAnalyzer
        from news_intelligence.signal_composer import NewsSignalComposer

        # 1. Cache (SQLite persistence)
        cache = NewsCache(
            retention_days=config.news_cache_retention_days,
        )

        # 2. News Ingestion
        ingestion = NewsIngestionEngine(
            finnhub_api_key=config.finnhub_api_key,
            news_api_key=config.news_api_key,
            rss_feeds=config.news_rss_feeds or None,
            polling_interval=config.news_polling_interval,
        )

        # 3. AI Analyzer (only if Claude key is set)
        analyzer = None
        if config.claude_api_key:
            analyzer = ClaudeNewsAnalyzer(api_key=config.claude_api_key)

        # 4. Signal Composer
        composer = NewsSignalComposer(
            lookback_hours=config.news_lookback_hours,
            min_confidence=config.news_min_confidence,
            min_impact=config.news_min_impact,
            signal_threshold=config.news_signal_threshold,
        )

        engine = {
            "ingestion": ingestion,
            "cache": cache,
            "analyzer": analyzer,
            "composer": composer,
            "analysis_paused": _load_ai_toggle_state(),
        }

        # Start ingestion
        ingestion.start()
        logger.info("News Intelligence: ingestion engine started.")

        # Start the analysis processing loop in a background thread
        import threading

        def _news_processing_loop():
            """Background loop that analyses new articles and composes signals."""
            import time
            _proc_logger = logging.getLogger("news_processor")
            _proc_logger.info("News processing loop started.")

            if not analyzer:
                _proc_logger.warning(
                    "⚠️  CLAUDE_API_KEY not set — articles will be ingested "
                    "but NOT analyzed.  Add your key to .env to enable AI analysis."
                )

            # ── Rehydrate composer from SQLite on startup ─────────────
            # This restores signals that were persisted before the last
            # restart so the dashboard doesn't show "No active signals".
            try:
                recent = cache.get_recent_articles(limit=200, hours=12)
                reloaded = 0
                for art in recent:
                    if art.get("sentiment"):
                        composer.add_analysis({
                            "sentiment": art["sentiment"],
                            "confidence": art.get("analysis_confidence", 0),
                            "impact_score": art.get("impact_score", 0),
                            "affected_symbols": art.get("affected_symbols", []),
                            "reasoning": art.get("analysis_reasoning", ""),
                            "suggested_action": art.get("suggested_action", "HOLD"),
                        }, article_id=art["id"])
                        reloaded += 1
                if reloaded > 0:
                    restored_signals = composer.get_all_signals()
                    _proc_logger.info(
                        "♻️  Rehydrated %d analyses from DB → %d active signal(s).",
                        reloaded, len(restored_signals),
                    )
                else:
                    _proc_logger.info("No cached analyses to rehydrate.")
            except Exception as exc:
                _proc_logger.warning(
                    "Failed to rehydrate from cache: %s", exc,
                )

            while True:
                try:
                    # Get new articles from ingestion engine
                    new_articles = ingestion.get_latest(
                        max_age_hours=config.news_lookback_hours,
                    )

                    # Store ALL articles in cache (dedup handled by cache)
                    stored_count = 0
                    for article in new_articles:
                        if cache.store_article(article):
                            stored_count += 1
                    if stored_count > 0:
                        _proc_logger.info(
                            "Cached %d new articles (%d total from ingestion).",
                            stored_count, len(new_articles),
                        )

                    # Get unanalyzed articles from cache and analyze
                    # Rate-limited: max 5 per cycle, only recent articles
                    if analyzer and not engine["analysis_paused"]:
                        unanalyzed = cache.get_unanalyzed_articles(limit=5)
                        # Filter out old articles (>6h) to avoid burning API on stale news
                        from datetime import datetime, timezone, timedelta
                        cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
                        fresh = []
                        for a in unanalyzed:
                            try:
                                pub = a.get("published_at") or a.get("ingested_at", "")
                                if pub and datetime.fromisoformat(str(pub).replace('Z','+00:00')) > cutoff:
                                    fresh.append(a)
                            except (ValueError, TypeError):
                                fresh.append(a)  # include if we can't parse date
                        if fresh:
                            _proc_logger.info(
                                "Analysing %d articles (skipped %d old) …",
                                len(fresh), len(unanalyzed) - len(fresh),
                            )

                            # Track tokens before/after for persistent logging
                            tokens_before_in = analyzer.total_input_tokens
                            tokens_before_out = analyzer.total_output_tokens
                            calls_before = analyzer.total_calls

                            # Batch analyze
                            batch = [
                                {
                                    "headline": a["headline"],
                                    "summary": a.get("summary", ""),
                                    "source": a.get("source_name", ""),
                                }
                                for a in fresh
                            ]
                            results = analyzer.analyze_batch(batch)

                            for article_dict, analysis in zip(fresh, results):
                                cache.store_analysis(article_dict["id"], analysis)
                                composer.add_analysis(
                                    analysis, article_id=article_dict["id"],
                                )

                            # Persist token usage to DB
                            new_calls = analyzer.total_calls - calls_before
                            new_in = analyzer.total_input_tokens - tokens_before_in
                            new_out = analyzer.total_output_tokens - tokens_before_out
                            if new_calls > 0 and hasattr(cache, 'log_usage'):
                                cache.log_usage(new_calls, new_in, new_out)
                                _proc_logger.info(
                                    "API usage: +%d calls, +%d in/%d out tokens",
                                    new_calls, new_in, new_out,
                                )

                            # Store only actionable (BUY/SELL) signals in cache
                            # HOLD signals are derived on the fly and don't
                            # need to be persisted.
                            actionable_signals = composer.get_actionable_signals()
                            for symbol, signal_data in actionable_signals.items():
                                cache.store_signal(
                                    symbol=symbol,
                                    action=signal_data["action"],
                                    strength=signal_data["strength"],
                                    confidence=signal_data["confidence"],
                                    source_article_ids=[],
                                    reasoning=signal_data["reason"],
                                    expires_hours=config.news_lookback_hours,
                                )

                            all_signals = composer.get_all_signals()
                            if all_signals:
                                _proc_logger.info(
                                    "Generated %d signal(s) (%d actionable): %s",
                                    len(all_signals),
                                    len(actionable_signals),
                                    ", ".join(f"{s}={d['action']}" for s, d in all_signals.items()),
                                )

                    # Periodic cache cleanup
                    cache.cleanup()

                except Exception as exc:
                    _proc_logger.error(
                        "News processing error: %s", exc, exc_info=True,
                    )

                time.sleep(config.news_polling_interval)

        proc_thread = threading.Thread(
            target=_news_processing_loop,
            name="NewsProcessor",
            daemon=True,
        )
        proc_thread.start()
        logger.info("News Intelligence: processing loop started.")

        return engine

    except ImportError as exc:
        logger.warning(
            "News Intelligence: could not import modules (%s). "
            "Install requirements: pip install anthropic feedparser", exc,
        )
        return None
    except Exception as exc:
        logger.error(
            "News Intelligence: failed to initialise: %s", exc, exc_info=True,
        )
        return None


def main() -> None:
    """Entry point — print banner, initialise, and run the Flask server."""
    import os
    import threading
    import webbrowser

    banner = r"""
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║      ∞  RIDER INFINITY — Trading Dashboard       ║
    ║    Ride on the Infinity edge of Markets.          ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """
    print(banner)

    _init_subsystems()

    app = create_app()

    url = "http://localhost:5000"
    print("\n" + "=" * 52)
    print(f"  Dashboard running at {url}")
    print("  Opening browser automatically ...")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 52 + "\n")

    # Auto-open browser only once (skip in Flask reloader's child process)
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=True,
        threaded=True,
    )


if __name__ == "__main__":
    main()
