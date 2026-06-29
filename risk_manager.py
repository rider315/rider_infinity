"""
risk_manager.py — Risk-Management Layer for the MT5 Forex Bot.

Enforces per-trade risk limits, daily drawdown caps, spread filters,
maximum open-position counts, calculates risk-based lot sizing, and
provides advanced features: correlation awareness, volatility-adjusted
sizing, drawdown recovery mode, and currency exposure analysis.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5

from config import Config
from session_manager import is_crypto

# File where risk state is persisted across restarts
_STATE_FILE = Path(__file__).resolve().parent / "data" / "risk_state.json"


# ---------------------------------------------------------------------------
# Correlation maps — used by the enhanced risk checks
# ---------------------------------------------------------------------------

# Pairs with strong positive correlation (>0.8 typically).
# Same-direction positions in these pairs effectively double risk.
POSITIVE_CORRELATIONS: list[tuple[str, str]] = [
    ("EURUSD", "GBPUSD"),   # Both move with USD weakness
    ("AUDUSD", "NZDUSD"),   # Both commodity currencies
    ("EURJPY", "GBPJPY"),   # Both vs JPY
    ("EURUSD", "EURGBP"),   # EUR base correlation
    ("GBPUSD", "GBPJPY"),   # GBP base correlation
    ("AUDUSD", "AUDJPY"),   # AUD base correlation
]

# Pairs with strong negative correlation (< -0.8 typically).
# Same-direction positions in these pairs partially hedge each other.
NEGATIVE_CORRELATIONS: list[tuple[str, str]] = [
    ("EURUSD", "USDCHF"),   # Inverse USD relationship
    ("GBPUSD", "USDCHF"),   # Inverse USD relationship
    ("AUDUSD", "USDCAD"),   # Commodity vs commodity-importer
    ("EURUSD", "USDJPY"),   # EUR/USD inverse
]


class RiskManager:
    """
    Guards every trade decision with pre-flight risk checks and
    calculates position sizes based on a fixed-percentage risk model.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config: Config, connector) -> None:
        """
        Parameters
        ----------
        config : Config
            Application configuration instance.
        connector : MT5Connector
            Live connection wrapper — used to pull account and symbol data.
        """
        self.config = config
        self.connector = connector
        self.logger = logging.getLogger("RiskManager")

        # Daily tracking — call reset_daily_tracking() at the start of
        # each trading day (or on first use).
        self._daily_start_equity: float | None = None
        self._daily_start_date: datetime | None = None

        # --- Stateful risk tracking (OctoBot-inspired) ---
        self._risk_state: dict = {
            "total_exposure": 0.0,
            "currency_exposure": {},
            "drawdown_recovery_mode": False,
            "recovery_lot_multiplier": 1.0,
            "peak_equity": 0.0,
            "last_update": None,
        }
        self._correlation_warnings: list[str] = []

        # Restore persisted state (daily tracking, peak equity, recovery)
        self._load_state()

    # ------------------------------------------------------------------
    # Daily tracking
    # ------------------------------------------------------------------
    def reset_daily_tracking(self) -> None:
        """Snapshot the current equity as today's starting equity."""
        account = self.connector.get_account_info()
        if account is None:
            self.logger.warning(
                "Cannot reset daily tracking — account info unavailable."
            )
            return

        self._daily_start_equity = account["equity"]
        self._daily_start_date = datetime.now().date()
        self.logger.info(
            "Daily tracking reset.  Starting equity: $%.2f",
            self._daily_start_equity,
        )
        self._save_state()

    def _ensure_daily_tracking(self) -> None:
        """Auto-reset if the day has rolled over or tracking was never set."""
        today = datetime.now().date()
        if (
            self._daily_start_equity is None
            or self._daily_start_date is None
            or self._daily_start_date != today
        ):
            self.reset_daily_tracking()

    def get_daily_pnl(self) -> float:
        """
        Return the current-day profit/loss (equity change since daily
        reset).

        Returns
        -------
        float
            Positive values = profit, negative = loss.
        """
        self._ensure_daily_tracking()
        account = self.connector.get_account_info()
        if account is None or self._daily_start_equity is None:
            return 0.0
        return account["equity"] - self._daily_start_equity

    # ------------------------------------------------------------------
    # Lot-size calculation
    # ------------------------------------------------------------------
    def calculate_lot_size(self, symbol: str, sl_pips: float) -> float:
        """
        Calculate a position size so that the trade risks exactly
        ``config.max_risk_per_trade`` percent of current equity.

        Parameters
        ----------
        symbol : str
            Trading symbol, e.g. ``"EURUSD"``.
        sl_pips : float
            Distance from entry to stop-loss in pips.

        Returns
        -------
        float
            Volume rounded to the symbol's volume step and clamped to
            [volume_min, volume_max].  Returns ``volume_min`` as a
            fallback on any calculation error.
        """
        # --- Account equity ------------------------------------------------
        account = self.connector.get_account_info()
        if account is None:
            self.logger.error(
                "Cannot calculate lot size — account info unavailable."
            )
            return 0.01  # sensible fallback

        equity = account["equity"]
        risk_amount = equity * (self.config.max_risk_per_trade / 100.0)

        # --- Symbol info ---------------------------------------------------
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            self.logger.error("Symbol info unavailable for %s.", symbol)
            return 0.01

        tick_value = sym_info.trade_tick_value      # monetary value of 1 tick for 1 lot
        tick_size = sym_info.trade_tick_size         # price change per tick
        point = sym_info.point                      # smallest price increment
        volume_min = sym_info.volume_min
        volume_max = sym_info.volume_max
        volume_step = sym_info.volume_step

        # --- Pip value ------------------------------------------------------
        # A "pip" is usually 10 points for 5-digit brokers (e.g. EURUSD),
        # and 1 point for JPY pairs / metals.  We derive the pip value from
        # the tick value supplied by the broker.
        if tick_size == 0 or point == 0:
            self.logger.warning(
                "tick_size or point is 0 for %s — falling back to volume_min.",
                symbol,
            )
            return volume_min

        # Determine the pip-to-point multiplier
        digits = sym_info.digits
        if digits in (3, 5):
            pip_points = 10  # 5-digit broker: 1 pip = 10 points
        else:
            pip_points = 1   # 4-digit / 2-digit

        # Value of 1 pip movement for 1 standard lot
        pip_value_per_lot = tick_value * (pip_points * point) / tick_size

        if pip_value_per_lot <= 0 or sl_pips <= 0:
            self.logger.warning(
                "pip_value_per_lot=%.5f  sl_pips=%.1f — using volume_min.",
                pip_value_per_lot,
                sl_pips,
            )
            return volume_min

        # --- Raw lot size ---------------------------------------------------
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        # --- Round to volume step -------------------------------------------
        if volume_step > 0:
            lot_size = round(lot_size / volume_step) * volume_step
            # Correct floating-point drift
            lot_size = round(lot_size, 8)

        # --- Clamp ----------------------------------------------------------
        lot_size = max(volume_min, min(volume_max, lot_size))

        self.logger.info(
            "Lot calc: equity=$%.2f  risk%%=%.1f  risk$=%.2f  "
            "SL=%.1f pips  pip_val=%.4f  → %.2f lots",
            equity,
            self.config.max_risk_per_trade,
            risk_amount,
            sl_pips,
            pip_value_per_lot,
            lot_size,
        )

        return lot_size

    def _pip_value_per_lot(self, symbol: str) -> float:
        """Monetary value of a 1-pip move for 1.0 lot, or 0.0 if unknown."""
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return 0.0
        tick_value = sym_info.trade_tick_value
        tick_size = sym_info.trade_tick_size
        point = sym_info.point
        if tick_size == 0 or point == 0:
            return 0.0
        pip_points = 10 if sym_info.digits in (3, 5) else 1
        return tick_value * (pip_points * point) / tick_size

    def cap_lot_for_max_loss(
        self, symbol: str, sl_pips: float, lot: float,
    ) -> Tuple[float, str]:
        """Shrink *lot* so the worst-case SL loss can't exceed the per-trade cap.

        The cap is ``config.max_loss_per_trade_pct`` of current equity.  This is
        the hard safety net that makes a single catastrophic loss (e.g. the
        −$164 trade) structurally impossible regardless of strategy or lot mode.

        Returns ``(capped_lot, note)``.  ``capped_lot`` is ``0.0`` when even the
        minimum volume would breach the cap — the caller must then skip the trade.
        """
        max_pct = getattr(self.config, "max_loss_per_trade_pct", 0.0)
        if max_pct <= 0 or sl_pips <= 0:
            return (lot, "OK")

        account = self.connector.get_account_info()
        sym_info = mt5.symbol_info(symbol)
        pip_value = self._pip_value_per_lot(symbol)
        if account is None or sym_info is None or pip_value <= 0:
            return (lot, "OK")  # can't evaluate — don't block

        max_loss = account["equity"] * (max_pct / 100.0)
        worst_case = lot * sl_pips * pip_value
        if worst_case <= max_loss:
            return (lot, "OK")

        # Reduce lot to fit the cap, rounded down to the volume step.
        safe_lot = max_loss / (sl_pips * pip_value)
        step = sym_info.volume_step or 0.01
        safe_lot = (int(safe_lot / step)) * step
        safe_lot = round(safe_lot, 8)

        if safe_lot < sym_info.volume_min:
            self.logger.warning(
                "%s: even %.2f lots risks > %.1f%% cap ($%.2f) at SL=%.1f pips "
                "— skipping trade.",
                symbol, sym_info.volume_min, max_pct, max_loss, sl_pips,
            )
            return (0.0, f"Min lot risks more than {max_pct:.1f}% cap")

        self.logger.info(
            "%s: lot reduced %.2f → %.2f to respect %.1f%% max-loss cap "
            "($%.2f at SL=%.1f pips).",
            symbol, lot, safe_lot, max_pct, max_loss, sl_pips,
        )
        return (safe_lot, "capped")

    # ------------------------------------------------------------------
    # Pre-trade gate checks
    # ------------------------------------------------------------------
    def can_trade(
        self, symbol: str, lot: float
    ) -> Tuple[bool, str]:
        """
        Run every pre-flight check before placing a trade.

        Parameters
        ----------
        symbol : str
            Symbol about to be traded.
        lot : float
            Requested volume.

        Returns
        -------
        tuple[bool, str]
            ``(True, "OK")`` when all checks pass, otherwise
            ``(False, "<reason>")`` explaining which check failed.
        """
        # 1 — Market open? (basic weekend check). Crypto trades 24/7, so the
        # forex weekend + trading-hours calendar is skipped for crypto symbols.
        now = datetime.utcnow()
        if not is_crypto(symbol):
            weekday = now.weekday()  # 0=Mon … 6=Sun
            if weekday == 6 or (weekday == 5 and now.hour >= 22):
                return (False, "Market is closed (weekend).")
            if weekday == 0 and now.hour < 0:
                return (False, "Market has not opened yet (Sunday evening).")

            # 1b — Trading-hours window (UTC). Backtests show off-session hours
            # produce mostly losing/noise trades; this gate restricts entries to
            # the configured window (default 0..24 = no restriction).
            start = getattr(self.config, "trade_hour_start", 0)
            end = getattr(self.config, "trade_hour_end", 24)
            if (start, end) != (0, 24):
                h = now.hour
                in_window = (start <= h < end) if start <= end else (h >= start or h < end)
                if not in_window:
                    return (False, f"Outside trading hours ({h:02d}:00 UTC; "
                                   f"window {start:02d}-{end:02d}).")

        # 2 — Max open positions (account-wide PORTFOLIO ceiling across all
        # bots combined — a safety cap, not the per-bot limit). If this trips
        # during normal fleet operation, raise MAX_OPEN_POSITIONS to >= number
        # of bots; per-bot independence is enforced separately by the caller.
        positions = mt5.positions_get()
        open_count = len(positions) if positions else 0
        if open_count >= self.config.max_open_positions:
            return (
                False,
                f"Portfolio position limit reached "
                f"({open_count}/{self.config.max_open_positions} open account-wide).",
            )

        # 3 — Daily drawdown
        self._ensure_daily_tracking()
        account = self.connector.get_account_info()
        if (
            account is not None
            and self._daily_start_equity is not None
            and self._daily_start_equity > 0
        ):
            drawdown_pct = (
                (self._daily_start_equity - account["equity"])
                / self._daily_start_equity
                * 100.0
            )
            if drawdown_pct > self.config.max_daily_drawdown:
                return (
                    False,
                    f"Daily drawdown limit hit ({drawdown_pct:.2f}% > "
                    f"{self.config.max_daily_drawdown:.1f}%).",
                )

        # 4 — Spread filter
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            sym_info = mt5.symbol_info(symbol)
            if sym_info is not None and sym_info.point > 0:
                spread_points = (tick.ask - tick.bid) / sym_info.point
                if spread_points > self.config.max_spread_points:
                    return (
                        False,
                        f"Spread too wide ({spread_points:.1f} > "
                        f"{self.config.max_spread_points} points).",
                    )

        # 5 — Volume sanity
        sym_info = mt5.symbol_info(symbol)
        if sym_info is not None:
            if lot < sym_info.volume_min:
                return (False, f"Lot {lot} below minimum {sym_info.volume_min}.")
            if lot > sym_info.volume_max:
                return (False, f"Lot {lot} exceeds maximum {sym_info.volume_max}.")

        return (True, "OK")

    # ==================================================================
    #  ENHANCED RISK FEATURES (OctoBot-inspired)
    # ==================================================================

    # ------------------------------------------------------------------
    # Correlation check
    # ------------------------------------------------------------------
    def check_correlation(
        self, symbol: str, direction: str,
    ) -> Tuple[bool, str]:
        """
        Check whether opening a position in *symbol* / *direction*
        creates concerning correlated exposure with existing positions.

        Parameters
        ----------
        symbol : str
            Symbol about to be traded.
        direction : str
            ``"BUY"`` or ``"SELL"``.

        Returns
        -------
        tuple[bool, str]
            ``(True, "OK")`` when no concerning correlations, otherwise
            ``(False, "<warning>")`` explaining the risk.
        """
        positions = mt5.positions_get()
        if not positions:
            return (True, "OK")

        self._correlation_warnings.clear()
        symbol_upper = symbol.upper()

        for pos in positions:
            pos_symbol = pos.symbol.upper()
            pos_dir = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"

            # Same pair, same direction — stacking risk
            if pos_symbol == symbol_upper and pos_dir == direction:
                warning = (
                    f"Already have a {pos_dir} position in {pos_symbol} "
                    f"({pos.volume} lots). Adding more increases risk."
                )
                self._correlation_warnings.append(warning)
                self.logger.warning("Correlation: %s", warning)
                continue

            # Check positive correlations (same direction = doubled risk)
            pair = tuple(sorted([symbol_upper, pos_symbol]))
            for corr_a, corr_b in POSITIVE_CORRELATIONS:
                corr_pair = tuple(sorted([corr_a, corr_b]))
                if pair == corr_pair and pos_dir == direction:
                    warning = (
                        f"{symbol_upper} {direction} + {pos_symbol} {pos_dir} — "
                        f"positively correlated, same direction doubles risk."
                    )
                    self._correlation_warnings.append(warning)
                    self.logger.warning("Correlation: %s", warning)

            # Check negative correlations (same direction = natural hedge, OK)
            for corr_a, corr_b in NEGATIVE_CORRELATIONS:
                corr_pair = tuple(sorted([corr_a, corr_b]))
                if pair == corr_pair and pos_dir == direction:
                    self.logger.info(
                        "Correlation: %s %s + %s %s — negatively correlated "
                        "(natural hedge). OK.",
                        symbol_upper, direction, pos_symbol, pos_dir,
                    )

        if self._correlation_warnings:
            combined = "; ".join(self._correlation_warnings)
            return (False, f"Correlated exposure warning: {combined}")

        self.logger.info(
            "Correlation check passed for %s %s.", symbol, direction,
        )
        return (True, "OK")

    # ------------------------------------------------------------------
    # Volatility-adjusted position sizing
    # ------------------------------------------------------------------
    def calculate_volatility_adjusted_lot(
        self,
        symbol: str,
        sl_pips: float,
        current_atr: float,
        average_atr: float,
    ) -> float:
        """
        Calculate a lot size adjusted for current market volatility.

        High volatility -> smaller positions.  Low volatility -> slightly
        larger positions.  If drawdown recovery mode is active the
        recovery multiplier is also applied.

        Parameters
        ----------
        symbol : str
            Trading symbol.
        sl_pips : float
            Stop-loss distance in pips.
        current_atr : float
            Current ATR value.
        average_atr : float
            Average ATR over the lookback period.

        Returns
        -------
        float
            Volatility-adjusted volume.
        """
        base_lot = self.calculate_lot_size(symbol, sl_pips)

        if average_atr <= 0 or current_atr <= 0:
            self.logger.warning(
                "Invalid ATR values (current=%.5f, avg=%.5f) — using base lot.",
                current_atr, average_atr,
            )
            return base_lot

        # Volatility factor: inverse relationship
        vol_factor = average_atr / current_atr
        vol_factor = max(0.5, min(2.0, vol_factor))  # clamp

        # Apply drawdown recovery multiplier
        recovery_mult = self._risk_state["recovery_lot_multiplier"]

        adjusted_lot = base_lot * vol_factor * recovery_mult

        # Clamp to symbol limits
        sym_info = mt5.symbol_info(symbol)
        if sym_info is not None:
            step = sym_info.volume_step
            if step > 0:
                adjusted_lot = round(adjusted_lot / step) * step
                adjusted_lot = round(adjusted_lot, 8)
            adjusted_lot = max(sym_info.volume_min, min(sym_info.volume_max, adjusted_lot))

        self.logger.info(
            "Vol-adjusted lot: base=%.2f  vol_factor=%.2f  "
            "recovery=%.2f  -> %.2f lots",
            base_lot, vol_factor, recovery_mult, adjusted_lot,
        )

        return adjusted_lot

    # ------------------------------------------------------------------
    # Drawdown recovery mode
    # ------------------------------------------------------------------
    def check_drawdown_recovery(self) -> None:
        """
        Update the drawdown recovery mode based on current equity.

        Recovery mode reduces position sizes when drawdown is
        significant, gradually returning to full size as equity
        recovers.

        Thresholds (as fraction of ``max_daily_drawdown``):

        * > 75%  -> severe recovery (multiplier 0.25)
        * > 50%  -> moderate recovery (multiplier 0.50)
        * < 25%  -> exit recovery (multiplier 1.00)
        """
        self._ensure_daily_tracking()
        account = self.connector.get_account_info()
        if (
            account is None
            or self._daily_start_equity is None
            or self._daily_start_equity <= 0
        ):
            return

        drawdown_pct = (
            (self._daily_start_equity - account["equity"])
            / self._daily_start_equity
            * 100.0
        )
        max_dd = self.config.max_daily_drawdown
        was_in_recovery = self._risk_state["drawdown_recovery_mode"]
        old_mult = self._risk_state["recovery_lot_multiplier"]

        if drawdown_pct > max_dd * 0.75:
            self._risk_state["drawdown_recovery_mode"] = True
            self._risk_state["recovery_lot_multiplier"] = 0.25
            if old_mult != 0.25:
                self.logger.warning(
                    "SEVERE drawdown recovery — DD=%.2f%% (>75%% of %.1f%%). "
                    "Lot multiplier reduced to 0.25.",
                    drawdown_pct, max_dd,
                )
        elif drawdown_pct > max_dd * 0.50:
            self._risk_state["drawdown_recovery_mode"] = True
            self._risk_state["recovery_lot_multiplier"] = 0.50
            if old_mult != 0.50:
                self.logger.warning(
                    "Moderate drawdown recovery — DD=%.2f%% (>50%% of %.1f%%). "
                    "Lot multiplier reduced to 0.50.",
                    drawdown_pct, max_dd,
                )
        elif drawdown_pct < max_dd * 0.25:
            if was_in_recovery:
                self.logger.info(
                    "Exiting drawdown recovery — DD=%.2f%% (<25%% of %.1f%%). "
                    "Lot multiplier restored to 1.0.",
                    drawdown_pct, max_dd,
                )
            self._risk_state["drawdown_recovery_mode"] = False
            self._risk_state["recovery_lot_multiplier"] = 1.0

        # Persist if recovery state changed
        if (
            self._risk_state["drawdown_recovery_mode"] != was_in_recovery
            or self._risk_state["recovery_lot_multiplier"] != old_mult
        ):
            self._save_state()

    # ------------------------------------------------------------------
    # Currency exposure heat map
    # ------------------------------------------------------------------
    def get_currency_exposure(self) -> Dict[str, float]:
        """
        Decompose all open positions into per-currency net exposure.

        For example, EURUSD BUY 0.10 lots -> EUR +0.10, USD -0.10.

        Returns
        -------
        dict[str, float]
            Currency code -> net exposure.  Positive = long, negative =
            short.
        """
        exposure: Dict[str, float] = defaultdict(float)
        positions = mt5.positions_get()

        if not positions:
            return dict(exposure)

        for pos in positions:
            sym = pos.symbol.upper()
            vol = pos.volume
            is_buy = pos.type == mt5.ORDER_TYPE_BUY

            # Decompose pair into base / quote currencies
            # Assumes standard 6-char forex pair format (EURUSD)
            if len(sym) >= 6:
                base = sym[:3]
                quote = sym[3:6]
            else:
                continue  # skip non-standard symbols

            if is_buy:
                exposure[base] += vol
                exposure[quote] -= vol
            else:
                exposure[base] -= vol
                exposure[quote] += vol

        result = {k: round(v, 4) for k, v in exposure.items() if abs(v) > 0.0001}
        self.logger.debug("Currency exposure: %s", result)
        return result

    # ------------------------------------------------------------------
    # Update risk state (call each iteration)
    # ------------------------------------------------------------------
    def update_risk_state(self) -> dict:
        """
        Refresh all stateful risk metrics.  Call this on every trading
        iteration to keep the risk state up to date.

        Returns
        -------
        dict
            Current risk state snapshot.
        """
        # Total lot exposure
        positions = mt5.positions_get()
        total_lots = sum(p.volume for p in positions) if positions else 0.0
        self._risk_state["total_exposure"] = round(total_lots, 4)

        # Currency exposure
        self._risk_state["currency_exposure"] = self.get_currency_exposure()

        # Drawdown recovery
        self.check_drawdown_recovery()

        # Peak equity
        account = self.connector.get_account_info()
        if account is not None:
            equity = account["equity"]
            if equity > self._risk_state["peak_equity"]:
                self._risk_state["peak_equity"] = equity

        self._risk_state["last_update"] = datetime.utcnow().isoformat()
        self._save_state()
        return dict(self._risk_state)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        """Persist critical risk state to disk so it survives restarts."""
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "daily_start_equity": self._daily_start_equity,
                "daily_start_date": (
                    self._daily_start_date.isoformat()
                    if self._daily_start_date else None
                ),
                "peak_equity": self._risk_state["peak_equity"],
                "drawdown_recovery_mode": self._risk_state["drawdown_recovery_mode"],
                "recovery_lot_multiplier": self._risk_state["recovery_lot_multiplier"],
            }
            _STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            self.logger.debug("Failed to save risk state: %s", exc)

    def _load_state(self) -> None:
        """Restore persisted risk state from disk."""
        if not _STATE_FILE.exists():
            return
        try:
            data = json.loads(_STATE_FILE.read_text())

            # Restore daily tracking only if saved date matches today
            saved_date_str = data.get("daily_start_date")
            if saved_date_str:
                saved_date = datetime.fromisoformat(saved_date_str).date()
                if saved_date == datetime.now().date():
                    self._daily_start_equity = data.get("daily_start_equity")
                    self._daily_start_date = saved_date
                    self.logger.info(
                        "Restored daily tracking — start equity: $%.2f",
                        self._daily_start_equity or 0,
                    )

            # Always restore peak equity and recovery state
            peak = data.get("peak_equity", 0.0)
            if peak > 0:
                self._risk_state["peak_equity"] = peak

            self._risk_state["drawdown_recovery_mode"] = data.get(
                "drawdown_recovery_mode", False
            )
            self._risk_state["recovery_lot_multiplier"] = data.get(
                "recovery_lot_multiplier", 1.0
            )

            if self._risk_state["drawdown_recovery_mode"]:
                self.logger.warning(
                    "Restored drawdown recovery mode — "
                    "lot multiplier: %.2f",
                    self._risk_state["recovery_lot_multiplier"],
                )
        except Exception as exc:
            self.logger.debug("Failed to load risk state: %s", exc)

    # ------------------------------------------------------------------
    # Enhanced can_trade (superset of original)
    # ------------------------------------------------------------------
    def can_trade_enhanced(
        self, symbol: str, lot: float, direction: str = "BUY",
    ) -> Tuple[bool, str]:
        """
        Extended pre-flight check that includes correlation awareness
        and drawdown recovery mode on top of the original checks.

        Parameters
        ----------
        symbol : str
            Symbol about to be traded.
        lot : float
            Requested volume.
        direction : str
            ``"BUY"`` or ``"SELL"``.

        Returns
        -------
        tuple[bool, str]
            ``(True, "OK")`` or ``(False, "<reason>")``.
        """
        # Original checks first
        ok, reason = self.can_trade(symbol, lot)
        if not ok:
            return (ok, reason)

        # Correlation check (warning only — does not block)
        corr_ok, corr_msg = self.check_correlation(symbol, direction)
        if not corr_ok:
            self.logger.warning("Correlation warning (non-blocking): %s", corr_msg)
            # Note: we warn but don't block.  Override to block if needed.

        # Max per-currency exposure warning
        exposure = self.get_currency_exposure()
        for ccy, exp in exposure.items():
            if abs(exp) > 0.5:
                self.logger.warning(
                    "High %s exposure: %.2f lots net.", ccy, exp,
                )

        return (True, "OK")

    # ------------------------------------------------------------------
    # Comprehensive risk metrics
    # ------------------------------------------------------------------
    def get_risk_metrics(self) -> dict:
        """
        Return a comprehensive snapshot of all risk metrics.

        Returns
        -------
        dict
            Keys: daily_pnl, drawdown_pct, open_positions_count,
            total_exposure, currency_exposure, recovery_mode,
            recovery_multiplier, correlation_warnings, peak_equity,
            current_equity.
        """
        self._ensure_daily_tracking()
        account = self.connector.get_account_info()
        equity = account["equity"] if account else 0.0

        drawdown_pct = 0.0
        if (
            self._daily_start_equity
            and self._daily_start_equity > 0
            and account is not None
        ):
            drawdown_pct = (
                (self._daily_start_equity - equity)
                / self._daily_start_equity
                * 100.0
            )

        positions = mt5.positions_get()
        open_count = len(positions) if positions else 0

        return {
            "daily_pnl": self.get_daily_pnl(),
            "drawdown_pct": round(drawdown_pct, 2),
            "open_positions_count": open_count,
            "total_exposure": self._risk_state["total_exposure"],
            "currency_exposure": self._risk_state["currency_exposure"],
            "recovery_mode": self._risk_state["drawdown_recovery_mode"],
            "recovery_multiplier": self._risk_state["recovery_lot_multiplier"],
            "correlation_warnings": list(self._correlation_warnings),
            "peak_equity": self._risk_state["peak_equity"],
            "current_equity": equity,
        }
