"""
order_manager.py — Trade Execution Layer for the MT5 Forex Bot.

Handles all order operations: opening, closing, modifying positions,
and querying open trades.  Supports a dry-run mode that logs simulated
orders without touching the live account.
"""

from __future__ import annotations

import logging
import uuid
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

import MetaTrader5 as mt5

from config import Config


# ---------------------------------------------------------------------------
# Human-readable retcode map
# ---------------------------------------------------------------------------
_RETCODE_MESSAGES: Dict[int, str] = {
    10004: "Requote — prices have changed, retry.",
    10006: "Request rejected by server.",
    10007: "Request cancelled by trader.",
    10008: "Order placed (pending).",
    10009: "Order executed successfully.",
    10010: "Order executed partially.",
    10011: "Request processing error.",
    10012: "Request timed out.",
    10013: "Invalid request structure.",
    10014: "Invalid volume.",
    10015: "Invalid price.",
    10016: "Invalid stop-loss / take-profit.",
    10017: "Trading disabled for this account.",
    10018: "Market is closed.",
    10019: "Insufficient margin.",
    10020: "Prices changed (requote).",
    10021: "No quotes available.",
    10022: "Invalid order expiration.",
    10023: "Order state changed.",
    10024: "Too many requests — throttled.",
    10025: "No changes in request (modify ignored).",
    10026: "Autotrading disabled by server.",
    10027: "Autotrading disabled by client terminal.",
    10028: "Request locked for processing.",
    10029: "Order/position frozen.",
    10030: "Invalid order filling type.",
    10031: "No connection to trade server.",
    10032: "Operation allowed only for live accounts.",
    10033: "Pending orders limit reached.",
    10034: "Volume limit for symbol/position reached.",
    10035: "Incorrect or prohibited order type.",
    10036: "Position with the specified ticket not found.",
    10038: "Close order already exists for this position.",
    10039: "Close volume exceeds current position volume.",
    10040: "Position already has a close order.",
    10041: "Max positions reached.",
    10042: "Pending order activation rejected (conflict).",
    10043: "Only long positions allowed.",
    10044: "Only short positions allowed.",
    10045: "Only position close allowed.",
    10046: "Close by FIFO rule only.",
}


# ---------------------------------------------------------------------------
# Order enums
# ---------------------------------------------------------------------------
class OrderState(Enum):
    """Lifecycle states for a tracked order."""

    PENDING_CREATION = "pending_creation"
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    FAILED = "failed"


class OrderType(Enum):
    """RI-specific order types that map to MT5 constants."""

    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    STOP_BUY = "stop_buy"
    STOP_SELL = "stop_sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


# ---------------------------------------------------------------------------
# Tracked order / group dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrackedOrder:
    """In-memory representation of an order with full lifecycle history.

    Attributes
    ----------
    order_id : str
        Unique internal identifier (UUID).
    ticket : int
        MT5 ticket number.  ``0`` if the order has not yet been placed.
    symbol : str
        Trading symbol.
    order_type : OrderType
        RI-specific order type.
    state : OrderState
        Current lifecycle state.
    lot : float
        Requested volume.
    price : float
        Requested price.
    sl : float
        Stop-loss price.
    tp : float
        Take-profit price.
    comment : str
        Free-text comment.
    group_id : str | None
        OCO / bracket group this order belongs to.
    created_at : datetime
        Timestamp when the tracked order was created.
    updated_at : datetime
        Timestamp of the last state update.
    filled_at : datetime | None
        Timestamp when the order was filled (if applicable).
    fill_price : float
        Actual fill price (``0.0`` until filled).
    history : list[tuple[datetime, OrderState, str]]
        Chronological log of state transitions.
    """

    order_id: str
    ticket: int
    symbol: str
    order_type: OrderType
    state: OrderState
    lot: float
    price: float
    sl: float
    tp: float
    comment: str
    group_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    fill_price: float = 0.0
    history: list = field(default_factory=list)


@dataclass
class OrderGroup:
    """Logical grouping of orders (e.g. OCO, bracket).

    Attributes
    ----------
    group_id : str
        Unique identifier (UUID).
    name : str
        Human-readable name for the group.
    orders : list[str]
        ``order_id`` values belonging to this group.
    cancel_on_fill : bool
        When *True*, filling any order in the group cancels the rest
        (OCO behaviour).
    created_at : datetime
        Timestamp when the group was created.
    """

    group_id: str
    name: str
    orders: list = field(default_factory=list)
    cancel_on_fill: bool = True
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# OrderAdapter — price / volume normalisation & pre-flight validation
# ---------------------------------------------------------------------------
class OrderAdapter:
    """Utility class that normalises prices and volumes to MT5 symbol specs.

    All methods are static so the adapter is stateless and can be used from
    anywhere without instantiation.
    """

    _logger = logging.getLogger("OrderAdapter")

    @staticmethod
    def adapt_price(price: float, symbol: str) -> float:
        """Round *price* to the symbol's digit precision.

        Parameters
        ----------
        price : float
            Raw price value.
        symbol : str
            MT5 symbol name.

        Returns
        -------
        float
            Price rounded to the correct number of decimal places.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            OrderAdapter._logger.warning(
                "symbol_info(%s) returned None — returning raw price", symbol
            )
            return price
        return round(price, info.digits)

    @staticmethod
    def adapt_volume(volume: float, symbol: str) -> float:
        """Round *volume* to the symbol's volume step and clamp to min/max.

        Parameters
        ----------
        volume : float
            Raw volume value.
        symbol : str
            MT5 symbol name.

        Returns
        -------
        float
            Volume snapped to the nearest valid step, clamped within the
            broker's allowed range.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            OrderAdapter._logger.warning(
                "symbol_info(%s) returned None — returning raw volume", symbol
            )
            return volume

        step = info.volume_step
        if step > 0:
            volume = round(round(volume / step) * step, 10)

        volume = max(info.volume_min, min(volume, info.volume_max))
        return volume

    @staticmethod
    def validate_order(
        symbol: str,
        order_type: int,
        lot: float,
        price: float,
        sl: float,
        tp: float,
    ) -> Tuple[bool, str]:
        """Run comprehensive pre-flight checks against MT5 symbol limits.

        Parameters
        ----------
        symbol : str
            MT5 symbol name.
        order_type : int
            MT5 order type constant.
        lot : float
            Requested trade volume.
        price : float
            Requested entry price.
        sl : float
            Stop-loss price.
        tp : float
            Take-profit price.

        Returns
        -------
        tuple[bool, str]
            ``(True, "OK")`` when all checks pass, or
            ``(False, "<reason>")`` on failure.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            return False, f"Symbol '{symbol}' not found or not available."

        # -- tradability ---------------------------------------------------
        if not info.visible:
            return False, f"Symbol '{symbol}' is not visible in Market Watch."
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False, f"Trading is disabled for '{symbol}'."

        # -- volume --------------------------------------------------------
        if lot < info.volume_min:
            return False, (
                f"Volume {lot} below minimum {info.volume_min} for '{symbol}'."
            )
        if lot > info.volume_max:
            return False, (
                f"Volume {lot} exceeds maximum {info.volume_max} for '{symbol}'."
            )
        step = info.volume_step
        if step > 0:
            remainder = round(lot % step, 10)
            if remainder > 1e-10 and abs(remainder - step) > 1e-10:
                return False, (
                    f"Volume {lot} not a multiple of step "
                    f"{step} for '{symbol}'."
                )

        # -- price / stops -------------------------------------------------
        stops_level = info.trade_stops_level  # in points
        point = info.point
        min_distance = stops_level * point

        if price <= 0:
            return False, "Price must be positive."

        if sl != 0.0:
            if abs(price - sl) < min_distance:
                return False, (
                    f"SL too close to price.  Minimum distance is "
                    f"{min_distance:.{info.digits}f} ({stops_level} pts)."
                )

        if tp != 0.0:
            if abs(price - tp) < min_distance:
                return False, (
                    f"TP too close to price.  Minimum distance is "
                    f"{min_distance:.{info.digits}f} ({stops_level} pts)."
                )

        return True, "OK"


class OrderManager:
    """Manages all order operations against MetaTrader 5."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config: Config, dry_run: bool = False) -> None:
        """
        Parameters
        ----------
        config : Config
            Application configuration instance.
        dry_run : bool
            When *True* orders are only logged — nothing is sent to the
            broker.
        """
        self.config = config
        self.dry_run = dry_run
        self.logger = logging.getLogger("OrderManager")

        # -- Tracking / OCO state (thread-safe) ---------------------------
        self._tracked_orders: Dict[str, TrackedOrder] = {}
        self._order_groups: Dict[str, OrderGroup] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_retcode_message(retcode: int) -> str:
        """Return a human-readable description for a MT5 trade retcode."""
        return _RETCODE_MESSAGES.get(retcode, f"Unknown retcode ({retcode})")

    @staticmethod
    def _get_filling_type(symbol: str) -> int:
        """Auto-detect the correct order filling type for a symbol.

        Brokers advertise which filling modes they support via the
        ``filling_mode`` bitmask in symbol info.  We check in order
        of preference: FOK → IOC → RETURN.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_FOK  # safest default

        filling = info.filling_mode

        # Bit 1 = FOK, Bit 2 = IOC, Bit 4 = RETURN (broker-dependent)
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------
    def open_position(
        self,
        symbol: str,
        order_type: int,
        lot: float,
        sl_price: float,
        tp_price: float,
        comment: str = "",
    ) -> dict:
        """
        Open a new market position.

        Parameters
        ----------
        symbol : str
            Trading symbol, e.g. ``"EURUSD"``.
        order_type : int
            ``mt5.ORDER_TYPE_BUY`` or ``mt5.ORDER_TYPE_SELL``.
        lot : float
            Trade volume.
        sl_price : float
            Stop-loss price level.
        tp_price : float
            Take-profit price level.
        comment : str, optional
            Free-text comment attached to the order.

        Returns
        -------
        dict
            Result dictionary with at least ``retcode`` and ``comment`` keys.
        """
        direction = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"

        # --- Dry-run mode --------------------------------------------------
        if self.dry_run:
            self.logger.info(
                "[DRY RUN] %s %s %.2f lots | SL=%.5f  TP=%.5f  comment='%s'",
                direction,
                symbol,
                lot,
                sl_price,
                tp_price,
                comment,
            )
            return {
                "retcode": 10009,
                "deal": 0,
                "order": 0,
                "volume": lot,
                "price": 0.0,
                "comment": "Dry-run simulated order",
                "request_id": 0,
            }

        # --- Live order ----------------------------------------------------
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            msg = f"Failed to get tick for {symbol}: {mt5.last_error()}"
            self.logger.error(msg)
            return {"retcode": -1, "comment": msg}

        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        request: dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(symbol),
        }

        self.logger.info(
            "Sending %s order: %s %.2f lots @ %.5f  SL=%.5f  TP=%.5f",
            direction,
            symbol,
            lot,
            price,
            sl_price,
            tp_price,
        )

        result = mt5.order_send(request)
        if result is None:
            msg = f"order_send returned None: {mt5.last_error()}"
            self.logger.error(msg)
            return {"retcode": -1, "comment": msg}

        result_dict = result._asdict()

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(
                "✔ Order executed — ticket #%s, deal #%s, vol=%.2f @ %.5f",
                result.order,
                result.deal,
                result.volume,
                result.price,
            )
        else:
            self.logger.warning(
                "✘ Order FAILED — retcode=%s (%s)  server='%s'",
                result.retcode,
                self.get_retcode_message(result.retcode),
                result.comment,
            )

        return result_dict

    # ------------------------------------------------------------------
    # Close single position
    # ------------------------------------------------------------------
    def close_position(self, ticket: int) -> bool:
        """
        Close an open position by its ticket number.

        Parameters
        ----------
        ticket : int
            Position ticket to close.

        Returns
        -------
        bool
            *True* if the position was closed successfully.
        """
        if self.dry_run:
            self.logger.info("[DRY RUN] Close position ticket #%s", ticket)
            return True

        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            self.logger.warning("Position #%s not found.", ticket)
            return False

        pos = positions[0]
        close_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            self.logger.error(
                "Cannot get tick for %s while closing #%s", pos.symbol, ticket
            )
            return False

        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request: dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(pos.symbol),
        }

        self.logger.info(
            "Closing position #%s (%s %.2f lots) @ %.5f",
            ticket,
            pos.symbol,
            pos.volume,
            price,
        )

        result = mt5.order_send(request)
        if result is None:
            self.logger.error(
                "Close order_send returned None: %s", mt5.last_error()
            )
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info("✔ Position #%s closed successfully.", ticket)
            return True

        self.logger.warning(
            "✘ Failed to close #%s — retcode=%s (%s)",
            ticket,
            result.retcode,
            self.get_retcode_message(result.retcode),
        )
        return False

    # ------------------------------------------------------------------
    # Close all positions
    # ------------------------------------------------------------------
    def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """
        Close every open position (optionally filtered by *symbol*).

        Returns
        -------
        int
            Number of positions that were successfully closed.
        """
        if self.dry_run:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            count = len(positions) if positions else 0
            self.logger.info(
                "[DRY RUN] Would close %d position(s)%s.",
                count,
                f" for {symbol}" if symbol else "",
            )
            return count

        positions = (
            mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        )
        if positions is None or len(positions) == 0:
            self.logger.info("No open positions to close.")
            return 0

        closed = 0
        for pos in positions:
            if self.close_position(pos.ticket):
                closed += 1

        self.logger.info(
            "Closed %d / %d positions%s.",
            closed,
            len(positions),
            f" for {symbol}" if symbol else "",
        )
        return closed

    # ------------------------------------------------------------------
    # Modify position SL / TP
    # ------------------------------------------------------------------
    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """
        Modify the stop-loss and/or take-profit of an open position.

        Parameters
        ----------
        ticket : int
            Position ticket to modify.
        sl : float or None
            New stop-loss price.  *None* keeps the current value.
        tp : float or None
            New take-profit price.  *None* keeps the current value.

        Returns
        -------
        bool
            *True* if the modification succeeded.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY RUN] Modify #%s — SL=%s  TP=%s", ticket, sl, tp
            )
            return True

        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            self.logger.warning("Position #%s not found for modification.", ticket)
            return False

        pos = positions[0]
        new_sl = float(sl) if sl is not None else pos.sl
        new_tp = float(tp) if tp is not None else pos.tp

        request: dict = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
        }

        self.logger.info(
            "Modifying #%s — SL: %.5f → %.5f | TP: %.5f → %.5f",
            ticket,
            pos.sl,
            new_sl,
            pos.tp,
            new_tp,
        )

        result = mt5.order_send(request)
        if result is None:
            self.logger.error(
                "Modify order_send returned None: %s", mt5.last_error()
            )
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info("✔ Position #%s modified.", ticket)
            return True

        self.logger.warning(
            "✘ Modify failed for #%s — retcode=%s (%s)",
            ticket,
            result.retcode,
            self.get_retcode_message(result.retcode),
        )
        return False

    # ------------------------------------------------------------------
    # Partial close
    # ------------------------------------------------------------------
    def partial_close_position(
        self, ticket: int, close_pct: float
    ) -> bool:
        """Close a fraction of an open position.

        Parameters
        ----------
        ticket : int
            Position ticket to partially close.
        close_pct : float
            Percentage of the position to close (1-99).

        Returns
        -------
        bool
            *True* if the partial close succeeded.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY RUN] Partial close %d%% of position #%s", close_pct, ticket
            )
            return True

        if not (1 <= close_pct <= 99):
            self.logger.warning(
                "Invalid close_pct=%s — must be 1-99.", close_pct
            )
            return False

        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            self.logger.warning("Position #%s not found for partial close.", ticket)
            return False

        pos = positions[0]
        close_volume = OrderAdapter.adapt_volume(
            pos.volume * (close_pct / 100.0), pos.symbol
        )

        if close_volume <= 0 or close_volume >= pos.volume:
            self.logger.warning(
                "Partial close volume %.2f invalid for position #%s (vol=%.2f).",
                close_volume, ticket, pos.volume,
            )
            return False

        close_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            self.logger.error(
                "Cannot get tick for %s while partial-closing #%s",
                pos.symbol, ticket,
            )
            return False

        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request: dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": f"partial_close_{close_pct}%",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(pos.symbol),
        }

        self.logger.info(
            "Partial closing %d%% of #%s — %.2f / %.2f lots @ %.5f",
            close_pct, ticket, close_volume, pos.volume, price,
        )

        result = mt5.order_send(request)
        if result is None:
            self.logger.error(
                "Partial close order_send returned None: %s", mt5.last_error()
            )
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(
                "✔ Partial close of #%s succeeded — closed %.2f lots.",
                ticket, close_volume,
            )
            return True

        self.logger.warning(
            "✘ Partial close failed for #%s — retcode=%s (%s)",
            ticket, result.retcode,
            self.get_retcode_message(result.retcode),
        )
        return False

    # ------------------------------------------------------------------
    # Set breakeven
    # ------------------------------------------------------------------
    def set_breakeven(
        self, ticket: int, offset_pips: float = 0.0
    ) -> bool:
        """Move the stop-loss of a position to its entry price.

        Optionally add a small offset in pips for spread protection.

        Parameters
        ----------
        ticket : int
            Position ticket.
        offset_pips : float, optional
            Extra pips above/below entry to account for spread.

        Returns
        -------
        bool
            *True* if the breakeven modification succeeded.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY RUN] Set breakeven for #%s (offset=%.1f pips)",
                ticket, offset_pips,
            )
            return True

        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            self.logger.warning("Position #%s not found for breakeven.", ticket)
            return False

        pos = positions[0]
        sym_info = mt5.symbol_info(pos.symbol)
        if sym_info is None:
            self.logger.error(
                "Cannot get symbol info for %s.", pos.symbol
            )
            return False

        point = sym_info.point
        # Calculate offset in price
        # For JPY pairs, 1 pip = 100 points; for others, 1 pip = 10 points
        pip_value = point * 10 if sym_info.digits in (3, 5) else point
        offset_price = offset_pips * pip_value

        if pos.type == mt5.ORDER_TYPE_BUY:
            be_sl = round(pos.price_open + offset_price, sym_info.digits)
        else:
            be_sl = round(pos.price_open - offset_price, sym_info.digits)

        # Check position is in profit (or at least not going to close on us)
        current_price = pos.price_current
        if pos.type == mt5.ORDER_TYPE_BUY and current_price <= be_sl:
            self.logger.warning(
                "Cannot set breakeven for BUY #%s — current price %.5f "
                "<= proposed SL %.5f.",
                ticket, current_price, be_sl,
            )
            return False
        if pos.type == mt5.ORDER_TYPE_SELL and current_price >= be_sl:
            self.logger.warning(
                "Cannot set breakeven for SELL #%s — current price %.5f "
                ">= proposed SL %.5f.",
                ticket, current_price, be_sl,
            )
            return False

        return self.modify_position(ticket, sl=be_sl)

    # ------------------------------------------------------------------
    # Modify pending order
    # ------------------------------------------------------------------
    def modify_pending_order(
        self,
        ticket: int,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        expiration: Optional[datetime] = None,
    ) -> bool:
        """Modify a pending order's price, SL, TP, or expiration.

        Parameters
        ----------
        ticket : int
            Pending order ticket to modify.
        price : float or None
            New order price.  *None* keeps the current value.
        sl : float or None
            New stop-loss.  *None* keeps the current value.
        tp : float or None
            New take-profit.  *None* keeps the current value.
        expiration : datetime or None
            New expiration time.  *None* keeps the current value.

        Returns
        -------
        bool
            *True* if the modification succeeded.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY RUN] Modify pending #%s — price=%s SL=%s TP=%s exp=%s",
                ticket, price, sl, tp, expiration,
            )
            return True

        orders = mt5.orders_get(ticket=ticket)
        if orders is None or len(orders) == 0:
            self.logger.warning(
                "Pending order #%s not found for modification.", ticket
            )
            return False

        order = orders[0]
        new_price = float(price) if price is not None else order.price_open
        new_sl = float(sl) if sl is not None else order.sl
        new_tp = float(tp) if tp is not None else order.tp

        # Normalise via adapter
        new_price = OrderAdapter.adapt_price(new_price, order.symbol)
        if new_sl:
            new_sl = OrderAdapter.adapt_price(new_sl, order.symbol)
        if new_tp:
            new_tp = OrderAdapter.adapt_price(new_tp, order.symbol)

        request: dict = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": ticket,
            "price": new_price,
            "sl": new_sl,
            "tp": new_tp,
        }

        if expiration is not None:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = int(expiration.timestamp())
        else:
            # Keep existing expiration
            if order.time_expiration > 0:
                request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                request["expiration"] = order.time_expiration
            else:
                request["type_time"] = mt5.ORDER_TIME_GTC

        self.logger.info(
            "Modifying pending #%s — price: %.5f → %.5f | "
            "SL: %.5f → %.5f | TP: %.5f → %.5f",
            ticket,
            order.price_open, new_price,
            order.sl, new_sl,
            order.tp, new_tp,
        )

        result = mt5.order_send(request)
        if result is None:
            self.logger.error(
                "Modify pending order_send returned None: %s",
                mt5.last_error(),
            )
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info("✔ Pending order #%s modified.", ticket)
            return True

        self.logger.warning(
            "✘ Modify pending failed for #%s — retcode=%s (%s)",
            ticket, result.retcode,
            self.get_retcode_message(result.retcode),
        )
        return False

    # ------------------------------------------------------------------
    # Query open positions
    # ------------------------------------------------------------------
    def get_open_positions(
        self, symbol: Optional[str] = None
    ) -> List[dict]:
        """
        Return a list of dicts describing every open position.

        Each dict contains: ticket, symbol, type (str), volume, price_open,
        sl, tp, profit, swap, time, magic, comment.
        """
        positions = (
            mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        )

        if positions is None or len(positions) == 0:
            return []

        result: List[dict] = []
        for pos in positions:
            result.append(
                {
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "time": datetime.fromtimestamp(pos.time).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "magic": pos.magic,
                    "comment": pos.comment,
                }
            )

        return result

    # ==================================================================
    #  PENDING ORDERS
    # ==================================================================

    def place_pending_order(
        self,
        symbol: str,
        order_type: int,
        lot: float,
        price: float,
        sl_price: float = 0.0,
        tp_price: float = 0.0,
        comment: str = "",
        expiration: Optional[datetime] = None,
    ) -> dict:
        """Place a pending (limit / stop) order.

        Parameters
        ----------
        symbol : str
            Trading symbol, e.g. ``"EURUSD"``.
        order_type : int
            One of ``mt5.ORDER_TYPE_BUY_LIMIT``,
            ``mt5.ORDER_TYPE_SELL_LIMIT``,
            ``mt5.ORDER_TYPE_BUY_STOP``,
            ``mt5.ORDER_TYPE_SELL_STOP``.
        lot : float
            Trade volume.
        price : float
            Desired entry price.
        sl_price : float, optional
            Stop-loss price level.
        tp_price : float, optional
            Take-profit price level.
        comment : str, optional
            Free-text comment.
        expiration : datetime or None, optional
            Order expiry time.  When *None* the order is GTC.

        Returns
        -------
        dict
            Result dictionary with at least ``retcode`` and ``comment`` keys.
        """
        _type_names = {
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP",
        }
        label = _type_names.get(order_type, f"UNKNOWN({order_type})")

        # Normalise price & volume via adapter
        price = OrderAdapter.adapt_price(price, symbol)
        lot = OrderAdapter.adapt_volume(lot, symbol)
        if sl_price:
            sl_price = OrderAdapter.adapt_price(sl_price, symbol)
        if tp_price:
            tp_price = OrderAdapter.adapt_price(tp_price, symbol)

        # --- Dry-run mode -------------------------------------------------
        if self.dry_run:
            self.logger.info(
                "[DRY RUN] Pending %s %s %.2f lots @ %.5f | "
                "SL=%.5f  TP=%.5f  exp=%s  comment='%s'",
                label, symbol, lot, price, sl_price, tp_price,
                expiration, comment,
            )
            return {
                "retcode": 10008,
                "order": 0,
                "volume": lot,
                "price": price,
                "comment": "Dry-run simulated pending order",
                "request_id": 0,
            }

        # --- Validate -----------------------------------------------------
        ok, reason = OrderAdapter.validate_order(
            symbol, order_type, lot, price, sl_price, tp_price
        )
        if not ok:
            self.logger.warning(
                "Pending order validation failed: %s", reason
            )
            return {"retcode": -1, "comment": reason}

        # --- Build request ------------------------------------------------
        request: dict = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_filling": self._get_filling_type(symbol),
        }

        if expiration is not None:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = int(expiration.timestamp())
        else:
            request["type_time"] = mt5.ORDER_TIME_GTC

        self.logger.info(
            "Sending pending %s: %s %.2f lots @ %.5f  SL=%.5f  TP=%.5f",
            label, symbol, lot, price, sl_price, tp_price,
        )

        result = mt5.order_send(request)
        if result is None:
            msg = f"order_send (pending) returned None: {mt5.last_error()}"
            self.logger.error(msg)
            return {"retcode": -1, "comment": msg}

        result_dict = result._asdict()

        if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            self.logger.info(
                "✔ Pending %s placed — order #%s, vol=%.2f @ %.5f",
                label, result.order, result.volume, price,
            )
        else:
            self.logger.warning(
                "✘ Pending %s FAILED — retcode=%s (%s)  server='%s'",
                label, result.retcode,
                self.get_retcode_message(result.retcode),
                result.comment,
            )

        return result_dict

    # ------------------------------------------------------------------
    def cancel_pending_order(self, ticket: int) -> bool:
        """Cancel a pending order by its ticket.

        Parameters
        ----------
        ticket : int
            Pending order ticket to cancel.

        Returns
        -------
        bool
            *True* if the cancellation succeeded.
        """
        if self.dry_run:
            self.logger.info(
                "[DRY RUN] Cancel pending order ticket #%s", ticket
            )
            return True

        request: dict = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }

        self.logger.info("Cancelling pending order #%s", ticket)

        result = mt5.order_send(request)
        if result is None:
            self.logger.error(
                "Cancel order_send returned None: %s", mt5.last_error()
            )
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info("✔ Pending order #%s cancelled.", ticket)
            return True

        self.logger.warning(
            "✘ Cancel failed for #%s — retcode=%s (%s)",
            ticket,
            result.retcode,
            self.get_retcode_message(result.retcode),
        )
        return False

    # ------------------------------------------------------------------
    def get_pending_orders(
        self, symbol: Optional[str] = None
    ) -> List[dict]:
        """Return pending orders from MT5.

        Parameters
        ----------
        symbol : str or None
            Filter by symbol.  ``None`` returns all pending orders.

        Returns
        -------
        list[dict]
            Each dict contains: ``ticket``, ``symbol``, ``type``,
            ``volume``, ``price``, ``sl``, ``tp``, ``time_setup``,
            ``expiration``, ``magic``, ``comment``.
        """
        _type_names = {
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP",
            mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY_STOP_LIMIT",
            mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL_STOP_LIMIT",
        }

        orders = (
            mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        )
        if orders is None or len(orders) == 0:
            return []

        result: List[dict] = []
        for o in orders:
            result.append({
                "ticket": o.ticket,
                "symbol": o.symbol,
                "type": _type_names.get(o.type, f"UNKNOWN({o.type})"),
                "volume_initial": o.volume_initial,
                "volume_current": o.volume_current,
                "price": o.price_open,
                "sl": o.sl,
                "tp": o.tp,
                "time_setup": datetime.fromtimestamp(o.time_setup).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "expiration": (
                    datetime.fromtimestamp(o.time_expiration).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if o.time_expiration > 0
                    else None
                ),
                "magic": o.magic,
                "comment": o.comment,
            })

        return result

    # ==================================================================
    #  ORDER TRACKING
    # ==================================================================

    def track_order(
        self,
        order_id: Optional[str] = None,
        ticket: int = 0,
        symbol: str = "",
        order_type: OrderType = OrderType.MARKET_BUY,
        lot: float = 0.0,
        price: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
        group_id: Optional[str] = None,
    ) -> TrackedOrder:
        """Create and register a new tracked order.

        Parameters
        ----------
        order_id : str or None
            Unique order id.  Auto-generated UUID if *None*.
        ticket : int
            MT5 order ticket (``0`` if not yet placed).
        symbol : str
            Trading symbol.
        order_type : OrderType
            RI order type.
        lot : float
            Volume.
        price : float
            Requested price.
        sl : float
            Stop-loss price.
        tp : float
            Take-profit price.
        comment : str
            Comment.
        group_id : str or None
            Group to associate the order with.

        Returns
        -------
        TrackedOrder
            The newly created tracked order.
        """
        if order_id is None:
            order_id = str(uuid.uuid4())

        now = datetime.now()
        initial_state = OrderState.PENDING_CREATION

        tracked = TrackedOrder(
            order_id=order_id,
            ticket=ticket,
            symbol=symbol,
            order_type=order_type,
            state=initial_state,
            lot=lot,
            price=price,
            sl=sl,
            tp=tp,
            comment=comment,
            group_id=group_id,
            created_at=now,
            updated_at=now,
            filled_at=None,
            fill_price=0.0,
            history=[(now, initial_state, "Order created")],
        )

        with self._lock:
            self._tracked_orders[order_id] = tracked

        self.logger.debug(
            "Tracking order %s — %s %s %.2f @ %.5f",
            order_id, order_type.value, symbol, lot, price,
        )
        return tracked

    # ------------------------------------------------------------------
    def update_order_state(
        self, order_id: str, new_state: OrderState, reason: str = ""
    ) -> bool:
        """Transition a tracked order to *new_state*.

        Parameters
        ----------
        order_id : str
            Internal order identifier.
        new_state : OrderState
            Target state.
        reason : str, optional
            Human-readable reason for the transition.

        Returns
        -------
        bool
            *True* if the order was found and updated.
        """
        with self._lock:
            tracked = self._tracked_orders.get(order_id)
            if tracked is None:
                self.logger.warning(
                    "update_order_state: order_id=%s not found.", order_id
                )
                return False

            old_state = tracked.state
            now = datetime.now()
            tracked.state = new_state
            tracked.updated_at = now
            tracked.history.append((now, new_state, reason))

            if new_state == OrderState.FILLED:
                tracked.filled_at = now

        self.logger.info(
            "Order %s: %s → %s  reason='%s'",
            order_id, old_state.value, new_state.value, reason,
        )
        return True

    # ------------------------------------------------------------------
    def get_tracked_order(self, order_id: str) -> Optional[TrackedOrder]:
        """Return a tracked order by its *order_id*, or ``None``."""
        with self._lock:
            return self._tracked_orders.get(order_id)

    # ------------------------------------------------------------------
    def get_tracked_orders(
        self, state: Optional[OrderState] = None
    ) -> List[TrackedOrder]:
        """Return tracked orders, optionally filtered by *state*.

        Parameters
        ----------
        state : OrderState or None
            Filter.  ``None`` returns all orders.

        Returns
        -------
        list[TrackedOrder]
        """
        with self._lock:
            orders = list(self._tracked_orders.values())
        if state is not None:
            orders = [o for o in orders if o.state == state]
        return orders

    # ==================================================================
    #  OCO / ORDER GROUPS
    # ==================================================================

    def create_order_group(
        self,
        name: str,
        order_ids: List[str],
        cancel_on_fill: bool = True,
    ) -> OrderGroup:
        """Link several tracked orders into a named group.

        Parameters
        ----------
        name : str
            Human-readable group name.
        order_ids : list[str]
            Order identifiers to include.
        cancel_on_fill : bool
            When *True*, filling one order cancels the others (OCO).

        Returns
        -------
        OrderGroup
        """
        group_id = str(uuid.uuid4())
        now = datetime.now()

        group = OrderGroup(
            group_id=group_id,
            name=name,
            orders=list(order_ids),
            cancel_on_fill=cancel_on_fill,
            created_at=now,
        )

        with self._lock:
            self._order_groups[group_id] = group
            # Tag each order with its group
            for oid in order_ids:
                tracked = self._tracked_orders.get(oid)
                if tracked is not None:
                    tracked.group_id = group_id

        self.logger.info(
            "Created order group '%s' (%s) with %d orders  OCO=%s",
            name, group_id, len(order_ids), cancel_on_fill,
        )
        return group

    # ------------------------------------------------------------------
    def handle_group_fill(self, order_id: str) -> None:
        """Process a fill event for a grouped order.

        If the filled order belongs to a group with ``cancel_on_fill=True``,
        every *other* order in that group is cancelled.

        Parameters
        ----------
        order_id : str
            The order that was filled.
        """
        with self._lock:
            tracked = self._tracked_orders.get(order_id)
            if tracked is None or tracked.group_id is None:
                return

            group = self._order_groups.get(tracked.group_id)
            if group is None or not group.cancel_on_fill:
                return

            siblings = [
                oid for oid in group.orders if oid != order_id
            ]

        self.logger.info(
            "OCO trigger: order %s filled → cancelling %d sibling(s) "
            "in group '%s'",
            order_id, len(siblings), group.name,
        )

        for sibling_id in siblings:
            sibling = self.get_tracked_order(sibling_id)
            if sibling is None:
                continue

            # Cancel in MT5 if still a live pending order
            if (
                sibling.ticket > 0
                and sibling.state
                in (OrderState.PENDING, OrderState.PENDING_CREATION)
            ):
                self.cancel_pending_order(sibling.ticket)

            self.update_order_state(
                sibling_id,
                OrderState.CANCELED,
                reason=f"OCO: sibling {order_id} filled",
            )

    # ------------------------------------------------------------------
    def get_order_group(
        self, group_id: str
    ) -> Optional[OrderGroup]:
        """Return an order group by its *group_id*, or ``None``."""
        with self._lock:
            return self._order_groups.get(group_id)

    # ==================================================================
    #  BRACKET ORDER
    # ==================================================================

    def place_bracket_order(
        self,
        symbol: str,
        direction: str,
        lot: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        comment: str = "",
    ) -> dict:
        """Place an entry with linked SL and TP as an OCO group.

        For a **market** entry (``direction`` is ``'BUY'`` or ``'SELL'``),
        a market order is sent immediately with ``sl_price`` / ``tp_price``
        attached.

        For a **pending** entry (``direction`` is ``'BUY_LIMIT'``,
        ``'SELL_LIMIT'``, ``'BUY_STOP'``, or ``'SELL_STOP'``), the pending
        order is placed with SL/TP embedded in the MT5 request, and a
        tracking group is created for lifecycle management.

        Parameters
        ----------
        symbol : str
            Trading symbol.
        direction : str
            ``'BUY'``, ``'SELL'``, ``'BUY_LIMIT'``, ``'SELL_LIMIT'``,
            ``'BUY_STOP'``, ``'SELL_STOP'``.
        lot : float
            Volume.
        entry_price : float
            Entry price (used for pending orders; ignored for market).
        sl_price : float
            Stop-loss price.
        tp_price : float
            Take-profit price.
        comment : str, optional
            Comment.

        Returns
        -------
        dict
            Keys: ``entry_result`` (order send result dict),
            ``group`` (OrderGroup or None),
            ``tracked_ids`` (list of internal order ids).
        """
        lot = OrderAdapter.adapt_volume(lot, symbol)
        entry_price = OrderAdapter.adapt_price(entry_price, symbol)
        sl_price = OrderAdapter.adapt_price(sl_price, symbol)
        tp_price = OrderAdapter.adapt_price(tp_price, symbol)

        direction_upper = direction.upper()
        _pending_map = {
            "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
            "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
            "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
            "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
        }

        tracked_ids: List[str] = []

        # ----- Market entry -----------------------------------------------
        if direction_upper in ("BUY", "SELL"):
            mt5_type = (
                mt5.ORDER_TYPE_BUY
                if direction_upper == "BUY"
                else mt5.ORDER_TYPE_SELL
            )
            ri_type = (
                OrderType.MARKET_BUY
                if direction_upper == "BUY"
                else OrderType.MARKET_SELL
            )

            entry_result = self.open_position(
                symbol, mt5_type, lot, sl_price, tp_price, comment=comment,
            )

            ticket = entry_result.get("order", 0)
            fill_price = entry_result.get("price", 0.0)

            entry_tracked = self.track_order(
                ticket=ticket,
                symbol=symbol,
                order_type=ri_type,
                lot=lot,
                price=fill_price if fill_price else entry_price,
                sl=sl_price,
                tp=tp_price,
                comment=comment,
            )
            tracked_ids.append(entry_tracked.order_id)

            success_codes = {10009, 10010}  # DONE, PARTIAL
            if entry_result.get("retcode") in success_codes:
                entry_tracked.fill_price = fill_price
                self.update_order_state(
                    entry_tracked.order_id,
                    OrderState.FILLED,
                    reason="Market order filled",
                )
            else:
                self.update_order_state(
                    entry_tracked.order_id,
                    OrderState.FAILED,
                    reason=entry_result.get("comment", "Unknown failure"),
                )

            # SL and TP are embedded in the position — track them
            if sl_price:
                sl_tracked = self.track_order(
                    symbol=symbol,
                    order_type=OrderType.STOP_LOSS,
                    lot=lot,
                    price=sl_price,
                    sl=0.0,
                    tp=0.0,
                    comment=f"SL for {entry_tracked.order_id}",
                )
                self.update_order_state(
                    sl_tracked.order_id, OrderState.OPEN,
                    reason="Attached SL active",
                )
                tracked_ids.append(sl_tracked.order_id)

            if tp_price:
                tp_tracked = self.track_order(
                    symbol=symbol,
                    order_type=OrderType.TAKE_PROFIT,
                    lot=lot,
                    price=tp_price,
                    sl=0.0,
                    tp=0.0,
                    comment=f"TP for {entry_tracked.order_id}",
                )
                self.update_order_state(
                    tp_tracked.order_id, OrderState.OPEN,
                    reason="Attached TP active",
                )
                tracked_ids.append(tp_tracked.order_id)

            # Group the SL + TP as OCO (if both exist)
            group = None
            if len(tracked_ids) > 1:
                group = self.create_order_group(
                    name=f"Bracket-{symbol}-{direction_upper}",
                    order_ids=tracked_ids[1:],  # SL & TP only
                    cancel_on_fill=True,
                )

            return {
                "entry_result": entry_result,
                "group": group,
                "tracked_ids": tracked_ids,
            }

        # ----- Pending entry ----------------------------------------------
        if direction_upper in _pending_map:
            mt5_pending_type = _pending_map[direction_upper]

            ri_type_map = {
                "BUY_LIMIT": OrderType.LIMIT_BUY,
                "SELL_LIMIT": OrderType.LIMIT_SELL,
                "BUY_STOP": OrderType.STOP_BUY,
                "SELL_STOP": OrderType.STOP_SELL,
            }
            ri_type = ri_type_map[direction_upper]

            entry_result = self.place_pending_order(
                symbol=symbol,
                order_type=mt5_pending_type,
                lot=lot,
                price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                comment=comment,
            )

            ticket = entry_result.get("order", 0)

            entry_tracked = self.track_order(
                ticket=ticket,
                symbol=symbol,
                order_type=ri_type,
                lot=lot,
                price=entry_price,
                sl=sl_price,
                tp=tp_price,
                comment=comment,
            )
            tracked_ids.append(entry_tracked.order_id)

            placed_codes = {10008, 10009}  # PLACED, DONE
            if entry_result.get("retcode") in placed_codes:
                self.update_order_state(
                    entry_tracked.order_id,
                    OrderState.PENDING,
                    reason="Pending order placed",
                )
            else:
                self.update_order_state(
                    entry_tracked.order_id,
                    OrderState.FAILED,
                    reason=entry_result.get("comment", "Unknown failure"),
                )

            # SL/TP are embedded in the pending order itself;
            # track them for lifecycle awareness
            if sl_price:
                sl_tracked = self.track_order(
                    symbol=symbol,
                    order_type=OrderType.STOP_LOSS,
                    lot=lot,
                    price=sl_price,
                    sl=0.0,
                    tp=0.0,
                    comment=f"SL for pending {entry_tracked.order_id}",
                )
                self.update_order_state(
                    sl_tracked.order_id, OrderState.PENDING,
                    reason="Attached to pending entry",
                )
                tracked_ids.append(sl_tracked.order_id)

            if tp_price:
                tp_tracked = self.track_order(
                    symbol=symbol,
                    order_type=OrderType.TAKE_PROFIT,
                    lot=lot,
                    price=tp_price,
                    sl=0.0,
                    tp=0.0,
                    comment=f"TP for pending {entry_tracked.order_id}",
                )
                self.update_order_state(
                    tp_tracked.order_id, OrderState.PENDING,
                    reason="Attached to pending entry",
                )
                tracked_ids.append(tp_tracked.order_id)

            group = None
            if len(tracked_ids) > 1:
                group = self.create_order_group(
                    name=f"Bracket-{symbol}-{direction_upper}",
                    order_ids=tracked_ids,
                    cancel_on_fill=True,
                )

            return {
                "entry_result": entry_result,
                "group": group,
                "tracked_ids": tracked_ids,
            }

        # ----- Unknown direction ------------------------------------------
        msg = (
            f"Invalid direction '{direction}'. Expected one of: "
            "BUY, SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP."
        )
        self.logger.error(msg)
        return {
            "entry_result": {"retcode": -1, "comment": msg},
            "group": None,
            "tracked_ids": [],
        }
