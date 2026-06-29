"""
mt5_connector.py - MetaTrader 5 terminal connection manager.

Provides the ``MT5Connector`` class which wraps the MetaTrader5 Python
package with robust error handling, automatic logging, and reconnection
support.  Every public method logs its outcome so issues can be traced
through the bot's log file.
"""

import logging
from typing import Optional

import MetaTrader5 as mt5

from config import Config
from utils import setup_logger

# Use the project logger so connector messages (incl. the Algo-Trading
# permission warning) are written to logs/bot.log, not just stderr.
logger = setup_logger("mt5_connector")


class MT5Connector:
    """Manages the lifecycle of a MetaTrader 5 terminal connection.

    Usage::

        cfg = Config()
        connector = MT5Connector(cfg)
        if connector.connect():
            info = connector.get_account_info()
            print(info)
        connector.disconnect()
    """

    def __init__(self, config: Config) -> None:
        """Initialise the connector with application configuration.

        Args:
            config: A ``Config`` instance holding MT5 credentials and
                    the path to the terminal executable.
        """
        self.config = config
        self._connected: bool = False

    # ── Connection lifecycle ─────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise the MT5 terminal and log in.

        Calls ``mt5.initialize()`` with the path, login, password and
        server from the loaded configuration.  On success the account
        summary is logged at INFO level.

        Returns:
            ``True`` if the connection was established, ``False`` otherwise.
        """
        logger.info("Initialising MT5 terminal at '%s' …", self.config.mt5_path)

        if not mt5.initialize(
            path=self.config.mt5_path,
            login=self.config.mt5_login,
            password=self.config.mt5_password,
            server=self.config.mt5_server,
        ):
            error_code, error_msg = mt5.last_error()
            logger.error(
                "MT5 initialisation failed  [%s] %s", error_code, error_msg
            )
            self._connected = False
            return False

        self._connected = True

        # Log a quick account summary so the operator can verify the
        # correct account was connected.
        account = mt5.account_info()
        if account is not None:
            logger.info(
                "Connected to account %s (%s) on %s  |  "
                "Balance: %.2f %s  |  Leverage: 1:%d",
                account.login,
                account.name,
                account.server,
                account.balance,
                account.currency,
                account.leverage,
            )
        else:
            logger.warning("Connected but could not retrieve account info.")

        # Surface the Algo-Trading permission of THIS terminal instance up
        # front. retcode 10027 on every order means this is False — and on a
        # server the instance the API bound to may not be the window a person
        # toggled. The path makes it obvious which terminal to fix.
        status = self.terminal_status()
        if not status.get("trade_allowed", False):
            data_path = str(status.get("data_path", "") or "")
            is_system = (
                "systemprofile" in data_path.lower()
                or "system32" in data_path.lower()
            )
            extra = ""
            if is_system:
                extra = (
                    " This terminal runs under the Windows SYSTEM account "
                    "(a service, Session 0) — it has NO visible window, so "
                    "Algo Trading can't be toggled there. Run the dashboard "
                    "from an interactive Administrator session instead, after "
                    "enabling Algo Trading in the Administrator's MT5 terminal."
                )
            logger.warning(
                "⚠ Algo Trading is DISABLED in the terminal this session is "
                "bound to (path=%s, data_path=%s). Orders (incl. Close All) "
                "will be rejected with retcode 10027 until you enable "
                "'Algo Trading' in THAT terminal instance.%s",
                status.get("path", "?"), status.get("data_path", "?"), extra,
            )
        else:
            logger.info(
                "Algo Trading is ENABLED (terminal build %s, path=%s).",
                status.get("build", "?"), status.get("path", "?"),
            )

        return True

    def disconnect(self) -> None:
        """Shut down the MT5 terminal connection gracefully."""
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 terminal connection shut down.")

    def reconnect(self) -> bool:
        """Disconnect and re-establish the MT5 connection.

        Useful after network hiccups or when the terminal becomes
        unresponsive.

        Returns:
            ``True`` if reconnection succeeded, ``False`` otherwise.
        """
        logger.info("Attempting reconnection …")
        self.disconnect()
        return self.connect()

    def terminal_status(self) -> dict:
        """Return the live trading-permission state of the attached terminal.

        ``trade_allowed`` is the single most important field: it is ``True``
        only when the 'Algo Trading' switch is ON **in the exact terminal
        instance this Python session is bound to**. On a server, ``initialize``
        can launch its *own* terminal instance (separate from any one a person
        opened interactively), so toggling Algo Trading in a visible window has
        no effect on the instance that actually sends orders. ``path`` /
        ``data_path`` identify which instance that is, so the right one can be
        fixed.
        """
        info = mt5.terminal_info()
        if info is None:
            return {"connected": False, "trade_allowed": False}
        return {
            "connected": bool(getattr(info, "connected", False)),
            "trade_allowed": bool(getattr(info, "trade_allowed", False)),
            "trade_expert": bool(getattr(info, "trade_expert", False)),
            "path": getattr(info, "path", ""),
            "data_path": getattr(info, "data_path", ""),
            "name": getattr(info, "name", ""),
            "build": getattr(info, "build", 0),
        }

    def is_connected(self) -> bool:
        """Check whether the terminal is still reachable.

        Performs a lightweight ``mt5.terminal_info()`` call to verify
        that the underlying connection is alive.

        Returns:
            ``True`` if the terminal responds, ``False`` otherwise.
        """
        if not self._connected:
            return False

        info = mt5.terminal_info()
        if info is None:
            logger.warning("Terminal health-check failed – connection lost.")
            self._connected = False
            return False
        return True

    # ── Symbol management ────────────────────────────────────────────────

    def ensure_symbol(self, symbol: str) -> bool:
        """Make sure a symbol is visible in Market Watch.

        Args:
            symbol: The instrument name (e.g. ``"EURUSD"``).

        Returns:
            ``True`` if the symbol is now selected, ``False`` on error.
        """
        # Check if the symbol exists on this server at all.
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error("Symbol '%s' not found on the server.", symbol)
            return False

        # If it's not currently visible, select it.
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                error_code, error_msg = mt5.last_error()
                logger.error(
                    "Failed to select symbol '%s'  [%s] %s",
                    symbol,
                    error_code,
                    error_msg,
                )
                return False
            logger.info("Symbol '%s' enabled in Market Watch.", symbol)
        else:
            logger.debug("Symbol '%s' is already visible.", symbol)

        return True

    # ── Account information ──────────────────────────────────────────────

    def get_account_info(self) -> Optional[dict]:
        """Retrieve key account metrics.

        Returns:
            A dictionary with the following keys on success::

                balance, equity, margin, free_margin, leverage, currency

            Returns ``None`` if the information cannot be retrieved.
        """
        account = mt5.account_info()
        if account is None:
            error_code, error_msg = mt5.last_error()
            logger.error(
                "Failed to get account info  [%s] %s", error_code, error_msg
            )
            return None

        return {
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "free_margin": account.margin_free,
            "leverage": account.leverage,
            "currency": account.currency,
        }
