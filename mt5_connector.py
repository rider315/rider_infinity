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

logger = logging.getLogger("mt5_connector")


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
