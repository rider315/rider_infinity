"""
utils.py - Shared utility functions for the Rider Infinity Trading Platform.

Provides:
* Colourised console + file logger setup
* Pip ↔ price conversion helpers
* Pretty-print formatters for signals and account info
* Trade journal CSV writer
* Terminal helpers (clear screen, banner, interactive menus)
"""

import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from colorama import Fore, Style, init as colorama_init

# Pip arithmetic lives in ``instruments`` (dependency-free, so the
# cloud worker can ship it) and is re-exported here for the many
# callers that already do ``from utils import pips_to_price``.
from instruments import pip_size, pips_to_price, price_to_pips  # noqa: F401

# Initialise colorama once for the process (autoreset keeps every
# print/log statement self-contained).
colorama_init(autoreset=True)

# Resolve the project root so log paths work regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_DIR = _PROJECT_ROOT / "logs"


# =====================================================================
# Logging
# =====================================================================

class _ColouredFormatter(logging.Formatter):
    """Logging formatter that adds ANSI colours per severity level."""

    _LEVEL_COLOURS = {
        logging.DEBUG:    Fore.CYAN,
        logging.INFO:     Fore.GREEN,
        logging.WARNING:  Fore.YELLOW,
        logging.ERROR:    Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = self._LEVEL_COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname:<8}{Style.RESET_ALL}"
        record.msg = f"{colour}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a named logger with console and file handlers.

    * **Console** – colourised via *colorama* for quick visual scanning.
    * **File** – plain text, appended to ``mt5_forex_bot/logs/bot.log``.
      The ``logs/`` directory is created automatically if it does not
      exist.

    Calling this function multiple times with the same *name* returns the
    same logger without adding duplicate handlers.

    Args:
        name:  Logger name (typically the module's ``__name__``).
        level: Minimum severity to emit (default ``logging.INFO``).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers when called more than once.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # ── Console handler (colourised) ─────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColouredFormatter(fmt, datefmt=datefmt))
    logger.addHandler(console)

    # ── File handler (plain text) ────────────────────────────────────
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / "bot.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    return logger


# =====================================================================
# Pip / price helpers
# =====================================================================

def compute_atr(df, period: int = 14) -> float:
    """Return the latest Average True Range (a *price* value) from OHLC data.

    Uses Wilder's true range = max(high-low, |high-prev_close|, |low-prev_close|)
    smoothed by a simple moving average over *period* bars.  Returns ``0.0`` if
    the data is insufficient or the required columns are missing.
    """
    try:
        if df is None or len(df) < period + 1:
            return 0.0
        if not {"high", "low", "close"}.issubset(df.columns):
            return 0.0
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = (high - low).abs()
        tr = tr.combine((high - prev_close).abs(), max)
        tr = tr.combine((low - prev_close).abs(), max)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return float(atr) if atr == atr else 0.0  # NaN guard
    except Exception:
        return 0.0


# =====================================================================
# Pretty-print formatters
# =====================================================================

def format_signal(signal: dict) -> str:
    """Return a colourised, human-readable string for a trade signal.

    Expected *signal* keys (all optional – missing keys are shown as
    ``"—"``)::

        type, symbol, price, sl, tp, lot, reason

    Args:
        signal: Dictionary describing the trade signal.

    Returns:
        A multi-line formatted string.
    """
    direction = signal.get("type", "—")
    colour = Fore.GREEN if direction == "BUY" else Fore.RED if direction == "SELL" else Fore.WHITE

    lines = [
        f"{colour}{Style.BRIGHT}{'═' * 44}",
        f"  ⚡  SIGNAL: {direction}  {signal.get('symbol', '—')}",
        f"{'═' * 44}{Style.RESET_ALL}",
        f"  Price : {signal.get('price', '—')}",
        f"  SL    : {signal.get('sl', '—')}",
        f"  TP    : {signal.get('tp', '—')}",
        f"  Lot   : {signal.get('lot', '—')}",
        f"  Reason: {signal.get('reason', '—')}",
        f"{colour}{'─' * 44}{Style.RESET_ALL}",
    ]
    return "\n".join(lines)


def format_account_info(info: dict) -> str:
    """Return a neatly formatted table of account information.

    Args:
        info: Dictionary as returned by
              ``MT5Connector.get_account_info()``.

    Returns:
        A multi-line string suitable for console output.
    """
    if not info:
        return f"{Fore.RED}  No account information available.{Style.RESET_ALL}"

    currency = info.get("currency", "USD")
    rows = [
        ("Balance",     f"{info.get('balance', 0):>12,.2f} {currency}"),
        ("Equity",      f"{info.get('equity', 0):>12,.2f} {currency}"),
        ("Margin",      f"{info.get('margin', 0):>12,.2f} {currency}"),
        ("Free Margin", f"{info.get('free_margin', 0):>12,.2f} {currency}"),
        ("Leverage",    f"{'1:' + str(info.get('leverage', '—')):>14}"),
    ]

    width = 42
    lines = [
        f"{Fore.CYAN}{Style.BRIGHT}{'═' * width}",
        f"  📊  ACCOUNT OVERVIEW",
        f"{'═' * width}{Style.RESET_ALL}",
    ]
    for label, value in rows:
        lines.append(f"  {label:<14} {value}")
    lines.append(f"{Fore.CYAN}{'─' * width}{Style.RESET_ALL}")
    return "\n".join(lines)


# =====================================================================
# Trade journal
# =====================================================================

def save_trade_to_journal(
    trade_data: dict,
    filepath: str = "logs/trade_journal.csv",
) -> None:
    """Append a trade record to a CSV journal file.

    If the file does not exist it is created with a header row derived
    from the keys of *trade_data*.

    Args:
        trade_data: Dictionary of trade fields (e.g. ``symbol``,
                    ``type``, ``lot``, ``price``, ``sl``, ``tp``,
                    ``profit``, ``timestamp``).
        filepath:   Relative or absolute path to the CSV file.  Defaults
                    to ``logs/trade_journal.csv`` inside the project.
    """
    target = Path(filepath)
    if not target.is_absolute():
        target = _PROJECT_ROOT / target

    target.parent.mkdir(parents=True, exist_ok=True)

    # Add a timestamp if the caller did not supply one.
    if "timestamp" not in trade_data:
        trade_data["timestamp"] = datetime.now(timezone.utc).isoformat()

    file_exists = target.exists()

    with open(target, mode="a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trade_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_data)


# =====================================================================
# Terminal helpers
# =====================================================================

def clear_screen() -> None:
    """Clear the terminal screen (Windows ``cls``, Unix ``clear``)."""
    os.system("cls" if os.name == "nt" else "clear")


def print_banner() -> None:
    """Print a stylish ASCII-art banner for Rider Infinity."""
    banner = rf"""
{Fore.YELLOW}{Style.BRIGHT}
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   ██████  ██ ██████  ███████ ██████                  ║
    ║   ██   ██ ██ ██   ██ ██      ██   ██                 ║
    ║   ██████  ██ ██   ██ █████   ██████                  ║
    ║   ██   ██ ██ ██   ██ ██      ██   ██                 ║
    ║   ██   ██ ██ ██████  ███████ ██   ██                 ║
    ║                                                      ║
    ║      ██ ███    ██ ███████ ██ ███    ██ ██ ████████    ║
    ║      ██ ████   ██ ██      ██ ████   ██ ██    ██      ║
    ║      ██ ██ ██  ██ █████   ██ ██ ██  ██ ██    ██      ║
    ║      ██ ██  ██ ██ ██      ██ ██  ██ ██ ██    ██      ║
    ║      ██ ██   ████ ██      ██ ██   ████ ██    ██      ║
    ║                                                      ║
    ║{Fore.WHITE}    Ride on the Infinity edge of Markets.{Fore.YELLOW}      ║
    ║{Fore.MAGENTA}              ━━━ RI v1.0 ━━━{Fore.YELLOW}                     ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""
    print(banner)


def print_menu(title: str, options: list[tuple[str, str]]) -> str:
    """Display a numbered menu and return the user's choice value.

    Each item in *options* is a ``(value, display_text)`` tuple.  The
    function loops until the user enters a valid number.

    Args:
        title:   Heading printed above the menu.
        options: List of ``(value, display_text)`` tuples.

    Returns:
        The *value* element of the selected option.
    """
    print(f"\n{Fore.CYAN}{Style.BRIGHT}  {title}{Style.RESET_ALL}")
    print(f"  {'─' * 36}")
    for idx, (_, display) in enumerate(options, start=1):
        print(f"  {Fore.YELLOW}{idx}.{Style.RESET_ALL} {display}")
    print(f"  {'─' * 36}")

    while True:
        try:
            raw = input(f"\n  {Fore.GREEN}Enter choice [1-{len(options)}]: {Style.RESET_ALL}").strip()
            choice = int(raw)
            if 1 <= choice <= len(options):
                selected_value, selected_display = options[choice - 1]
                print(f"  ✔ {selected_display}")
                return selected_value
            print(f"  {Fore.RED}Please enter a number between 1 and {len(options)}.{Style.RESET_ALL}")
        except ValueError:
            print(f"  {Fore.RED}Invalid input – enter a number.{Style.RESET_ALL}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {Fore.YELLOW}Cancelled.{Style.RESET_ALL}")
            sys.exit(0)
