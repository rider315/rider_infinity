"""
wsgi.py — WSGI entry point for running the dashboard behind a real server.

Used where ``python dashboard.py`` is not appropriate — e.g. gunicorn or
waitress on a Linux host, where the MetaTrader 5 terminal may live on a
different machine (or not exist at all, with ``MetaTrader5.py`` supplying
a stub).

Difference from ``dashboard.main()``: a WSGI worker that exits on a failed
broker connection just gets restarted by its supervisor in a crash loop,
so this entry point starts in a **degraded** state instead.  The app comes
up, every MT5-backed endpoint reports its error, and
``GET /api/terminal-status`` says what is wrong.

This replaces an earlier version that monkey-patched
``MT5Connector.connect`` to swallow exceptions and ``sys.exit`` to ignore
exit code 1.  Those patches hid a real failure: the app booted with
``_connected = False`` and every request failed with a confusing error
instead of one clear startup message.  ``_init_subsystems(require_mt5=…)``
now expresses that choice directly.

A missing or malformed configuration is still fatal — there is no useful
degraded mode without credentials.

Run with::

    gunicorn  -w 1 --threads 8 wsgi:app
    waitress-serve --threads=8 wsgi:app

Use a **single worker**: the bot threads, the MT5 connection and the
in-memory fleet state live in the worker process, so a second worker
would run the whole fleet a second time against one account.  Put
authentication in front of it (see :mod:`web_security`).
"""

from __future__ import annotations

import sys

from config import ConfigError
from dashboard import _init_subsystems, create_app

try:
    # Degraded start: serve even when the terminal is unreachable.
    _init_subsystems(require_mt5=False)
except ConfigError as exc:
    print(f"[STARTUP FAILED] {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

app = create_app()
