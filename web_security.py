"""
web_security.py — Access control for the Rider Infinity dashboard.

The dashboard is not a read-only monitor: it can start and stop bots,
flip a bot from dry-run to live, close every open position, move stops,
place orders and rewrite risk limits in ``.env``.  Reaching it is
therefore equivalent to holding the trading account, and it must not be
reachable without a credential.

Three independent protections live here:

1. **HTTP Basic authentication** (:func:`install_auth`).  Enforced on
   every route whenever ``DASHBOARD_PASSWORD`` is set.  Basic auth is
   used deliberately: the browser handles the challenge natively and
   then attaches the credential to every same-origin ``fetch`` the
   single-page UI makes, so the existing front-end needs no changes.
2. **Same-origin enforcement** on state-changing requests.  Because the
   browser attaches that credential ambiently, a page on another origin
   could otherwise POST to the dashboard on the operator's behalf.  Any
   mutating request carrying a cross-origin ``Origin``/``Referer`` is
   rejected.
3. **A safe bind** (:func:`resolve_bind`).  The listen address defaults
   to loopback, a non-loopback bind requires a password, and the
   Werkzeug debugger is never allowed off loopback.

Configuration (all via ``.env`` / environment):

==============================  ==========================================
``DASHBOARD_HOST``              Listen address.  Default ``127.0.0.1``.
``DASHBOARD_PORT``              Listen port.  Default ``5000``.
``DASHBOARD_USER``              Basic-auth user.  Default ``rider``.
``DASHBOARD_PASSWORD``          Basic-auth password.  Empty = auth off,
                                which then forces a loopback bind.
``DASHBOARD_DEBUG``             Werkzeug debugger + reloader.  Default
                                off; refused on a non-loopback bind.
``DASHBOARD_ALLOW_INSECURE_BIND``  Escape hatch to bind off-loopback with
                                no password.  Never use on a public host.
==============================  ==========================================
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request

logger = logging.getLogger("web_security")

#: Methods that change server state and therefore need CSRF protection.
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Realm shown in the browser's credential prompt.
_REALM = "Rider Infinity Dashboard"


class InsecureConfigError(RuntimeError):
    """Raised when the requested bind/debug combination is unsafe."""


def _is_env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("true", "1", "yes", "on")


def is_loopback(host: str) -> bool:
    """Return ``True`` if *host* can only be reached from this machine.

    Treats the empty host and ``localhost`` as loopback, and otherwise
    defers to :mod:`ipaddress`.  A hostname that is not an IP literal is
    assumed to be routable (so it is treated as *not* loopback) because
    we cannot prove otherwise without resolving it.
    """
    host = (host or "").strip().lower()
    if host in ("", "localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------
# Bind resolution
# ---------------------------------------------------------------------

def resolve_bind() -> tuple[str, int, bool]:
    """Return the ``(host, port, debug)`` the server may safely use.

    Returns
    -------
    tuple[str, int, bool]
        Listen host, listen port, and whether the Werkzeug debugger /
        reloader may be enabled.

    Raises
    ------
    InsecureConfigError
        If the dashboard would be exposed beyond this machine without a
        password, or the debugger would be exposed off loopback.  Both
        combinations hand control of a live trading account to anyone who
        can reach the port, so they fail closed rather than warn.
    """
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.getenv("DASHBOARD_PORT", "5000"))
    except ValueError:
        raise InsecureConfigError(
            f"DASHBOARD_PORT must be a whole number, got "
            f"{os.getenv('DASHBOARD_PORT')!r}."
        ) from None

    password = os.getenv("DASHBOARD_PASSWORD", "")
    debug = _is_env_true("DASHBOARD_DEBUG")
    loopback = is_loopback(host)

    if not loopback and not password:
        if not _is_env_true("DASHBOARD_ALLOW_INSECURE_BIND"):
            raise InsecureConfigError(
                f"Refusing to listen on '{host}' with no password: the "
                f"dashboard can place orders, flip bots to live and close "
                f"all positions, so anyone who can reach the port controls "
                f"the account. Set DASHBOARD_PASSWORD in .env, or bind to "
                f"127.0.0.1 and use an SSH tunnel. To override anyway set "
                f"DASHBOARD_ALLOW_INSECURE_BIND=true."
            )
        logger.error(
            "DASHBOARD_ALLOW_INSECURE_BIND is set — listening on %s with NO "
            "authentication. Anyone who can reach this port controls the "
            "trading account.", host,
        )

    if debug and not loopback:
        raise InsecureConfigError(
            f"Refusing to enable the Werkzeug debugger while listening on "
            f"'{host}'. The debugger allows arbitrary code execution. Either "
            f"bind to 127.0.0.1 or unset DASHBOARD_DEBUG."
        )

    return host, port, debug


# ---------------------------------------------------------------------
# Authentication / CSRF
# ---------------------------------------------------------------------

def _unauthorized() -> Response:
    resp = jsonify({
        "error": "Authentication required.",
        "detail": "Supply the DASHBOARD_USER / DASHBOARD_PASSWORD credential.",
    })
    resp.status_code = 401
    resp.headers["WWW-Authenticate"] = f'Basic realm="{_REALM}", charset="UTF-8"'
    return resp


def _credentials_match(supplied_user: str, supplied_password: str,
                       expected_user: str, expected_password: str) -> bool:
    """Compare credentials without leaking length/content via timing.

    Both halves are always compared so the work done is independent of
    which one is wrong.
    """
    user_ok = secrets.compare_digest(supplied_user, expected_user)
    password_ok = secrets.compare_digest(supplied_password, expected_password)
    return user_ok and password_ok


def _same_origin(origin_header: str, host_header: str) -> bool:
    """Return ``True`` if *origin_header* points at *host_header*.

    Compared on the ``host:port`` authority only.  The scheme is ignored
    because the dashboard is typically served over plain HTTP on
    loopback or behind a TLS-terminating tunnel, so requiring a matching
    scheme would reject legitimate same-origin requests.
    """
    if not origin_header or not host_header:
        return False
    parsed = urlsplit(origin_header)
    # A bare authority (no "//") parses into .path, not .netloc.
    authority = parsed.netloc or parsed.path
    return authority.strip("/").lower() == host_header.strip("/").lower()


def install_auth(app: Flask) -> bool:
    """Attach the authentication and CSRF checks to *app*.

    Returns
    -------
    bool
        ``True`` if Basic authentication is active, ``False`` if no
        password is configured (in which case :func:`resolve_bind` has
        already confined the server to loopback).
    """
    expected_user = os.getenv("DASHBOARD_USER", "rider")
    expected_password = os.getenv("DASHBOARD_PASSWORD", "")
    auth_enabled = bool(expected_password)

    @app.before_request
    def _enforce_access():  # noqa: ANN202 - Flask hook
        # -- CSRF: reject cross-origin state changes ------------------
        # The browser attaches Basic-auth credentials to any request it
        # makes to this origin, including one triggered by a page the
        # operator did not author, so a mutating request must prove it
        # came from the dashboard itself. Requests with no Origin and no
        # Referer are non-browser clients (curl, scripts), which carry no
        # ambient credential and so cannot be driven this way.
        if request.method in _MUTATING_METHODS:
            stated_origin = request.headers.get("Origin") or ""
            if not stated_origin:
                referer = request.headers.get("Referer") or ""
                stated_origin = referer
            if stated_origin and not _same_origin(
                stated_origin, request.headers.get("Host", "")
            ):
                logger.warning(
                    "Blocked cross-origin %s %s from Origin/Referer %r",
                    request.method, request.path, stated_origin,
                )
                return jsonify({
                    "error": "Cross-origin request blocked.",
                    "detail": (
                        "State-changing requests must originate from the "
                        "dashboard itself."
                    ),
                }), 403

        # -- Authentication -------------------------------------------
        if not auth_enabled:
            return None

        auth = request.authorization
        if auth is None or auth.type.lower() != "basic":
            return _unauthorized()
        if not _credentials_match(
            auth.username or "", auth.password or "",
            expected_user, expected_password,
        ):
            logger.warning(
                "Rejected dashboard login for user %r from %s",
                auth.username, request.remote_addr,
            )
            return _unauthorized()
        return None

    if auth_enabled:
        logger.info(
            "Dashboard authentication enabled (user=%r).", expected_user,
        )
    else:
        logger.warning(
            "DASHBOARD_PASSWORD is not set — the dashboard is unauthenticated "
            "and confined to loopback. Set it in .env before exposing the "
            "dashboard to any other machine.",
        )
    return auth_enabled
