"""
logic_version.py — Trade-logic fingerprinting.

Every live trade is stamped with a short **logic version** hash so exported
trade history can be split into "before code change X / after code change X"
without any manual bookkeeping.

The fingerprint covers everything that shapes *how* a trade is taken and
managed:

* the strategy's own source file (e.g. ``strategies/scalper_strategy.py``),
* the shared engine files (order execution, risk gates, trade management),
* the bot's strategy parameters, and
* the trade-management settings in effect (quick exit, cooldown, ATR stops…).

Change any of those → the hash changes → new trades carry the new version.
Identical setup → identical hash, across restarts and machines.

Each first-seen version is appended to ``data/logic_versions.json`` with a
UTC timestamp and the full settings snapshot, so a version hash in an export
can always be traced back to *what* changed and *when*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_REGISTRY_PATH = _ROOT / "data" / "logic_versions.json"
_REGISTRY_LOCK = threading.Lock()

# Engine files whose code determines how ANY strategy's trades are executed
# and managed (entry gates, SL/TP resolution, quick exit, breakeven, …).
_ENGINE_FILES = (
    "multi_bot.py",
    "risk_manager.py",
    "order_manager.py",
    "strategies/base.py",
)

# path -> (mtime, sha1) cache so repeated calls don't re-read unchanged files.
_file_cache: dict[str, tuple[float, str]] = {}


def _hash_file(path: Path) -> str:
    """SHA-1 of a file's bytes, cached by mtime. '' if unreadable."""
    key = str(path)
    try:
        mtime = path.stat().st_mtime
        cached = _file_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        _file_cache[key] = (mtime, digest)
        return digest
    except OSError:
        return ""


def _strategy_source(strategy_name: str) -> Path | None:
    """Resolve a registry key (e.g. 'scalper') to its strategy source file."""
    try:
        import inspect

        from strategies import STRATEGY_REGISTRY

        cls, _ = STRATEGY_REGISTRY[strategy_name]
        src = inspect.getsourcefile(cls)
        return Path(src) if src else None
    except Exception:
        return None


def compute_logic_version(
    strategy_name: str,
    strategy_params: dict | None = None,
    settings: dict | None = None,
) -> str:
    """Return a short stable fingerprint like ``'a3f9c1d208'``.

    Parameters
    ----------
    strategy_name : str
        Strategy registry key (``'scalper'``, ``'rsi'``, …).
    strategy_params : dict, optional
        The bot's per-strategy parameter overrides.
    settings : dict, optional
        Trade-management settings in effect (quick exit, cooldown, ATR
        stops…). Include anything that changes trade behaviour without a
        code edit.
    """
    h = hashlib.sha1()
    h.update(strategy_name.encode())

    src = _strategy_source(strategy_name)
    if src is not None:
        h.update(_hash_file(src).encode())

    for rel in _ENGINE_FILES:
        h.update(_hash_file(_ROOT / rel).encode())

    h.update(json.dumps(strategy_params or {}, sort_keys=True, default=str).encode())
    h.update(json.dumps(settings or {}, sort_keys=True, default=str).encode())

    version = h.hexdigest()[:10]
    _register(version, strategy_name, strategy_params, settings)
    return version


def _register(
    version: str,
    strategy_name: str,
    strategy_params: dict | None,
    settings: dict | None,
) -> None:
    """Append a first-seen version to the registry file (idempotent)."""
    with _REGISTRY_LOCK:
        try:
            registry: dict = {}
            if _REGISTRY_PATH.exists():
                registry = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
            if version in registry:
                return
            registry[version] = {
                "strategy": strategy_name,
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "strategy_params": strategy_params or {},
                "settings": settings or {},
            }
            _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            _REGISTRY_PATH.write_text(
                json.dumps(registry, indent=2, default=str), encoding="utf-8"
            )
            logger.info(
                "New logic version %s registered for strategy '%s'.",
                version, strategy_name,
            )
        except Exception as exc:  # never let bookkeeping break trading
            logger.warning("logic version registry update failed: %s", exc)


def load_registry() -> dict:
    """Return the {version: details} registry (empty dict if none yet)."""
    try:
        if _REGISTRY_PATH.exists():
            return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("logic version registry read failed: %s", exc)
    return {}
