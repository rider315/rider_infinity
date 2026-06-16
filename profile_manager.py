"""
profile_manager.py — Configuration Profile Management for Rider Infinity.

Allows users to save, load, switch, and share complete strategy
configurations (strategy + risk params + session filters + trailing
profiles).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class TradingProfile:
    """
    A complete, self-contained trading configuration.

    Parameters
    ----------
    name : str
        Human-readable name, e.g. ``'London Scalp'``.
    strategy_name : str
        Key from ``STRATEGY_REGISTRY`` (e.g. ``'ema'``).
    """

    name: str
    strategy_name: str

    # Identifiers
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""

    # Strategy parameters
    strategy_params: dict[str, Any] = field(default_factory=dict)

    # Risk parameters
    risk_params: dict[str, Any] = field(default_factory=lambda: {
        "max_risk_per_trade": 1.0,
        "max_daily_drawdown": 5.0,
        "max_open_positions": 5,
        "max_spread_points": 30,
        "default_sl_pips": 50,
        "default_tp_pips": 100,
    })

    # Session filter (None = all sessions)
    session_filter: list[str] | None = None

    # Symbols
    symbols: list[str] = field(default_factory=lambda: ["EURUSD"])

    # Timeframe
    timeframe: str = "H1"

    # Lot sizing
    lot_mode: str = "auto"  # 'fixed' or 'auto'
    lot_size: float = 0.01  # used when lot_mode='fixed'

    # Trailing stop
    trailing_profile: str | None = None
    trailing_params: dict[str, Any] = field(default_factory=dict)

    # Automation rules (serialised)
    automation_rules: list[dict] | None = None

    # Metadata
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    is_active: bool = False


# ----------------------------------------------------------------------
# Profile manager
# ----------------------------------------------------------------------

class ProfileManager:
    """
    Manage trading profiles — save, load, switch, share.

    Parameters
    ----------
    profiles_dir : str
        Directory for profile JSON files (default ``data/profiles``).
    """

    def __init__(self, profiles_dir: str = "data/profiles") -> None:
        self._dir = Path(profiles_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("ProfileManager — directory: %s", self._dir.resolve())

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_profile(self, profile: TradingProfile) -> bool:
        """Persist a profile to disk."""
        profile.updated_at = datetime.utcnow().isoformat()
        path = self._dir / f"{profile.profile_id}.json"
        try:
            path.write_text(
                json.dumps(asdict(profile), indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Profile saved: '%s' → %s", profile.name, path)
            return True
        except Exception:
            logger.exception("Failed to save profile '%s'.", profile.name)
            return False

    def load_profile(self, profile_id: str) -> TradingProfile | None:
        """Load a profile by ID."""
        path = self._dir / f"{profile_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._dict_to_profile(data)
        except Exception:
            logger.exception("Failed to load profile %s.", profile_id)
            return None

    def load_profile_by_name(self, name: str) -> TradingProfile | None:
        """Find a profile by name (case-insensitive)."""
        for profile in self.list_profiles():
            if profile.name.lower() == name.lower():
                return profile
        return None

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile from disk."""
        path = self._dir / f"{profile_id}.json"
        if path.exists():
            path.unlink()
            logger.info("Profile deleted: %s", profile_id)
            return True
        return False

    def list_profiles(self) -> list[TradingProfile]:
        """Return all saved profiles."""
        profiles: list[TradingProfile] = []
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles.append(self._dict_to_profile(data))
            except Exception:
                logger.warning("Skipping corrupt profile: %s", path.name)
        return profiles

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def get_active_profile(self) -> TradingProfile | None:
        """Return the currently active profile, or None."""
        for profile in self.list_profiles():
            if profile.is_active:
                return profile
        return None

    def activate_profile(self, profile_id: str) -> bool:
        """
        Set one profile as active, deactivating all others.
        """
        profiles = self.list_profiles()
        found = False
        for p in profiles:
            if p.profile_id == profile_id:
                p.is_active = True
                found = True
            else:
                p.is_active = False
            self.save_profile(p)

        if found:
            logger.info("Profile activated: %s", profile_id)
        return found

    def get_profile_for_session(
        self, session_name: str,
    ) -> TradingProfile | None:
        """Auto-select a profile based on the current trading session."""
        for profile in self.list_profiles():
            if (
                profile.session_filter
                and session_name.lower() in [
                    s.lower() for s in profile.session_filter
                ]
            ):
                return profile
        return None

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_profile(self, profile_id: str, filepath: str) -> bool:
        """Export a profile to a shareable JSON file."""
        profile = self.load_profile(profile_id)
        if profile is None:
            return False
        try:
            Path(filepath).write_text(
                json.dumps(asdict(profile), indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Profile exported: '%s' → %s", profile.name, filepath)
            return True
        except Exception:
            logger.exception("Export failed.")
            return False

    def import_profile(self, filepath: str) -> TradingProfile | None:
        """Import a profile from a JSON file."""
        try:
            data = json.loads(Path(filepath).read_text(encoding="utf-8"))
            # Assign new ID to avoid conflicts
            data["profile_id"] = str(uuid.uuid4())
            data["is_active"] = False
            profile = self._dict_to_profile(data)
            self.save_profile(profile)
            logger.info("Profile imported: '%s'", profile.name)
            return profile
        except Exception:
            logger.exception("Import failed from %s.", filepath)
            return None

    # ------------------------------------------------------------------
    # Defaults factory
    # ------------------------------------------------------------------

    def create_default_profiles(self) -> list[TradingProfile]:
        """
        Create built-in default profiles.

        Returns
        -------
        list[TradingProfile]
            Three pre-configured profiles.
        """
        profiles = [
            TradingProfile(
                name="Conservative Trend",
                description="H1 EMA crossover with low risk — steady trend following.",
                strategy_name="ema",
                strategy_params={"fast_period": 12, "slow_period": 26},
                risk_params={
                    "max_risk_per_trade": 0.5,
                    "max_daily_drawdown": 3.0,
                    "max_open_positions": 3,
                    "max_spread_points": 25,
                    "default_sl_pips": 50,
                    "default_tp_pips": 100,
                },
                symbols=["EURUSD", "GBPUSD", "USDJPY"],
                timeframe="H1",
                lot_mode="auto",
                trailing_profile="breakeven",
                trailing_params={"breakeven_pips": 25, "trail_pips": 15},
                session_filter=["london", "new_york"],
            ),
            TradingProfile(
                name="Aggressive Scalp",
                description="M5 RSI scalping with higher risk — quick entries.",
                strategy_name="rsi",
                strategy_params={"period": 7, "overbought": 75, "oversold": 25},
                risk_params={
                    "max_risk_per_trade": 1.5,
                    "max_daily_drawdown": 5.0,
                    "max_open_positions": 5,
                    "max_spread_points": 15,
                    "default_sl_pips": 25,
                    "default_tp_pips": 50,
                },
                symbols=["EURUSD"],
                timeframe="M5",
                lot_mode="auto",
                trailing_profile="step",
                trailing_params={"step_pips": 10, "activation_pips": 15},
                session_filter=["london"],
            ),
            TradingProfile(
                name="Range Grid",
                description="M15 grid strategy for ranging pairs.",
                strategy_name="grid",
                strategy_params={
                    "grid_size_pips": 20,
                    "num_levels": 5,
                    "max_open_orders": 8,
                },
                risk_params={
                    "max_risk_per_trade": 0.3,
                    "max_daily_drawdown": 4.0,
                    "max_open_positions": 10,
                    "max_spread_points": 20,
                    "default_sl_pips": 40,
                    "default_tp_pips": 20,
                },
                symbols=["EURGBP", "AUDNZD", "EURCHF"],
                timeframe="M15",
                lot_mode="fixed",
                lot_size=0.01,
            ),
        ]

        for p in profiles:
            self.save_profile(p)
            logger.info("Default profile created: '%s'", p.name)

        return profiles

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_profile(data: dict) -> TradingProfile:
        """Convert a dict to a TradingProfile."""
        return TradingProfile(
            profile_id=data.get("profile_id", str(uuid.uuid4())),
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            strategy_name=data.get("strategy_name", "ema"),
            strategy_params=data.get("strategy_params", {}),
            risk_params=data.get("risk_params", {}),
            session_filter=data.get("session_filter"),
            symbols=data.get("symbols", ["EURUSD"]),
            timeframe=data.get("timeframe", "H1"),
            lot_mode=data.get("lot_mode", "auto"),
            lot_size=data.get("lot_size", 0.01),
            trailing_profile=data.get("trailing_profile"),
            trailing_params=data.get("trailing_params", {}),
            automation_rules=data.get("automation_rules"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            is_active=data.get("is_active", False),
        )
