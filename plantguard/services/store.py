"""Firebase Realtime Database wrapper with graceful degradation.

Implements ROADMAP [4.1]: the insecure auto-download of a service-account key
from a public Google Drive link is **removed**. The key must be provided via
``FIREBASE_KEY_PATH``; if it (or the SDK, or the URL) is missing, the app logs a
warning and runs fully without persistence instead of crashing or fetching
credentials from the internet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, db
except ImportError:  # pragma: no cover - optional dependency
    firebase_admin = None  # type: ignore[assignment]
    credentials = None  # type: ignore[assignment]
    db = None  # type: ignore[assignment]


class Store:
    """Thin wrapper around Firebase RTDB. ``ok`` is False when unavailable."""

    def __init__(self, key_path: Path, database_url: str):
        self.ok = False

        if firebase_admin is None:
            log.warning("firebase-admin not installed; running without persistence.")
            return
        if not key_path.exists():
            log.warning(
                "Firebase key not found at %s; running without persistence. "
                "(Insecure auto-download was removed — provide the key file yourself.)",
                key_path,
            )
            return
        if not database_url:
            log.warning("DATABASE_URL not set; running without persistence.")
            return

        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(key_path))
                firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            db.reference("health").set({"status": "ok"})
            self.ok = True
            log.info("Firebase connected.")
        except Exception as exc:  # noqa: BLE001 - never crash the app on telemetry
            log.warning("Firebase init failed (%s); continuing without persistence.", exc)

    def set(self, path: str, value: Any) -> bool:
        """Write ``value`` at ``path``; returns False (never raises) on failure."""
        if not self.ok:
            return False
        try:
            db.reference(path).set(value)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Firebase write to '%s' failed: %s", path, exc)
            return False

    def get(self, path: str) -> Any:
        """Read the value at ``path``; returns None on failure or when disabled."""
        if not self.ok:
            return None
        try:
            return db.reference(path).get()
        except Exception as exc:  # noqa: BLE001
            log.warning("Firebase read from '%s' failed: %s", path, exc)
            return None
