"""Per-user diagnosis history (ROADMAP feature [2.3]).

Keeps an in-memory log per user (always available) and mirrors it to Firebase
when persistence is enabled, so history survives restarts for named users.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from .store import Store

log = logging.getLogger(__name__)

MAX_ENTRIES = 50


@dataclass
class DiagnosisEntry:
    plant: str
    disease: str
    confidence: float
    healthy: bool
    image_path: str = ""
    timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def caption(self) -> str:
        status = "✅ Healthy" if self.healthy else f"⚠️ {self.disease}"
        return f"{self.plant} — {status} ({self.confidence:.0%}) · {self.timestamp}"


class HistoryService:
    def __init__(self, store: Store):
        self.store = store
        self._mem: dict[str, list[DiagnosisEntry]] = defaultdict(list)

    def record(self, user_id: str, entry: DiagnosisEntry) -> None:
        uid = user_id or "guest"
        self._mem[uid].insert(0, entry)
        self._mem[uid] = self._mem[uid][:MAX_ENTRIES]
        if uid != "guest":
            self.store.set(f"history/{uid}", [e.to_dict() for e in self._mem[uid]])

    def list(self, user_id: str) -> list[DiagnosisEntry]:
        uid = user_id or "guest"
        if uid in self._mem:
            return self._mem[uid]
        # Lazily hydrate from Firebase for named users.
        raw = self.store.get(f"history/{uid}") if uid != "guest" else None
        if raw:
            self._mem[uid] = [DiagnosisEntry(**d) for d in raw if isinstance(d, dict)]
        return self._mem[uid]
