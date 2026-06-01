"""Per-user, persistent gamification engine (ROADMAP [1.1], [1.2], [1.3]).

The original code kept a single module-level ``game_system`` instance, so with
``share=True`` every visitor mutated the **same** points and level. Here all
state lives in a plain :class:`GameState` dataclass that the UI stores per
session (``gr.State``) and that is persisted per username to Firebase.

Daily missions reset once per calendar day; cumulative points, level and
achievements persist so re-completing missions each day keeps rewarding users.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field, asdict

from ..i18n import t
from .store import Store

log = logging.getLogger(__name__)

POINTS_PER_LEVEL = 50

# Mission catalogue. ``id -> metadata``. Names are localised in :func:`status_md`.
TASKS: dict[str, dict] = {
    "upload_image": {"points": 10, "icon": "🖼️", "en": "Upload Disease Image", "he": "העלאת תמונת מחלה"},
    "check_sensors": {"points": 15, "icon": "📊", "en": "Check Sensor Data", "he": "בדיקת נתוני חיישנים"},
    "research_query": {"points": 20, "icon": "🔍", "en": "Perform Research Query", "he": "ביצוע שאילתת מחקר"},
}


@dataclass
class GameState:
    """Serializable per-user game state (safe to keep in ``gr.State``)."""

    user_id: str = "guest"
    points: int = 0
    level: int = 1
    completed: dict = field(default_factory=dict)  # task_id -> bool
    achievements: list = field(default_factory=list)
    last_reset: str = field(default_factory=lambda: dt.date.today().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "GameState":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class GamificationService:
    """Mutates :class:`GameState` objects and persists them via :class:`Store`."""

    def __init__(self, store: Store):
        self.store = store

    # -- persistence ------------------------------------------------------
    def load(self, user_id: str = "guest") -> GameState:
        """Load a user's state (or a fresh one), applying the daily reset."""
        raw = self.store.get(f"gamification/{user_id}") if user_id else None
        state = GameState.from_dict(raw) if raw else GameState(user_id=user_id or "guest")
        state.user_id = user_id or "guest"
        self._apply_daily_reset(state)
        return state

    def save(self, state: GameState) -> None:
        if state.user_id and state.user_id != "guest":
            self.store.set(f"gamification/{state.user_id}", state.to_dict())

    # -- logic ------------------------------------------------------------
    def _apply_daily_reset(self, state: GameState) -> None:
        today = dt.date.today().isoformat()
        if state.last_reset != today:
            state.completed = {}
            state.last_reset = today

    def complete(self, state: GameState, task_id: str) -> tuple[bool, int]:
        """Award points for ``task_id`` once per day. Returns (success, points)."""
        if task_id not in TASKS or state.completed.get(task_id):
            return False, 0

        state.completed[task_id] = True
        pts = TASKS[task_id]["points"]
        state.points += pts

        new_level = (state.points // POINTS_PER_LEVEL) + 1
        if new_level > state.level:
            state.level = new_level
            state.achievements.append(f"🎉 Level {state.level}!")

        if all(state.completed.get(tid) for tid in TASKS) and "🏆 Master!" not in state.achievements:
            state.achievements.append("🏆 Master!")

        self.save(state)
        return True, pts

    def reset(self, state: GameState) -> str:
        """Manual reset of the daily missions (keeps the user id)."""
        state.completed = {}
        state.last_reset = dt.date.today().isoformat()
        self.save(state)
        return "🔄 Missions reset!"

    def status_md(self, state: GameState, lang: str = "he") -> str:
        lines = [
            f"### 🎮 {t('missions_header', lang)}",
            f"**{t('points_label', lang)}:** {state.points} 🌟 | **Level:** {state.level}",
            "",
        ]
        for tid, meta in TASKS.items():
            mark = "✅" if state.completed.get(tid) else "⬜"
            name = meta.get(lang, meta["en"])
            lines.append(f"{mark} {meta['icon']} **{name}** — {meta['points']}pts")
        if state.achievements:
            lines.append("\n### 🏆 Achievements")
            lines += [f"- {a}" for a in state.achievements]
        return "\n".join(lines)
