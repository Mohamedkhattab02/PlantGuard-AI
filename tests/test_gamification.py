"""Tests for the per-user gamification engine (ROADMAP [1.1]-[1.3])."""

from pathlib import Path

from plantguard.services.gamification import GameState, GamificationService
from plantguard.services.store import Store


def _service() -> GamificationService:
    # A Store with a missing key file → disabled (no Firebase, no network).
    return GamificationService(Store(Path("__no_such_key__.json"), ""))


def test_complete_awards_points_once_per_day():
    svc = _service()
    state = GameState()
    ok, pts = svc.complete(state, "research_query")
    assert ok and pts == 20 and state.points == 20

    ok_again, pts_again = svc.complete(state, "research_query")
    assert not ok_again and pts_again == 0 and state.points == 20


def test_unknown_task_is_ignored():
    svc = _service()
    state = GameState()
    assert svc.complete(state, "nope") == (False, 0)


def test_master_achievement_when_all_done():
    svc = _service()
    state = GameState()
    for task in ("upload_image", "check_sensors", "research_query"):
        svc.complete(state, task)
    assert state.points == 45
    assert "🏆 Master!" in state.achievements


def test_level_tracks_points():
    svc = _service()
    state = GameState()
    state.points = 0
    svc.complete(state, "research_query")  # 20 → level 1
    assert state.level == 1


def test_manual_reset_clears_completion_keeps_points():
    svc = _service()
    state = GameState()
    svc.complete(state, "upload_image")
    svc.reset(state)
    assert state.completed == {}


def test_daily_reset_clears_completion():
    svc = _service()
    state = GameState()
    svc.complete(state, "upload_image")
    state.last_reset = "2000-01-01"
    svc._apply_daily_reset(state)
    assert state.completed == {}


def test_state_roundtrip():
    state = GameState(user_id="alice", points=30, level=1)
    restored = GameState.from_dict(state.to_dict())
    assert restored.user_id == "alice" and restored.points == 30
