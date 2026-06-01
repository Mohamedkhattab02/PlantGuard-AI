"""Smoke tests: modules import and the full Gradio UI graph builds.

This exercises every Gradio component call in ``build_app`` without loading the
model, hitting the network, or needing API keys — so it catches component-API
mismatches that pure-logic tests cannot.
"""

from pathlib import Path

import gradio as gr

from plantguard.app import Services, build_app
from plantguard.config import Settings
from plantguard.services.gamification import GamificationService
from plantguard.services.store import Store


def _lite_services() -> Services:
    settings = Settings(
        gemini_api_key="test-key",
        database_url="",
        base_url="",
        firebase_key_path=Path("__no_such_key__.json"),
    )
    store = Store(settings.firebase_key_path, settings.database_url)
    # image/rag/iot/history are only referenced inside (un-invoked) handler
    # closures during layout, so None is fine for a build-only smoke test.
    return Services(
        settings=settings,
        store=store,
        image=None,  # type: ignore[arg-type]
        rag=None,  # type: ignore[arg-type]
        iot=None,  # type: ignore[arg-type]
        game=GamificationService(store),
        history=None,  # type: ignore[arg-type]
    )


def test_api_module_imports():
    import plantguard.api as api  # noqa: F401

    assert hasattr(api, "create_app")


def test_build_app_returns_blocks():
    demo = build_app(_lite_services())
    assert isinstance(demo, gr.Blocks)
