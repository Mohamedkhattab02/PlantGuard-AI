"""Tests for agronomic alerting (ROADMAP feature [2.2])."""

from types import SimpleNamespace

from plantguard.services.alerts import evaluate

SETTINGS = SimpleNamespace(
    soil_min=20, soil_max=80,
    temp_min=10, temp_max=35,
    humidity_min=30, humidity_max=85,
)


def test_low_soil_is_critical():
    alerts = evaluate({"soil": {"avg": 10}}, SETTINGS)
    assert alerts[0].sensor == "soil" and alerts[0].level == "critical"


def test_high_temperature_is_critical():
    alerts = evaluate({"temperature": {"avg": 42}}, SETTINGS)
    assert any(a.sensor == "temperature" and a.level == "critical" for a in alerts)


def test_normal_reading_is_info():
    alerts = evaluate({"humidity": {"avg": 50}}, SETTINGS)
    assert alerts[0].level == "info"


def test_empty_input():
    assert evaluate({}, SETTINGS) == []


def test_alerts_sorted_critical_first():
    reduced = {"humidity": {"avg": 50}, "soil": {"avg": 5}}
    alerts = evaluate(reduced, SETTINGS)
    assert alerts[0].level == "critical"
