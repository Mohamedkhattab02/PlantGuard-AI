"""Agronomic threshold alerting (ROADMAP feature [2.2]).

Pure, dependency-free rules over the aggregated sensor stats so they are trivial
to unit-test. The Dashboard surfaces the results; a notifier (email/Telegram)
can be layered on top later.
"""

from __future__ import annotations

from dataclasses import dataclass

# Severity ordering for sorting/printing.
_LEVELS = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Alert:
    level: str  # "critical" | "warning" | "info"
    sensor: str
    message: str

    @property
    def icon(self) -> str:
        return {"critical": "🔴", "warning": "🟠", "info": "🟢"}.get(self.level, "ℹ️")


def evaluate(reduced: dict, settings) -> list[Alert]:
    """Map aggregated stats (``{sensor: {avg,min,max,...}}``) to alerts.

    ``settings`` supplies the thresholds (soil_min/max, temp_min/max, ...).
    """
    alerts: list[Alert] = []
    if not reduced:
        return alerts

    rules = {
        "soil": (settings.soil_min, settings.soil_max, "Soil moisture"),
        "temperature": (settings.temp_min, settings.temp_max, "Temperature"),
        "humidity": (settings.humidity_min, settings.humidity_max, "Humidity"),
    }

    for sensor, (lo, hi, label) in rules.items():
        stats = reduced.get(sensor)
        if not stats:
            continue
        avg = stats.get("avg")
        if avg is None:
            continue
        if avg < lo:
            sev = "critical" if sensor == "soil" else "warning"
            alerts.append(Alert(sev, sensor, f"{label} low ({avg}) — below {lo}."))
        elif avg > hi:
            sev = "critical" if sensor == "temperature" else "warning"
            alerts.append(Alert(sev, sensor, f"{label} high ({avg}) — above {hi}."))
        else:
            alerts.append(Alert("info", sensor, f"{label} normal ({avg})."))

    alerts.sort(key=lambda a: _LEVELS.get(a.level, 9))
    return alerts


def alerts_md(alerts: list[Alert]) -> str:
    """Render alerts as Markdown for the dashboard."""
    if not alerts:
        return "_No sensor data to evaluate yet._"
    return "\n".join(f"{a.icon} **{a.sensor}** — {a.message}" for a in alerts)
