"""Weather & agronomic context (ROADMAP feature [2.8]).

Uses the free **Open-Meteo** API, which needs no API key, so this works out of
the box. Returns ``None`` (handled gracefully by the UI) when offline.
"""

from __future__ import annotations

import logging

from ..http_client import get_json

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(city: str) -> dict | None:
    """Return current weather for ``city`` or ``None`` if unavailable."""
    try:
        geo = get_json(GEOCODE_URL, {"name": city, "count": 1, "language": "en"}, timeout=15)
        results = geo.get("results")
        if not results:
            return None
        loc = results[0]
        forecast = get_json(
            FORECAST_URL,
            {
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation",
            },
            timeout=15,
        )
        cur = forecast.get("current", {})
        return {
            "city": loc.get("name", city),
            "country": loc.get("country", ""),
            "temperature": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "precipitation": cur.get("precipitation"),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Weather lookup for '%s' failed: %s", city, exc)
        return None


def disease_risk(weather: dict | None) -> str:
    """Translate humidity/precipitation into a simple fungal-risk hint."""
    if not weather:
        return ""
    humidity = weather.get("humidity") or 0
    precip = weather.get("precipitation") or 0
    if humidity >= 80 or precip > 0:
        return "🍄 High humidity/rain — elevated risk of fungal disease. Inspect foliage."
    if humidity <= 30:
        return "🌵 Dry conditions — watch for water stress."
    return "🌤️ Conditions are moderate."


def weather_md(weather: dict | None) -> str:
    if not weather:
        return "_Weather unavailable._"
    return (
        f"**{weather['city']}, {weather['country']}** — "
        f"🌡️ {weather.get('temperature')}°C · "
        f"💧 {weather.get('humidity')}% · "
        f"🌧️ {weather.get('precipitation')} mm\n\n{disease_risk(weather)}"
    )
