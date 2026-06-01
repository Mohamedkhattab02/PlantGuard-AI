"""IoT sensor service: history fetch, concurrent dashboard load, MapReduce.

Implements ROADMAP [3.1] (retrying session), [2.4] (concurrent feed fetch) and
[3.4] (the MapReduce aggregate is now returned for the UI instead of being
written to Firebase and forgotten). The map/reduce functions are pure and
unit-tested.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..http_client import get_json
from .store import Store

log = logging.getLogger(__name__)

FEEDS = ("temperature", "humidity", "soil")


# --------------------------------------------------------------------------
# Pure MapReduce over local IoT JSON exports (testable without I/O mocks).
# --------------------------------------------------------------------------
def map_sensor_data(records: list[dict]) -> list[tuple[str, float]]:
    """Flatten records into ``(sensor, value)`` pairs."""
    mapped: list[tuple[str, float]] = []
    for rec in records:
        try:
            raw = rec.get("value")
            if isinstance(raw, str):
                raw = json.loads(raw)
            if not isinstance(raw, dict):
                continue
            for sensor in FEEDS:
                if sensor in raw:
                    mapped.append((sensor, float(raw[sensor])))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return mapped


def reduce_sensor_data(mapped: list[tuple[str, float]]) -> dict[str, dict]:
    """Aggregate mapped pairs into ``{sensor: {count, min, max, avg}}``."""
    if not mapped:
        return {}
    buckets: dict[str, list[float]] = defaultdict(list)
    for sensor, val in mapped:
        buckets[sensor].append(val)
    return {
        sensor: {
            "count": len(vals),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "avg": round(sum(vals) / len(vals), 2),
        }
        for sensor, vals in buckets.items()
    }


def load_iot_json(folder: Path) -> list[dict]:
    """Read every ``*.json`` export in ``folder`` into a flat record list."""
    records: list[dict] = []
    if not folder.is_dir():
        log.warning("IoT folder not found: %s", folder)
        return records
    for path in folder.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
            records.extend(data if isinstance(data, list) else [data])
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read IoT file %s: %s", path, exc)
    log.info("Loaded %d IoT records.", len(records))
    return records


def run_mapreduce(folder: Path) -> dict[str, dict]:
    """Convenience: load → map → reduce. Returns the aggregated stats."""
    return reduce_sensor_data(map_sensor_data(load_iot_json(folder)))


# --------------------------------------------------------------------------
# Live service (talks to the Render REST backend).
# --------------------------------------------------------------------------
class IotService:
    def __init__(self, base_url: str, store: Store | None = None):
        self.base_url = base_url.rstrip("/")
        self.store = store

    def get_history(self, feed: str, limit: int) -> list:
        """Return raw sample values for a feed (used by the Sensor tab)."""
        data = get_json(f"{self.base_url}/history", {"feed": feed, "limit": limit})
        if "data" not in data:
            raise ValueError("Unexpected response from sensor server (no 'data').")
        return data["data"]

    def fetch_series(self, feed: str, limit: int) -> pd.DataFrame:
        """Return a time-sorted DataFrame (``created_at``, ``value``) for a feed."""
        data = get_json(f"{self.base_url}/history", {"feed": feed, "limit": limit})
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame(columns=["created_at", "value"])
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["value"]).sort_values("created_at")

    def fetch_all_series(self, limit: int) -> dict[str, pd.DataFrame]:
        """Fetch all feeds **concurrently** (ROADMAP [2.4])."""
        with ThreadPoolExecutor(max_workers=len(FEEDS)) as pool:
            futures = {feed: pool.submit(self._safe_series, feed, limit) for feed in FEEDS}
            return {feed: fut.result() for feed, fut in futures.items()}

    def _safe_series(self, feed: str, limit: int) -> pd.DataFrame:
        try:
            return self.fetch_series(feed, limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("Feed '%s' fetch failed: %s", feed, exc)
            return pd.DataFrame(columns=["created_at", "value"])
