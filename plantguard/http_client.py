"""Shared HTTP session with retry/backoff (see ROADMAP [3.1]).

The IoT backend runs on a Render free dyno that cold-starts (~30-60 s), so the
first request after idle frequently times out. A retrying session with
exponential backoff turns that transient failure into a short wait.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Return a process-wide session that retries idempotent failures."""
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=1.5,  # waits ~1.5s, 3s, 6s, 12s between attempts
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _session = s
    return _session


def get_json(url: str, params: dict | None = None, timeout: float = 45.0) -> Any:
    """GET ``url`` and parse JSON, retrying transient errors."""
    resp = get_session().get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
