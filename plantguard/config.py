"""Central configuration, settings, and logging for PlantGuard AI.

Every environment-driven knob lives here so the rest of the codebase imports
typed settings instead of sprinkling ``os.getenv()`` calls everywhere.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# UTF-8 console (Windows defaults to cp1252, which crashes on the emoji used
# throughout the UI/logs). Force UTF-8 so logging never raises.
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Load .env (replaces Colab secrets / hard-coded keys).
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    logging.getLogger(__name__).warning(
        "python-dotenv not installed; relying on the real environment."
    )

# Project root = the directory that contains this package.
BASE_DIR = Path(__file__).resolve().parents[1]


def _flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(env_value: str) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    p = Path(env_value)
    return p if p.is_absolute() else (BASE_DIR / p)


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-resolved runtime configuration."""

    # --- Secrets / endpoints ---
    gemini_api_key: str
    database_url: str
    base_url: str
    firebase_key_path: Path

    # --- Models ---
    gemini_model: str = "gemini-2.5-flash"
    embed_model: str = "models/text-embedding-004"
    image_model: str = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

    # --- Data folders ---
    articles_dir: Path = BASE_DIR / "articles"
    iot_dir: Path = BASE_DIR / "IOT_DETAILS"
    cache_dir: Path = BASE_DIR / ".cache"
    reports_dir: Path = BASE_DIR / ".cache" / "reports"

    # --- Launch behaviour (see [4.2] in ROADMAP) ---
    share: bool = False
    debug: bool = False
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    default_lang: str = "he"

    # --- Image diagnosis ---
    confidence_threshold: float = 0.45
    max_image_mb: float = 10.0
    max_image_dim: int = 1024

    # --- Gemini guardrails ---
    gemini_max_calls_per_min: int = 30

    # --- Agronomic alert thresholds (sensible defaults) ---
    soil_min: float = 20.0
    soil_max: float = 80.0
    temp_min: float = 10.0
    temp_max: float = 35.0
    humidity_min: float = 30.0
    humidity_max: float = 85.0

    # --- Weather context ---
    weather_default_city: str = "Tel Aviv"


def load_settings() -> Settings:
    """Build :class:`Settings` from the environment. Fails loudly on missing key."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Copy .env.example to .env and fill it in."
        )
    return Settings(
        gemini_api_key=key,
        database_url=os.getenv(
            "DATABASE_URL",
            "https://plant-disease-index-default-rtdb.firebaseio.com/",
        ),
        base_url=os.getenv("BASE_URL", "https://server-cloud-v645.onrender.com/").rstrip("/"),
        firebase_key_path=_resolve(os.getenv("FIREBASE_KEY_PATH", "firebase-key.json")),
        share=_flag("GRADIO_SHARE", False),
        debug=_flag("DEBUG", False),
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        default_lang=os.getenv("DEFAULT_LANG", "he"),
    )


def configure_logging(debug: bool = False) -> None:
    """Configure structured logging once for the whole process (see [3.3])."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "urllib3", "google", "matplotlib", "PIL", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
