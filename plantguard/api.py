"""FastAPI service exposing the PlantGuard engines (ROADMAP feature [2.6]).

Decouples the AI/data logic from the Gradio UI so the same engines can power a
mobile app, automations, or third-party integrations.

Run with::

    uvicorn plantguard.api:create_app --factory --port 8000
"""

from __future__ import annotations

import io
import logging

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel

from .app import Services, build_services
from .config import configure_logging, load_settings
from .services import alerts as alerts_svc
from .services.iot import reduce_sensor_data, map_sensor_data, load_iot_json

log = logging.getLogger(__name__)


class ResearchRequest(BaseModel):
    question: str
    top_k: int = 2


def create_api(svc: Services) -> FastAPI:
    """Build a FastAPI app around already-constructed :class:`Services`."""
    api = FastAPI(title="PlantGuard AI API", version="1.0.0")
    s = svc.settings

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "firebase": svc.store.ok}

    @api.post("/diagnose")
    async def diagnose(file: UploadFile = File(...)) -> dict:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        result = svc.image.diagnose(img)
        payload = {k: v for k, v in result.items() if k != "image"}
        if not result["healthy"]:
            payload["advice"] = svc.rag.explain_and_treat(result["plant"], result["disease"])
        return payload

    @api.post("/research")
    def research(req: ResearchRequest) -> dict:
        return svc.rag.answer(req.question, req.top_k)

    @api.get("/sensors/{feed}")
    def sensors(feed: str, limit: int = 10) -> dict:
        return {"feed": feed, "data": svc.iot.get_history(feed, limit)}

    @api.get("/dashboard")
    def dashboard() -> dict:
        reduced = reduce_sensor_data(map_sensor_data(load_iot_json(s.iot_dir)))
        alerts = alerts_svc.evaluate(reduced, s)
        return {
            "stats": reduced,
            "alerts": [a.__dict__ for a in alerts],
        }

    return api


def create_app() -> FastAPI:
    """Factory for ``uvicorn --factory`` (builds services from the environment)."""
    settings = load_settings()
    configure_logging(settings.debug)
    return create_api(build_services(settings))
