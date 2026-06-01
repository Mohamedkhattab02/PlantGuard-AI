"""Image disease classifier (ROADMAP [5.3], [5.2], [4.3]).

- **Label parsing** is backed by a known-plant table + overrides instead of the
  fragile "first word is the plant" heuristic ([5.3]).
- **Confidence threshold + top-3** results so low-confidence guesses are flagged
  rather than asserted ([5.2]).
- **Upload validation** caps image size and rejects non-images ([4.3]).

The model is loaded lazily with an explicit processor because transformers 5.x
can't auto-detect this older model's image processor.
"""

from __future__ import annotations

import logging
import re

from PIL import Image

log = logging.getLogger(__name__)

# Known PlantVillage crops, matched longest-first so "Corn (Maize)" wins over
# "Corn" and multi-word crops aren't truncated to their first token.
KNOWN_PLANTS = [
    "Corn (Maize)", "Corn", "Bell Pepper", "Pepper bell", "Pepper, bell",
    "Apple", "Blueberry", "Cherry", "Grape", "Orange", "Peach", "Potato",
    "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato",
]
_PLANTS_SORTED = sorted(KNOWN_PLANTS, key=len, reverse=True)

# Labels whose plant is NOT the first word need an explicit mapping.
OVERRIDES: dict[str, tuple[str, str, bool]] = {
    "cedar apple rust": ("Apple", "Cedar Apple Rust", False),
    "orange haunglongbing (citrus greening)": ("Orange", "Huanglongbing (Citrus Greening)", False),
}


def parse_label(label: str) -> tuple[str, str, bool]:
    """Parse a model label into ``(plant, disease, is_healthy)``.

    Handles "Tomato with Late Blight", "Healthy Apple", "Apple Scab",
    "Corn (Maize) with Common Rust" and the overridden tricky cases.
    """
    label = (label or "").strip()
    low = label.lower()

    if low in OVERRIDES:
        return OVERRIDES[low]

    if low.startswith("healthy"):
        plant = re.sub(r"^healthy\s+", "", label, flags=re.IGNORECASE)
        plant = re.sub(r"\s+plant$", "", plant, flags=re.IGNORECASE).strip() or "Plant"
        return plant, "Healthy", True

    if " with " in label:
        plant, disease = label.split(" with ", 1)
        return plant.strip(), disease.strip(), False

    for plant in _PLANTS_SORTED:
        if low.startswith(plant.lower()):
            disease = label[len(plant):].strip(" -,") or label
            return plant, disease, False

    parts = label.split()
    plant = parts[0] if parts else "Plant"
    disease = " ".join(parts[1:]).strip() or label
    return plant, disease, False


class ImageService:
    """Lazily-loaded MobileNetV2 disease classifier."""

    def __init__(self, settings):
        self.settings = settings
        self._clf = None
        self.id2label: dict = {}

    def _ensure_loaded(self) -> None:
        if self._clf is not None:
            return
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            pipeline,
        )

        log.info("Loading image classifier: %s", self.settings.image_model)
        model = AutoModelForImageClassification.from_pretrained(self.settings.image_model)
        try:
            processor = AutoImageProcessor.from_pretrained(self.settings.image_model)
        except Exception:  # noqa: BLE001
            from transformers import MobileNetV2ImageProcessor

            processor = MobileNetV2ImageProcessor.from_pretrained(self.settings.image_model)

        device = 0 if torch.cuda.is_available() else -1
        self._clf = pipeline(
            "image-classification", model=model, image_processor=processor, device=device
        )
        self.id2label = getattr(model.config, "id2label", {})
        log.info("Image classifier ready (%d classes).", len(self.id2label))

    def validate_image(self, img: Image.Image | None) -> Image.Image:
        """Reject non-images and downscale very large uploads ([4.3])."""
        if img is None:
            raise ValueError("No image provided.")
        if not isinstance(img, Image.Image):
            raise ValueError("Uploaded file is not a valid image.")
        max_dim = self.settings.max_image_dim
        if max(img.size) > max_dim:
            img = img.copy()
            img.thumbnail((max_dim, max_dim))
        return img.convert("RGB")

    def classify(self, img: Image.Image, top_k: int = 3) -> list[dict]:
        self._ensure_loaded()
        return self._clf(img, top_k=top_k)

    def diagnose(self, img: Image.Image | None) -> dict:
        """Full diagnosis result with top-3 and a low-confidence flag ([5.2])."""
        img = self.validate_image(img)
        preds = self.classify(img, top_k=3)
        if not preds:
            raise ValueError("Could not classify the image.")

        top = preds[0]
        plant, disease, healthy = parse_label(top["label"])
        confidence = float(top.get("score", 0.0))
        return {
            "plant": plant,
            "disease": disease,
            "healthy": healthy,
            "confidence": confidence,
            "low_confidence": confidence < self.settings.confidence_threshold,
            "top_k": [(p["label"], float(p["score"])) for p in preds],
            "image": img,
        }
