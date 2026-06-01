"""Curated treatment knowledge base (ROADMAP feature [2.9]).

A small, citable table of common diseases with organic and chemical options and
indicative dosages. It grounds the LLM treatment generation so advice is
consistent and actionable instead of free-text only. Always defer to local
regulations and product labels for exact rates.
"""

from __future__ import annotations

# Keyed by (plant_lower, disease_keyword). Disease keyword is matched as a
# substring of the detected disease, so "Late Blight" matches "late blight".
KB: dict[tuple[str, str], dict] = {
    ("tomato", "late blight"): {
        "organic": "Copper-based fungicide (e.g. copper hydroxide) every 7–10 days.",
        "chemical": "Chlorothalonil 0.2% or mancozeb 0.25%, repeat after rain.",
        "cultural": "Remove infected foliage; improve airflow; avoid overhead irrigation.",
    },
    ("tomato", "early blight"): {
        "organic": "Neem oil or copper soap weekly at first symptoms.",
        "chemical": "Mancozeb 0.25% or azoxystrobin per label.",
        "cultural": "Mulch to prevent soil splash; rotate crops 2–3 years.",
    },
    ("potato", "late blight"): {
        "organic": "Copper hydroxide preventively; destroy volunteer plants.",
        "chemical": "Mancozeb / cymoxanil per label on a 7-day cycle in wet weather.",
        "cultural": "Hill soil over tubers; harvest in dry conditions.",
    },
    ("apple", "scab"): {
        "organic": "Sulfur or potassium bicarbonate from green-tip stage.",
        "chemical": "Captan or myclobutanil at 7–10 day intervals in spring.",
        "cultural": "Rake and destroy fallen leaves to reduce inoculum.",
    },
    ("apple", "cedar apple rust"): {
        "organic": "Remove nearby junipers/galls; sulfur sprays.",
        "chemical": "Myclobutanil from pink bud through early summer.",
        "cultural": "Plant resistant cultivars where rust pressure is high.",
    },
    ("grape", "black rot"): {
        "organic": "Copper or sulfur sprays from early shoot growth.",
        "chemical": "Mancozeb or myclobutanil pre-bloom and post-bloom.",
        "cultural": "Remove mummified berries and infected canes during pruning.",
    },
    ("corn (maize)", "common rust"): {
        "organic": "Usually not needed; choose resistant hybrids.",
        "chemical": "Azoxystrobin / propiconazole if severe before tasseling.",
        "cultural": "Avoid excessive nitrogen; ensure good spacing.",
    },
}


def lookup(plant: str, disease: str) -> dict | None:
    """Return the KB entry matching ``plant`` + a disease keyword, if any."""
    p = (plant or "").strip().lower()
    d = (disease or "").strip().lower()
    for (kb_plant, kb_kw), entry in KB.items():
        if kb_plant == p and kb_kw in d:
            return entry
    return None


def kb_context(plant: str, disease: str) -> str:
    """Format a KB entry as grounding text for the LLM prompt (or '')."""
    entry = lookup(plant, disease)
    if not entry:
        return ""
    lines = [f"Reference treatment guidance for {plant} / {disease}:"]
    for key in ("organic", "chemical", "cultural"):
        if entry.get(key):
            lines.append(f"- {key.capitalize()}: {entry[key]}")
    return "\n".join(lines)
