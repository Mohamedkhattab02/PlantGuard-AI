"""One-click PDF diagnosis reports (ROADMAP feature [2.4]).

Built with ``fpdf2``. The core fonts are Latin-1, so non-Latin text is
transliterated to '?'; diagnosis fields (plant/disease/treatment) are English,
so reports render cleanly.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from fpdf import FPDF

log = logging.getLogger(__name__)


def _latin(text: str) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


def build_report_pdf(entry: dict, out_dir: Path) -> str:
    """Render a diagnosis ``entry`` to a PDF and return the file path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _latin("PlantGuard AI - Diagnosis Report"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, _latin(dt.datetime.now().strftime("%Y-%m-%d %H:%M")), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    image_path = entry.get("image_path")
    if image_path and Path(image_path).exists():
        try:
            pdf.image(image_path, w=80)
            pdf.ln(4)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not embed image in report: %s", exc)

    def field(label: str, value: str) -> None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(40, 8, _latin(label))
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, _latin(value))

    field("Plant:", entry.get("plant", "-"))
    field("Diagnosis:", entry.get("disease", "-"))
    field("Confidence:", f"{entry.get('confidence', 0):.1%}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _latin("Treatment"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _latin(entry.get("treatment", "-")))

    if entry.get("sources"):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _latin("Sources"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin(entry.get("sources", "")))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"plantguard_report_{stamp}.pdf"
    pdf.output(str(out_path))
    log.info("Report written to %s", out_path)
    return str(out_path)
