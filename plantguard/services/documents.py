"""PDF corpus loading — each file parsed exactly once (ROADMAP [2.5]).

The original code read every PDF twice (PyPDF2 for the keyword index and pypdf
for the TF-IDF retriever). Here we standardise on ``pypdf``, parse each file
once, and expose both the full document text and pre-computed chunks so every
downstream index reuses the same extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

log = logging.getLogger(__name__)


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Split normalized text into overlapping character windows."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    i = 0
    step = max(1, size - overlap)
    while i < len(text):
        chunks.append(text[i : i + size])
        i += step
    return chunks


def read_pdf(path: Path) -> str:
    """Extract all text from a PDF, returning '' on failure."""
    try:
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read PDF %s: %s", path, exc)
        return ""


@dataclass
class Corpus:
    """A parsed PDF library shared by every retriever."""

    texts: dict[int, str] = field(default_factory=dict)  # doc_id -> full text
    names: dict[int, str] = field(default_factory=dict)  # doc_id -> filename
    chunks: list[tuple[int, str]] = field(default_factory=list)  # (doc_id, chunk)

    def __bool__(self) -> bool:
        return bool(self.texts)


def load_corpus(folder: Path, chunk_size: int = 1200, overlap: int = 200) -> Corpus:
    """Read every ``*.pdf`` in ``folder`` once and build a :class:`Corpus`."""
    corpus = Corpus()
    if not folder.is_dir():
        log.warning("PDF folder not found: %s", folder)
        return corpus

    for doc_id, name in enumerate(sorted(p.name for p in folder.glob("*.pdf"))):
        text = read_pdf(folder / name)
        if not text.strip():
            continue
        corpus.texts[doc_id] = text
        corpus.names[doc_id] = name
        for chunk in chunk_text(text, chunk_size, overlap):
            corpus.chunks.append((doc_id, chunk))

    log.info("Loaded %d PDFs into %d chunks.", len(corpus.texts), len(corpus.chunks))
    return corpus
