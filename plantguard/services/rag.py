"""Retrieval-Augmented Generation engine (ROADMAP [5.1], [5.4], [2.1], [2.2], [4.4]).

Improvements over the prototype:

- **Semantic retrieval** via Gemini embeddings + cosine similarity, replacing the
  fixed keyword index ([5.1]); transparently falls back to TF-IDF if embeddings
  are unavailable.
- **Real ``system_instruction``** for the model instead of a fake user turn ([5.4]).
- **One structured call** returns explanation *and* treatment together ([2.1]).
- **Caching** of answers/treatments ([2.2]) and a **rate limiter** for Gemini ([4.4]).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque

import numpy as np

from .documents import Corpus
from .knowledge_base import kb_context

log = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are an academic research assistant specialising in plant diseases and "
    "image classification. Answer using ONLY the provided documents, cite the "
    "document name, and clearly state when information is missing."
)


# --------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` within ``period`` seconds."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > self.period:
            self._calls.popleft()
        if len(self._calls) < self.max_calls:
            self._calls.append(now)
            return True
        return False


# --------------------------------------------------------------------------
class TfidfIndex:
    """Lexical fallback retriever over corpus chunks."""

    def __init__(self, corpus: Corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.corpus = corpus
        self._chunks = corpus.chunks or [(0, "")]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c for _, c in self._chunks])

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
        from sklearn.metrics.pairwise import cosine_similarity

        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix).flatten()
        order = sims.argsort()[::-1][:top_k]
        return [(self._chunks[i][0], float(sims[i]), self._chunks[i][1]) for i in order]


class EmbeddingIndex:
    """Semantic retriever using Gemini embeddings + cosine similarity."""

    def __init__(self, corpus: Corpus, embed_model: str, cache_dir):
        import google.generativeai as genai

        self._genai = genai
        self.corpus = corpus
        self.embed_model = embed_model
        self._chunks = corpus.chunks
        self._matrix = self._build_matrix(cache_dir)

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):  # batch to keep payloads small
            batch = texts[start : start + 100]
            res = self._genai.embed_content(
                model=self.embed_model, content=batch, task_type=task_type
            )
            emb = res["embedding"]
            vectors.extend(emb if isinstance(emb[0], list) else [emb])
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)

    def _build_matrix(self, cache_dir) -> np.ndarray:
        texts = [c for _, c in self._chunks]
        digest = hashlib.md5(
            (self.embed_model + "|" + "|".join(texts)).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = cache_dir / f"embeddings_{digest}.npy"
        if cache_path.exists():
            try:
                arr = np.load(cache_path)
                if arr.shape[0] == len(texts):
                    log.info("Loaded cached embeddings (%s).", cache_path.name)
                    return arr
            except Exception:  # noqa: BLE001
                pass
        log.info("Embedding %d chunks with %s ...", len(texts), self.embed_model)
        arr = self._embed(texts, "retrieval_document")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, arr)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not cache embeddings: %s", exc)
        return arr

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
        q = self._embed([query], "retrieval_query")[0]
        sims = self._matrix @ q
        order = np.argsort(sims)[::-1][:top_k]
        return [(self._chunks[i][0], float(sims[i]), self._chunks[i][1]) for i in order]


# --------------------------------------------------------------------------
class RagService:
    def __init__(self, settings, corpus: Corpus):
        import google.generativeai as genai

        self.settings = settings
        self.corpus = corpus
        self._genai = genai
        self._limiter = RateLimiter(settings.gemini_max_calls_per_min)
        self._answer_cache: dict[tuple[str, int], dict] = {}
        self._treat_cache: dict[tuple[str, str], dict] = {}

        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={"temperature": 0.3, "top_p": 0.8, "max_output_tokens": 2048},
        )
        self.index = self._build_index()

    def _build_index(self):
        if not self.corpus.chunks:
            log.warning("Empty corpus — RAG will report no documents.")
            return TfidfIndex(self.corpus)
        try:
            return EmbeddingIndex(self.corpus, self.settings.embed_model, self.settings.cache_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("Embeddings unavailable (%s); falling back to TF-IDF.", exc)
            return TfidfIndex(self.corpus)

    # -- low-level generation --------------------------------------------
    def _generate(self, prompt, max_output_tokens: int = 512, json_mode: bool = False):
        if not self._limiter.allow():
            return "⏳ Rate limit reached — please wait a moment and try again."
        cfg = {"max_output_tokens": max_output_tokens}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        try:
            resp = self.model.generate_content(prompt, generation_config=cfg)
            return (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Gemini generation failed: %s", exc)
            return f"⚠️ Generation error: {exc}"

    def generate(self, prompt: str, max_output_tokens: int = 512) -> str:
        """Public ad-hoc generation helper."""
        return self._generate(prompt, max_output_tokens)

    # -- retrieval-grounded Q&A ------------------------------------------
    def _context(self, hits: list[tuple[int, float, str]], max_chars: int = 1500) -> str:
        parts = []
        for doc_id, score, chunk in hits:
            name = self.corpus.names.get(doc_id, f"doc-{doc_id}")
            parts.append(f"=== {name} (score {score:.3f}) ===\n{chunk[:max_chars]}")
        return "\n\n".join(parts)

    def answer(self, question: str, top_k: int = 2) -> dict:
        question = (question or "").strip()
        if not question:
            return {"answer": "⚠️ Please enter a question.", "sources": ""}

        key = (question.lower(), top_k)
        if key in self._answer_cache:
            return self._answer_cache[key]

        hits = self.index.search(question, top_k)
        if not hits or all(score <= 0 for _, score, _ in hits):
            result = {"answer": "❌ No relevant documents found.", "sources": ""}
            self._answer_cache[key] = result
            return result

        prompt = (
            "Answer the question using ONLY the documents below. Cite document "
            f"names and structure the answer clearly.\n\nQUESTION:\n{question}\n\n"
            f"DOCUMENTS:\n\n{self._context(hits)}\n\nANSWER:"
        )
        answer = self._generate(prompt, max_output_tokens=1024)
        sources = "\n".join(
            f"📄 {self.corpus.names.get(d, f'doc-{d}')} (score {s:.3f})" for d, s, _ in hits
        )
        result = {"answer": answer, "sources": sources}
        self._answer_cache[key] = result
        return result

    # -- merged explanation + treatment (one call) -----------------------
    def explain_and_treat(self, plant: str, disease: str) -> dict:
        """Return ``{explanation, severity, treatment[]}`` in a single call ([2.1])."""
        key = (plant.lower(), disease.lower())
        if key in self._treat_cache:
            return self._treat_cache[key]

        reference = kb_context(plant, disease)
        grounding = f"\n\nUse this reference guidance when relevant:\n{reference}" if reference else ""
        prompt = (
            f"For the plant '{plant}' affected by '{disease}', respond with JSON "
            "having keys: explanation (2-3 sentences), severity (one of "
            "low/medium/high), treatment (array of 5-7 short action strings, "
            "preferring concrete products/dosages where appropriate). "
            f"Base it on standard agronomic knowledge.{grounding}"
        )
        raw = self._generate(prompt, max_output_tokens=700, json_mode=True)
        try:
            data = json.loads(raw)
            result = {
                "explanation": str(data.get("explanation", "")).strip(),
                "severity": str(data.get("severity", "")).strip(),
                "treatment": [str(s) for s in data.get("treatment", []) if str(s).strip()],
            }
        except (json.JSONDecodeError, TypeError):
            result = {"explanation": raw, "severity": "", "treatment": []}

        self._treat_cache[key] = result
        return result

    # -- floating chatbot -------------------------------------------------
    def chat(self, message: str, history: list[dict] | None, lang: str = "he") -> str:
        if not message or not message.strip():
            return ""
        if not self._limiter.allow():
            return "⏳ Rate limit reached — please wait a moment."

        lang_name = "Hebrew" if lang == "he" else "English"
        persona = self._genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            system_instruction=(
                "You are a friendly assistant for the PlantGuard plant-disease "
                f"platform. Explain features briefly in {lang_name}."
            ),
        )
        contents = []
        for turn in history or []:
            role = "user" if turn.get("role") == "user" else "model"
            text = turn.get("content")
            if text:
                contents.append({"role": role, "parts": [{"text": str(text)}]})
        contents.append({"role": "user", "parts": [{"text": message}]})
        try:
            return (persona.generate_content(contents).text or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Chat failed: %s", exc)
            return f"😅 {exc}"
