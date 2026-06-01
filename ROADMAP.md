<div align="center">

# 🗺️ PlantGuard AI — Engineering Roadmap

### From a working prototype to a production-grade, hi-tech product

</div>

This document is a **technical improvement plan**. It has two parts:

1. **[Part 1 — Hardening & Improving Existing Features](#part-1--hardening--improving-existing-features)** — concrete issues in the current code and how to fix them so each feature reaches production quality.
2. **[Part 2 — Proposed New Features](#part-2--proposed-new-features)** — high-impact capabilities to add next.

A **[phased delivery plan](#-phased-delivery-plan)** and **[prioritization matrix](#-prioritization-matrix)** close the document.

> **Legend** — Priority: `P0` critical · `P1` high · `P2` nice-to-have. Effort: `S` <½ day · `M` 1–3 days · `L` >3 days.

---

## Part 1 — Hardening & Improving Existing Features

### 🔴 1. Correctness & Concurrency

| # | Issue | Where | Fix | Priority | Effort |
|---|-------|-------|-----|----------|--------|
| 1.1 | **Global shared state across all users.** `game_system` is a single module-level instance, so with `share=True` **every visitor mutates the same points/level**. The floating chatbot history is similarly global. | [`game_system`](micro_final.py#L264), [`GamificationSystem`](micro_final.py#L222) | Move per-user state into Gradio [`gr.State`](https://www.gradio.app/docs/gradio/state) (already used for the chat toggle at [L902](micro_final.py#L902)). Pass the user's game object through event handlers instead of touching a global. | **P0** | M |
| 1.2 | **Gamification is in-memory only** — a restart wipes all points/levels/achievements. | [`reset_daily_tasks`](micro_final.py#L256) | Persist per-user progress to Firebase (the [`db_set`](micro_final.py#L115) helper already exists but gamification never calls it). | **P1** | M |
| 1.3 | **"Daily" missions never reset by day** — only a manual button resets them. | [`reset_daily_tasks`](micro_final.py#L256) | Store a `last_reset` date per user; auto-reset on first interaction of a new day. | P2 | S |

### 🟠 2. Performance

| # | Issue | Where | Fix | Priority | Effort |
|---|-------|-------|-----|----------|--------|
| 2.1 | **Two sequential Gemini round-trips per diagnosis** (`rag_explain` → then `gemini_treatment`), both blocking the UI with no progress indicator. | [`diagnose`](micro_final.py#L698) calls [L710](micro_final.py#L710) + [L711](micro_final.py#L711) | Merge into **one** structured Gemini call that returns explanation **and** treatment as JSON. Add a `gr.Progress()` bar. | **P1** | M |
| 2.2 | **No caching.** Identical RAG questions and treatment lookups re-hit Gemini every time. | [`gemini_treatment`](micro_final.py#L693), [`query`](micro_final.py#L456) | Wrap pure lookups in `functools.lru_cache` (or a small TTL cache) keyed on the normalized input. | **P1** | S |
| 2.3 | **Matplotlib figure leak.** `plot_all_feeds` creates figures with `plt.subplots` but never calls `plt.close(fig)` → memory grows with every dashboard refresh. | [`plot_all_feeds`](micro_final.py#L743) | `plt.close(fig)` after handing the figure to Gradio, or reuse a single figure. | **P1** | S |
| 2.4 | **Dashboard fires 3 sequential HTTP calls** (temperature, humidity, soil) on each refresh. | [`plot_all_feeds`](micro_final.py#L743) | Fetch the feeds concurrently (`concurrent.futures.ThreadPoolExecutor`) — ~3× faster. | P2 | S |
| 2.5 | **PDFs are parsed twice** — once by `DocumentService` (PyPDF2) for the keyword index, again by `build_pdf_retriever` (pypdf) for the TF-IDF retriever. Double I/O and double memory. | [`load_documents`](micro_final.py#L278) + [`build_pdf_retriever`](micro_final.py#L597) | Parse each PDF once, share the extracted text between both indexes. Standardize on **one** PDF library (drop either PyPDF2 or pypdf). | P1 | M |
| 2.6 | **Dead NLTK download.** `punkt`/`punkt_tab` are downloaded at startup but a regex tokenizer is what's actually used (the code comment admits this). | [L162–L169](micro_final.py#L162) | Remove the NLTK download and the `PorterStemmer`/`nltk` imports if unused, or commit to using them. | P2 | S |

### 🟡 3. Reliability & Error Handling

| # | Issue | Where | Fix | Priority | Effort |
|---|-------|-------|-----|----------|--------|
| 3.1 | **No retry/backoff for the IoT backend.** The Render free tier cold-starts (~30–60 s); the first request frequently times out and the user just sees an error. | [`get_sensor_data`](micro_final.py#L723), [`plot_all_feeds`](micro_final.py#L743) | Use a `requests.Session` with `urllib3` `Retry` (exponential backoff) and a friendly "waking the server…" message. | **P1** | S |
| 3.2 | **Errors are returned as user-facing strings** (`f"❌ {e}"`) and mixed into results — no real logging or telemetry. | throughout Cell 5 | Log exceptions via the `logging` module; show the user a clean message, keep the stack trace in logs. | **P1** | M |
| 3.3 | **`print()` everywhere despite `logging.basicConfig` being configured.** Inconsistent and unstructured. | [L96](micro_final.py#L96) + ~40 `print` calls | Replace `print` with `logging.info/warning/error`. Adopt structured logging for production. | P1 | M |
| 3.4 | **Reduced IoT MapReduce result is computed and written to Firebase but never surfaced in the UI.** | [`run_mapreduce`](micro_final.py#L407) | Either show the aggregated min/max/avg on the Dashboard, or remove the dead path. | P2 | S |

### 🔵 4. Security & Configuration

| # | Issue | Where | Fix | Priority | Effort |
|---|-------|-------|-----|----------|--------|
| 4.1 | **A Firebase service-account key is auto-downloaded from a hardcoded public Google Drive link.** Service-account credentials must never live on a public link or be fetched at runtime. | [`DRIVE_JSON_ID`](micro_final.py#L102), [L134–L145](micro_final.py#L134) | Remove the auto-download entirely. Require the key via `FIREBASE_KEY_PATH` (or a secrets manager) and fail loudly if missing. Rotate the exposed key. | **P0** | S |
| 4.2 | **`share=True, debug=True` hardcoded at launch** — exposes a public tunnel and dev tracebacks on every run. | [`demo.launch`](micro_final.py#L934) | Drive both from env vars (`GRADIO_SHARE`, `DEBUG`), default to `False` in production. | **P1** | S |
| 4.3 | **No upload validation** — any file/size is accepted into the classifier. | [`diagnose`](micro_final.py#L698) | Validate type, cap dimensions/size, reject non-images early. | P2 | S |
| 4.4 | **No rate limiting on Gemini** — a burst of requests can exhaust quota and incur cost. | RAG + chatbot | Add per-session throttling / a token budget. | P2 | M |

### 🟣 5. AI / ML Quality

| # | Issue | Where | Fix | Priority | Effort |
|---|-------|-------|-----|----------|--------|
| 5.1 | **Keyword-count retrieval, not semantic.** The RAG index matches stemmed keywords from a fixed `KEY_TERMS` list — it misses paraphrases and synonyms. | [`TextProcessingService`](micro_final.py#L293), [`IndexService`](micro_final.py#L312) | Replace with **vector embeddings** (Gemini `text-embedding-004` or `sentence-transformers`) + cosine similarity for true semantic retrieval. | **P1** | L |
| 5.2 | **No confidence threshold.** `diagnose` always trusts the top prediction even at, say, 4 % confidence. | [`diagnose`](micro_final.py#L707) | If `confidence < threshold`, return "uncertain — please retake the photo" and show the **top-3** with confidence bars. | **P1** | S |
| 5.3 | **Fragile label parser.** The "first word = plant" heuristic mislabels e.g. `"Cedar Apple Rust"` → plant `"Cedar"`. | [`parse_label`](micro_final.py#L631) | Back it with an explicit label→(plant, disease) mapping table for the model's 38 known classes. | P1 | S |
| 5.4 | **Chatbot system prompt is injected as a fake "user" turn.** `gemini-2.5-flash` supports a real `system_instruction`. | [`helper_bot_reply`](micro_final.py#L774) | Pass the persona via `system_instruction` on the model instead of a leading user message. | P2 | S |

### ⚙️ 6. Code Quality & Project Structure

| # | Issue | Fix | Priority | Effort |
|---|-------|-----|----------|--------|
| 6.1 | **934-line monolith.** Despite "microservice-style" classes, everything lives in one file — hard to test, review, and reuse. | Split into a package: `services/` (rag, image, iot, gamification), `app.py` (UI), `config.py`. | **P1** | M |
| 6.2 | **No `requirements.txt`.** Dependencies are a `pip install` one-liner inside a docstring — versions unpinned and non-reproducible. | Add a **pinned** `requirements.txt` (and ideally `pyproject.toml`). | **P0** | S |
| 6.3 | **Zero tests.** | Add `pytest` unit tests for pure logic (`parse_label`, MapReduce reduce, gamification scoring) and a smoke test that the app imports. | **P1** | M |
| 6.4 | **No `LICENSE` file** though the README declares MIT. | Add the MIT `LICENSE`. | P1 | S |
| 6.5 | **No formatting/linting.** | Adopt `ruff` + `black`; enforce via pre-commit. | P2 | S |

---

## Part 2 — Proposed New Features

> Ordered roughly by impact-to-effort. Each is a meaningful step toward a market-ready product.

### 🧑‍🌾 2.1 User Accounts & Persistent Progress · `P1` · `L`
Authentication (Firebase Auth) with **per-user** diagnosis history, saved missions, and a profile. Turns the demo into a real multi-tenant app and is the natural home for the fix in [1.1](#-1-correctness--concurrency).

### 🚨 2.2 Smart Sensor Alerting · `P1` · `M`
Threshold rules on the IoT feeds (e.g. *soil moisture < 20 % → "irrigate now"*). Push notifications via email/Telegram. The MapReduce min/max/avg pipeline ([L389](micro_final.py#L389)) already computes the stats needed to power this.

### 🗂️ 2.3 Diagnosis History & Timeline · `P1` · `M`
Persist every diagnosis (image thumbnail, result, confidence, timestamp) per user, with a gallery and trend view ("Tomato blight detected 3× this month").

### 📄 2.4 Exportable Reports (PDF) · `P2` · `S`
One-click export of a diagnosis (image + disease + treatment + sources) to a branded PDF — useful for agronomists and record-keeping.

### 🌐 2.5 Bilingual UI (Hebrew ⇄ English) · `P1` · `M`
The chatbot already replies in Hebrew; extend a full **RTL Hebrew/English toggle** across all tabs. High value for the target users.

### 🔌 2.6 Decoupled REST API (FastAPI) · `P1` · `L`
Expose the diagnosis/RAG/IoT engines as a **FastAPI** service, with Gradio as one client. Enables a future mobile app, third-party integrations, and clean separation of UI from logic.

### 📷 2.7 Live Camera Capture & Batch Diagnosis · `P2` · `M`
Diagnose straight from a webcam/phone camera, and accept **multiple images** at once for field surveys.

### 🌦️ 2.8 Weather & Agronomic Context · `P2` · `M`
Integrate a weather API to correlate disease risk with humidity/rainfall and offer preventative advice.

### 🧪 2.9 Treatment Knowledge Base with Dosages · `P2` · `M`
Move from free-text Gemini treatment to a curated, citable knowledge base (products, dosages, organic alternatives) grounded by the RAG corpus.

---

## 🚀 Phased Delivery Plan

> A pragmatic order that ships safety and stability first, then quality, then growth.

### Phase 0 — Stabilize (the "must-fix" sprint)
- [ ] **4.1** Remove the service-account auto-download · `P0`
- [ ] **1.1** Per-user state via `gr.State` · `P0`
- [ ] **6.2** Pinned `requirements.txt` · `P0`
- [ ] **4.2** Env-driven `share`/`debug` flags · `P1`

### Phase 1 — Productionize
- [ ] **3.1–3.3** Retries, logging, clean error handling
- [ ] **2.2/2.3** Caching + fix the matplotlib leak
- [ ] **6.1/6.3/6.4** Modularize, add tests, add `LICENSE`
- [ ] **5.2** Confidence threshold + top-3 results

### Phase 2 — Elevate (AI & UX)
- [ ] **5.1** Embeddings-based semantic RAG
- [ ] **2.1** User accounts & persistent progress
- [ ] **2.5** Bilingual RTL UI
- [ ] **2.2** Smart sensor alerting

### Phase 3 — Scale & Productize
- [ ] **2.6** FastAPI service + Dockerization + CI/CD (GitHub Actions)
- [ ] **2.3/2.4** History timeline + PDF reports
- [ ] Observability: structured logs, metrics, error tracking

---

## 📊 Prioritization Matrix

```
        HIGH IMPACT
            ▲
   2.1 User accounts   │  4.1 Remove key download (P0)
   5.1 Semantic RAG    │  1.1 Per-user state (P0)
   2.6 FastAPI API     │  3.1 Retry/backoff
                       │  2.2 Caching · 5.2 Confidence
 ──────────────────────┼────────────────────────────────►
   2.8 Weather         │  6.2 requirements.txt
   2.9 KB dosages      │  2.3 Fig leak fix
   2.7 Batch capture   │  6.4 LICENSE · 5.3 Label table
            │
        LOW IMPACT
     HIGH EFFORT  ◄───────────────►  LOW EFFORT
```

**Top-left (big bets):** semantic RAG, user accounts, the FastAPI split — schedule deliberately.
**Top-right (quick wins, do first):** the two `P0`s, retries, caching, confidence threshold.

---

<div align="center">

*This roadmap is a living document — update it as items ship.* 🌱

</div>
