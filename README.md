<div align="center">

# 🌱 PlantGuard AI

### AI-Powered Plant Disease Diagnosis, Research & Field Monitoring

*Computer-vision diagnosis · semantic RAG research assistant · live IoT telemetry with smart alerts · weather context · gamification — in one elegant app, plus a headless REST API.*

<p>
  <img alt="Python"        src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Gradio"        src="https://img.shields.io/badge/Gradio-6.x-FF7C00?logo=gradio&logoColor=white">
  <img alt="Transformers"  src="https://img.shields.io/badge/🤗%20Transformers-5.x-FFD21E">
  <img alt="Gemini"        src="https://img.shields.io/badge/Google-Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white">
  <img alt="FastAPI"       src="https://img.shields.io/badge/FastAPI-REST-009688?logo=fastapi&logoColor=white">
  <img alt="Firebase"      src="https://img.shields.io/badge/Firebase-RTDB-FFCA28?logo=firebase&logoColor=black">
  <img alt="Tests"         src="https://img.shields.io/badge/tests-33%20passing-3fb950">
  <img alt="License"       src="https://img.shields.io/badge/License-MIT-blue">
</p>

</div>

---

## 📖 Overview

**PlantGuard AI** helps growers and researchers **detect plant diseases from a photo**,
**ask grounded questions over a corpus of scientific papers**, and **monitor live field
conditions** with smart alerts — behind a single, polished web UI **and** a decoupled
REST API.

| Capability | Engine | What it does |
|------------|--------|--------------|
| 🔬 **Image diagnosis** | `MobileNetV2` (38-class) | Classifies a leaf photo into plant + disease with a confidence score, top-3, and a low-confidence guard. |
| 📚 **Semantic RAG** | Gemini **embeddings** + **Gemini 2.5 Flash** | Vector retrieval over a local PDF library; answers grounded in sources with citations. |
| 🩺 **Diagnosis + treatment** | One structured Gemini call + curated **knowledge base** | Returns explanation, severity, and a dosage-aware treatment plan in a single round-trip. |
| 📡 **IoT + alerts** | REST backend + Firebase RTDB | Concurrent feed fetch, live dashboards, and agronomic threshold alerts. |
| 🌦️ **Weather context** | Open-Meteo (no key) | Correlates humidity/rain with fungal-disease risk. |
| 🗂️ **History & PDF** | Firebase + `fpdf2` | Per-user diagnosis history and one-click PDF reports. |
| 🎮 **Gamification** | Per-user, persistent | Daily missions and points that persist across restarts. |

> **Origin:** started as a single-file Colab port (`micro_final.py`) and was refactored into
> a modular, tested, production-grade package. See **[ROADMAP.md](ROADMAP.md)** for the full
> engineering plan — **every item in it has been implemented**.

---

## ✨ Features

- **🔬 Disease Diagnosis** — upload **or capture from webcam**; get plant, disease, confidence,
  **top-3 predictions**, a **low-confidence warning**, a grounded explanation, and a treatment
  plan. **Batch mode** diagnoses many images at once for field surveys.
- **📚 Research Assistant** — **semantic** (embedding-based) Q&A over your PDF library, with
  citations and answer caching.
- **📊 Sensors & 📈 Dashboard** — fetch raw feeds, or render concurrent live time-series with
  **aggregated stats** and **threshold alerts** (e.g. *soil low → irrigate*), plus weather.
- **🗂️ History** — every diagnosis saved per user, with an image gallery and timeline.
- **📄 PDF Reports** — export any diagnosis (image + disease + treatment) to a branded PDF.
- **🎮 Daily Missions** — per-user, **persistent** points/levels that auto-reset daily.
- **🤖 Elegant Floating Assistant** — a redesigned chat widget explaining the platform.
- **🔌 REST API** — the same engines exposed via **FastAPI** for automations or a future mobile app.
- **🛡️ Graceful degradation** — missing Firebase / offline sensors / unavailable embeddings all
  degrade to safe fallbacks instead of crashing.

---

## 🏗️ Architecture

A clean, microservice-style **package** with a thin UI and API on top. Heavy initialization
happens once at startup; nothing expensive runs at import time, so the logic is fully testable.

```
                        ┌─────────────────────────────────────────┐
                        │   plantguard/app.py   (Gradio UI)         │
                        │   plantguard/api.py   (FastAPI REST)      │
                        └───────────────┬───────────────────────────┘
                                        │  build_services()
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼               ▼                ▼                ▼               ▼
  ┌───────────┐  ┌────────────┐   ┌────────────┐   ┌───────────┐  ┌────────────┐
  │  image    │  │    rag     │   │    iot     │   │gamification│  │  history   │
  │ MobileNet │  │ embeddings │   │ retry +    │   │ per-user   │  │ per-user   │
  │ +labels   │  │ +Gemini    │   │ concurrent │   │ +persist   │  │ +PDF       │
  │ +top-3    │  │ +KB +cache │   │ +MapReduce │   └───────────┘  └────────────┘
  └───────────┘  └─────┬──────┘   └─────┬──────┘
                       │                 │
                 ┌─────▼─────┐    ┌──────▼──────┐   ┌──────────┐  ┌──────────┐
                 │ documents │    │   alerts    │   │ weather  │  │  store   │
                 │ parse-once│    │ thresholds  │   │OpenMeteo │  │ Firebase │
                 └───────────┘    └─────────────┘   └──────────┘  └──────────┘

  Cross-cutting:  config.py (settings/logging) · http_client.py (retry session) · i18n.py
```

| Module | Responsibility |
|--------|----------------|
| [config.py](plantguard/config.py) | Typed settings from env, logging, UTF-8 console |
| [http_client.py](plantguard/http_client.py) | Shared `requests` session with exponential-backoff retry |
| [services/store.py](plantguard/services/store.py) | Firebase RTDB wrapper (no insecure key download) |
| [services/image.py](plantguard/services/image.py) | Classifier, robust label parsing, confidence, validation |
| [services/rag.py](plantguard/services/rag.py) | Embedding/TF-IDF retrieval, system-instruction, caching, rate limiting |
| [services/documents.py](plantguard/services/documents.py) | Parse each PDF once into a shared corpus |
| [services/knowledge_base.py](plantguard/services/knowledge_base.py) | Curated treatment guidance with dosages |
| [services/iot.py](plantguard/services/iot.py) | History, concurrent fetch, MapReduce aggregates |
| [services/alerts.py](plantguard/services/alerts.py) | Pure agronomic threshold rules |
| [services/gamification.py](plantguard/services/gamification.py) | Per-user, persistent missions/points |
| [services/history.py](plantguard/services/history.py) · [reports.py](plantguard/services/reports.py) · [weather.py](plantguard/services/weather.py) | History, PDF reports, weather context |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11**
- A **Google Gemini API key** ([get one](https://aistudio.google.com/app/apikey))
- *(Optional)* a **Firebase** service-account JSON for persistence

### 1. Clone & enter
```powershell
git clone https://github.com/Mohamedkhattab02/PlantGuard-AI
cd "PlantGuard AI"
```

### 2. Virtual environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install (pinned, reproducible)
```powershell
pip install -r requirements.txt
```

### 4. Configure secrets
```powershell
Copy-Item .env.example .env
```
```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=https://your-project-default-rtdb.firebaseio.com/
FIREBASE_KEY_PATH=firebase-key.json
BASE_URL=https://your-server.onrender.com/
# Optional launch controls (default to local/safe):
# GRADIO_SHARE=false
# DEBUG=false
```
Place your Firebase key as `firebase-key.json` in the project root **(it is never
auto-downloaded — provide it yourself)**.

> 🔒 `.env`, `firebase-key.json`, and `*-key.json` are git-ignored. Only `.env.example` is tracked.

### 5. Run the app
```powershell
python micro_final.py        # backward-compatible entry point
# or:  python -m plantguard.app
```
The UI launches at **`http://127.0.0.1:7860`**. A public link is created only if `GRADIO_SHARE=true`.

---

## 🔌 REST API (headless)

```powershell
uvicorn plantguard.api:create_app --factory --port 8000
```

| Method & path | Purpose |
|---------------|---------|
| `GET /health` | Liveness + Firebase status |
| `POST /diagnose` | Multipart image upload → diagnosis (+ advice if diseased) |
| `POST /research` | `{ "question": "...", "top_k": 2 }` → grounded answer + sources |
| `GET /sensors/{feed}?limit=10` | Raw sensor history |
| `GET /dashboard` | Aggregated stats + alerts |

Interactive docs at `http://127.0.0.1:8000/docs`.

---

## 📂 Project Structure

```
PlantGuard AI/
├── micro_final.py          # Backward-compatible launcher (delegates to the package)
├── plantguard/             # The application package
│   ├── config.py · http_client.py · i18n.py
│   ├── app.py              # Gradio UI
│   ├── api.py              # FastAPI service
│   └── services/           # image · rag · documents · knowledge_base · iot ·
│                           #   alerts · gamification · history · reports · weather · store
├── tests/                  # pytest suite (pure logic + UI build smoke test)
├── articles/               # 📚 Research PDFs — indexed for RAG
├── IOT_DETAILS/            # 📡 IoT JSON exports
├── requirements.txt        # Pinned dependencies
├── pyproject.toml          # Packaging + ruff/black + pytest config
├── ROADMAP.md              # Engineering plan (fully implemented)
└── LICENSE                 # MIT
```

---

## ⚙️ Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | — (required) | Gemini key; the app fails loudly without it |
| `DATABASE_URL` | demo RTDB | Firebase Realtime DB URL |
| `FIREBASE_KEY_PATH` | `firebase-key.json` | Service-account key path |
| `BASE_URL` | demo Render URL | IoT REST backend |
| `GRADIO_SHARE` | `false` | Create a public tunnel |
| `DEBUG` | `false` | Verbose logs + Gradio debug |
| `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` | `127.0.0.1` / `7860` | Bind address |

---

## 🧪 Testing & Quality

```powershell
pytest          # 33 tests: label parsing, gamification, MapReduce, alerts, i18n, + UI build smoke test
ruff check .    # lint
black .         # format
```
The UI build smoke test constructs the entire Gradio graph **without** the model, network,
or API keys — catching component-API regressions in CI.

---

## 🧠 Model & API Reference

| Component | Identifier / Endpoint |
|-----------|-----------------------|
| **Image classifier** | [`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification) |
| **LLM** | `gemini-2.5-flash` (system-instruction, JSON mode for structured advice) |
| **Embeddings** | `models/text-embedding-004` (cosine retrieval, disk-cached; TF-IDF fallback) |
| **IoT API** | `GET {BASE_URL}/history?feed=<feed>&limit=<n>` |
| **Weather** | Open-Meteo (geocoding + current conditions, no key) |

> **Label parsing:** the model emits human-readable labels (`"Tomato with Late Blight"`,
> `"Healthy Apple"`). [`parse_label`](plantguard/services/image.py) uses a known-plant table +
> overrides (e.g. `"Cedar Apple Rust"` → *Apple*) instead of a naive first-word heuristic.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **UI** | Gradio 6.x (custom theme + CSS, redesigned chat widget) |
| **API** | FastAPI · Uvicorn · Pydantic |
| **Vision** | 🤗 Transformers 5.x · PyTorch · MobileNetV2 |
| **LLM / RAG** | Google Gemini 2.5 Flash · Gemini embeddings · scikit-learn (TF-IDF fallback) · NumPy |
| **Docs / Data** | pypdf · pandas · matplotlib · fpdf2 |
| **Backend** | Firebase Admin · `requests` (retry session) |
| **Tooling** | pytest · ruff · black · python-dotenv |

---

## ⚠️ Compatibility Notes

- **`transformers` 5.x dropped `text2text-generation`** → local FLAN-T5 replaced by Gemini.
- **`transformers` 5.x can't auto-detect the image processor** → explicit
  `AutoModelForImageClassification` + `AutoImageProcessor`.
- **Gradio 6 moved `theme`/`css`** from `Blocks(...)` to `launch()` — applied accordingly.
- **Gradio 6 chatbot** uses the messages format (`{"role","content"}`) by default.
- **Windows console is `cp1252`** → UTF-8 is forced at startup so emoji never crash logging.

---

## 🗺️ Roadmap Status

All **[ROADMAP.md](ROADMAP.md)** items are implemented. Pragmatic scoping notes:
- **User accounts** use lightweight username-based persistence (per-user history/progress) rather
  than full OAuth.
- **Sensor alerting** ships the rule engine + in-app alerts; email/Telegram push is left as a hook.
- **Bilingual UI** was intentionally consolidated to **English-only** per product decision.

---

## 📜 License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

<div align="center">

**Built with 🌱 for healthier crops.**

</div>
