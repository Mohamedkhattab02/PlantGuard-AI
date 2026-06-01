<div align="center">

# 🌱 PlantGuard AI

### AI-Powered Plant Disease Research & Monitoring Platform

*Computer-vision diagnosis · RAG research assistant · live IoT telemetry · interactive dashboards — in one Gradio app.*

<p>
  <img alt="Python"        src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Gradio"        src="https://img.shields.io/badge/Gradio-6.x-FF7C00?logo=gradio&logoColor=white">
  <img alt="Transformers"  src="https://img.shields.io/badge/🤗%20Transformers-5.x-FFD21E">
  <img alt="Gemini"        src="https://img.shields.io/badge/Google-Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white">
  <img alt="Firebase"      src="https://img.shields.io/badge/Firebase-RTDB-FFCA28?logo=firebase&logoColor=black">
  <img alt="License"       src="https://img.shields.io/badge/License-MIT-blue">
</p>

</div>

---

## 📖 Overview

**PlantGuard AI** is a research-and-monitoring platform that helps growers and researchers
**detect plant diseases from a photo**, **ask grounded questions over a corpus of scientific
papers**, and **monitor live field conditions** from IoT sensors — all behind a single,
themed [Gradio](https://www.gradio.app/) web UI.

The application fuses four AI/data capabilities:

| Capability | Engine | What it does |
|------------|--------|--------------|
| 🔬 **Image diagnosis** | `MobileNetV2` (38-class, fine-tuned) | Classifies a leaf photo into plant + disease, with a confidence score. |
| 📚 **RAG research assistant** | Custom retriever + **Gemini 2.5 Flash** | Answers questions grounded **only** in a local library of research PDFs, with citations. |
| 📡 **IoT telemetry** | Remote REST API + Firebase RTDB | Pulls temperature / humidity / soil readings and renders live dashboards. |
| 🎮 **Gamification** | In-memory mission engine | Awards points for completing tasks across tabs to drive engagement. |

> **Origin:** ported from a Google Colab notebook into a self-contained local application,
> with the LLM swapped to the Gemini API and the model-loading path hardened for
> `transformers` 5.x.

---

## ✨ Features

- **🔬 Disease Diagnosis** — upload a leaf image → get plant, disease, confidence, a
  RAG-grounded explanation, and a Gemini-generated treatment plan.
- **📚 Research Assistant** — natural-language Q&A over your own PDF library; every answer
  is sourced strictly from the indexed documents and lists its citations.
- **📊 Sensor Data** — fetch raw `temperature` / `humidity` / `soil` feeds from the IoT
  backend on demand.
- **📈 Visual Dashboard** — time-series plots of all sensor feeds, refreshed live.
- **🎮 Daily Missions** — point-based gamification that rewards using each feature.
- **🤖 Floating Assistant** — an always-on Gemini chatbot (replies in Hebrew) that explains
  how to use the platform.
- **🛡️ Graceful degradation** — if Firebase, the IoT server, or a download is unavailable,
  the app logs a warning and **keeps running** instead of crashing.

---

## 🏗️ Architecture

The codebase follows a **microservice-style separation of concerns**, organized into six
logical "cells" within [`micro_final.py`](micro_final.py):

```
┌──────────────────────────────────────────────────────────────────┐
│                          Gradio UI (Cell 6)                        │
│   🔬 Diagnosis · 📚 Research · 📊 Sensors · 📈 Dashboard · 🎮 Game │
└───────────────┬───────────────────────────────┬──────────────────┘
                │                                │
   ┌────────────▼────────────┐      ┌────────────▼─────────────┐
   │   Image Pipeline         │      │   RAG System (GeminiRAG)  │
   │   MobileNetV2 + processor│      │   AcademicSearchEngine    │
   │   (image-classification) │      │   ├─ DocumentService      │
   └────────────┬────────────┘      │   ├─ IndexService         │
                │                    │   ├─ SearchService        │
   ┌────────────▼────────────┐      │   └─ ResultService        │
   │   parse_label()          │      │        + TfidfRetriever   │
   │   rag_explain()          │      └────────────┬─────────────┘
   │   gemini_treatment()     │                   │
   └─────────────────────────┘      ┌────────────▼─────────────┐
                                     │   Gemini 2.5 Flash (LLM)  │
   ┌──────────────────────────┐     └───────────────────────────┘
   │   IoT Layer               │
   │   get_sensor_data()       │──────► REST: {BASE_URL}/history
   │   plot_all_feeds()        │──────► Firebase RTDB (telemetry)
   └──────────────────────────┘
```

**Cell breakdown** (top-to-bottom in the script):

| Cell | Lines | Responsibility |
|------|-------|----------------|
| **1 — Imports** | [L32](micro_final.py#L32) | Dependencies + forced UTF-8 console (Windows `cp1252` fix). |
| **2 — Config & Setup** | [L99](micro_final.py#L99) | Loads `.env`, initializes Firebase, resolves local data folders (Drive fallback). |
| **3 — Domain Engines** | [L218](micro_final.py#L218) | Gamification, academic search microservices, IoT MapReduce. |
| **4 — RAG & Models** | [L421](micro_final.py#L421) | Builds the RAG pipeline + loads the image classifier. |
| **5 — Business Logic** | [L628](micro_final.py#L628) | `diagnose`, `query_handler`, sensor fetch/plot, chatbot. |
| **6 — UI** | [L801](micro_final.py#L801) | Gradio `Blocks`, tabs, floating chatbot, event wiring. |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11**
- A **Google Gemini API key** ([get one here](https://aistudio.google.com/app/apikey))
- *(Optional)* A **Firebase** service-account JSON for telemetry persistence

### 1. Clone & enter the project

```powershell
git clone <https://github.com/Mohamedkhattab02/PlantGuard-AI>
cd "PlantGuard AI"
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install gradio nltk PyPDF2 transformers torch pillow pypdf scikit-learn `
            huggingface_hub sentencepiece datasets google-generativeai pandas `
            matplotlib requests gdown python-dotenv firebase-admin
```

### 4. Configure secrets

Copy the example env file and fill in your real values:

```powershell
Copy-Item .env.example .env
```

```dotenv
# .env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=https://your-project-default-rtdb.firebaseio.com/
FIREBASE_KEY_PATH=firebase-key.json
BASE_URL=https://your-server.onrender.com/
```

Place your Firebase service-account file as `firebase-key.json` in the project root.

> 🔒 **`.env`, `firebase-key.json`, and all `*-key.json` files are git-ignored** — never
> commit secrets. Only `.env.example` is tracked.

### 5. Run

```powershell
python micro_final.py
```

The app launches at **`http://127.0.0.1:7860`** and prints a temporary public
`share=True` link.

---

## 📂 Data Layout

The app prefers **local data folders** and only falls back to Google Drive if they are empty.

```
PlantGuard AI/
├── micro_final.py          # The application (single file, 6 cells)
├── .env                    # Secrets (git-ignored)
├── .env.example            # Template for .env
├── firebase-key.json       # Firebase service account (git-ignored)
├── articles/               # 📚 Research PDFs — indexed for the RAG assistant
│   ├── 1-s2.0-S2772899424000417-main.pdf
│   └── ... (5 papers)
└── IOT_DETAILS/            # 📡 IoT JSON exports (temperature / humidity / soil)
    └── json-20251229-2107.json
```

- **`articles/`** — drop in any plant-disease research PDFs; they are chunked and indexed
  on startup for retrieval.
- **`IOT_DETAILS/`** — historical sensor exports. Each record's `value` is a JSON string
  containing `temperature`, `humidity`, and `soil`.

---

## 🧠 Model & API Reference

| Component | Identifier / Endpoint |
|-----------|-----------------------|
| **Image classifier** | [`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification) |
| **LLM** | `gemini-2.5-flash` (temperature `0.3`, `top_p 0.8`, 2048 max tokens) |
| **Retrievers** | Keyword index (RAG Q&A) + TF-IDF + cosine similarity (diagnosis explanations) |
| **IoT API** | `GET {BASE_URL}/history?feed=<feed>&limit=<n>` |
| **Telemetry store** | Firebase Realtime Database |

> **Label format:** this model emits human-readable labels such as `"Tomato with Late
> Blight"` or `"Healthy Apple"` — **not** the PlantVillage `Plant___Disease` format. The
> [`parse_label()`](micro_final.py#L631) function splits on `" with "` and detects the
> `Healthy` prefix.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **UI** | Gradio 6.x (`Blocks`, themed, messages-format chatbot) |
| **Computer Vision** | 🤗 Transformers 5.x · PyTorch · MobileNetV2 |
| **LLM / Generation** | Google Gemini 2.5 Flash (`google-generativeai`) |
| **Retrieval** | scikit-learn (TF-IDF), custom keyword index, NLTK |
| **Document parsing** | PyPDF2 / pypdf |
| **Data / Plotting** | pandas · matplotlib (`Agg` backend) |
| **Backend / Telemetry** | Firebase Admin SDK · `requests` (REST) |
| **Config** | `python-dotenv` |

---

## ⚠️ Compatibility Notes

This project runs on **modern, breaking-change-prone** library versions. Key adaptations
already baked in:

- **`transformers` 5.x dropped `text2text-generation`** → the local FLAN-T5 LLM was
  replaced with a `gemini_generate()` helper.
- **`transformers` 5.x can't auto-detect the image processor** (the model's
  `preprocessor_config.json` lacks `image_processor_type`) → the pipeline is built with an
  explicit `AutoModelForImageClassification` + `AutoImageProcessor`.
- **Gradio 6.x removed `gr.Chatbot(type=...)`** → uses the default messages format
  (`{"role", "content"}` dicts).
- **Windows console is `cp1252`** → `sys.stdout/stderr.reconfigure(encoding="utf-8")` is
  forced at startup so emoji never raise `UnicodeEncodeError`.

---

## 🗺️ Usage Guide

| Tab | How to use it |
|-----|---------------|
| 🔬 **Disease Diagnosis** | Upload a leaf photo → **Analyze** → read the diagnosis, confidence, explanation & treatment. |
| 📚 **Research Assistant** | Type a question, pick how many docs to retrieve → **Search**. Answers cite their sources. |
| 📊 **Sensor Data** | Choose a feed and sample count → **Fetch** raw readings. |
| 📈 **Dashboard** | Set the number of points → **Refresh** to render live time-series plots. |
| 🎮 **Daily Missions** | Complete tasks across tabs to earn points; **Reset** to start a new day. |
| 💬 **Floating Assistant** | Click the bubble (bottom-right) for help in Hebrew. |

---

## 🤝 Contributing

1. Fork the repo and create a feature branch (`git checkout -b feature/my-feature`).
2. Keep the **six-cell structure** and the microservice separation intact.
3. Ensure the app still launches cleanly (`python micro_final.py`) before opening a PR.
4. **Never** commit `.env`, `firebase-key.json`, or any credential file.

---


<div align="center">

**Built with 🌱 for healthier crops.**

</div>
