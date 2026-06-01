# -*- coding: utf-8 -*-
"""
Plant Disease Research & Monitoring System
Converted from Google Colab to run locally in VS Code.

==========================================================
SETUP (run once in terminal before running this file):
==========================================================
    pip install gradio nltk PyPDF2 transformers torch pillow pypdf scikit-learn huggingface_hub sentencepiece datasets google-generativeai pandas matplotlib requests gdown python-dotenv firebase-admin

Create a file named ".env" next to this script with:
    GEMINI_API_KEY=your_gemini_api_key_here
    DATABASE_URL=https://plant-disease-index-default-rtdb.firebaseio.com/
    FIREBASE_KEY_PATH=firebase-key.json
    BASE_URL=https://server-cloud-v645.onrender.com/

Put your Firebase service account JSON in the same folder as "firebase-key.json".
(If you don't have the file, it will be auto-downloaded from the Drive ID below.)

Data folders:
    - Put research PDFs in a local "articles" folder (next to this script).
    - Put IoT JSON exports in a local "IOT_DETAILS" folder.
    If those folders are missing, the script falls back to downloading from Google Drive.

Run with:
    python micro_final.py
==========================================================
"""

# ==========================================================
# Cell 1: Imports
# ==========================================================
import os
import re
import sys
import json
import logging
import shutil
from collections import defaultdict

# On Windows the console often defaults to cp1252, which crashes on the emoji
# used throughout this script. Force UTF-8 so prints never raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe non-interactive backend for server/gradio rendering
import matplotlib.pyplot as plt
import torch
from PIL import Image
import nltk
from nltk.stem import PorterStemmer
from PyPDF2 import PdfReader
from pypdf import PdfReader as PdfReader2
import google.generativeai as genai
import gradio as gr
from transformers import (
    pipeline,
    AutoModelForImageClassification,
    AutoImageProcessor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional dependencies -- the app still runs if these are missing / offline.
try:
    import firebase_admin
    from firebase_admin import credentials, db
except ImportError:
    firebase_admin = None
    credentials = None
    db = None

try:
    import gdown
except ImportError:
    gdown = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# Load environment variables from .env (replaces Colab secrets / hardcoded keys)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

logging.basicConfig(level=logging.INFO)

# ==========================================================
# Cell 2: Firebase Setup + Local Folders + Config
# ==========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "https://plant-disease-index-default-rtdb.firebaseio.com/")
DRIVE_JSON_ID = "1hikKKqjePaBDbYSYmWqKTFYgspJDHpCA"

# In VS Code we use local paths instead of /content/...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve the Firebase key path relative to this script if it is given as a bare name.
_key_env = os.getenv("FIREBASE_KEY_PATH", "firebase-key.json")
SERVICE_KEY_PATH = _key_env if os.path.isabs(_key_env) else os.path.join(BASE_DIR, _key_env)

# Global flag: is Firebase available for reads/writes?
FIREBASE_OK = False


def db_set(path, value):
    """Write to Firebase only if it is available; never crash the app."""
    if not FIREBASE_OK or db is None:
        return False
    try:
        db.reference(path).set(value)
        return True
    except Exception as e:
        print(f"⚠️ Firebase write to '{path}' failed: {e}")
        return False


def init_firebase():
    global FIREBASE_OK
    if firebase_admin is None:
        print("⚠️ firebase-admin not installed -- skipping Firebase (app still works).")
        return

    if not os.path.exists(SERVICE_KEY_PATH):
        print("⬇️ Downloading Firebase key...")
        try:
            r = requests.get(
                f"https://drive.google.com/uc?export=download&id={DRIVE_JSON_ID}",
                timeout=30,
            )
            r.raise_for_status()
            with open(SERVICE_KEY_PATH, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"⚠️ Could not download Firebase key: {e}. Skipping Firebase.")
            return

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(SERVICE_KEY_PATH)
            firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
            print("🔥 Firebase connected")
        db.reference("test").set({"status": "ok"})
        FIREBASE_OK = True
    except Exception as e:
        print(f"⚠️ Firebase init failed: {e}. Continuing without Firebase.")


init_firebase()

# NLTK data is not strictly required (we use a regex tokenizer below), but we try
# to fetch it quietly so anything relying on it keeps working.
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

# Base URLs and Keys
BASE_URL = os.getenv("BASE_URL", "https://server-cloud-v645.onrender.com/").rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found. Add it to your .env file.")

PDF_LINK = "https://drive.google.com/drive/folders/1_Y3ZzVa19Pa562XMGwaHZhVBsh6l5DxZ"
IOT_LINK = "https://drive.google.com/drive/folders/1E61DvdF4kaW01kKBjDx3bfpFh_Tt1up0"


def drive_folder_to_local(url: str, local: str, clean: bool = True) -> str:
    if gdown is None:
        print("⚠️ gdown not installed -- cannot download from Drive.")
        os.makedirs(local, exist_ok=True)
        return local
    if clean and os.path.exists(local):
        shutil.rmtree(local)
    os.makedirs(local, exist_ok=True)
    try:
        gdown.download_folder(url=url, output=local, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"⚠️ Drive download failed for {url}: {e}")
    return local


def ensure_data_folder(local_name: str, drive_url: str) -> str:
    """Prefer an existing local folder with data; otherwise download from Drive."""
    local = os.path.join(BASE_DIR, local_name)
    has_files = os.path.isdir(local) and any(os.scandir(local))
    if has_files:
        print(f"📁 Using local folder: {local_name}")
        return local
    print(f"⬇️ Local folder '{local_name}' is empty/missing -- trying Google Drive...")
    return drive_folder_to_local(drive_url, local)


# Use the local data folders that ship with the project; fall back to Drive.
PDF_FOLDER = ensure_data_folder("articles", PDF_LINK)
IOT_FOLDER = ensure_data_folder("IOT_DETAILS", IOT_LINK)

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
KEY_TERMS = ["plant", "diseas", "leaf", "infect", "fungal", "virus", "detect", "imag", "symptom", "classifi"]

# Configure Gemini once, up front.
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================================
# Cell 3: Gamification, Academic Engine, IoT MapReduce
# ==========================================================

# Gamification
class GamificationSystem:
    def __init__(self):
        self.tasks = {
            "upload_image": {"name": "Upload Disease Image", "points": 10, "completed": False, "icon": "🖼️"},
            "check_sensors": {"name": "Check Sensor Data", "points": 15, "completed": False, "icon": "📊"},
            "research_query": {"name": "Perform Research Query", "points": 20, "completed": False, "icon": "🔍"}
        }
        self.total_points = 0
        self.level = 1
        self.achievements = []

    def complete_task(self, task_id):
        if task_id in self.tasks and not self.tasks[task_id]["completed"]:
            self.tasks[task_id]["completed"] = True
            pts = self.tasks[task_id]["points"]
            self.total_points += pts
            new_lvl = (self.total_points // 50) + 1
            if new_lvl > self.level:
                self.level = new_lvl
                self.achievements.append(f"🎉 Level {self.level}!")
            if sum(1 for t in self.tasks.values() if t["completed"]) == len(self.tasks):
                if "🏆 Master" not in self.achievements:
                    self.achievements.append("🏆 Master!")
            return True, pts
        return False, 0

    def get_status(self):
        s = f"### 🎮 Status\n**Points:** {self.total_points} 🌟\n**Level:** {self.level}\n\n### ✅ Missions\n"
        for tid, t in self.tasks.items():
            s += f"{'✅' if t['completed'] else '⬜'} {t['icon']} **{t['name']}** - {t['points']}pts\n"
        if self.achievements:
            s += "\n### 🏆 Achievements\n" + "\n".join(f"- {a}" for a in self.achievements)
        return s

    def reset_daily_tasks(self):
        for t in self.tasks.values():
            t["completed"] = False
        self.total_points = 0
        self.level = 1
        self.achievements = []
        return "🔄 Reset!"

game_system = GamificationSystem()

# ===== DocumentService =====
class DocumentService:
    def __init__(self):
        self.documents = {}
        self.doc_names = {}

    def read_pdf(self, path):
        try:
            return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        except Exception:
            return ""

    def load_documents(self, folder):
        if not os.path.isdir(folder):
            print(f"⚠️ PDF folder not found: {folder}")
            return self.documents
        for i, f in enumerate(sorted(os.listdir(folder))):
            if f.lower().endswith(".pdf"):
                text = self.read_pdf(os.path.join(folder, f))
                if text.strip():
                    self.documents[i] = text
                    self.doc_names[i] = f
        print(f"✅ Loaded {len(self.documents)} PDFs")
        return self.documents


# ===== TextProcessingService =====
class TextProcessingService:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = {
            'the', 'and', 'over', 'under', 'between', 'among', 'through',
            'for', 'with', 'are', 'was', 'were', 'from', 'all', 'any', 'some'
        }

    def process(self, text, key_terms):
        words = re.findall(r"\w+", text.lower())
        stems = [
            self.stemmer.stem(w)
            for w in words
            if w not in self.stop_words and len(w) > 2
        ]
        return [w for w in stems if any(k in w for k in key_terms)]


# ===== IndexService =====
class IndexService:
    def __init__(self, text_service):
        self.index = defaultdict(lambda: defaultdict(int))
        self.text_service = text_service

    def build_index(self, documents, key_terms):
        for doc_id, text in documents.items():
            terms = self.text_service.process(text, key_terms)
            for term in terms:
                self.index[term][doc_id] += 1
        print(f"🔨 Index built with {len(self.index)} terms")


# ===== SearchService =====
class SearchService:
    def __init__(self, index_service, text_service):
        self.index_service = index_service
        self.text_service = text_service

    def search(self, query, key_terms):
        scores = defaultdict(int)
        terms = self.text_service.process(query, key_terms)

        for term in terms:
            for doc_id, count in self.index_service.index.get(term, {}).items():
                scores[doc_id] += count

        return scores


# ===== ResultService =====
class ResultService:
    def __init__(self, document_service):
        self.document_service = document_service

    def rank(self, scores, top_k=3):
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (doc_id, score, self.document_service.doc_names[doc_id])
            for doc_id, score in ranked
        ]


# IoT MapReduce
def load_iot_json(folder):
    records = []
    if not os.path.isdir(folder):
        print(f"⚠️ IoT folder not found: {folder}")
        return records
    for f in os.listdir(folder):
        if f.endswith(".json"):
            try:
                with open(os.path.join(folder, f), encoding="utf-8") as fp:
                    data = json.load(fp)
                    records.extend(data if isinstance(data, list) else [data])
            except Exception:
                pass
    print(f"✅ Loaded {len(records)} IoT records")
    return records


def map_sensor_data(records):
    mapped = []
    for r in records:
        try:
            raw = r.get("value")
            if isinstance(raw, str):
                raw = json.loads(raw)
            for sensor in ["temperature", "humidity", "soil"]:
                if sensor in raw:
                    mapped.append((sensor, float(raw[sensor])))
        except Exception:
            pass
    print(f"🗺️ Mapped {len(mapped)} values")
    return mapped


def reduce_sensor_data(mapped):
    if not mapped:
        return {}
    buckets = defaultdict(list)
    for sensor, val in mapped:
        buckets[sensor].append(val)
    result = {}
    for sensor, vals in buckets.items():
        result[sensor] = {
            "count": len(vals),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "avg": round(sum(vals) / len(vals), 2)
        }
    print(f"📉 Reduced {len(result)} sensors")
    return result


def run_mapreduce(folder):
    records = load_iot_json(folder)
    if not records:
        return
    mapped = map_sensor_data(records)
    reduced = reduce_sensor_data(mapped)
    if reduced:
        if db_set("iot_analysis", reduced):
            print("🔥 Saved to Firebase")


run_mapreduce(IOT_FOLDER)

# ==========================================================
# Cell 4: RAG System, PDF/HF Retriever, Image + LLM Pipeline
# ==========================================================

# ---------- RAG SYSTEM ----------
class GeminiRAG:
    def __init__(self, engine, api_key: str):
        print("🔧 Configuring Gemini...")
        genai.configure(api_key=api_key)
        self.engine = engine

        model_name = "gemini-2.5-flash"
        print(f"🎯 Using model: {model_name}")

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.8,
                "max_output_tokens": 2048,
            }
        )
        print("✅ Gemini configured successfully\n")

    def create_context(self, ranked_docs, max_chars_per_doc=1500):
        parts = []
        for doc_id, score, filename in ranked_docs:
            content = self.engine.document_service.documents[doc_id]
            snippet = content[:max_chars_per_doc]
            if len(content) > max_chars_per_doc:
                snippet += "..."
            parts.append(
                f"=== DOCUMENT: {filename} (Score: {score}) ===\n{snippet}\n"
            )
        return "\n\n".join(parts)

    def query(self, question, top_k=2):
        ranked_docs = self.engine.search(question, top_k)
        if not ranked_docs:
            return {"answer": "❌ No relevant documents found.", "sources": "None"}

        context = self.create_context(ranked_docs)

        prompt = f"""You are an academic research assistant specializing in plant diseases and image classification.

TASK: Answer the question using ONLY the information from the provided documents.

RULES:
1. Base your answer EXCLUSIVELY on the provided documents
2. If the information is not in the documents, clearly state that
3. Cite your sources (mention document name or ID)
4. Structure your answer with clear points
5. Be concise but comprehensive

QUESTION:
{question}

DOCUMENTS:

{context}

ANSWER:"""
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"⚠️ Gemini error: {e}"

        sources = "\n".join(
            [f"📄 {name} (Score: {score})" for _, score, name in ranked_docs]
        )

        return {"answer": answer, "sources": sources}


# ---------- SIMPLE SEARCH ENGINE (Microservices Wrapper) ----------
class AcademicSearchEngine:
    def __init__(self, document_service, search_service, result_service):
        self.document_service = document_service
        self.search_service = search_service
        self.result_service = result_service

    def search(self, query, top_k=3):
        scores = self.search_service.search(query, KEY_TERMS)
        return self.result_service.rank(scores, top_k)


print("\n🚀 INITIALIZING...\n")

document_service = DocumentService()
text_service = TextProcessingService()
index_service = IndexService(text_service)
search_service = SearchService(index_service, text_service)
result_service = ResultService(document_service)

documents = document_service.load_documents(PDF_FOLDER)
index_service.build_index(documents, KEY_TERMS)

engine = AcademicSearchEngine(
    document_service,
    search_service,
    result_service
)

rag_system = GeminiRAG(engine, GEMINI_API_KEY)


def gemini_generate(prompt: str, max_output_tokens: int = 512) -> str:
    """Single helper for ad-hoc Gemini generations (replaces the local flan-t5 model)."""
    try:
        resp = rag_system.model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_output_tokens},
        )
        return (resp.text or "").strip()
    except Exception as e:
        return f"⚠️ {e}"


# ---------- IMAGE PIPELINE ----------
IMAGE_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
device = 0 if torch.cuda.is_available() else -1


def build_image_classifier():
    """Build the image classifier with an explicit image processor.

    transformers 5.x can no longer auto-detect this older model's image
    processor (its preprocessor_config.json has no `image_processor_type`),
    so we load the model and processor explicitly.
    """
    model = AutoModelForImageClassification.from_pretrained(IMAGE_MODEL)
    try:
        processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
    except Exception:
        # Fall back to the MobileNetV2 processor referenced by the base model.
        from transformers import MobileNetV2ImageProcessor
        processor = MobileNetV2ImageProcessor.from_pretrained(IMAGE_MODEL)
    return pipeline(
        "image-classification",
        model=model,
        image_processor=processor,
        device=device,
    )


print("🖼️ Loading image classifier...")
clf = build_image_classifier()
print("✅ Image classifier ready")


# ---------- TFIDF RETRIEVERS (used by rag_explain) ----------
class TfidfRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.X = self.vectorizer.fit_transform(docs)

    def search(self, query, top_k=3):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.X).flatten()
        return [self.docs[i] for i in sims.argsort()[::-1][:top_k]]


def pdf_to_text(path):
    return "\n".join([p.extract_text() or "" for p in PdfReader2(path).pages])


def chunk_text(text, size=1200, overlap=200):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


def build_pdf_retriever(folder):
    if not os.path.isdir(folder):
        return None
    paths = [os.path.join(folder, n) for n in os.listdir(folder) if n.endswith(".pdf")]
    chunks = []
    for p in paths:
        try:
            chunks.extend(chunk_text(pdf_to_text(p)))
        except Exception:
            pass
    return TfidfRetriever(chunks) if chunks else None


def load_hf_chunks(model_id):
    if hf_hub_download is None:
        return []
    try:
        path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(path, encoding="utf-8") as f:
            return chunk_text(f.read())
    except Exception:
        return []


pdf_retriever = build_pdf_retriever(PDF_FOLDER)
hf_chunks = load_hf_chunks(IMAGE_MODEL)
hf_retriever = TfidfRetriever(hf_chunks) if hf_chunks else None

print("✅ READY - Move to Gradio interface")

# ==========================================================
# Cell 5: Functions for Diagnosis, RAG explain, IoT, Helper Bot
# ==========================================================

def parse_label(label):
    """Parse the model's human-readable labels.

    This model emits labels like:
        "Tomato with Late Blight", "Apple Scab", "Healthy Apple",
        "Corn (Maize) with Common Rust", "Tomato Yellow Leaf Curl Virus"
    (NOT the PlantVillage "Plant___Disease" format). We split on " with "
    when present and detect the "Healthy" prefix.
    """
    label = label.strip()
    lower = label.lower()

    is_healthy = lower.startswith("healthy")

    if " with " in label:
        plant, disease_raw = label.split(" with ", 1)
        plant = plant.strip()
        disease_clean = disease_raw.strip()
    elif is_healthy:
        # "Healthy Apple" / "Healthy Tomato Plant" -> plant = the rest
        plant = re.sub(r"^healthy\s+", "", label, flags=re.IGNORECASE)
        plant = re.sub(r"\s+plant$", "", plant, flags=re.IGNORECASE).strip()
        disease_raw = "healthy"
        disease_clean = "Healthy"
    else:
        # e.g. "Apple Scab", "Tomato Yellow Leaf Curl Virus", "Cedar Apple Rust"
        # Heuristic: first word is the plant/crop, the rest is the condition.
        parts = label.split()
        plant = parts[0] if parts else "Plant"
        disease_raw = label
        disease_clean = " ".join(parts[1:]).strip() if len(parts) > 1 else label

    if is_healthy:
        disease_raw = "healthy"

    return plant, disease_raw, disease_clean


def rag_explain(plant, disease_raw, disease_clean):
    if "healthy" in disease_raw.lower():
        return f"✅ Healthy plant\n🌿 **{plant}**"

    query = f"{plant} {disease_clean} symptoms treatment"
    pdf_ctx = hf_ctx = ""

    if pdf_retriever:
        pdf_ctx = "\n\n".join([h[:700] for h in pdf_retriever.search(query, 2)])
    if hf_retriever:
        hf_ctx = "\n\n".join([h[:700] for h in hf_retriever.search(query, 1)])

    if not pdf_ctx and not hf_ctx:
        return f"⚠ Disease detected\n🌿 **{plant}**\n🩺 **{disease_clean}**"

    sources = (f"PDF:\n{pdf_ctx}\n\n" if pdf_ctx else "") + (f"HF:\n{hf_ctx}" if hf_ctx else "")
    prompt = (
        "Explain briefly using these sources. Keep it short and clear.\n"
        f"Plant: {plant}\nDisease: {disease_clean}\n\nSOURCES:\n{sources}\n\nAnswer:"
    )
    out = gemini_generate(prompt, max_output_tokens=300)
    return f"⚠ **{plant}** - **{disease_clean}**\n\n{out}"


def gemini_treatment(plant, disease):
    prompt = f"Treatment for {plant} {disease}. Give 5-7 bullet points."
    return gemini_generate(prompt, max_output_tokens=512)


def diagnose(img):
    if img is None:
        return "⚠️ Upload image first"
    try:
        pred = clf(img)
    except Exception as e:
        return f"❌ Classification failed: {e}"
    if not pred:
        return "❌ Could not classify the image."
    top = pred[0]
    confidence = top.get("score", 0)
    plant, raw, clean = parse_label(top["label"])
    expl = rag_explain(plant, raw, clean)
    treat = "✅ No treatment needed" if "healthy" in raw.lower() else gemini_treatment(plant, clean)
    success, pts = game_system.complete_task("upload_image")
    result = (
        f"{expl}\n\n"
        f"🔎 Confidence: {confidence:.1%}\n\n"
        f"{'=' * 40}\n✅ **Treatment**\n{treat}"
    )
    if success:
        result += f"\n\n🎮 +{pts}pts!"
    return result


def get_sensor_data(feed, limit):
    try:
        r = requests.get(
            f"{BASE_URL}/history",
            params={"feed": feed, "limit": limit},
            timeout=30,
        )
        data = r.json()
        if "data" not in data:
            return "❌ Error: unexpected response from server."
        success, pts = game_system.complete_task("check_sensors")
        vals = [str(s.get("value")) for s in data["data"]]
        res = f"Feed: {feed}\nSamples: {len(vals)}\n\n" + "\n".join(vals)
        if success:
            res += f"\n\n🎮 +{pts}pts!"
        return res
    except Exception as e:
        return f"❌ {e}"


def plot_all_feeds(limit):
    figs = []
    for feed in ["temperature", "humidity", "soil"]:
        try:
            r = requests.get(
                f"{BASE_URL}/history",
                params={"feed": feed, "limit": limit},
                timeout=30,
            )
            data = r.json()
            if "data" not in data or not data["data"]:
                figs.append(None)
                continue
            df = pd.DataFrame(data["data"])
            df["created_at"] = pd.to_datetime(df["created_at"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.sort_values("created_at")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(df["created_at"], df["value"], marker="o", color="#4CAF50", lw=2)
            ax.set_title(feed.capitalize(), fontsize=14, weight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            figs.append(fig)
        except Exception:
            figs.append(None)
    return figs


def helper_bot_reply(msg, history):
    sys_prompt = "You're a helpful assistant for Plant Disease System. Explain features in Hebrew briefly."
    try:
        # history is a list of {"role": ..., "content": ...} (Gradio 6 messages format)
        contents = [{"role": "user", "parts": [{"text": sys_prompt}]}]
        for turn in history or []:
            role = "user" if turn.get("role") == "user" else "model"
            text = turn.get("content", "")
            if text:
                contents.append({"role": role, "parts": [{"text": str(text)}]})
        contents.append({"role": "user", "parts": [{"text": msg}]})
        resp = rag_system.model.generate_content(contents)
        return resp.text
    except Exception as e:
        return f"😅 {e}"


def query_handler(q, top_k):
    if not q or not q.strip():
        return "⚠️ Enter question", ""
    res = rag_system.query(q.strip(), int(top_k))
    success, pts = game_system.complete_task("research_query")
    ans = res["answer"]
    if success:
        ans += f"\n\n🎮 +{pts}pts!"
    return ans, res["sources"]

# ==========================================================
# CELL 6: GRADIO UI
# ==========================================================

with gr.Blocks(
    title="🌱 Plant Disease System",
    theme=gr.themes.Soft(primary_hue="green"),
    css="""
        .main-header {text-align:center; padding:20px; background:linear-gradient(135deg,#667eea,#764ba2); color:white; border-radius:10px; margin-bottom:20px;}
        .section-header {background:linear-gradient(90deg,#11998e,#38ef7d); color:white; padding:15px; border-radius:8px; margin:15px 0;}
        .game-card {background:linear-gradient(135deg,#11998e,#38ef7d); color:white; padding:25px; border-radius:15px; margin:15px 0; box-shadow:0 5px 20px rgba(0,0,0,0.2);}
        .floating-chat-btn {position:fixed!important; bottom:25px!important; right:25px!important; width:60px!important; height:60px!important; border-radius:50%!important; background:linear-gradient(135deg,#667eea,#764ba2)!important; color:white!important; font-size:28px!important; z-index:10000!important; box-shadow:0 4px 20px rgba(102,126,234,0.5)!important;}
        .chat-window {position:fixed!important; bottom:100px!important; right:25px!important; width:400px!important; height:550px!important; background:white!important; border-radius:20px!important; box-shadow:0 10px 50px rgba(0,0,0,0.3)!important; z-index:9999!important; border:3px solid #667eea!important;}
    """
) as demo:

    gr.HTML('<div class="main-header"><h1>🌱 Plant Disease Research & Monitoring System</h1><p>AI-Powered Platform with Gamification</p></div>')

    with gr.Tabs():
        # Tab 1: Disease Diagnosis
        with gr.Tab("🔬 Disease Diagnosis"):
            gr.HTML('<div class="section-header">🔬 AI Disease Detection</div>')
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="📷 Upload Image", height=400)
                    diag_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                with gr.Column():
                    diag_output = gr.Textbox(label="📋 Report", lines=20)
            diag_btn.click(diagnose, img_input, diag_output)

        # Tab 2: Research Assistant
        with gr.Tab("📚 Research Assistant"):
            gr.HTML('<div class="section-header">🔍 Ask Questions</div>')
            with gr.Row():
                query_input = gr.Textbox(label="🔍 Question", lines=3, scale=2)
                topk_slider = gr.Slider(1, 5, value=2, step=1, label="📄 Docs", scale=1)
            submit_btn = gr.Button("🚀 Search", variant="primary", size="lg")
            with gr.Row():
                ans_output = gr.Markdown("*Answer here...*")
                src_output = gr.Textbox(label="📚 Sources", lines=4)
            submit_btn.click(query_handler, [query_input, topk_slider], [ans_output, src_output])

        # Tab 3: Sensor Data
        with gr.Tab("📊 Sensor Data"):
            gr.HTML('<div class="section-header">📡 Real-time Data</div>')
            with gr.Row():
                feed_input = gr.Dropdown(["humidity", "soil", "temperature", "json"], value="humidity", label="🌡️ Feed")
                limit_input = gr.Slider(1, 100, value=10, step=1, label="📊 Samples")
            fetch_btn = gr.Button("📥 Fetch", variant="primary", size="lg")
            output_box = gr.Textbox(label="📋 Values", lines=15)
            fetch_btn.click(get_sensor_data, [feed_input, limit_input], output_box)

        # Tab 4: Visual Dashboard
        with gr.Tab("📈 Dashboard"):
            gr.HTML('<div class="section-header">📈 Monitoring Dashboard</div>')
            limit_slider = gr.Slider(5, 50, value=20, step=1, label="📊 Points")
            btn = gr.Button("🔄 Refresh", variant="primary", size="lg")
            with gr.Row():
                temp_plot = gr.Plot(label="🌡️ Temperature")
                hum_plot = gr.Plot(label="💧 Humidity")
                soil_plot = gr.Plot(label="🌱 Soil")
            btn.click(plot_all_feeds, limit_slider, [temp_plot, hum_plot, soil_plot])

        # Tab 5: Daily Missions
        with gr.Tab("🎮 Daily Missions"):
            gr.HTML('<div class="section-header">🎮 Daily Missions</div>')
            gr.HTML('<div class="game-card"><h2>🏆 Welcome!</h2><p>Complete missions and earn points!</p></div>')
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("""
## 📋 Your Tasks

**How to Play:**
1. Complete tasks in different tabs
2. Earn points for each task
3. Level up and unlock achievements!

**Daily Tasks:**

🖼️ **Upload Image** (10pts) - Go to Disease Diagnosis
📊 **Check Sensors** (15pts) - Go to Sensor Data
🔍 **Research Query** (20pts) - Go to Research Assistant
                    """)
                with gr.Column(scale=1):
                    status_display = gr.Markdown(game_system.get_status())
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Status", variant="secondary", size="lg")
                        reset_btn = gr.Button("🔄 Reset", variant="stop", size="lg")
                    reset_msg = gr.Textbox(visible=False)

            def refresh_status():
                return game_system.get_status()

            def reset_tasks():
                msg = game_system.reset_daily_tasks()
                return game_system.get_status(), msg

            refresh_btn.click(refresh_status, outputs=status_display)
            reset_btn.click(reset_tasks, outputs=[status_display, reset_msg])

    # Floating Chatbot
    chat_visible = gr.State(False)

    with gr.Column(visible=False, elem_classes="chat-window") as chat_window:
        gr.HTML('<div style="background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:15px; font-weight:bold;">🤖 Personal Assistant</div>')
        chatbot_display = gr.Chatbot(label="", height=400, show_label=False)
        with gr.Row():
            helper_input = gr.Textbox(placeholder="Ask me...", show_label=False, scale=4, container=False)
            helper_send = gr.Button("📤", scale=1, variant="primary")

    toggle_btn = gr.Button("💬", elem_classes="floating-chat-btn")

    def toggle_chat(visible):
        return not visible, gr.update(visible=not visible)

    toggle_btn.click(toggle_chat, chat_visible, [chat_visible, chat_window])

    def respond(msg, hist):
        hist = hist or []
        if not msg or not msg.strip():
            return "", hist
        bot_resp = helper_bot_reply(msg, hist)
        hist.append({"role": "user", "content": msg})
        hist.append({"role": "assistant", "content": bot_resp})
        return "", hist

    helper_send.click(respond, [helper_input, chatbot_display], [helper_input, chatbot_display])
    helper_input.submit(respond, [helper_input, chatbot_display], [helper_input, chatbot_display])

    gr.Markdown("---\n**System Status:** ✅ RAG | ✅ Image AI | ✅ IoT | ✅ Chatbot | ✅ Gamification")

if __name__ == "__main__":
    print("🌐 Launching...")
    demo.launch(share=True, debug=True)
