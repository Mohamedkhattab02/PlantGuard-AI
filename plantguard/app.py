"""Gradio application — an elegant, English-only UI over the service layer.

Implements the UI-facing roadmap items: per-user ``gr.State`` ([1.1]), progress
bars ([2.1]), webcam + batch capture ([2.7]), surfaced aggregates & alerts
([3.4]/[2.2]), history gallery ([2.3]), PDF export ([2.4]), weather ([2.8]) and
an env-driven launch ([4.2]). Heavy services are built once in
:func:`build_services`; nothing expensive happens at import time.

Note: in Gradio 6 the ``theme`` and ``css`` arguments live on ``launch()`` rather
than the ``Blocks`` constructor, so they are applied in :func:`main`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from matplotlib.figure import Figure
from PIL import Image

from .config import Settings, configure_logging, load_settings
from .i18n import t
from .services import alerts as alerts_svc
from .services import weather as weather_svc
from .services.documents import load_corpus
from .services.gamification import GamificationService, GameState
from .services.history import DiagnosisEntry, HistoryService
from .services.image import ImageService
from .services.iot import FEEDS, IotService
from .services.rag import RagService
from .services.reports import build_report_pdf
from .services.store import Store

log = logging.getLogger(__name__)

# Elegant theme (applied at launch in Gradio 6).
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#eef3f0",
    block_background_fill="white",
    block_radius="18px",
    block_shadow="0 6px 24px rgba(16,24,40,0.05)",
    block_border_width="1px",
    button_primary_background_fill="linear-gradient(135deg,#10b981,#0d9488)",
    button_primary_background_fill_hover="linear-gradient(135deg,#0d9488,#047857)",
    button_primary_text_color="white",
)

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.gradio-container { font-family:'Inter',system-ui,sans-serif !important; max-width:1180px !important; margin:auto !important; }
footer { display:none !important; }

/* Hero header */
.pg-hero {
  background:linear-gradient(135deg,#065f46 0%,#0d9488 55%,#10b981 100%);
  color:#fff; border-radius:24px; padding:34px 28px; text-align:center;
  box-shadow:0 18px 50px rgba(13,148,136,.35); margin-bottom:18px; position:relative; overflow:hidden;
}
.pg-hero::after { content:""; position:absolute; inset:0; background:radial-gradient(circle at 82% -20%,rgba(255,255,255,.25),transparent 60%); pointer-events:none; }
.pg-hero-badge { font-size:42px; line-height:1; }
.pg-hero h1 { margin:8px 0 4px; font-size:34px; font-weight:800; letter-spacing:-.5px; }
.pg-hero p { margin:0; opacity:.92; font-size:15px; }
.pg-pills { margin-top:16px; display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }
.pg-pill { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); padding:6px 14px; border-radius:999px; font-size:13px; font-weight:500; }

/* Section headers */
.pg-section { font-size:18px; font-weight:700; color:#0f766e; border-left:4px solid #10b981; padding:2px 0 2px 12px; margin:4px 0 12px; }

/* Cards */
.pg-card { background:#fff !important; border:1px solid #eef1f0 !important; border-radius:18px !important; box-shadow:0 6px 24px rgba(16,24,40,.05) !important; }

/* Footer */
.pg-footer { text-align:center; color:#6b7280; font-size:13px; margin-top:18px; padding:14px; border-top:1px solid #e5e7eb; }
.pg-footer b { color:#0f766e; }

/* Floating chat button */
.pg-chat-btn {
  position:fixed !important; bottom:26px !important; right:26px !important;
  width:62px !important; height:62px !important; min-width:62px !important; border-radius:50% !important;
  background:linear-gradient(135deg,#10b981,#0d9488) !important; color:#fff !important; font-size:26px !important;
  border:none !important; z-index:10000 !important; box-shadow:0 10px 30px rgba(13,148,136,.5) !important;
  transition:transform .15s ease;
}
.pg-chat-btn:hover { transform:translateY(-3px) scale(1.05); }

/* Chat window */
.pg-chat-window {
  position:fixed !important; bottom:100px !important; right:26px !important;
  width:380px !important; max-width:92vw !important; height:560px !important;
  background:#fff !important; border-radius:22px !important; overflow:hidden !important;
  box-shadow:0 24px 70px rgba(16,24,40,.32) !important; z-index:9999 !important; border:1px solid #e5e7eb !important;
}
.pg-chat-header { background:linear-gradient(135deg,#065f46,#0d9488); color:#fff; padding:15px 18px; font-weight:700; display:flex; align-items:center; gap:10px; font-size:16px; }
.pg-chat-avatar { width:34px; height:34px; border-radius:50%; background:rgba(255,255,255,.2); display:inline-flex; align-items:center; justify-content:center; font-size:18px; }
.pg-chat-status { margin-left:auto; font-size:11px; font-weight:500; opacity:.92; }
.pg-chat-status::before { content:"●"; color:#6ee7b7; margin-right:4px; }
.pg-chat-window .message.user { background:linear-gradient(135deg,#10b981,#0d9488) !important; color:#fff !important; border-radius:16px 16px 4px 16px !important; }
.pg-chat-window .message.bot { background:#f3f4f6 !important; border-radius:16px 16px 16px 4px !important; }
"""


@dataclass
class Services:
    settings: Settings
    store: Store
    image: ImageService
    rag: RagService
    iot: IotService
    game: GamificationService
    history: HistoryService


def build_services(settings: Settings) -> Services:
    """Construct every service once (loads corpus, embeddings, Firebase)."""
    store = Store(settings.firebase_key_path, settings.database_url)
    corpus = load_corpus(settings.articles_dir)
    return Services(
        settings=settings,
        store=store,
        image=ImageService(settings),
        rag=RagService(settings, corpus),
        iot=IotService(settings.base_url, store),
        game=GamificationService(store),
        history=HistoryService(store),
    )


def _make_figure(df, title: str, color: str) -> Figure:
    """Build a standalone Figure (no pyplot global registry → no leak, [2.3])."""
    fig = Figure(figsize=(6, 3))
    ax = fig.subplots()
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#9ca3af")
    else:
        ax.plot(df["created_at"], df["value"], marker="o", color=color, lw=2)
        ax.fill_between(df["created_at"], df["value"], alpha=0.08, color=color)
        ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=13, weight="bold", color="#374151")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def build_app(svc: Services) -> gr.Blocks:
    s = svc.settings
    lang = "en"

    def H(key: str) -> str:
        return f'<div class="pg-section">{t(key, lang)}</div>'

    def _persist_image(img: Image.Image) -> str:
        folder = s.cache_dir / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{int(time.time() * 1000)}.png"
        img.save(path)
        return str(path)

    def _history_view(user_id: str):
        entries = svc.history.list(user_id or "guest")
        if not entries:
            return [], t("history_empty")
        gallery = [(e.image_path, e.caption) for e in entries if e.image_path]
        return gallery, "\n".join(f"- {e.caption}" for e in entries)

    # ---------------- handlers ----------------
    def on_diagnose(img, game: GameState, progress=gr.Progress()):
        try:
            progress(0.15, desc="Validating image")
            result = svc.image.diagnose(img)
        except ValueError as exc:
            msg = t("no_image") if img is None else f"❌ {exc}"
            return (msg, game, None, svc.game.status_md(game), gr.update(), gr.update())
        except Exception as exc:  # noqa: BLE001
            log.exception("Diagnosis failed")
            return (f"❌ {exc}", game, None, svc.game.status_md(game), gr.update(), gr.update())

        plant, disease, healthy = result["plant"], result["disease"], result["healthy"]
        conf = result["confidence"]
        img_path = _persist_image(result["image"])

        lines: list[str] = []
        if result["low_confidence"]:
            lines.append(f"> {t('low_confidence')}")
        if healthy:
            lines.append(f"### {t('healthy')} — 🌿 **{plant}**")
            treatment_plain = "No treatment needed."
        else:
            progress(0.55, desc="Explanation & treatment")
            et = svc.rag.explain_and_treat(plant, disease)
            lines.append(f"### {t('disease_detected')}: **{plant}** — *{disease}*")
            if et.get("severity"):
                lines.append(f"**Severity:** {et['severity']}")
            if et.get("explanation"):
                lines.append(et["explanation"])
            bullets = et.get("treatment") or []
            treatment_plain = "\n".join(f"- {b}" for b in bullets) if bullets else "See documents."

        lines.append(f"🔎 **{t('confidence')}:** {conf:.1%}")
        top3 = " · ".join(f"{lbl} ({sc:.0%})" for lbl, sc in result["top_k"])
        lines.append(f"**Top-3:** {top3}")
        lines.append(f"\n---\n#### {t('treatment')}\n{treatment_plain}")

        success, pts = svc.game.complete(game, "upload_image")
        if success:
            lines.append(f"\n🎮 **+{pts} {t('points_earned')}!**")

        svc.history.record(
            game.user_id,
            DiagnosisEntry(
                plant=plant,
                disease="Healthy" if healthy else disease,
                confidence=conf,
                healthy=healthy,
                image_path=img_path,
            ),
        )
        last_entry = {
            "plant": plant,
            "disease": "Healthy" if healthy else disease,
            "confidence": conf,
            "healthy": healthy,
            "image_path": img_path,
            "treatment": treatment_plain,
            "sources": "",
        }
        gallery, hist_md = _history_view(game.user_id)
        progress(1.0, desc="Done")
        return (
            "\n\n".join(lines),
            game,
            last_entry,
            svc.game.status_md(game),
            gr.update(value=gallery),
            gr.update(value=hist_md),
        )

    def on_batch(files, progress=gr.Progress()):
        if not files:
            return [], "_No images uploaded._"
        gallery, rows = [], []
        for i, f in enumerate(files):
            progress((i + 1) / len(files), desc=f"Image {i + 1}/{len(files)}")
            path = f if isinstance(f, str) else getattr(f, "name", None)
            try:
                res = svc.image.diagnose(Image.open(path))
                status = "✅ Healthy" if res["healthy"] else f"⚠️ {res['disease']}"
                caption = f"{res['plant']} — {status} ({res['confidence']:.0%})"
            except Exception as exc:  # noqa: BLE001
                caption = f"❌ {exc}"
            gallery.append((path, caption))
            rows.append(f"- {caption}")
        return gallery, "\n".join(rows)

    def on_export(last_entry):
        if not last_entry:
            return gr.update(value=None, visible=False)
        path = build_report_pdf(last_entry, s.reports_dir)
        return gr.update(value=path, visible=True)

    def on_search(question, top_k, game: GameState):
        res = svc.rag.answer(question, int(top_k))
        ans = res["answer"]
        success, pts = svc.game.complete(game, "research_query")
        if success:
            ans += f"\n\n🎮 **+{pts} {t('points_earned')}!**"
        return ans, res["sources"], game, svc.game.status_md(game)

    def on_fetch(feed, limit, game: GameState):
        try:
            rows = svc.iot.get_history(feed, int(limit))
        except Exception as exc:  # noqa: BLE001
            log.warning("Sensor fetch failed: %s", exc)
            return (f"❌ Could not reach sensor server: {exc}", game, svc.game.status_md(game))
        vals = "\n".join(str(r.get("value")) for r in rows)
        success, pts = svc.game.complete(game, "check_sensors")
        text = f"Feed: {feed}\nSamples: {len(rows)}\n\n{vals}"
        if success:
            text += f"\n\n🎮 +{pts} {t('points_earned')}!"
        return text, game, svc.game.status_md(game)

    def on_dashboard(limit, city, progress=gr.Progress()):
        progress(0.2, desc="Fetching feeds")
        series = svc.iot.fetch_all_series(int(limit))
        temp_fig = _make_figure(series.get("temperature"), t("temp_label"), "#ef4444")
        hum_fig = _make_figure(series.get("humidity"), t("humidity_label"), "#3b82f6")
        soil_fig = _make_figure(series.get("soil"), t("soil_label"), "#10b981")
        reduced = {
            feed: {
                "count": int(df.shape[0]),
                "min": round(float(df["value"].min()), 2),
                "max": round(float(df["value"].max()), 2),
                "avg": round(float(df["value"].mean()), 2),
            }
            for feed, df in series.items()
            if df is not None and not df.empty
        }
        stats_md = "\n".join(
            f"**{feed.capitalize()}** — avg {v['avg']} (min {v['min']}, max {v['max']}, n={v['count']})"
            for feed, v in reduced.items()
        ) or "_No live data._"
        alerts = alerts_svc.evaluate(reduced, s)

        progress(0.8, desc="Weather")
        weather = weather_svc.get_weather(city or s.weather_default_city)
        return temp_fig, hum_fig, soil_fig, stats_md, alerts_svc.alerts_md(alerts), weather_svc.weather_md(weather)

    def on_user_change(user_id):
        game = svc.game.load(user_id or "guest")
        gallery, hist_md = _history_view(game.user_id)
        return game, svc.game.status_md(game), gr.update(value=gallery), gr.update(value=hist_md)

    def on_chat(message, history):
        history = history or []
        if not message or not message.strip():
            return "", history
        reply = svc.rag.chat(message, history, lang)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return "", history

    # ---------------- UI ----------------
    with gr.Blocks(title="🌱 PlantGuard AI", analytics_enabled=False) as demo:
        game_state = gr.State(GameState(user_id="guest"))
        last_entry_state = gr.State(None)
        chat_visible = gr.State(False)

        gr.HTML(
            '<div class="pg-hero">'
            '<div class="pg-hero-badge">🌱</div>'
            "<h1>PlantGuard&nbsp;AI</h1>"
            f"<p>{t('app_subtitle')}</p>"
            '<div class="pg-pills">'
            '<span class="pg-pill">🔬 Vision AI</span>'
            '<span class="pg-pill">📚 Semantic RAG</span>'
            '<span class="pg-pill">📡 IoT + Alerts</span>'
            '<span class="pg-pill">🗂️ History</span>'
            '<span class="pg-pill">🌦️ Weather</span>'
            "</div></div>"
        )
        user_box = gr.Textbox(label=t("user_label"), placeholder="guest")

        with gr.Tabs():
            with gr.Tab(t("tab_diagnosis")):
                gr.HTML(H("diag_header"))
                with gr.Row():
                    with gr.Column(elem_classes="pg-card"):
                        img_input = gr.Image(type="pil", sources=["upload", "webcam"],
                                             label=t("upload_label"), height=360)
                        with gr.Row():
                            analyze_btn = gr.Button(t("analyze_btn"), variant="primary", size="lg", scale=2)
                            export_btn = gr.Button(t("export_pdf_btn"), size="lg", scale=1)
                        export_file = gr.File(label="PDF report", visible=False)
                    with gr.Column(elem_classes="pg-card"):
                        diag_output = gr.Markdown("*Upload a leaf image and press Analyze.*")
                with gr.Accordion(t("batch_label"), open=False):
                    batch_input = gr.File(file_count="multiple", file_types=["image"])
                    batch_btn = gr.Button(t("batch_btn"), variant="primary")
                    batch_gallery = gr.Gallery(columns=4, height=240)
                    batch_md = gr.Markdown()

            with gr.Tab(t("tab_research")):
                gr.HTML(H("research_header"))
                with gr.Group(elem_classes="pg-card"):
                    with gr.Row():
                        query_input = gr.Textbox(label=t("question_label"), lines=3, scale=3)
                        topk_slider = gr.Slider(1, 5, value=2, step=1, label=t("docs_label"), scale=1)
                    search_btn = gr.Button(t("search_btn"), variant="primary", size="lg")
                with gr.Row():
                    ans_output = gr.Markdown("*Ask a question grounded in the research library.*")
                    src_output = gr.Textbox(label=t("sources_label"), lines=5)

            with gr.Tab(t("tab_sensors")):
                gr.HTML(H("sensors_header"))
                with gr.Group(elem_classes="pg-card"):
                    with gr.Row():
                        feed_input = gr.Dropdown(list(FEEDS) + ["json"], value="humidity", label=t("feed_label"))
                        limit_input = gr.Slider(1, 100, value=10, step=1, label=t("samples_label"))
                    fetch_btn = gr.Button(t("fetch_btn"), variant="primary", size="lg")
                values_box = gr.Textbox(label=t("values_label"), lines=14)

            with gr.Tab(t("tab_dashboard")):
                gr.HTML(H("dashboard_header"))
                with gr.Group(elem_classes="pg-card"):
                    with gr.Row():
                        limit_slider = gr.Slider(5, 50, value=20, step=1, label=t("points_label"))
                        city_box = gr.Textbox(value=s.weather_default_city, label="🌦️ City")
                    dash_refresh_btn = gr.Button(t("refresh_btn"), variant="primary", size="lg")
                with gr.Row():
                    temp_plot = gr.Plot(label=t("temp_label"))
                    hum_plot = gr.Plot(label=t("humidity_label"))
                    soil_plot = gr.Plot(label=t("soil_label"))
                with gr.Row():
                    stats_md = gr.Markdown()
                    alerts_md = gr.Markdown()
                weather_md = gr.Markdown()

            with gr.Tab(t("tab_missions")):
                gr.HTML(H("missions_header"))
                with gr.Group(elem_classes="pg-card"):
                    status_display = gr.Markdown(svc.game.status_md(GameState()))
                    with gr.Row():
                        status_btn = gr.Button(t("status_btn"), variant="secondary")
                        reset_btn = gr.Button(t("reset_btn"), variant="stop")

            with gr.Tab(t("tab_history")):
                gr.HTML(H("history_header"))
                hist_refresh_btn = gr.Button(t("refresh_btn"), variant="secondary")
                hist_gallery = gr.Gallery(columns=4, height=300)
                hist_md = gr.Markdown(t("history_empty"))

        # Floating chatbot
        with gr.Column(visible=False, elem_classes="pg-chat-window") as chat_window:
            gr.HTML(
                '<div class="pg-chat-header">'
                '<span class="pg-chat-avatar">🤖</span>'
                f"<span>{t('chat_title')}</span>"
                '<span class="pg-chat-status">online</span>'
                "</div>"
            )
            chatbot_display = gr.Chatbot(height=400, show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder=t("chat_placeholder"), show_label=False, scale=4, container=False)
                chat_send = gr.Button("➤", scale=1, variant="primary")
        toggle_btn = gr.Button("💬", elem_classes="pg-chat-btn")

        gr.HTML(
            '<div class="pg-footer">🌱 <b>PlantGuard AI</b> · Semantic RAG · Vision AI · '
            "IoT + Alerts · Assistant · Gamification · History · Weather</div>"
        )

        # ---------------- wiring ----------------
        analyze_btn.click(
            on_diagnose, [img_input, game_state],
            [diag_output, game_state, last_entry_state, status_display, hist_gallery, hist_md],
        )
        export_btn.click(on_export, last_entry_state, export_file)
        batch_btn.click(on_batch, batch_input, [batch_gallery, batch_md])
        search_btn.click(on_search, [query_input, topk_slider, game_state],
                         [ans_output, src_output, game_state, status_display])
        fetch_btn.click(on_fetch, [feed_input, limit_input, game_state],
                        [values_box, game_state, status_display])
        dash_refresh_btn.click(on_dashboard, [limit_slider, city_box],
                               [temp_plot, hum_plot, soil_plot, stats_md, alerts_md, weather_md])
        status_btn.click(lambda g: svc.game.status_md(g), game_state, status_display)
        reset_btn.click(lambda g: (svc.game.reset(g), svc.game.status_md(g))[1], game_state, status_display)
        hist_refresh_btn.click(
            lambda g: tuple(gr.update(value=v) for v in _history_view(g.user_id)),
            game_state, [hist_gallery, hist_md],
        )
        user_box.submit(on_user_change, user_box, [game_state, status_display, hist_gallery, hist_md])

        chat_send.click(on_chat, [chat_input, chatbot_display], [chat_input, chatbot_display])
        chat_input.submit(on_chat, [chat_input, chatbot_display], [chat_input, chatbot_display])
        toggle_btn.click(lambda v: (not v, gr.update(visible=not v)), chat_visible, [chat_visible, chat_window])

    return demo


def main() -> None:
    settings = load_settings()
    configure_logging(settings.debug)
    log.info("Starting PlantGuard AI v%s", __import__("plantguard").__version__)
    svc = build_services(settings)
    demo = build_app(svc)
    demo.launch(
        theme=THEME,
        css=_CSS,
        share=settings.share,
        debug=settings.debug,
        server_name=settings.server_name,
        server_port=settings.server_port,
    )


if __name__ == "__main__":
    main()
