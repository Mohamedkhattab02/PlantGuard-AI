"""Centralized UI string table (English).

The product UI is English-only. Strings live here (rather than inline) so copy
is consistent and easy to adjust in one place. ``t()`` keeps a ``lang`` argument
for forward-compatibility but always resolves to English today.
"""

from __future__ import annotations

STRINGS: dict[str, str] = {
    "app_subtitle": "AI-powered plant disease diagnosis, research & field monitoring",
    "user_label": "👤 Username (optional — saves your progress & history)",
    # Tabs
    "tab_diagnosis": "🔬 Diagnosis",
    "tab_research": "📚 Research",
    "tab_sensors": "📊 Sensors",
    "tab_dashboard": "📈 Dashboard",
    "tab_missions": "🎮 Missions",
    "tab_history": "🗂️ History",
    # Diagnosis
    "diag_header": "AI Disease Detection",
    "upload_label": "📷 Upload or capture a leaf image",
    "analyze_btn": "🔍 Analyze",
    "report_label": "Report",
    "export_pdf_btn": "📄 Export PDF report",
    "batch_label": "🗂️ Batch images (field survey)",
    "batch_btn": "🔍 Analyze batch",
    # Research
    "research_header": "Ask Grounded Questions",
    "question_label": "Question",
    "docs_label": "Docs to retrieve",
    "search_btn": "🚀 Search",
    "answer_label": "Answer",
    "sources_label": "📚 Sources",
    # Sensors
    "sensors_header": "Real-time Sensor Data",
    "feed_label": "Feed",
    "samples_label": "Samples",
    "fetch_btn": "📥 Fetch",
    "values_label": "Values",
    # Dashboard
    "dashboard_header": "Monitoring Dashboard",
    "points_label": "Data points",
    "refresh_btn": "🔄 Refresh",
    "temp_label": "🌡️ Temperature",
    "humidity_label": "💧 Humidity",
    "soil_label": "🌱 Soil",
    "stats_label": "Aggregated stats",
    "alerts_label": "Alerts",
    # Missions
    "missions_header": "Daily Missions",
    "status_btn": "🔄 Status",
    "reset_btn": "♻️ Reset missions",
    # History
    "history_header": "Your Diagnosis History",
    "history_empty": "No diagnoses yet — analyze an image to start your history.",
    # Chat
    "chat_title": "Plant Assistant",
    "chat_placeholder": "Ask me anything about the platform…",
    # Messages
    "no_image": "⚠️ Please upload an image first.",
    "low_confidence": "🤔 Low confidence — please retake the photo in good light.",
    "healthy": "✅ Healthy plant",
    "disease_detected": "⚠️ Disease detected",
    "treatment": "Treatment",
    "confidence": "Confidence",
    "points_earned": "points",
}


def t(key: str, lang: str = "en") -> str:
    """Return the UI string for ``key`` (English), or the key if unknown."""
    return STRINGS.get(key, key)
