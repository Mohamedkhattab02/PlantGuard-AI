"""Tests for the (English) UI string table."""

from plantguard.i18n import t


def test_known_key():
    assert t("analyze_btn") == "🔍 Analyze"


def test_unknown_key_returns_key():
    assert t("totally_unknown_key") == "totally_unknown_key"


def test_lang_argument_is_accepted():
    # The lang arg is kept for forward-compat; today everything is English.
    assert t("analyze_btn", "en") == t("analyze_btn", "he")
