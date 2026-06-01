"""Tests for the label parser (ROADMAP [5.3])."""

from plantguard.services.image import parse_label


def test_with_split():
    assert parse_label("Tomato with Late Blight") == ("Tomato", "Late Blight", False)


def test_healthy_prefix():
    assert parse_label("Healthy Apple") == ("Apple", "Healthy", True)


def test_healthy_plant_suffix():
    plant, disease, healthy = parse_label("Healthy Tomato plant")
    assert plant == "Tomato" and healthy is True


def test_multiword_crop():
    assert parse_label("Corn (Maize) with Common Rust") == ("Corn (Maize)", "Common Rust", False)


def test_first_word_fallback():
    assert parse_label("Apple Scab") == ("Apple", "Scab", False)


def test_known_plant_prefix_multiword_disease():
    assert parse_label("Tomato Yellow Leaf Curl Virus") == (
        "Tomato",
        "Yellow Leaf Curl Virus",
        False,
    )


def test_override_cedar_apple_rust():
    # "Cedar Apple Rust" is an *apple* disease — the naive first-word rule fails.
    assert parse_label("Cedar Apple Rust") == ("Apple", "Cedar Apple Rust", False)


def test_empty_label():
    plant, disease, healthy = parse_label("")
    assert plant == "Plant" and healthy is False
