"""Tests for PDF text chunking (ROADMAP [2.5])."""

from plantguard.services.documents import chunk_text


def test_chunks_respect_size():
    chunks = chunk_text("word " * 1000, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_overlap_creates_shared_text():
    chunks = chunk_text("abcdefghij" * 30, size=100, overlap=20)
    # consecutive chunks overlap, so the tail of one appears at the head of the next
    assert chunks[0][-20:] == chunks[1][:20]


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
