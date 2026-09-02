"""
tests/test_text_processing.py

Unit tests for text cleaning, Unicode normalization, and sentence-boundary chunking.
"""
from app.utils.text import (
    chunk_text,
    clean_text,
    expand_abbreviations,
    normalize_unicode,
    preprocess,
    split_into_sentences,
)


def test_unicode_normalization():
    text = "Héllo \u0000 World"
    cleaned = normalize_unicode(text)
    assert "Héllo" in cleaned
    assert "\u0000" not in cleaned


def test_url_and_email_replacement():
    text = "Contact support@example.com or visit https://my-tts-api.com for docs."
    cleaned = clean_text(text)
    assert "[email]" in cleaned
    assert "[link]" in cleaned
    assert "https://" not in cleaned
    assert "support@" not in cleaned


def test_abbreviation_expansion():
    text = "Dr. Smith met Mr. Brown on 5th Ave."
    expanded = expand_abbreviations(text)
    assert "Doctor" in expanded
    assert "Mister" in expanded
    assert "Avenue" in expanded


def test_sentence_splitting_respects_abbreviations():
    text = "Dr. Smith arrived at 5 p.m. He was very punctual."
    sentences = split_into_sentences(text)
    # Shouldn't split between Dr. and Smith
    assert len(sentences) == 2


def test_chunking_bounds():
    # Long text with multiple sentences
    text = "Sentence one is clear. " * 30
    chunks = chunk_text(text, max_chunk_size=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 250  # sentence accumulated within reasonable limit
