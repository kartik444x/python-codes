"""
app/utils/text.py

Text preprocessing and intelligent chunking for TTS synthesis.

Key responsibilities:
- Clean and normalize raw text (Unicode, whitespace, etc.)
- Detect and handle special patterns (URLs, emails, numbers, abbreviations)
- Split long text into synthesizable chunks at sentence boundaries
  (never mid-word or mid-sentence if avoidable)
- Concatenate chunked audio segments

Design note: TTS models have a practical limit on input length.
Long inputs must be segmented and synthesized in chunks, then concatenated.
The chunking algorithm always prefers sentence-boundary splits.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# ─── Constants ────────────────────────────────────────────────────────────────

# Maximum characters per synthesis chunk sent to the model
# Kokoro handles up to ~500 chars cleanly; keep chunks smaller for quality
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 0  # No overlap — audio chunks concatenate cleanly

# Pattern library for text normalization
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"{}|\\^`\[\]]+"
    r"|www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s<>\"{}|\\^`\[\]]*",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
)
_NUMBER_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")
_EXCESSIVE_WHITESPACE = re.compile(r"[ \t]{2,}")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_SENTENCE_ENDINGS = re.compile(r'([.!?]+(?:["\'])?)\s+')

# Common abbreviations that should NOT trigger a sentence split
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc",
    "ltd", "corp", "co", "dept", "est", "gov", "approx", "e.g", "i.e",
    "fig", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "oct", "nov", "dec", "st", "ave", "blvd",
}

# Emoji pattern for stripping or replacing emojis
_EMOJI_PATTERN = re.compile(
    "[\U00010000-\U0010FFFF"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


# ─── Data Types ───────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single chunk of text ready for TTS synthesis."""
    index: int
    text: str
    char_start: int
    char_end: int

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class ProcessedText:
    """The result of preprocessing a raw text input."""
    original: str
    cleaned: str
    chunks: list[TextChunk] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def total_chars(self) -> int:
        return len(self.cleaned)


# ─── Core Functions ───────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form and remove control characters.
    Keeps standard printable characters, spaces, and newlines.
    """
    # NFC normalization: compose precomposed forms
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except newline (\n) and tab (\t)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C"
        or ch in ("\n", "\t")
    )
    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/tabs to single space.
    Collapse 3+ newlines to 2 (preserve paragraph breaks).
    Strip leading/trailing whitespace.
    """
    text = _EXCESSIVE_WHITESPACE.sub(" ", text)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()


def replace_urls(text: str) -> str:
    """Replace URLs with a readable description."""
    return _URL_PATTERN.sub("[link]", text)


def replace_emails(text: str) -> str:
    """Replace email addresses with a readable description."""
    return _EMAIL_PATTERN.sub("[email]", text)


def normalize_emojis(text: str, replace_with: str = "") -> str:
    """
    Remove or replace emoji characters.
    TTS models may mispronounce or skip emojis entirely.
    """
    return _EMOJI_PATTERN.sub(replace_with, text)


def expand_abbreviations(text: str) -> str:
    """
    Expand common abbreviations for more natural TTS pronunciation.
    This is a lightweight rule-based approach; for production-grade
    normalization, consider a dedicated TTS text normalizer.
    """
    expansions = {
        r"\bDr\.": "Doctor",
        r"\bMr\.": "Mister",
        r"\bMrs\.": "Missus",
        r"\bMs\.": "Miss",
        r"\bProf\.": "Professor",
        r"\bvs\.": "versus",
        r"\betc\.": "et cetera",
        r"\be\.g\.": "for example",
        r"\bi\.e\.": "that is",
        r"\bSt\.": "Saint",
        r"\bAve\.": "Avenue",
        r"\bBlvd\.": "Boulevard",
    }
    for pattern, replacement in expansions.items():
        text = re.sub(pattern, replacement, text)
    return text


def clean_text(
    text: str,
    *,
    replace_urls: bool = True,
    replace_emails: bool = True,
    strip_emojis: bool = True,
    expand_abbrevs: bool = True,
) -> str:
    """
    Full text cleaning pipeline.

    Args:
        text: Raw input text.
        replace_urls: Replace URL strings with '[link]'.
        replace_emails: Replace email addresses with '[email]'.
        strip_emojis: Remove emoji characters.
        expand_abbrevs: Expand common abbreviations.

    Returns:
        Cleaned, normalized text ready for chunking.
    """
    if not text:
        return text

    text = normalize_unicode(text)

    if strip_emojis:
        text = normalize_emojis(text, replace_with=" ")

    if replace_urls:
        from app.utils import text as _self  # Avoid shadowing local name
        text = _URL_PATTERN.sub("[link]", text)

    if replace_emails:
        text = _EMAIL_PATTERN.sub("[email]", text)

    if expand_abbrevs:
        text = expand_abbreviations(text)

    text = normalize_whitespace(text)

    return text


def _is_abbreviation_end(text: str, match_end: int) -> bool:
    """
    Check if a period at `match_end` is likely an abbreviation,
    not a sentence end. Returns True if it should NOT be split.
    """
    # Find the word before the period
    before = text[:match_end].rstrip(". ")
    last_word = re.split(r"\s+", before)[-1].lower().rstrip(".")
    return last_word in _ABBREVIATIONS


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into a list of sentences using heuristic rules.
    Respects abbreviations to avoid false splits.
    Handles paragraph breaks as sentence boundaries.

    Returns:
        List of sentences (stripped, non-empty).
    """
    # First, split on double newlines (paragraph breaks)
    paragraphs = text.split("\n\n")
    sentences: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Replace single newlines with spaces within a paragraph
        para = para.replace("\n", " ")

        # Find sentence boundaries
        parts: list[str] = []
        last_end = 0

        for match in _SENTENCE_ENDINGS.finditer(para):
            end = match.end()
            candidate = para[last_end:end].strip()

            # Check if this period is an abbreviation
            if "." in match.group(1) and _is_abbreviation_end(para, match.start() + 1):
                continue  # Skip — keep going

            if candidate:
                parts.append(candidate)
            last_end = end

        # Remainder after last match
        remainder = para[last_end:].strip()
        if remainder:
            parts.append(remainder)

        sentences.extend(s for s in parts if s)

    return sentences


def chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[TextChunk]:
    """
    Split text into synthesis chunks at sentence boundaries.

    Algorithm:
    1. Split text into sentences.
    2. Greedily accumulate sentences into a chunk until size limit is reached.
    3. If a single sentence exceeds max_chunk_size, split at word boundaries.

    Args:
        text: Cleaned text to chunk.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of TextChunk objects with position metadata.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_len = 0
    char_cursor = 0
    chunk_idx = 0

    def flush_chunk() -> None:
        nonlocal char_cursor, chunk_idx, current_len
        if not current_parts:
            return
        chunk_text_str = " ".join(current_parts)
        chunks.append(TextChunk(
            index=chunk_idx,
            text=chunk_text_str,
            char_start=char_cursor,
            char_end=char_cursor + len(chunk_text_str),
        ))
        char_cursor += len(chunk_text_str) + 1  # +1 for separator space
        chunk_idx += 1
        current_parts.clear()
        current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds max_chunk_size, split at word boundaries
        if sentence_len > max_chunk_size:
            flush_chunk()
            words = sentence.split()
            word_parts: list[str] = []
            word_len = 0

            for word in words:
                if word_len + len(word) + 1 > max_chunk_size and word_parts:
                    chunk_str = " ".join(word_parts)
                    chunks.append(TextChunk(
                        index=chunk_idx,
                        text=chunk_str,
                        char_start=char_cursor,
                        char_end=char_cursor + len(chunk_str),
                    ))
                    char_cursor += len(chunk_str) + 1
                    chunk_idx += 1
                    word_parts = [word]
                    word_len = len(word)
                else:
                    word_parts.append(word)
                    word_len += len(word) + 1

            if word_parts:
                chunk_str = " ".join(word_parts)
                chunks.append(TextChunk(
                    index=chunk_idx,
                    text=chunk_str,
                    char_start=char_cursor,
                    char_end=char_cursor + len(chunk_str),
                ))
                char_cursor += len(chunk_str) + 1
                chunk_idx += 1
            continue

        # Normal case: accumulate sentences into chunk
        if current_len + sentence_len + 1 > max_chunk_size and current_parts:
            flush_chunk()

        current_parts.append(sentence)
        current_len += sentence_len + 1

    flush_chunk()
    return chunks


def preprocess(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    *,
    replace_urls: bool = True,
    replace_emails: bool = True,
    strip_emojis: bool = True,
    expand_abbrevs: bool = True,
) -> ProcessedText:
    """
    Full preprocessing pipeline: clean + chunk.

    Args:
        text: Raw user input text.
        max_chunk_size: Maximum characters per synthesis chunk.

    Returns:
        ProcessedText with cleaned text and list of TextChunks.
    """
    cleaned = clean_text(
        text,
        replace_urls=replace_urls,
        replace_emails=replace_emails,
        strip_emojis=strip_emojis,
        expand_abbrevs=expand_abbrevs,
    )
    chunks = chunk_text(cleaned, max_chunk_size=max_chunk_size)
    return ProcessedText(original=text, cleaned=cleaned, chunks=chunks)
