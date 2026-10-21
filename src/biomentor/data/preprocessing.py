"""Text normalisation and tokenisation.

Design rule: change whitespace and invisible characters only. Scientific content (Greek letters,
superscripts, units, hyphenated gene/virus names such as ``SARS-CoV-2`` or ``IL-6``) is preserved,
so passages stay faithful enough to quote and cite. We use Unicode NFC (not NFKC, which would turn
"m²" into "m2" and change meaning).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")
_SPACE_LIKE_RE = re.compile(r"[   -   　]")
_PARA_SPLIT_RE = re.compile(r"\n[ \t]*\n+")
_DEHYPHEN_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


def _base_clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return _SPACE_LIKE_RE.sub(" ", text)


def normalize_whitespace(text: Any, keep_paragraphs: bool = False) -> str:
    """Collapse whitespace. Handles ``None`` and non-string input (returns ``""``).

    With ``keep_paragraphs=True`` blank-line paragraph breaks become a single ``"\\n\\n"`` while
    hard line wraps inside a paragraph become spaces.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _base_clean(text).replace("\r\n", "\n").replace("\r", "\n")
    if not keep_paragraphs:
        return re.sub(r"\s+", " ", text).strip()
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in _PARA_SPLIT_RE.split(text)]
    return "\n\n".join(p for p in paragraphs if p)


def clean_pdf_text(text: Any) -> str:
    """Normalise text extracted from a PDF: repair end-of-line hyphenation, keep paragraphs."""
    if not text:
        return ""
    text = _base_clean(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    # "exam-\nple" -> "example" (only when the next line continues in lowercase, so
    # "SARS-\nCoV-2" keeps its hyphen).
    text = _DEHYPHEN_RE.sub("", text)
    return normalize_whitespace(text, keep_paragraphs=True)


def clean_pdf_lines(text: Any) -> str:
    """Like :func:`clean_pdf_text` but keeps line breaks (needed to find headings such as "Abstract"
    that PDF extractors put on their own line)."""
    if not text:
        return ""
    text = _base_clean(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    text = _DEHYPHEN_RE.sub("", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalize_for_match(text: Any) -> str:
    """Lower-cased, whitespace-collapsed text for substring checks (e.g. verifying quotes)."""
    return normalize_whitespace(text).casefold()


# ---------------------------------------------------------------------------------------------
# Tokenisation (shared by BM25, abstention and the lexical support heuristic)
# ---------------------------------------------------------------------------------------------
STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself just
    me more most my of off on once only or other our out over own same she should so some such
    than that the their them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your""".split()
)

# A token is a run of letters/digits, optionally joined by hyphens or apostrophes
# ("il-6", "sars-cov-2", "crohn's"), optionally with a decimal part ("0.05").
_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*(?:\.\d+)?")
_SPLIT_RE = re.compile(r"[-'’]")


def tokenize(text: Any, remove_stopwords: bool = True) -> list[str]:
    """Lower-case tokens. Hyphenated compounds are kept whole *and* split into parts, so a query
    for "cov" can match "SARS-CoV-2" while "sars-cov-2" still matches exactly."""
    if not text:
        return []
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalize_whitespace(text).casefold()):
        tok = match.group(0)
        if remove_stopwords and tok in STOPWORDS:
            continue
        tokens.append(tok)
        parts = _SPLIT_RE.split(tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if len(p) >= 2 and not (remove_stopwords and p in STOPWORDS))
    return tokens


_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s")


def stem(token: str) -> str:
    """A deliberately crude suffix stripper, used only for the lexical-overlap heuristics."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def content_stems(text: Any) -> set[str]:
    """Distinct stemmed content tokens (length >= 3, not purely numeric)."""
    return {stem(t) for t in tokenize(text) if len(t) >= 3 and not t.replace(".", "").isdigit()}


def stable_hash(*parts: Any, length: int = 12) -> str:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return digest[:length]


def title_key(title: str | None, year: int | None = None) -> str:
    """Key used to detect duplicate documents that lack a PMID/DOI."""
    letters = re.sub(r"[^a-z0-9]+", " ", (title or "").casefold()).strip()
    return f"{letters}|{year or ''}"
