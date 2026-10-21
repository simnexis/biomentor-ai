"""Sentence-aware chunking with overlap.

Strategy
--------
1. Split text into sentences (regex + an abbreviation guard; biomedical text is full of
   "et al.", "Fig. 2", "p < 0.05", "i.v." that naive splitters break on).
2. Greedily pack whole sentences into chunks of at most ``max_chars`` characters.
3. Start each following chunk with the last ``overlap_sentences`` sentences of the previous
   one, so that a fact straddling a boundary appears intact in at least one chunk.
4. A single sentence longer than ``max_chars`` is split at word boundaries.

Invariant (tested): for every passage, ``normalize_whitespace(source)[start:end] == passage.text``.
Passages are therefore exact quotations of the source, which is what citations need.

Trade-offs
----------
* Smaller chunks -> more precise evidence and better fit for embedding models with short context
  windows (MiniLM truncates around 256 word pieces, roughly 1000 characters), but less context per
  passage and more passages to index.
* Larger chunks -> more context, but diluted embeddings and noisier citations.
* Overlap -> protects boundary facts at the cost of a larger index and some duplicate evidence.
Default: 800 characters, 1 sentence overlap (typical abstracts become 1-3 passages).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from biomentor.data.preprocessing import normalize_whitespace
from biomentor.schemas import Document, Passage


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 800
    overlap_sentences: int = 1

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError("max_chars must be at least 100")
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")

    def signature(self) -> str:
        return f"max{self.max_chars}-ov{self.overlap_sentences}"

    def to_dict(self) -> dict:
        return asdict(self)


_CLOSERS = "\"')\\]”’"
_BOUNDARY_RE = re.compile(
    rf"(?:(?<=[.!?])|(?<=[.!?][{_CLOSERS}]))\s+(?=[\"“‘(\[]?[A-Z0-9])|\n{{2,}}"
)
_ABBREV_END_RE = re.compile(
    r"(?:\b(?:et al|e\.g|i\.e|vs|cf|Fig|Figs|Eq|Eqs|Dr|Mr|Ms|Prof|approx|ca|No|Nos|resp|spp|sp|subsp|St|Ref|Refs)\.|\b[A-Z]\.)$"
)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of each sentence (paragraph breaks always split)."""
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    for match in _BOUNDARY_RE.finditer(text):
        if match.start() <= start:
            continue
        segment = text[start : match.start()]
        is_paragraph_break = "\n" in match.group(0)
        if not is_paragraph_break and _ABBREV_END_RE.search(segment):
            continue  # "et al. The ..." / "Fig. 2" - keep the sentence going
        spans.append((start, start + len(segment.rstrip())))
        start = match.end()
    tail = text[start:].rstrip()
    if tail:
        spans.append((start, start + len(tail)))
    return spans


def _split_long_span(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    pos = start
    while end - pos > max_chars:
        cut = text.rfind(" ", pos + 1, pos + max_chars + 1)
        if cut <= pos:
            cut = pos + max_chars
        pieces.append((pos, cut))
        pos = cut
        while pos < end and text[pos] == " ":
            pos += 1
    if pos < end:
        pieces.append((pos, end))
    return pieces


def chunk_spans(text: str, config: ChunkConfig) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans of the chunks of ``text``."""
    units: list[tuple[int, int]] = []
    for s, e in sentence_spans(text):
        if e - s <= config.max_chars:
            units.append((s, e))
        else:
            units.extend(_split_long_span(text, s, e, config.max_chars))

    chunks: list[tuple[int, int]] = []
    n, i = len(units), 0
    while i < n:
        j = i
        while j + 1 < n and units[j + 1][1] - units[i][0] <= config.max_chars:
            j += 1
        chunks.append((units[i][0], units[j][1]))
        if j >= n - 1:
            break
        nxt = max(j + 1 - config.overlap_sentences, i + 1)
        # Never let the overlap crowd out the first new sentence (would yield an overlap-only chunk).
        while nxt < j + 1 and units[j + 1][1] - units[nxt][0] > config.max_chars:
            nxt += 1
        i = nxt
    return chunks


def chunk_text_to_passages(
    document: Document, text: str | None, config: ChunkConfig | None = None, keep_paragraphs: bool = False
) -> list[Passage]:
    """Chunk ``text`` (already belonging to ``document``) into :class:`Passage` objects."""
    config = config or ChunkConfig()
    normalized = normalize_whitespace(text, keep_paragraphs=keep_paragraphs)
    if not normalized:
        return []
    passages: list[Passage] = []
    seen: set[str] = set()
    for start, end in chunk_spans(normalized, config):
        chunk = normalized[start:end]
        key = chunk.casefold()
        if not chunk.strip() or key in seen:
            continue
        seen.add(key)
        idx = len(passages)
        passages.append(
            Passage(
                passage_id=f"{document.document_id}-p{idx:03d}",
                document_id=document.document_id,
                chunk_index=idx,
                text=chunk,
                start_char=start,
                end_char=end,
                title=document.title or "",
                source_name=document.source_name,
                source_type=document.source_type,
                pmid=document.pmid,
                doi=document.doi,
                journal=document.journal,
                publication_year=document.publication_year,
                source_url=document.source_url,
            )
        )
    return passages


def chunk_document(document: Document, config: ChunkConfig | None = None) -> list[Passage]:
    """Chunk a document's abstract. Documents without an abstract yield no passages
    (a title alone is not citable evidence)."""
    return chunk_text_to_passages(document, document.abstract, config)
