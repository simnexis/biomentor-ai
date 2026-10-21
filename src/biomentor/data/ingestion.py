"""Raw documents -> cleaned, de-duplicated documents -> passages.

Layout::

    data/raw/*.jsonl          raw records exactly as fetched (one Document per line)
    data/processed/documents.jsonl   cleaned, de-duplicated documents
    data/processed/passages.jsonl    derived chunks (rebuildable at any time)
    data/indexes/             search indexes and manifest (rebuildable)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from biomentor.data.chunking import ChunkConfig, chunk_document
from biomentor.data.preprocessing import normalize_whitespace
from biomentor.data.storage import (
    deduplicate_documents,
    load_documents,
    save_documents,
    save_passages,
)
from biomentor.schemas import SOURCE_PUBMED, SOURCE_SYNTHETIC, Document, Passage

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    raw_files: int
    raw_documents: int
    duplicates_removed: int
    documents: int
    documents_without_abstract: int
    passages: int
    corpus_kind: str


def clean_document(doc: Document) -> Document:
    """Normalise whitespace in the text fields; empty strings become ``None``."""
    update = {
        "title": normalize_whitespace(doc.title) or doc.title,
        "abstract": normalize_whitespace(doc.abstract) or None,
    }
    return doc.model_copy(update=update)


def corpus_kind_of(documents: Iterable[Document]) -> str:
    kinds = {d.source_type for d in documents}
    if not kinds:
        return "empty"
    if kinds == {SOURCE_SYNTHETIC}:
        return "demo_synthetic"
    if kinds == {SOURCE_PUBMED}:
        return "pubmed"
    return "mixed"


def documents_to_passages(documents: Sequence[Document], config: ChunkConfig | None = None) -> list[Passage]:
    config = config or ChunkConfig()
    passages: list[Passage] = []
    for doc in documents:
        passages.extend(chunk_document(doc, config))
    return passages


def find_raw_files(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    return sorted(p for p in raw_dir.glob("*.jsonl") if p.is_file()) if raw_dir.exists() else []


def process_corpus(
    raw_paths: Sequence[Path], processed_dir: Path, config: ChunkConfig | None = None
) -> tuple[list[Document], list[Passage], IngestionStats]:
    """Load raw JSONL files, clean + de-duplicate, chunk, and write ``data/processed``."""
    config = config or ChunkConfig()
    raw_docs: list[Document] = []
    for path in raw_paths:
        loaded = load_documents(Path(path))
        logger.info("Loaded %d documents from %s", len(loaded), path)
        raw_docs.extend(loaded)
    cleaned = [clean_document(d) for d in raw_docs]
    unique, removed = deduplicate_documents(cleaned)
    passages = documents_to_passages(unique, config)

    processed_dir = Path(processed_dir)
    save_documents(processed_dir / "documents.jsonl", unique)
    save_passages(processed_dir / "passages.jsonl", passages)
    stats = IngestionStats(
        raw_files=len(raw_paths),
        raw_documents=len(raw_docs),
        duplicates_removed=removed,
        documents=len(unique),
        documents_without_abstract=sum(1 for d in unique if not d.abstract),
        passages=len(passages),
        corpus_kind=corpus_kind_of(unique),
    )
    return unique, passages, stats


def append_raw_documents(path: Path, documents: Iterable[Document]) -> int:
    """Merge ``documents`` into a raw JSONL file (de-duplicating against what is already there)."""
    path = Path(path)
    existing = load_documents(path)
    merged, _ = deduplicate_documents([*existing, *documents])
    save_documents(path, merged)
    return len(merged) - len(existing)
