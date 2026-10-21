"""Local file storage: JSONL for documents/passages, JSON for small records.

Writes are atomic (temp file + ``os.replace``) so an interrupted run never leaves a truncated file.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from biomentor.data.preprocessing import title_key
from biomentor.schemas import Document, Passage


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, data: Any) -> None:
    _atomic_write_text(Path(path), json.dumps(data, indent=2, ensure_ascii=False, default=str))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    lines = []
    for rec in records:
        data = rec.model_dump() if hasattr(rec, "model_dump") else rec
        lines.append(json.dumps(data, ensure_ascii=False))
    _atomic_write_text(Path(path), "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file, skipping blank lines. Raises on malformed lines."""
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc


def save_documents(path: Path, documents: Iterable[Document]) -> int:
    return write_jsonl(path, documents)


def load_documents(path: Path, *, skip_invalid: bool = True) -> list[Document]:
    path = Path(path)
    if not path.exists():
        return []
    docs: list[Document] = []
    for record in iter_jsonl(path):
        try:
            docs.append(Document.model_validate(record))
        except ValidationError:
            if not skip_invalid:
                raise
    return docs


def save_passages(path: Path, passages: Iterable[Passage]) -> int:
    return write_jsonl(path, passages)


def load_passages(path: Path) -> list[Passage]:
    path = Path(path)
    if not path.exists():
        return []
    return [Passage.model_validate(r) for r in iter_jsonl(path)]


def deduplicate_documents(documents: Iterable[Document]) -> tuple[list[Document], int]:
    """Drop duplicates (same PMID, same DOI, or same normalised title+year).

    When two records collide the one with the longer abstract is kept. Order of first appearance
    is preserved. Returns ``(unique_documents, number_removed)``.
    """
    kept: dict[int, Document] = {}
    order: list[int] = []
    index: dict[str, int] = {}
    removed = 0
    for doc in documents:
        keys = []
        if doc.pmid:
            keys.append(f"pmid:{doc.pmid}")
        if doc.doi:
            keys.append(f"doi:{doc.doi.lower()}")
        if doc.title:
            keys.append(f"title:{title_key(doc.title, doc.publication_year)}")
        slot = next((index[k] for k in keys if k in index), None)
        if slot is None:
            slot = len(order)
            order.append(slot)
            kept[slot] = doc
        else:
            removed += 1
            if len(doc.abstract or "") > len(kept[slot].abstract or ""):
                kept[slot] = doc
        for k in keys:
            index.setdefault(k, slot)
    return [kept[s] for s in order], removed
