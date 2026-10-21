"""Benchmark format, loading/validation, and the pooled-annotation workflow.

A benchmark is a JSONL file; each line is a :class:`BenchmarkRecord`. To build a *real* benchmark
over a PubMed corpus you fetched:

1. ``python scripts/make_benchmark.py candidates --queries my_queries.txt``  pools the top results of
   BM25 / dense / hybrid for each query into ``annotation_sheet.csv`` (so every system's picks are judged).
2. A human reads each candidate's title + abstract and fills ``relevance`` (0 = not relevant,
   1 = partially relevant, 2 = highly relevant).
3. ``python scripts/make_benchmark.py compile --sheet annotation_sheet.csv`` writes the benchmark JSONL.

Caveat: pooled judgments only cover documents that some system retrieved, which slightly favours
the pooled systems; with a small corpus and 3 systems this bias is modest but should be stated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator

from biomentor.data.storage import write_jsonl
from biomentor.exceptions import BenchmarkError, RetrievalUnavailableError
from biomentor.retrieval.index_manager import RetrievalIndex
from biomentor.schemas import Document

logger = logging.getLogger(__name__)


class BenchmarkRecord(BaseModel):
    question_id: str
    query: str
    topic: str
    relevant_document_ids: list[str]
    relevant_passage_ids: list[str] | None = None
    relevance_grades: dict[str, int] | None = None
    annotation_notes: str | None = None
    provenance: str

    @field_validator("query", "question_id", "provenance")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @field_validator("relevant_document_ids")
    @classmethod
    def _has_relevant(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("needs at least one relevant document id")
        return list(dict.fromkeys(v))

    @field_validator("relevance_grades")
    @classmethod
    def _grades(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        if v is not None and any((not isinstance(g, int)) or g < 0 for g in v.values()):
            raise ValueError("relevance grades must be non-negative integers")
        return v

    def grades_for_documents(self) -> dict[str, int]:
        """Graded relevance if provided, else binary (1) for every relevant document."""
        return dict(self.relevance_grades) if self.relevance_grades else {d: 1 for d in self.relevant_document_ids}

    @property
    def has_grades(self) -> bool:
        return bool(self.relevance_grades)


def load_benchmark(path: Path | str) -> list[BenchmarkRecord]:
    path = Path(path)
    if not path.exists():
        raise BenchmarkError(f"Benchmark file not found: {path}")
    records: list[BenchmarkRecord] = []
    ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = BenchmarkRecord.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise BenchmarkError(f"{path.name} line {line_no}: invalid benchmark record ({type(exc).__name__}).") from exc
            if record.question_id in ids:
                raise BenchmarkError(f"{path.name} line {line_no}: duplicate question_id '{record.question_id}'.")
            ids.add(record.question_id)
            records.append(record)
    if not records:
        raise BenchmarkError(f"{path.name} contains no records.")
    return records


def save_benchmark(path: Path | str, records: Iterable[BenchmarkRecord | dict]) -> int:
    validated = [r if isinstance(r, BenchmarkRecord) else BenchmarkRecord.model_validate(r) for r in records]
    return write_jsonl(Path(path), validated)


def benchmark_digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def is_synthetic(records: Sequence[BenchmarkRecord]) -> bool:
    return any("synthetic" in r.provenance.lower() for r in records)


def validate_against_corpus(records: Sequence[BenchmarkRecord], document_ids: set[str], passage_ids: set[str] | None = None) -> list[str]:
    """Return human-readable problems (relevant ids that do not exist in the corpus)."""
    problems = []
    for r in records:
        for d in r.relevant_document_ids:
            if d not in document_ids:
                problems.append(f"{r.question_id}: relevant document '{d}' is not in the corpus")
        if passage_ids is not None:
            for p in r.relevant_passage_ids or []:
                if p not in passage_ids:
                    problems.append(f"{r.question_id}: relevant passage '{p}' is not in the corpus")
    return problems


# ---------------------------------------------------------------------------------------------
# Pooled annotation workflow
# ---------------------------------------------------------------------------------------------
SHEET_COLUMNS = ["question_id", "query", "topic", "document_id", "title", "abstract_snippet", "found_by", "relevance", "notes"]


def write_annotation_sheet(
    path: Path | str,
    queries: Sequence[tuple[str, str]],
    index: RetrievalIndex,
    documents: dict[str, Document],
    top_k: int = 10,
    modes: Sequence[str] = ("bm25", "dense", "hybrid"),
) -> int:
    """Pool the top-``top_k`` documents of each mode for each ``(query, topic)`` into a CSV for a
    human to label. Returns the number of candidate rows. Modes that are unavailable are skipped."""
    rows = []
    for q_index, (query, topic) in enumerate(queries, start=1):
        qid = f"q{q_index:03d}"
        pooled: dict[str, dict[str, int]] = {}
        for mode in modes:
            try:
                results = index.search(query, mode, top_k * 3)
            except RetrievalUnavailableError as exc:
                logger.warning("Skipping mode %s for annotation pooling: %s", mode, exc)
                continue
            seen: list[str] = []
            for r in results:
                if r.passage.document_id not in seen:
                    seen.append(r.passage.document_id)
            for rank, doc_id in enumerate(seen[:top_k], start=1):
                pooled.setdefault(doc_id, {})[mode] = rank
        for doc_id, found in sorted(pooled.items(), key=lambda kv: min(kv[1].values())):
            doc = documents.get(doc_id)
            rows.append({
                "question_id": qid, "query": query, "topic": topic, "document_id": doc_id,
                "title": doc.title if doc else "", "abstract_snippet": ((doc.abstract or "")[:600] if doc else ""),
                "found_by": "; ".join(f"{m}@{r}" for m, r in found.items()), "relevance": "", "notes": "",
            })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def compile_annotations(sheet_path: Path | str, provenance: str) -> tuple[list[BenchmarkRecord], list[str]]:
    """Turn a labelled sheet into benchmark records. Returns ``(records, warnings)``.
    Queries with no document labelled >= 1 are skipped (and reported)."""
    warnings: list[str] = []
    grouped: dict[str, dict] = {}
    with open(sheet_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qid = row["question_id"]
            entry = grouped.setdefault(qid, {"query": row["query"], "topic": row["topic"], "grades": {}, "labelled": 0, "notes": []})
            label = (row.get("relevance") or "").strip()
            if label == "":
                continue
            if label not in {"0", "1", "2", "3"}:
                raise BenchmarkError(f"{qid}/{row['document_id']}: relevance must be 0, 1, 2 or 3 (got '{label}').")
            entry["labelled"] += 1
            if int(label) > 0:
                entry["grades"][row["document_id"]] = int(label)
            if row.get("notes"):
                entry["notes"].append(row["notes"].strip())
    records = []
    for qid, e in grouped.items():
        if e["labelled"] == 0:
            warnings.append(f"{qid}: no candidates labelled - skipped")
        elif not e["grades"]:
            warnings.append(f"{qid}: no relevant documents (all 0) - skipped")
        else:
            records.append(BenchmarkRecord(
                question_id=qid, query=e["query"], topic=e["topic"], relevant_document_ids=list(e["grades"]),
                relevance_grades=e["grades"], annotation_notes="; ".join(e["notes"]) or None, provenance=provenance,
            ))
    return records, warnings
