"""Run the retrieval benchmark.

Procedure (identical for every method)
1. For each benchmark query, retrieve the top ``passage_depth`` passages.
2. Collapse passages to documents: a document's rank is the rank of its best passage.
3. Score the ranked document list against the human relevance labels.
4. Average over queries. Latency is the wall-clock time of ``search`` per query (milliseconds).

The metrics are computed at *document* level because abstracts are chunked into 1-3 passages and
the labels are per document. A method that cannot run (e.g. no embedding model) is recorded as
skipped with its reason - never given placeholder numbers.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from datetime import datetime, timezone

from biomentor.evaluation.benchmark import BenchmarkRecord, benchmark_digest, is_synthetic
from biomentor.evaluation.metrics import (
    dedupe,
    first_relevant_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from biomentor.exceptions import RetrievalUnavailableError
from biomentor.retrieval.index_manager import MODE_LABELS, RetrievalIndex

METRIC_KEYS = ["recall@5", "recall@10", "precision@5", "mrr", "ndcg@10"]


def evaluate_method(
    index: RetrievalIndex, records: Sequence[BenchmarkRecord], mode: str, passage_depth: int = 50, reranker=None
) -> dict:
    """Evaluate one retrieval mode. Returns ``{"status": "ok", "queries": [...], "aggregate": {...}}``
    or ``{"status": "skipped", "reason": ...}``."""
    # warm-up so one-off costs (lazy model load, index build) do not pollute query latency
    try:
        index.search(records[0].query, mode, 5, reranker=reranker)
    except RetrievalUnavailableError as exc:
        return {"status": "skipped", "mode": mode, "label": MODE_LABELS[mode], "reason": str(exc)}
    except Exception as exc:  # model download failure etc.
        return {"status": "skipped", "mode": mode, "label": MODE_LABELS[mode], "reason": f"{type(exc).__name__}: {exc}"}

    per_query = []
    for rec in records:
        t0 = time.perf_counter()
        results = index.search(rec.query, mode, passage_depth, reranker=reranker)
        latency_ms = (time.perf_counter() - t0) * 1000
        ranked_docs = dedupe(r.passage.document_id for r in results)
        relevant = set(rec.relevant_document_ids)
        grades = rec.grades_for_documents()
        row = {
            "question_id": rec.question_id,
            "query": rec.query,
            "topic": rec.topic,
            "n_relevant": len(relevant),
            "recall@5": recall_at_k(ranked_docs, relevant, 5),
            "recall@10": recall_at_k(ranked_docs, relevant, 10),
            "precision@5": precision_at_k(ranked_docs, relevant, 5),
            "mrr": reciprocal_rank(ranked_docs, relevant),
            "ndcg@10": ndcg_at_k(ranked_docs, grades, 10),
            "first_relevant_rank": first_relevant_rank(ranked_docs, relevant),
            "latency_ms": latency_ms,
            "top5_documents": ranked_docs[:5],
            "relevant_documents": sorted(relevant),
        }
        per_query.append(row)

    aggregate = {k: statistics.fmean(q[k] for q in per_query) for k in METRIC_KEYS}
    lat = sorted(q["latency_ms"] for q in per_query)
    aggregate["latency_ms_mean"] = statistics.fmean(lat)
    aggregate["latency_ms_median"] = statistics.median(lat)
    aggregate["latency_ms_p95"] = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    return {"status": "ok", "mode": mode, "label": MODE_LABELS[mode], "queries": per_query, "aggregate": aggregate}


def error_analysis(method_result: dict, k: int = 5, limit: int = 10) -> list[dict]:
    """Queries where the first relevant document is outside the top ``k`` (or never retrieved)."""
    if method_result.get("status") != "ok":
        return []
    misses = [q for q in method_result["queries"] if q["first_relevant_rank"] is None or q["first_relevant_rank"] > k]
    misses.sort(key=lambda q: (q["first_relevant_rank"] is not None, -(q["first_relevant_rank"] or 0)))
    return misses[:limit]


def run_evaluation(
    index: RetrievalIndex,
    records: Sequence[BenchmarkRecord],
    benchmark_path,
    modes: Sequence[str] = ("bm25", "dense", "hybrid", "hybrid_rerank"),
    config: dict | None = None,
    reranker=None,
    passage_depth: int = 50,
) -> dict:
    """Evaluate each mode and return one JSON-serialisable results document."""
    methods = {}
    for mode in modes:
        if mode == "hybrid_rerank" and reranker is None and index.reranker is None:
            methods[mode] = {"status": "skipped", "mode": mode, "label": MODE_LABELS[mode], "reason": "No reranker configured (use --rerank)."}
            continue
        methods[mode] = evaluate_method(index, records, mode, passage_depth, reranker)
    for m in methods.values():
        if m.get("status") == "ok":
            m["error_analysis"] = error_analysis(m)
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": {
            "n_queries": len(records),
            "digest": benchmark_digest(benchmark_path),
            "path": str(benchmark_path),
            "synthetic": is_synthetic(records),
            "graded": all(r.has_grades for r in records),
        },
        "corpus": {"n_passages": len(index.passages), "n_documents": len({p.document_id for p in index.passages}), "kind": index.corpus_kind},
        "config": config or {},
        "metric_level": "document (passages collapsed to their document; rank = best passage)",
        "methods": methods,
    }
