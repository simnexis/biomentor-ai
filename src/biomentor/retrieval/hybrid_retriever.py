"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

    RRF(d) = sum over systems s of  1 / (c + rank_s(d))

* ``rank_s(d)`` is the 1-based position of passage ``d`` in system ``s``'s ranking; a system that
  did not retrieve ``d`` contributes nothing.
* ``c`` (default 60, from the original RRF paper) damps the influence of the very top ranks.

RRF uses ranks only, so BM25 scores (unbounded) and cosine similarities (0..1) never need to be
put on a common scale - adding the raw scores would let whichever scale is larger dominate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from biomentor.retrieval.bm25_retriever import BM25Retriever
from biomentor.retrieval.dense_retriever import DenseRetriever
from biomentor.schemas import Passage, RetrievedPassage


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievedPassage]],
    c: int = 60,
    top_k: int | None = None,
    method: str = "hybrid",
) -> list[RetrievedPassage]:
    """Fuse several ranked lists. Duplicates are merged by ``passage_id``; component ranks and
    scores are preserved. Ties are broken by best component rank, then first appearance."""
    if c < 0:
        raise ValueError("RRF constant c must be non-negative")
    fused: dict[str, float] = {}
    passages: dict[str, Passage] = {}
    comp_ranks: dict[str, dict[str, int]] = {}
    comp_scores: dict[str, dict[str, float]] = {}
    first_seen: dict[str, int] = {}

    for system, results in rankings.items():
        seen: set[str] = set()
        position = 0
        for item in results:
            pid = item.passage.passage_id
            if pid in seen:
                continue
            seen.add(pid)
            position += 1
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (c + position)
            passages.setdefault(pid, item.passage)
            comp_ranks.setdefault(pid, {})[system] = position
            comp_scores.setdefault(pid, {})[system] = item.score
            first_seen.setdefault(pid, len(first_seen))

    order = sorted(fused, key=lambda pid: (-fused[pid], min(comp_ranks[pid].values()), first_seen[pid]))
    if top_k is not None:
        order = order[: max(0, top_k)]
    return [
        RetrievedPassage(
            passage=passages[pid],
            score=fused[pid],
            rank=i,
            method=method,
            component_ranks=comp_ranks[pid],
            component_scores=comp_scores[pid],
        )
        for i, pid in enumerate(order, start=1)
    ]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, c: int = 60, candidate_k: int = 50):
        self.bm25, self.dense, self.c, self.candidate_k = bm25, dense, c, candidate_k

    def search(self, query: str, k: int = 10) -> list[RetrievedPassage]:
        if k <= 0 or not query or not query.strip():
            return []
        depth = max(self.candidate_k, k)
        return reciprocal_rank_fusion(
            {"bm25": self.bm25.search(query, depth), "dense": self.dense.search(query, depth)},
            c=self.c,
            top_k=k,
            method=self.name,
        )
