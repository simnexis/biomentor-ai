"""Optional cross-encoder reranking.

A bi-encoder (dense retrieval) embeds query and passage *separately*; a cross-encoder reads the
pair *together* and scores relevance directly - more accurate but too slow to run over a whole
corpus, so it only re-scores a short candidate list from the first-stage retriever.

The application works without it. Whether it helps is an empirical question - see the evaluation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from biomentor.config import DEFAULT_RERANKER_MODEL
from biomentor.exceptions import RerankerError
from biomentor.schemas import RetrievedPassage

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError("The 'sentence-transformers' package is not installed.") from exc
            try:
                logger.info("Loading cross-encoder %s", self.model_name)
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RerankerError(
                    f"Could not load the reranker model '{self.model_name}' ({type(exc).__name__}). "
                    "It is downloaded from Hugging Face on first use."
                ) from exc
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        scores = model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def rerank(
    query: str,
    candidates: Sequence[RetrievedPassage],
    reranker: Reranker,
    top_k: int = 10,
    method: str = "hybrid_rerank",
) -> list[RetrievedPassage]:
    """Re-order ``candidates`` by the reranker's score; earlier component ranks are preserved."""
    if not candidates or not query.strip():
        return []
    scores = reranker.score(query, [c.passage.retrieval_text() for c in candidates])
    if len(scores) != len(candidates):
        raise RerankerError("Reranker returned a different number of scores than candidates.")
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))[:top_k]
    out = []
    for new_rank, i in enumerate(order, start=1):
        c = candidates[i]
        out.append(
            RetrievedPassage(
                passage=c.passage,
                score=scores[i],
                rank=new_rank,
                method=method,
                component_ranks={**c.component_ranks, "rerank": new_rank},
                component_scores={**c.component_scores, "rerank": scores[i]},
            )
        )
    return out
