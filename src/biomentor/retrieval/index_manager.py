"""Ties BM25, dense and hybrid retrieval together and manages the on-disk index.

``RetrievalIndex`` is the object the rest of the app searches with; it is built either from disk
(``IndexManager.load``) or in memory (``RetrievalIndex.from_passages``, used for live PubMed
results and uploaded papers). The dense part is loaded/built lazily on first use, so starting the
app never downloads a model, and a dense failure degrades gracefully to BM25.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from biomentor.config import Settings
from biomentor.data.chunking import ChunkConfig
from biomentor.data.storage import load_passages, read_json, write_json
from biomentor.exceptions import (
    BioMentorError,
    IndexCorruptError,
    IndexIncompatibleError,
    IndexNotFoundError,
    RetrievalUnavailableError,
)
from biomentor.retrieval.bm25_retriever import BM25Retriever
from biomentor.retrieval.dense_retriever import DenseRetriever, Embedder, corpus_fingerprint
from biomentor.retrieval.hybrid_retriever import HybridRetriever
from biomentor.retrieval.reranker import Reranker, rerank
from biomentor.schemas import Passage, RetrievedPassage

logger = logging.getLogger(__name__)

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
MODE_LABELS = {
    "bm25": "Keyword (BM25)",
    "dense": "Semantic (embeddings)",
    "hybrid": "Hybrid (BM25 + semantic, RRF)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}

DenseLoader = Callable[[], DenseRetriever]


class RetrievalIndex:
    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        dense_loader: DenseLoader | None = None,
        reranker: Reranker | None = None,
        rrf_c: int = 60,
        candidate_k: int = 50,
        corpus_kind: str = "unknown",
    ):
        self.passages = list(passages)
        self.bm25 = bm25 or BM25Retriever(self.passages)
        self._dense = dense
        self._dense_loader = dense_loader
        self._dense_error: str | None = None
        self._dense_attempted = dense is not None
        self.reranker = reranker
        self.rrf_c, self.candidate_k = rrf_c, candidate_k
        self.corpus_kind = corpus_kind

    @classmethod
    def from_passages(
        cls,
        passages: Sequence[Passage],
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_c: int = 60,
        candidate_k: int = 50,
        corpus_kind: str = "unknown",
    ) -> RetrievalIndex:
        loader = (lambda: DenseRetriever(passages, embedder)) if embedder is not None else None
        return cls(passages, dense_loader=loader, reranker=reranker, rrf_c=rrf_c, candidate_k=candidate_k, corpus_kind=corpus_kind)

    # ------------------------------------------------------------------ dense access
    def get_dense(self) -> DenseRetriever:
        """Return the dense retriever, loading/building it on first call. Raises
        :class:`RetrievalUnavailableError` (with a user-readable reason) if not possible."""
        if self._dense is not None:
            return self._dense
        if self._dense_error:
            raise RetrievalUnavailableError(self._dense_error)
        if self._dense_loader is None:
            self._dense_error = "No dense index or embedding model is configured."
            raise RetrievalUnavailableError(self._dense_error)
        try:
            self._dense = self._dense_loader()
        except (BioMentorError, OSError, ValueError, RuntimeError) as exc:
            self._dense_error = f"Semantic search unavailable: {exc}"
            logger.warning(self._dense_error)
            raise RetrievalUnavailableError(self._dense_error) from exc
        return self._dense

    @property
    def dense_error(self) -> str | None:
        return self._dense_error

    def dense_available(self) -> bool:
        try:
            self.get_dense()
        except RetrievalUnavailableError:
            return False
        return True

    def available_modes(self, probe_dense: bool = False) -> list[str]:
        """Modes that can run. ``probe_dense`` forces the lazy dense load to find out."""
        modes = ["bm25"]
        if self._dense is not None or (probe_dense and self.dense_available()) or (
            not probe_dense and self._dense_loader is not None and not self._dense_error
        ):
            modes += ["dense", "hybrid"]
            if self.reranker is not None:
                modes.append("hybrid_rerank")
        return modes

    # ------------------------------------------------------------------ search
    def search(self, query: str, mode: str = "hybrid", top_k: int = 10, reranker: Reranker | None = None) -> list[RetrievedPassage]:
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from {MODES}.")
        if not query or not query.strip() or top_k <= 0 or not self.passages:
            return []
        if mode == "bm25":
            return self.bm25.search(query, top_k)
        dense = self.get_dense()
        if mode == "dense":
            return dense.search(query, top_k)
        hybrid = HybridRetriever(self.bm25, dense, c=self.rrf_c, candidate_k=self.candidate_k)
        if mode == "hybrid":
            return hybrid.search(query, top_k)
        reranker = reranker or self.reranker
        if reranker is None:
            raise RetrievalUnavailableError("No reranker is configured.")
        pool = hybrid.search(query, max(top_k, min(self.candidate_k, 30)))
        return rerank(query, pool, reranker, top_k=top_k)


# ---------------------------------------------------------------------------------------------
def _kind_from_passages(passages: Sequence[Passage]) -> str:
    """Corpus kind derived from the passages themselves (so it is right even without a manifest)."""
    kinds = {p.source_type for p in passages}
    if not kinds:
        return "empty"
    if kinds == {"synthetic_fixture"}:
        return "demo_synthetic"
    if kinds == {"pubmed"}:
        return "pubmed"
    return "mixed"


class IndexManager:
    """Build, load, inspect and clear the local index under ``settings.index_dir``."""

    MANIFEST = "manifest.json"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index_dir = Path(settings.index_dir)
        self.processed_dir = Path(settings.processed_dir)
        self.chunk_config = ChunkConfig(settings.chunk_max_chars, settings.chunk_overlap_sentences)

    @property
    def passages_path(self) -> Path:
        return self.processed_dir / "passages.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / self.MANIFEST

    def read_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        try:
            return read_json(self.manifest_path)
        except (OSError, ValueError):
            return None

    def build(
        self,
        passages: Sequence[Passage],
        n_documents: int,
        corpus_kind: str,
        embedder: Embedder | None = None,
    ) -> dict:
        """(Re)build the on-disk index. BM25 is cheap and rebuilt at load time; the dense index is
        embedded and saved when ``embedder`` is given. A dense failure is recorded, not raised."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        dense_dir = self.index_dir / "dense"
        if dense_dir.exists():
            shutil.rmtree(dense_dir)
        manifest: dict = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus_kind": corpus_kind,
            "n_documents": n_documents,
            "n_passages": len(passages),
            "fingerprint": corpus_fingerprint(passages),
            "chunk_config": self.chunk_config.to_dict(),
            "chunk_signature": self.chunk_config.signature(),
            "rrf_c": self.settings.rrf_c,
            "dense": {"built": False, "error": None},
        }
        if embedder is not None and passages:
            try:
                dense = DenseRetriever(passages, embedder)
                meta = dense.save(dense_dir, chunk_signature=self.chunk_config.signature())
                manifest["dense"] = {
                    "built": True,
                    "error": None,
                    "model_name": meta["model_name"],
                    "dimension": meta["dimension"],
                    "backend": meta["backend"],
                }
            except BioMentorError as exc:
                manifest["dense"] = {"built": False, "error": str(exc)}
        elif embedder is None:
            manifest["dense"]["error"] = "Dense index not requested."
        write_json(self.manifest_path, manifest)
        return manifest

    def load(self, embedder: Embedder | None = None, reranker: Reranker | None = None) -> RetrievalIndex:
        """Load the corpus and build the retrieval index. Raises :class:`IndexNotFoundError` if
        no corpus has been prepared; dense problems never raise here (they surface lazily)."""
        if not self.passages_path.exists():
            raise IndexNotFoundError("No local corpus found. Run `python scripts/prepare_demo_data.py` then `python scripts/build_index.py`.")
        try:
            passages = load_passages(self.passages_path)
        except (ValueError, OSError) as exc:
            raise IndexCorruptError(f"The processed corpus could not be read: {exc}") from exc
        dense_dir = self.index_dir / "dense"

        loader: DenseLoader | None = None
        if embedder is not None and (dense_dir / "meta.json").exists():
            loader = lambda: DenseRetriever.load(  # noqa: E731
                dense_dir, passages, embedder, chunk_signature=self.chunk_config.signature()
            )
        elif embedder is not None:
            # No saved dense index: embed in memory on first use (fine for small corpora).
            loader = lambda: DenseRetriever(passages, embedder)  # noqa: E731
        return RetrievalIndex(
            passages,
            dense_loader=loader,
            reranker=reranker,
            rrf_c=self.settings.rrf_c,
            candidate_k=self.settings.candidate_k,
            corpus_kind=_kind_from_passages(passages),
        )

    def status(self) -> dict:
        manifest = self.read_manifest()
        return {
            "has_corpus": self.passages_path.exists(),
            "has_manifest": manifest is not None,
            "manifest": manifest,
        }

    def clear(self) -> None:
        """Delete the derived index (corpus files in data/raw and data/processed are kept)."""
        if self.index_dir.exists():
            for child in self.index_dir.iterdir():
                if child.name == ".gitkeep":
                    continue
                shutil.rmtree(child) if child.is_dir() else child.unlink()


__all__ = [
    "MODES",
    "MODE_LABELS",
    "IndexIncompatibleError",
    "IndexManager",
    "RetrievalIndex",
]
