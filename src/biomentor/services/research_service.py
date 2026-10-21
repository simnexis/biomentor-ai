"""Search orchestration: local indexed corpus or live PubMed, with graceful fallbacks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from biomentor.config import Settings
from biomentor.data.chunking import ChunkConfig
from biomentor.data.ingestion import (
    IngestionStats,
    append_raw_documents,
    corpus_kind_of,
    documents_to_passages,
    find_raw_files,
    process_corpus,
)
from biomentor.data.preprocessing import STOPWORDS, tokenize
from biomentor.data.pubmed_client import PubMedClient
from biomentor.data.storage import deduplicate_documents, load_documents
from biomentor.exceptions import (
    IndexNotFoundError,
    PubMedError,
    RetrievalUnavailableError,
)
from biomentor.generation.abstention import validate_query
from biomentor.retrieval.dense_retriever import Embedder
from biomentor.retrieval.index_manager import MODE_LABELS, IndexManager, RetrievalIndex
from biomentor.retrieval.reranker import Reranker
from biomentor.schemas import Document, RetrievedPassage

logger = logging.getLogger(__name__)

SOURCE_LABELS = {
    "live_pubmed": "Live PubMed retrieval (NCBI E-utilities)",
    "local_corpus": "Local indexed corpus (PubMed records you fetched)",
    "demo_fixtures": "Demo fixtures - SYNTHETIC test data, not real publications",
    "local_mixed": "Local indexed corpus (includes synthetic demo fixtures)",
}


@dataclass
class SearchResponse:
    query: str
    mode_requested: str
    mode_used: str
    results: list[RetrievedPassage]
    source_kind: str
    source_label: str
    n_passages_searched: int
    warnings: list[str] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)
    documents: dict[str, Document] = field(default_factory=dict)
    pubmed_query_used: str | None = None


def pubmed_query_variants(query: str) -> list[str]:
    """Natural-language questions often return nothing from PubMed, which ANDs every term.
    Try the original query, then its content keywords (AND), then the keywords with OR."""
    original = " ".join(query.split())
    keywords = []
    for tok in tokenize(query):
        if tok not in STOPWORDS and len(tok) > 2 and tok not in keywords and "-" not in tok:
            keywords.append(tok)
    keywords = keywords[:8]
    variants = [original]
    if keywords and " ".join(keywords) != original.lower():
        variants.append(" ".join(keywords))
        if len(keywords) > 1:
            variants.append(" OR ".join(keywords))
    return variants


def _kind_to_source(corpus_kind: str) -> str:
    return {"demo_synthetic": "demo_fixtures", "pubmed": "local_corpus"}.get(corpus_kind, "local_mixed")


class SearchService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        pubmed_client: PubMedClient | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self._pubmed = pubmed_client
        self.index_manager = IndexManager(settings)
        self._local_index: RetrievalIndex | None = None
        self._local_docs: dict[str, Document] = {}
        self._stamp: tuple | None = None

    # ------------------------------------------------------------------ local corpus
    def _current_stamp(self) -> tuple:
        def mt(p: Path) -> float:
            return p.stat().st_mtime if p.exists() else 0.0

        return (mt(self.index_manager.passages_path), mt(self.index_manager.manifest_path))

    def has_local_corpus(self) -> bool:
        return self.index_manager.passages_path.exists()

    def local_index(self) -> RetrievalIndex:
        """Cached local index; reloaded automatically if the corpus/index files changed."""
        stamp = self._current_stamp()
        if self._local_index is None or stamp != self._stamp:
            self._local_index = self.index_manager.load(embedder=self.embedder, reranker=self.reranker)
            self._local_docs = {d.document_id: d for d in load_documents(self.settings.processed_dir / "documents.jsonl")}
            self._stamp = stamp
        return self._local_index

    def local_documents(self) -> dict[str, Document]:
        self.local_index()
        return self._local_docs

    def rebuild_local_index(self, build_dense: bool = True) -> tuple[IngestionStats, dict]:
        """Re-process ``data/raw/*.jsonl`` and rebuild the index (embedding model downloads on first use)."""
        raw = find_raw_files(self.settings.raw_dir)
        if not raw:
            raise IndexNotFoundError(
                "No raw documents found in data/raw. Run `python scripts/prepare_demo_data.py` "
                "or `python scripts/fetch_literature.py --query \"...\"` first."
            )
        cfg = ChunkConfig(self.settings.chunk_max_chars, self.settings.chunk_overlap_sentences)
        docs, passages, stats = process_corpus(raw, self.settings.processed_dir, cfg)
        manifest = self.index_manager.build(
            passages, len(docs), stats.corpus_kind, embedder=self.embedder if build_dense else None
        )
        self._local_index = None
        return stats, manifest

    # ------------------------------------------------------------------ live PubMed
    def pubmed_client(self) -> PubMedClient:
        if self._pubmed is None:
            self._pubmed = PubMedClient(
                email=self.settings.ncbi_email,
                api_key=self.settings.ncbi_api_key,
                tool=self.settings.ncbi_tool,
                cache_dir=self.settings.cache_dir,
            )
        return self._pubmed

    def save_live_documents(self, documents: list[Document], filename: str = "pubmed_live.jsonl") -> int:
        """Append live-fetched records to the raw corpus (rebuild the index afterwards to search them)."""
        return append_raw_documents(self.settings.raw_dir / filename, documents)

    # ------------------------------------------------------------------ search
    def _run(self, index: RetrievalIndex, query: str, mode: str, top_k: int, warnings: list[str]) -> tuple[str, list[RetrievedPassage]]:
        chain = {"hybrid_rerank": ["hybrid_rerank", "hybrid", "bm25"], "hybrid": ["hybrid", "bm25"], "dense": ["dense", "bm25"], "bm25": ["bm25"]}[mode]
        for attempt in chain:
            try:
                return attempt, index.search(query, attempt, top_k, reranker=self.reranker)
            except RetrievalUnavailableError as exc:
                if attempt == "bm25":
                    raise
                nxt = chain[chain.index(attempt) + 1]
                warnings.append(f"{MODE_LABELS[attempt]} is unavailable ({exc}). Falling back to {MODE_LABELS[nxt]}.")
        return "bm25", []  # unreachable

    def search(self, query: str, mode: str = "hybrid", top_k: int = 10, source: str = "local", live_fetch: int = 30) -> SearchResponse:
        query = validate_query(query)
        top_k = max(1, min(int(top_k), 50))
        if source == "auto":
            # Prefer fresh PubMed results; if PubMed is unreachable or finds nothing, use the local corpus.
            try:
                return self.search(query, mode, top_k, "live", live_fetch)
            except PubMedError as exc:
                if not self.has_local_corpus():
                    raise
                fallback = self.search(query, mode, top_k, "local", live_fetch)
                fallback.warnings.insert(0, f"Live PubMed search was not available ({exc}) so results come from your local corpus.")
                return fallback
        warnings: list[str] = []
        timing: dict[str, float] = {}
        t0 = time.perf_counter()

        if source == "live":
            client = self.pubmed_client()
            docs: list[Document] = []
            used_query = query
            for variant in pubmed_query_variants(query):
                docs = client.search(variant, max_results=live_fetch)
                used_query = variant
                if docs:
                    break
            if not docs:
                raise PubMedError("PubMed returned no records for this query. Try different or fewer keywords.")
            if used_query != query:
                warnings.append(f"PubMed returned nothing for the full question, so it was searched as: {used_query}")
            docs, _ = deduplicate_documents(docs)
            timing["pubmed_fetch"] = (time.perf_counter() - t0) * 1000
            passages = documents_to_passages(docs, ChunkConfig(self.settings.chunk_max_chars, self.settings.chunk_overlap_sentences))
            if not passages:
                raise PubMedError("PubMed returned records but none had abstracts to search.")
            index = RetrievalIndex.from_passages(
                passages, embedder=self.embedder, reranker=self.reranker,
                rrf_c=self.settings.rrf_c, candidate_k=self.settings.candidate_k, corpus_kind="pubmed",
            )
            doc_map = {d.document_id: d for d in docs}
            kind, label, pubmed_query = "live_pubmed", SOURCE_LABELS["live_pubmed"], used_query
        else:
            index = self.local_index()
            doc_map = self._local_docs
            src = _kind_to_source(index.corpus_kind)
            kind, label, pubmed_query = src, SOURCE_LABELS[src], None

        t1 = time.perf_counter()
        used, results = self._run(index, query, mode, top_k, warnings)
        timing["retrieval"] = (time.perf_counter() - t1) * 1000
        timing["total"] = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            query=query,
            mode_requested=mode,
            mode_used=used,
            results=results,
            source_kind=kind,
            source_label=label,
            n_passages_searched=len(index.passages),
            warnings=warnings,
            timing_ms={k: round(v, 1) for k, v in timing.items()},
            documents={r.passage.document_id: doc_map[r.passage.document_id] for r in results if r.passage.document_id in doc_map},
            pubmed_query_used=pubmed_query,
        )


def group_by_document(results: list[RetrievedPassage]) -> list[tuple[str, list[RetrievedPassage]]]:
    """Group passage hits by document, ordered by each document's best passage."""
    groups: dict[str, list[RetrievedPassage]] = {}
    for r in results:
        groups.setdefault(r.passage.document_id, []).append(r)
    return list(groups.items())


__all__ = ["SOURCE_LABELS", "SearchResponse", "SearchService", "corpus_kind_of", "group_by_document", "pubmed_query_variants"]
