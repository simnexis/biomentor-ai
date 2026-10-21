"""Dense semantic retrieval: Sentence-Transformers embeddings + FAISS inner-product search.

All vectors are L2-normalised, so the inner product equals cosine similarity. FAISS
(``IndexFlatIP``, exact search) is used when installed; otherwise an equivalent numpy matrix
product is used so that the retriever (and its tests) still work. Embeddings are always saved as
``vectors.npy`` (portable source of truth); a ``faiss.index`` file is written too when FAISS is
available.

Index compatibility is checked on load: embedding model, dimension, normalisation, prefixes,
chunk configuration and a fingerprint of the exact passages must match, else
:class:`IndexIncompatibleError` is raised and the caller can rebuild.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from biomentor.config import DEFAULT_EMBEDDING_MODEL
from biomentor.exceptions import EmbeddingModelError, IndexCorruptError, IndexIncompatibleError
from biomentor.schemas import Passage, RetrievedPassage

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Embedder(Protocol):
    model_name: str
    query_prefix: str
    passage_prefix: str

    @property
    def dimension(self) -> int: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Lazy wrapper: the model is imported/downloaded on first use, not at import time."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._cache: dict[tuple[str, str], np.ndarray] = {}  # (prefix, text) -> vector; avoids re-embedding on reruns

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingModelError(
                    "The 'sentence-transformers' package is not installed. Run `pip install -r requirements.txt`."
                ) from exc
            try:
                logger.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:  # network, disk, unknown model name ...
                raise EmbeddingModelError(
                    f"Could not load the embedding model '{self.model_name}' ({type(exc).__name__}). "
                    "The first run downloads it from Hugging Face, so an internet connection is needed. "
                    "Keyword (BM25) search still works."
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        cache = self._cache
        missing = [t for t in dict.fromkeys(texts) if (prefix, t) not in cache]
        if missing:
            model = self._load()
            vectors = l2_normalize(
                model.encode([prefix + t for t in missing], batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
            )
            for text, vector in zip(missing, vectors, strict=True):
                cache[(prefix, text)] = vector
            if len(cache) > 20000:  # bound memory: drop the oldest half
                for key in list(cache)[: len(cache) // 2]:
                    del cache[key]
        return np.stack([cache[(prefix, t)] for t in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)


# ---------------------------------------------------------------------------------------------
# Vector index backends
# ---------------------------------------------------------------------------------------------
class NumpyFlatIP:
    """Exact inner-product search with numpy (same results as ``faiss.IndexFlatIP``)."""

    backend = "numpy"

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = self.vectors @ query[0]
        k = min(k, len(sims))
        order = np.argsort(-sims, kind="stable")[:k]
        return sims[order], order


class FaissFlatIP:
    backend = "faiss"

    def __init__(self, vectors: np.ndarray | None = None, index=None):
        import faiss

        self._faiss = faiss
        if index is None:
            index = faiss.IndexFlatIP(int(vectors.shape[1]))
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.index = index

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(np.ascontiguousarray(query, dtype=np.float32), k)
        keep = ids[0] >= 0
        return scores[0][keep], ids[0][keep]

    def write(self, path: Path) -> None:
        self._faiss.write_index(self.index, str(path))


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def make_index(vectors: np.ndarray, prefer_faiss: bool = True) -> NumpyFlatIP | FaissFlatIP:
    if prefer_faiss and faiss_available():
        return FaissFlatIP(vectors)
    return NumpyFlatIP(vectors)


def corpus_fingerprint(passages: Sequence[Passage]) -> str:
    """Hash of passage ids + texts; changes if the corpus or chunking changes."""
    import hashlib

    h = hashlib.sha256()
    for p in passages:
        h.update(p.passage_id.encode())
        h.update(b"\x1f")
        h.update(p.retrieval_text().encode())
        h.update(b"\x1e")
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------------------------
class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        passages: Sequence[Passage],
        embedder: Embedder,
        vectors: np.ndarray | None = None,
        prefer_faiss: bool = True,
        _index=None,
    ):
        self.passages = list(passages)
        self.embedder = embedder
        if vectors is None:
            texts = [p.retrieval_text() for p in self.passages]
            vectors = embedder.encode_passages(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        self.vectors = l2_normalize(vectors) if len(vectors) else np.zeros((0, 1), dtype=np.float32)
        if len(self.vectors) != len(self.passages):
            raise IndexCorruptError("Number of vectors does not match number of passages.")
        self._index = _index if _index is not None else (make_index(self.vectors, prefer_faiss) if len(self.vectors) else None)

    @property
    def backend(self) -> str:
        return self._index.backend if self._index is not None else "empty"

    def __len__(self) -> int:
        return len(self.passages)

    def search(self, query: str, k: int = 10) -> list[RetrievedPassage]:
        if k <= 0 or not query or not query.strip() or self._index is None:
            return []
        qvec = l2_normalize(self.embedder.encode_queries([query]))
        scores, ids = self._index.search(qvec, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores, ids, strict=False), start=1):
            results.append(
                RetrievedPassage(
                    passage=self.passages[int(idx)],
                    score=float(score),
                    rank=rank,
                    method=self.name,
                    component_ranks={self.name: rank},
                    component_scores={self.name: float(score)},
                )
            )
        return results

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path, chunk_signature: str = "") -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        if isinstance(self._index, FaissFlatIP):
            self._index.write(directory / "faiss.index")
        meta = {
            "schema_version": SCHEMA_VERSION,
            "model_name": self.embedder.model_name,
            "dimension": int(self.vectors.shape[1]) if len(self.vectors) else 0,
            "normalized": True,
            "similarity": "cosine (inner product of L2-normalised vectors)",
            "query_prefix": self.embedder.query_prefix,
            "passage_prefix": self.embedder.passage_prefix,
            "n_vectors": len(self.passages),
            "passage_ids": [p.passage_id for p in self.passages],  # row i <-> passage_ids[i]
            "fingerprint": corpus_fingerprint(self.passages),
            "chunk_signature": chunk_signature,
            "backend": self.backend,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    @classmethod
    def load(
        cls,
        directory: Path,
        passages: Sequence[Passage],
        embedder: Embedder,
        chunk_signature: str | None = None,
        prefer_faiss: bool = True,
    ) -> DenseRetriever:
        directory = Path(directory)
        meta_path, vec_path = directory / "meta.json", directory / "vectors.npy"
        if not meta_path.exists() or not vec_path.exists():
            raise IndexCorruptError("Dense index files are missing. Rebuild the index.")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vectors = np.load(vec_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise IndexCorruptError(f"Dense index could not be read ({type(exc).__name__}). Rebuild the index.") from exc

        passages = list(passages)
        problems = []
        if meta.get("schema_version") != SCHEMA_VERSION:
            problems.append("index format version differs")
        if meta.get("model_name") != embedder.model_name:
            problems.append(f"embedding model differs (index: {meta.get('model_name')}, current: {embedder.model_name})")
        if meta.get("query_prefix", "") != embedder.query_prefix or meta.get("passage_prefix", "") != embedder.passage_prefix:
            problems.append("embedding prefixes differ")
        if chunk_signature is not None and meta.get("chunk_signature") != chunk_signature:
            problems.append("chunking configuration differs")
        if meta.get("fingerprint") != corpus_fingerprint(passages):
            problems.append("the corpus changed since the index was built")
        if vectors.ndim != 2 or vectors.shape[0] != len(passages) or vectors.shape[0] != meta.get("n_vectors"):
            problems.append("vector count does not match passages")
        elif meta.get("dimension") != vectors.shape[1]:
            problems.append("vector dimension mismatch")
        if problems:
            raise IndexIncompatibleError("Saved dense index is incompatible: " + "; ".join(problems) + ". Rebuild the index.")

        index = None
        faiss_path = directory / "faiss.index"
        if prefer_faiss and faiss_path.exists() and faiss_available():
            try:
                import faiss

                index = FaissFlatIP(index=faiss.read_index(str(faiss_path)))
                if index.index.ntotal != len(passages):
                    index = None
            except Exception as exc:  # corrupt faiss file -> rebuild from vectors.npy
                logger.warning("Ignoring unreadable faiss.index (%s); rebuilding from vectors.npy", type(exc).__name__)
                index = None
        return cls(passages, embedder, vectors=vectors, prefer_faiss=prefer_faiss, _index=index)
