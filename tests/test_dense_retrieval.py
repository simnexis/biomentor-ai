import json

import numpy as np

from biomentor.exceptions import IndexCorruptError, IndexIncompatibleError
from biomentor.retrieval.dense_retriever import DenseRetriever, NumpyFlatIP, l2_normalize


def test_l2_normalize_unit_length_and_zero_vector_safe():
    v = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(v[0]), 1.0) and np.all(v[1] == 0)


def test_numpy_index_matches_brute_force_cosine():
    rng = np.random.default_rng(0)
    vecs = l2_normalize(rng.normal(size=(20, 16)))
    q = l2_normalize(rng.normal(size=(1, 16)))
    scores, ids = NumpyFlatIP(vecs).search(q, 5)
    expected = np.argsort(-(vecs @ q[0]))[:5]
    assert list(ids) == list(expected) and np.allclose(scores, (vecs @ q[0])[expected])


def test_search_maps_vectors_to_correct_passages_with_metadata(passages, embedder):
    dense = DenseRetriever(passages, embedder, prefer_faiss=False)
    results = dense.search("malaria parasite fever red blood cells", k=3)
    assert results[0].passage.document_id == "syn-mal-01"
    r = results[0]
    assert r.method == "dense" and r.rank == 1 and "dense" in r.component_scores
    assert r.passage.passage_id == "syn-mal-01-p000" and r.passage.source_name
    assert -1.0001 <= r.score <= 1.0001  # cosine range


def test_vectors_are_normalized(passages, embedder):
    dense = DenseRetriever(passages, embedder, prefer_faiss=False)
    assert np.allclose(np.linalg.norm(dense.vectors, axis=1), 1.0, atol=1e-5)


def test_empty_query_and_empty_corpus(passages, embedder):
    assert DenseRetriever(passages, embedder, prefer_faiss=False).search("  ") == []
    empty = DenseRetriever([], embedder, prefer_faiss=False)
    assert empty.search("x") == [] and empty.backend == "empty"


def test_k_larger_than_corpus(passages, embedder):
    assert len(DenseRetriever(passages[:3], embedder, prefer_faiss=False).search("insulin", k=50)) == 3


def test_save_and_load_roundtrip(passages, embedder, tmp_path):
    a = DenseRetriever(passages, embedder, prefer_faiss=False)
    meta = a.save(tmp_path, chunk_signature="sig")
    assert meta["passage_ids"][0] == passages[0].passage_id and meta["dimension"] == embedder.dimension
    b = DenseRetriever.load(tmp_path, passages, embedder, chunk_signature="sig", prefer_faiss=False)
    ra, rb = a.search("asthma airway", 5), b.search("asthma airway", 5)
    assert [x.passage.passage_id for x in ra] == [x.passage.passage_id for x in rb]


def _load_expecting(error, tmp_path, passages, embedder, **kw):
    try:
        DenseRetriever.load(tmp_path, passages, embedder, prefer_faiss=False, **kw)
    except error as exc:
        return str(exc)
    raise AssertionError(f"expected {error.__name__}")


def test_incompatible_when_corpus_changes(passages, embedder, tmp_path):
    DenseRetriever(passages, embedder, prefer_faiss=False).save(tmp_path, "sig")
    msg = _load_expecting(IndexIncompatibleError, tmp_path, passages[:-1], embedder, chunk_signature="sig")
    assert "corpus" in msg or "vector count" in msg


def test_incompatible_when_model_or_chunking_differs(passages, embedder, tmp_path):
    DenseRetriever(passages, embedder, prefer_faiss=False).save(tmp_path, "sig")

    class Other(type(embedder)):
        model_name = "another-model"

    assert "embedding model differs" in _load_expecting(IndexIncompatibleError, tmp_path, passages, Other(), chunk_signature="sig")
    assert "chunking" in _load_expecting(IndexIncompatibleError, tmp_path, passages, embedder, chunk_signature="different")


def test_corrupt_and_missing_files(passages, embedder, tmp_path):
    _load_expecting(IndexCorruptError, tmp_path, passages, embedder)  # nothing saved
    DenseRetriever(passages, embedder, prefer_faiss=False).save(tmp_path, "sig")
    (tmp_path / "vectors.npy").write_bytes(b"garbage")
    _load_expecting(IndexCorruptError, tmp_path, passages, embedder, chunk_signature="sig")
    (tmp_path / "meta.json").write_text("{broken")
    _load_expecting(IndexCorruptError, tmp_path, passages, embedder, chunk_signature="sig")


def test_meta_records_vector_to_passage_mapping(passages, embedder, tmp_path):
    DenseRetriever(passages, embedder, prefer_faiss=False).save(tmp_path, "sig")
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["passage_ids"] == [p.passage_id for p in passages] and meta["normalized"] is True
