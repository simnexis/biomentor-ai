from biomentor.retrieval.bm25_retriever import BM25Retriever
from biomentor.retrieval.dense_retriever import DenseRetriever
from biomentor.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from biomentor.retrieval.index_manager import RetrievalIndex
from biomentor.retrieval.reranker import rerank
from biomentor.schemas import Passage, RetrievedPassage


def P(pid):
    return Passage(passage_id=pid, document_id=pid.split("-")[0], chunk_index=0, text=f"text {pid}", start_char=0, end_char=5)


def R(pid, rank, method="x", score=1.0):
    return RetrievedPassage(passage=P(pid), score=score, rank=rank, method=method)


def test_rrf_formula_and_order():
    fused = reciprocal_rank_fusion({"a": [R("d1-p0", 1), R("d2-p0", 2)], "b": [R("d2-p0", 1), R("d3-p0", 2)]}, c=60)
    scores = {r.passage.passage_id: r.score for r in fused}
    assert abs(scores["d2-p0"] - (1 / 62 + 1 / 61)) < 1e-12  # rank 2 in a, rank 1 in b
    assert abs(scores["d1-p0"] - 1 / 61) < 1e-12 and abs(scores["d3-p0"] - 1 / 62) < 1e-12
    assert [r.passage.passage_id for r in fused] == ["d2-p0", "d1-p0", "d3-p0"]
    assert [r.rank for r in fused] == [1, 2, 3]


def test_rrf_preserves_component_ranks_and_scores_and_dedupes():
    fused = reciprocal_rank_fusion({"bm25": [R("d1-p0", 1, score=7.5)], "dense": [R("d1-p0", 3, score=0.42)]})
    assert len(fused) == 1
    assert fused[0].component_ranks == {"bm25": 1, "dense": 1}  # positions within each list
    assert fused[0].component_scores == {"bm25": 7.5, "dense": 0.42}


def test_rrf_uses_position_not_the_rank_field():
    fused = reciprocal_rank_fusion({"a": [R("d1-p0", 99), R("d2-p0", 100)]}, c=0)
    assert fused[0].score == 1.0 and fused[1].score == 0.5


def test_rrf_constant_changes_scores_not_order_for_single_list():
    a = reciprocal_rank_fusion({"a": [R("d1-p0", 1), R("d2-p0", 2)]}, c=1)
    b = reciprocal_rank_fusion({"a": [R("d1-p0", 1), R("d2-p0", 2)]}, c=100)
    assert [r.passage.passage_id for r in a] == [r.passage.passage_id for r in b] and a[0].score != b[0].score


def test_rrf_top_k_ties_and_duplicates_within_a_list():
    fused = reciprocal_rank_fusion({"a": [R("d1-p0", 1), R("d1-p0", 2), R("d2-p0", 3)], "b": [R("d3-p0", 1)]}, top_k=2)
    assert len(fused) == 2
    # d1 (pos 1 in a) and d3 (pos 1 in b) tie; first-seen wins
    assert [r.passage.passage_id for r in fused] == ["d1-p0", "d3-p0"]


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"a": []}) == []


def test_rrf_negative_constant_rejected():
    try:
        reciprocal_rank_fusion({"a": []}, c=-1)
    except ValueError:
        return
    raise AssertionError


def test_hybrid_retriever_combines_both_systems(passages, embedder):
    hybrid = HybridRetriever(BM25Retriever(passages), DenseRetriever(passages, embedder, prefer_faiss=False))
    results = hybrid.search("airway inflammation asthma", k=5)
    assert results[0].passage.document_id.startswith("syn-asm")
    assert results[0].method == "hybrid" and set(results[0].component_ranks) <= {"bm25", "dense"}
    assert hybrid.search("", 5) == [] and hybrid.search("x", 0) == []


def test_retrieval_index_modes_and_fallback_reasons(passages, embedder):
    idx = RetrievalIndex.from_passages(passages, embedder=embedder)
    for mode in ("bm25", "dense", "hybrid"):
        assert idx.search("tuberculosis granuloma macrophages", mode, 3)[0].passage.document_id == "syn-tb-01"
    no_dense = RetrievalIndex(passages)
    assert no_dense.available_modes() == ["bm25"]
    try:
        no_dense.search("x", "hybrid", 3)
    except Exception as exc:
        assert type(exc).__name__ == "RetrievalUnavailableError"
    else:
        raise AssertionError
    try:
        idx.search("x", "nonsense", 3)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_dense_failure_is_remembered_and_bm25_still_works(passages):
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("model download failed")

    idx = RetrievalIndex(passages, dense_loader=boom)
    assert not idx.dense_available() and not idx.dense_available()
    assert len(calls) == 1 and "model download failed" in idx.dense_error
    assert idx.search("insulin", "bm25", 3)


class FakeReranker:
    model_name = "fake"

    def score(self, query, texts):
        return [float(len(t)) for t in texts]  # longest text first


def test_rerank_reorders_and_keeps_components():
    cands = [R("d1-p0", 1), R("d22222-p0", 2)]
    out = rerank("q", cands, FakeReranker(), top_k=2)
    assert out[0].passage.passage_id == "d22222-p0" and out[0].method == "hybrid_rerank"
    assert out[0].component_ranks["rerank"] == 1
    assert rerank("q", [], FakeReranker()) == [] and rerank("  ", cands, FakeReranker()) == []
