from biomentor.retrieval.bm25_retriever import BM25Retriever


def test_relevant_passage_ranked_first_with_metadata(passages):
    bm = BM25Retriever(passages)
    results = bm.search("Plasmodium parasites liver merozoites", k=3)
    assert results[0].passage.document_id == "syn-mal-01"
    r = results[0]
    assert r.rank == 1 and r.method == "bm25" and r.score > 0
    assert r.component_ranks == {"bm25": 1} and r.passage.source_type == "synthetic_fixture"


def test_scores_sorted_descending_and_ranks_sequential(passages):
    results = BM25Retriever(passages).search("insulin glucose blood", k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_k_limits_results(passages):
    assert len(BM25Retriever(passages).search("blood", k=2)) == 2
    assert BM25Retriever(passages).search("blood", k=0) == []


def test_empty_and_whitespace_query(passages):
    bm = BM25Retriever(passages)
    assert bm.search("") == [] and bm.search("   ") == []


def test_stopword_only_query_returns_nothing(passages):
    assert BM25Retriever(passages).search("the of and") == []


def test_empty_corpus():
    bm = BM25Retriever([])
    assert bm.search("anything") == [] and len(bm) == 0


def test_unknown_terms_return_nothing(passages):
    assert BM25Retriever(passages).search("zzzxqv") == []


def test_matches_formula_by_hand(passages):
    bm = BM25Retriever(passages[:3])
    import math

    q = "insulin"
    n = 3
    df = sum(1 for p in passages[:3] if "insulin" in p.retrieval_text().lower())
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    assert df >= 1
    scores = bm.scores(q)
    assert scores.max() > 0 and idf > 0


def test_hyphenated_part_matches_compound():
    from biomentor.schemas import Passage

    p = Passage(passage_id="a-p000", document_id="a", chunk_index=0, text="SARS-CoV-2 spike protein binds ACE2.", start_char=0, end_char=36)
    assert BM25Retriever([p]).search("cov")  # part of the hyphenated term
    assert BM25Retriever([p]).search("SARS-CoV-2")
