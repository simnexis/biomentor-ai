from biomentor.exceptions import QueryValidationError
from biomentor.generation.abstention import assess_evidence, check_query_safety, validate_query
from biomentor.retrieval.bm25_retriever import BM25Retriever


def test_validate_query_cleans_and_rejects():
    assert validate_query("  type   2 diabetes ") == "type 2 diabetes"
    for bad in (None, "", "   ", "the of and", "x" * 600):
        try:
            validate_query(bad)
        except QueryValidationError as exc:
            assert str(exc)
        else:
            raise AssertionError(f"accepted {bad!r}")


def test_personal_medical_questions_are_routed_to_safety():
    for q in ("Do I have diabetes?", "Should I take ibuprofen with my blood thinner?", "What dose of amoxicillin should I take",
              "My test results show high glucose, what now", "am I dying", "diagnose me", "I have a fever and chest pain"):
        assert check_query_safety(q).is_personal_medical, q
    assert "healthcare professional" in check_query_safety("Do I have diabetes?").message


def test_general_science_questions_are_not_blocked():
    for q in ("How does insulin resistance develop?", "What do studies say about aspirin and stroke prevention?",
              "Explain the malaria life cycle", "What is the mechanism of beta-2 agonists?", "Do I need to know anatomy first? no - what is a granuloma"):
        assert not check_query_safety(q).is_personal_medical, q


def test_no_results_means_insufficient():
    a = assess_evidence("malaria", [])
    assert not a.sufficient and "No relevant passages" in a.reason


def test_relevant_results_are_sufficient(passages):
    results = BM25Retriever(passages).search("malaria parasite liver", 5)
    a = assess_evidence("malaria parasite liver", results)
    assert a.sufficient and a.coverage > 0.9


def test_off_topic_query_abstains_even_though_bm25_returns_something(passages):
    results = BM25Retriever(passages).search("blood cancer chemotherapy leukemia", 5)  # matches only 'blood'
    assert results  # the retriever returns weak hits...
    a = assess_evidence("blood cancer chemotherapy leukemia", results)
    assert not a.sufficient and a.coverage < 0.34  # ...but coverage says the topic is not covered


def test_generic_only_query_skips_coverage_test(passages):
    results = BM25Retriever(passages).search("explain mechanism disease", 5) or BM25Retriever(passages).search("blood", 3)
    assert assess_evidence("explain how it works", results).sufficient
