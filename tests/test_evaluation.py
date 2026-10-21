import csv
import math

from biomentor.data.demo_fixtures import demo_benchmark_records
from biomentor.evaluation.answer_review import export_review_sheet, summarize_reviews
from biomentor.evaluation.benchmark import (
    BenchmarkRecord,
    compile_annotations,
    is_synthetic,
    load_benchmark,
    save_benchmark,
    validate_against_corpus,
    write_annotation_sheet,
)
from biomentor.evaluation.evaluate_retrieval import error_analysis, evaluate_method, run_evaluation
from biomentor.evaluation.metrics import (
    first_relevant_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from biomentor.evaluation.results import (
    aggregate_frame,
    load_results,
    save_results,
    skipped_methods,
)
from biomentor.exceptions import BenchmarkError
from biomentor.generation.citation_validator import validate_citations
from biomentor.generation.evidence import EvidenceSet
from biomentor.retrieval.index_manager import RetrievalIndex
from biomentor.schemas import Claim, ExplanationOutput


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# ---- metrics: hand-computed values ----
def test_recall_precision():
    ranked = ["a", "b", "c", "d", "e", "f"]
    rel = {"b", "e", "z"}
    assert approx(recall_at_k(ranked, rel, 5), 2 / 3)
    assert approx(recall_at_k(ranked, rel, 1), 0.0)
    assert approx(precision_at_k(ranked, rel, 5), 2 / 5)
    assert approx(precision_at_k(["a"], {"a"}, 5), 1 / 5)  # k is the denominator even with fewer results


def test_reciprocal_rank_and_first_rank():
    assert approx(reciprocal_rank(["x", "y", "r"], {"r"}), 1 / 3)
    assert reciprocal_rank(["x"], {"r"}) == 0.0 and first_relevant_rank(["x"], {"r"}) is None
    assert first_relevant_rank(["x", "r"], {"r"}) == 2


def test_duplicates_in_ranking_ignored():
    assert approx(recall_at_k(["a", "a", "b"], {"b"}, 2), 1.0)  # b is at position 2 after dedupe


def test_ndcg_binary_and_graded():
    assert approx(ndcg_at_k(["r", "x"], {"r": 1}, 10), 1.0)
    dcg = 1 / math.log2(2) + 0 + 1 / math.log2(4)  # relevant docs at rank 1 and 3
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert approx(ndcg_at_k(["a", "x", "b"], {"a": 1, "b": 1}, 10), dcg / idcg)
    graded = {"a": 2, "b": 1}
    dcg = 1 / math.log2(2) * 1 + 3 / math.log2(3)  # b(grade1) first, then a(grade2)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert approx(ndcg_at_k(["b", "a"], graded, 10), dcg / idcg)
    assert approx(ndcg_at_k(["x", "y"], {"a": 1}, 10), 0.0)


def test_metric_errors():
    for fn in (lambda: recall_at_k(["a"], set(), 5), lambda: precision_at_k(["a"], {"a"}, 0), lambda: ndcg_at_k(["a"], {"a": 0}, 5)):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError


# ---- benchmark file handling ----
def test_demo_benchmark_validates_against_demo_corpus(passages, docs):
    records = [BenchmarkRecord.model_validate(r) for r in demo_benchmark_records()]
    assert is_synthetic(records) and len(records) >= 10
    assert validate_against_corpus(records, {d.document_id for d in docs}, {p.passage_id for p in passages}) == []


def test_validate_against_corpus_reports_unknown_ids(docs):
    rec = BenchmarkRecord(question_id="q", query="x", topic="t", relevant_document_ids=["nope"], provenance="p")
    assert "nope" in validate_against_corpus([rec], {d.document_id for d in docs})[0]


def test_save_load_roundtrip_and_errors(tmp_path):
    path = tmp_path / "b.jsonl"
    save_benchmark(path, demo_benchmark_records())
    assert len(load_benchmark(path)) == len(demo_benchmark_records())
    for content in ("", "not json\n", '{"question_id": "a"}\n'):
        path.write_text(content)
        try:
            load_benchmark(path)
        except BenchmarkError:
            continue
        raise AssertionError(content)
    dup = demo_benchmark_records()[0]
    save_benchmark(path, [dup])
    path.write_text(path.read_text() * 2)
    try:
        load_benchmark(path)
    except BenchmarkError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError
    try:
        load_benchmark(tmp_path / "missing.jsonl")
    except BenchmarkError:
        pass
    else:
        raise AssertionError


def test_record_requires_relevant_docs_and_provenance():
    for kw in (dict(relevant_document_ids=[], provenance="p"), dict(relevant_document_ids=["a"], provenance=" ")):
        try:
            BenchmarkRecord(question_id="q", query="x", topic="t", **kw)
        except ValueError:
            continue
        raise AssertionError


# ---- running an evaluation ----
def build_index(passages, embedder):
    return RetrievalIndex.from_passages(passages, embedder=embedder, corpus_kind="demo_synthetic")


def records():
    return [BenchmarkRecord.model_validate(r) for r in demo_benchmark_records()]


def test_evaluate_method_computes_consistent_metrics(passages, embedder):
    idx = build_index(passages, embedder)
    out = evaluate_method(idx, records(), "bm25")
    assert out["status"] == "ok" and len(out["queries"]) == len(records())
    agg = out["aggregate"]
    for key in ("recall@5", "recall@10", "precision@5", "mrr", "ndcg@10"):
        assert 0.0 <= agg[key] <= 1.0
    q0 = out["queries"][0]
    assert q0["recall@5"] == recall_at_k(q0["top5_documents"], q0["relevant_documents"], 5)
    assert approx(agg["mrr"], sum(q["mrr"] for q in out["queries"]) / len(out["queries"]))
    assert agg["latency_ms_mean"] >= 0


def test_perfect_retriever_scores_one():
    from biomentor.schemas import Passage

    ps = [Passage(passage_id=f"d{i}-p000", document_id=f"d{i}", chunk_index=0, text=f"unique{i} marker{i} word", start_char=0, end_char=5) for i in range(5)]
    recs = [BenchmarkRecord(question_id=f"q{i}", query=f"unique{i} marker{i}", topic="t", relevant_document_ids=[f"d{i}"], provenance="synthetic test") for i in range(5)]
    out = evaluate_method(RetrievalIndex(ps), recs, "bm25")
    assert out["aggregate"]["recall@5"] == 1.0 and out["aggregate"]["mrr"] == 1.0 and out["aggregate"]["ndcg@10"] == 1.0


def test_unavailable_method_is_skipped_not_faked(passages):
    out = evaluate_method(RetrievalIndex(passages), records(), "dense")
    assert out["status"] == "skipped" and out["reason"] and "aggregate" not in out


def test_run_evaluation_all_modes_and_persistence(passages, embedder, tmp_path):
    bench = tmp_path / "bench.jsonl"
    save_benchmark(bench, demo_benchmark_records())
    res = run_evaluation(build_index(passages, embedder), records(), bench, config={"embedding_model": embedder.model_name})
    assert res["methods"]["bm25"]["status"] == res["methods"]["dense"]["status"] == res["methods"]["hybrid"]["status"] == "ok"
    assert res["methods"]["hybrid_rerank"]["status"] == "skipped"
    assert res["benchmark"]["synthetic"] and res["benchmark"]["n_queries"] == len(records()) and res["corpus"]["n_documents"] == 12
    save_results(res, tmp_path)
    loaded = load_results(tmp_path)
    assert loaded["config"]["embedding_model"] == "fake-hash-embedder"
    assert len(aggregate_frame(loaded)) == 3 and "Hybrid + cross-encoder rerank" in skipped_methods(loaded)
    assert (tmp_path / "results" / "aggregate_metrics.csv").exists() and (tmp_path / "results" / "query_level_metrics.csv").exists()


def test_load_results_missing_or_corrupt_is_none(tmp_path):
    assert load_results(tmp_path) is None
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "latest_results.json").write_text("{broken")
    assert load_results(tmp_path) is None


def test_error_analysis_lists_misses():
    fake = {"status": "ok", "queries": [
        {"first_relevant_rank": 1, "query": "ok"}, {"first_relevant_rank": 9, "query": "late"}, {"first_relevant_rank": None, "query": "never"}]}
    assert [q["query"] for q in error_analysis(fake, k=5)] == ["never", "late"]
    assert error_analysis({"status": "skipped"}) == []


# ---- annotation workflow ----
def test_annotation_sheet_and_compile(passages, docs, embedder, tmp_path):
    idx = build_index(passages, embedder)
    sheet = tmp_path / "sheet.csv"
    n = write_annotation_sheet(sheet, [("malaria parasite liver", "malaria"), ("asthma airway", "asthma")], idx, {d.document_id: d for d in docs}, top_k=3)
    assert n >= 2
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert {"question_id", "document_id", "relevance", "found_by"} <= set(rows[0])
    for r in rows:
        if r["question_id"] == "q001":
            r["relevance"] = "2" if r["document_id"] == "syn-mal-01" else "0"
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    recs, warnings = compile_annotations(sheet, "unit test")
    assert len(recs) == 1 and recs[0].relevance_grades == {"syn-mal-01": 2} and any("q002" in w for w in warnings)
    rows[0]["relevance"] = "7"
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:
        compile_annotations(sheet, "x")
    except BenchmarkError:
        pass
    else:
        raise AssertionError


# ---- human answer review ----
def test_review_sheet_export_and_summary(passages, tmp_path):
    ev = EvidenceSet.build([("retrieved", passages[:2])], nonce="n1")
    out = ExplanationOutput(
        short_definition=Claim(text="Insulin lowers blood glucose", basis="evidence", citation_ids=["E1", "E9"]),
        explanation=[], mechanism_steps=[], key_terms=[], key_findings=[], limitations=[], follow_up_questions=[],
        insufficient_evidence=False, insufficient_evidence_note="")
    clean, report = validate_citations(out, ev)
    path = tmp_path / "review.csv"
    assert export_review_sheet(path, "a1", clean.model_dump(), ev, report) == 1
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert rows[0]["cited_labels"] == "E1" and rows[0]["auto_invalid_ids_removed"] == "E9" and "[E1]" in rows[0]["cited_passages"]
    assert summarize_reviews(path)["statements_reviewed"] == 0 and summarize_reviews(path)["supported_rate"] is None
    rows[0].update(claim_support="supported", citation_valid="yes", uncertainty_ok="yes")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = summarize_reviews(path)
    assert s["supported_rate"] == 1.0 and s["citation_valid_rate"] == 1.0 and s["uncertainty_preserved_rate"] == 1.0


def test_paired_significance_test():
    from biomentor.evaluation.metrics import paired_significance_test

    baseline = [0.5, 0.6, 0.7, 0.8]
    treatment = [0.8, 0.9, 0.95, 1.0]
    res = paired_significance_test(baseline, treatment)
    assert res["mean_diff"] > 0
    assert "p_value" in res and "t_statistic" in res
    assert 0.0 <= res["p_value"] <= 1.0

