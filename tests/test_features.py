"""Extractive digest, references, Markdown export, HTML helpers, auto source, embedding cache, runner."""

import json

import numpy as np

from biomentor.data.pubmed_client import PubMedClient
from biomentor.evaluation.runner import find_benchmark, run_and_save
from biomentor.exceptions import BenchmarkError, BioMentorError, IndexNotFoundError, PubMedError
from biomentor.generation.evidence import EvidenceSet
from biomentor.retrieval.bm25_retriever import BM25Retriever
from biomentor.retrieval.dense_retriever import SentenceTransformerEmbedder
from biomentor.schemas import Claim, Document, ExplanationOutput, Term
from biomentor.services import learning_service as ls
from biomentor.services.digest import (
    build_references,
    extractive_digest,
    format_reference,
    result_to_markdown,
)
from biomentor.services.research_service import SearchService
from biomentor.ui.formatting import claim_html, highlight_html, meta_html, source_chip
from conftest import PUBMED_XML, FakeResponse, FakeSession
from test_services import prepare  # noqa: E402  (reuse the corpus helper)


# ---- extractive digest ----
def test_digest_returns_exact_sentences_that_match_the_query(passages):
    hits = BM25Retriever(passages).search("how do Plasmodium parasites cause fever", 5)
    digest = extractive_digest("how do Plasmodium parasites cause fever", hits, max_sentences=3)
    assert digest and all(d.sentence in d.passage.text for d in digest)  # exact quotations
    assert any("fever" in d.sentence for d in digest)
    assert digest[0].passage.document_id == "syn-mal-01"


def test_digest_limits_per_document_and_total(passages):
    hits = BM25Retriever(passages).search("malaria parasites mosquitoes", 10)
    digest = extractive_digest("malaria parasites mosquitoes", hits, max_sentences=4, max_per_document=1)
    docs = [d.passage.document_id for d in digest]
    assert len(docs) == len(set(docs)) and len(digest) <= 4


def test_digest_prefers_empty_over_irrelevant(passages):
    hits = BM25Retriever(passages).search("blood", 5)
    assert extractive_digest("leukemia chemotherapy resistance", hits) == []
    assert extractive_digest("", hits) == [] and extractive_digest("malaria", []) == []


# ---- references ----
def test_reference_uses_only_available_fields():
    d = Document(document_id="pmid-1", title="A title.", source_name="PubMed", source_type="pubmed", pmid="1", doi="10.1/x",
                 authors=["Lee A", "Kim B", "Roe C", "Poe D"], journal="J Test", publication_year=2020)
    ref = format_reference(d)
    assert ref == "Lee A, Kim B, Roe C, et al.. A title. J Test. 2020. PMID 1. doi:10.1/x"
    bare = Document(document_id="x", title="Only title", source_name="s", source_type="synthetic_fixture")
    assert format_reference(bare) == "Only title [synthetic demo text, not a real publication]"


def test_build_references_groups_labels_per_document(passages):
    ev = EvidenceSet.build([("retrieved", passages)], max_passages=12, max_chars=100000)
    refs = build_references(ev)
    assert len(refs) == len({p.document_id for p in passages})
    assert all(labels and ref for labels, ref, _ in refs)


# ---- markdown export ----
def explanation_result(settings, with_output):
    prepare(settings)
    svc = SearchService(settings)
    res = ls.explain_topic(svc, None, settings, "malaria parasite liver red blood cells", "beginner", mode="bm25")
    if with_output:
        res.output = ExplanationOutput(
            short_definition=Claim(text="Malaria is caused by Plasmodium", basis="evidence", citation_ids=["E1"]),
            explanation=[Claim(text="Parasites invade the liver", basis="evidence", citation_ids=["E1"])],
            mechanism_steps=[Claim(text="Sporozoites reach the liver", basis="evidence", citation_ids=["E1"]), Claim(text="Merozoites infect red cells", basis="evidence", citation_ids=["E1"])],
            key_terms=[Term(term="Merozoite", definition="A stage that infects red cells", citation_ids=["E1"])],
            key_findings=[], limitations=[Claim(text="Abstract-level only", basis="background", citation_ids=[])],
            follow_up_questions=["What is a vector?"], insufficient_evidence=False, insufficient_evidence_note="")
    return res


def test_markdown_export_with_generated_answer(settings):
    md = result_to_markdown(explanation_result(settings, True))
    for needle in ("# malaria parasite", "## Short definition", "Malaria is caused by Plasmodium [E1]", "1. Sporozoites reach the liver [E1]",
                   "2. Merozoites", "**Merozoite**", "(general background, not from the sources)", "## References", "[E1]", "synthetic demo text", "Not medical advice"):
        assert needle in md, needle


def test_markdown_export_without_gemini_uses_extractive_digest(settings):
    res = explanation_result(settings, False)
    assert res.status == ls.NO_GEMINI
    md = result_to_markdown(res)
    assert "extracted, not generated" in md and "## Evidence passages" in md and "Warning: the sources are synthetic" in md


# ---- html helpers ----
def test_highlight_escapes_html_and_marks_stems():
    out = highlight_html("Parasites infect <b>red</b> blood cells", "parasite infection")
    assert "<mark>Parasites</mark>" in out and "<mark>infect</mark>" in out
    assert "&lt;b&gt;" in out and "<b>" not in out  # source text cannot inject markup


def test_highlight_without_usable_terms_is_just_escaped():
    assert highlight_html("A & B", "the of") == "A &amp; B"
    assert highlight_html("", "anything") == ""


def test_claim_html_escapes_and_labels_basis():
    out = claim_html("x <script>alert(1)</script>", ["E1", "E2"], "analogy")
    assert "<script>" not in out and "&lt;script&gt;" in out and out.count('class="cite"') == 2 and "analogy" in out


def test_meta_html_only_available_fields_and_links():
    assert meta_html(None, None, None, None, None) == "No bibliographic details available"
    out = meta_html(["A B", "C D", "E F", "G H"], 2020, "J", "123", "10.1/x")
    assert "et al." in out and "pubmed.ncbi.nlm.nih.gov/123/" in out and "doi.org/10.1/x" in out
    assert "Synthetic" in source_chip("synthetic_fixture")


# ---- auto source fallback ----
def test_auto_source_uses_live_when_available(settings):
    session = FakeSession(FakeResponse(200, json.dumps({"esearchresult": {"count": "2", "idlist": ["99999901", "99999902"]}})), FakeResponse(200, PUBMED_XML))
    svc = SearchService(settings, pubmed_client=PubMedClient(session=session, sleep=lambda s: None))
    assert svc.search("synthetic abstract", "bm25", 5, source="auto").source_kind == "live_pubmed"


def test_auto_source_falls_back_to_local_with_a_warning(settings):
    prepare(settings)
    client = PubMedClient(session=FakeSession(*[FakeResponse(503)] * 4), sleep=lambda s: None)
    resp = SearchService(settings, pubmed_client=client).search("malaria parasite", "bm25", 3, source="auto")
    assert resp.source_kind == "demo_fixtures" and "not available" in resp.warnings[0]


def test_auto_source_without_local_corpus_raises_the_pubmed_error(settings):
    client = PubMedClient(session=FakeSession(*[FakeResponse(503)] * 4), sleep=lambda s: None)
    try:
        SearchService(settings, pubmed_client=client).search("malaria", "bm25", 3, source="auto")
    except PubMedError:
        return
    raise AssertionError


# ---- embedding cache ----
class CountingModel:
    def __init__(self):
        self.encoded = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kw):
        self.encoded.extend(texts)
        return np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)


def test_embedder_encodes_each_text_once_and_keeps_order():
    emb = SentenceTransformerEmbedder("fake")
    emb._model = CountingModel()
    a = emb.encode_passages(["alpha", "beta", "alpha"])
    assert emb._model.encoded == ["alpha", "beta"] and a.shape == (3, 4) and np.allclose(a[0], a[2])
    emb.encode_passages(["beta", "gamma"])
    assert emb._model.encoded == ["alpha", "beta", "gamma"]
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert emb.encode_passages([]).shape == (0, 4)


def test_embedder_cache_separates_prefixes():
    emb = SentenceTransformerEmbedder("fake", query_prefix="query: ", passage_prefix="passage: ")
    emb._model = CountingModel()
    emb.encode_queries(["x"])
    emb.encode_passages(["x"])
    assert emb._model.encoded == ["query: x", "passage: x"]


# ---- evaluation runner ----
def test_runner_end_to_end_and_errors(settings, embedder):
    try:
        run_and_save(settings, None)
    except BenchmarkError as exc:
        assert "No benchmark" in str(exc)
    else:
        raise AssertionError
    prepare(settings)
    from biomentor.data.demo_fixtures import demo_benchmark_records
    from biomentor.evaluation.benchmark import save_benchmark

    save_benchmark(settings.eval_dir / "demo_benchmark.jsonl", demo_benchmark_records())
    assert find_benchmark(settings).name == "demo_benchmark.jsonl"
    res = run_and_save(settings, embedder)
    assert res["methods"]["hybrid"]["status"] == "ok" and res["config"]["embedding_model"] == "fake-hash-embedder"
    only_bm25 = run_and_save(settings, None)
    assert list(only_bm25["methods"]) == ["bm25"]
    assert (settings.eval_dir / "results" / "latest_results.json").exists()


def test_runner_rejects_mismatched_benchmark_and_missing_corpus(settings):
    from biomentor.data.demo_fixtures import demo_benchmark_records
    from biomentor.evaluation.benchmark import save_benchmark

    save_benchmark(settings.eval_dir / "benchmark.jsonl", demo_benchmark_records())
    try:
        run_and_save(settings, None)
    except IndexNotFoundError:
        pass
    else:
        raise AssertionError
    save_documents_ok = Document(document_id="other-1", title="Other", abstract="Something unrelated entirely about kidneys.", source_name="s", source_type="synthetic_fixture")
    from biomentor.data.ingestion import process_corpus
    from biomentor.data.storage import save_documents

    save_documents(settings.raw_dir / "o.jsonl", [save_documents_ok])
    process_corpus([settings.raw_dir / "o.jsonl"], settings.processed_dir)
    try:
        run_and_save(settings, None)
    except BioMentorError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError
