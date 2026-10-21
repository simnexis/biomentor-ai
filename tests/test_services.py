"""Storage, index manager, search service, learning service, PDF handling, comparison."""

import json

from biomentor.data.demo_fixtures import demo_documents
from biomentor.data.ingestion import append_raw_documents, corpus_kind_of, process_corpus
from biomentor.data.pubmed_client import PubMedClient
from biomentor.data.storage import deduplicate_documents, iter_jsonl, load_documents, save_documents
from biomentor.exceptions import (
    BioMentorError,
    IndexNotFoundError,
    InvalidPDFError,
    NoExtractableTextError,
    PubMedError,
)
from biomentor.generation.gemini_client import GeminiClient
from biomentor.retrieval.index_manager import IndexManager
from biomentor.schemas import (
    Claim,
    ComparisonOutput,
    Document,
    ExplanationOutput,
    FieldValue,
    PaperComparison,
)
from biomentor.services import learning_service as ls
from biomentor.services import paper_service as ps
from biomentor.services.comparison_service import _numbers_supported, compare_papers
from biomentor.services.research_service import SearchService, pubmed_query_variants
from conftest import PUBMED_XML, FakeBackend, FakeEmbedder, FakeResponse, FakeSession


def prepare(settings):
    save_documents(settings.raw_dir / "demo.jsonl", demo_documents())
    return process_corpus([settings.raw_dir / "demo.jsonl"], settings.processed_dir)


# ---- storage ----
def test_dedup_by_pmid_doi_and_title_keeps_longer_abstract():
    def d(i, **kw):
        base = dict(document_id=i, title="T", source_name="s", source_type="pubmed")
        base.update(kw)
        return Document(**base)

    docs = [d("a", pmid="1", abstract="short", title="T1"), d("b", pmid="1", abstract="a much longer abstract", title="T2"),
            d("c", doi="10.1/X", title="T3"), d("e", doi="10.1/x", title="T4"),
            d("f", title="Same Title!", publication_year=2000), d("g", title="same title", publication_year=2000)]
    unique, removed = deduplicate_documents(docs)
    assert removed == 3 and len(unique) == 3
    assert next(u for u in unique if u.pmid == "1").abstract == "a much longer abstract"


def test_jsonl_roundtrip_skips_invalid_and_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    save_documents(path, demo_documents()[:2])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps({"document_id": "bad"}) + "\n")  # missing required fields
    assert len(load_documents(path)) == 2
    assert load_documents(tmp_path / "missing.jsonl") == []
    path.write_text("{oops\n")
    try:
        list(iter_jsonl(path))
    except ValueError as exc:
        assert "x.jsonl:1" in str(exc)
    else:
        raise AssertionError


def test_append_raw_documents_merges_without_duplicates(tmp_path):
    p = tmp_path / "raw.jsonl"
    assert append_raw_documents(p, demo_documents()[:3]) == 3
    assert append_raw_documents(p, demo_documents()[:5]) == 2


def test_empty_dataset_processing(settings):
    save_documents(settings.raw_dir / "empty.jsonl", [])
    docs, passages, stats = process_corpus([settings.raw_dir / "empty.jsonl"], settings.processed_dir)
    assert docs == [] and passages == [] and stats.corpus_kind == "empty" and corpus_kind_of([]) == "empty"


def test_documents_without_abstract_yield_no_passages(settings):
    d = Document(document_id="pmid-1", title="Only a title", source_name="PubMed", source_type="pubmed", pmid="1")
    save_documents(settings.raw_dir / "a.jsonl", [d])
    docs, passages, stats = process_corpus([settings.raw_dir / "a.jsonl"], settings.processed_dir)
    assert len(docs) == 1 and passages == [] and stats.documents_without_abstract == 1


# ---- index manager ----
def test_index_manager_build_load_status_clear(settings, embedder):
    docs, passages, stats = prepare(settings)
    mgr = IndexManager(settings)
    manifest = mgr.build(passages, len(docs), stats.corpus_kind, embedder=embedder)
    assert manifest["dense"]["built"] and manifest["corpus_kind"] == "demo_synthetic" and manifest["n_passages"] == len(passages)
    idx = mgr.load(embedder=embedder)
    assert idx.search("malaria liver", "hybrid", 3)[0].passage.document_id == "syn-mal-01"
    assert mgr.status()["has_manifest"]
    mgr.clear()
    assert not mgr.manifest_path.exists() and mgr.passages_path.exists()  # corpus kept, index removed


def test_index_load_without_corpus_raises(settings):
    try:
        IndexManager(settings).load()
    except IndexNotFoundError as exc:
        assert "prepare_demo_data" in str(exc)
    else:
        raise AssertionError


def test_stale_dense_index_is_detected_not_used(settings, embedder):
    docs, passages, stats = prepare(settings)
    mgr = IndexManager(settings)
    mgr.build(passages, len(docs), stats.corpus_kind, embedder=embedder)
    save_documents(settings.raw_dir / "more.jsonl", [Document(document_id="x-1", title="New", abstract="Brand new abstract about kidneys.", source_name="s", source_type="synthetic_fixture")])
    process_corpus([settings.raw_dir / "demo.jsonl", settings.raw_dir / "more.jsonl"], settings.processed_dir)  # corpus changed, index not rebuilt
    idx = mgr.load(embedder=embedder)
    assert not idx.dense_available() and "incompatible" in idx.dense_error.lower()
    assert idx.search("kidneys", "bm25", 3)  # keyword search is unaffected


# ---- search service ----
def test_local_search_labels_source_and_falls_back(settings):
    prepare(settings)
    svc = SearchService(settings, embedder=None)
    resp = svc.search("malaria parasite", mode="hybrid", top_k=3)
    assert resp.mode_requested == "hybrid" and resp.mode_used == "bm25"
    assert resp.source_kind == "demo_fixtures" and "SYNTHETIC" in resp.source_label
    assert resp.warnings and "Falling back" in resp.warnings[0]
    assert resp.documents["syn-mal-01"].title.startswith("[Synthetic fixture]")


def test_search_service_reloads_when_corpus_changes(settings):
    prepare(settings)
    svc = SearchService(settings)
    n1 = len(svc.local_index().passages)
    save_documents(settings.raw_dir / "more.jsonl", [Document(document_id="x-1", title="New", abstract="Kidney filtration happens in the glomerulus.", source_name="s", source_type="synthetic_fixture")])
    process_corpus([settings.raw_dir / "demo.jsonl", settings.raw_dir / "more.jsonl"], settings.processed_dir)
    import os
    import time
    os.utime(settings.processed_dir / "passages.jsonl", (time.time() + 5, time.time() + 5))
    assert len(svc.local_index().passages) == n1 + 1


def test_live_search_uses_pubmed_client_and_labels_source(settings):
    session = FakeSession(FakeResponse(200, json.dumps({"esearchresult": {"count": "2", "idlist": ["99999901", "99999902"]}})), FakeResponse(200, PUBMED_XML))
    client = PubMedClient(session=session, sleep=lambda s: None)
    svc = SearchService(settings, pubmed_client=client)
    resp = svc.search("synthetic abstract", mode="bm25", top_k=5, source="live")
    assert resp.source_kind == "live_pubmed" and resp.results and resp.results[0].passage.pmid == "99999901"
    assert resp.documents["pmid-99999901"].doi == "10.0000/test.1" and resp.pubmed_query_used


def test_live_search_failure_is_a_friendly_error(settings):
    client = PubMedClient(session=FakeSession(*[FakeResponse(503)] * 4), sleep=lambda s: None)
    try:
        SearchService(settings, pubmed_client=client).search("insulin", source="live")
    except PubMedError as exc:
        assert "Could not reach NCBI" in str(exc)
    else:
        raise AssertionError


def test_pubmed_query_variants():
    v = pubmed_query_variants("How does insulin resistance lead to type 2 diabetes?")
    assert v[0].startswith("How does") and " OR " in v[-1] and "insulin" in v[1]


def test_rebuild_requires_raw_data(settings):
    try:
        SearchService(settings).rebuild_local_index(build_dense=False)
    except IndexNotFoundError:
        pass
    else:
        raise AssertionError


# ---- learning service (explain_topic) ----
def explain_payload(**over):
    base = dict(
        short_definition=Claim(text="Malaria is caused by Plasmodium parasites", basis="evidence", citation_ids=["E1", "E99"]),
        explanation=[Claim(text="Parasites invade liver cells then red blood cells", basis="evidence", citation_ids=["E1"])],
        mechanism_steps=[], key_terms=[], key_findings=[], limitations=[], follow_up_questions=["What is a vector?"],
        insufficient_evidence=False, insufficient_evidence_note="")
    base.update(over)
    return ExplanationOutput(**base).model_dump_json()


def test_explain_topic_full_pipeline(settings):
    prepare(settings)
    svc = SearchService(settings)
    gem = GeminiClient("k", "m", backend=FakeBackend(explain_payload()), sleep=lambda s: None)
    res = ls.explain_topic(svc, gem, settings, "malaria parasite liver red blood cells", "beginner", mode="bm25")
    assert res.status == ls.OK and res.report.n_invalid_ids == 1  # E99 removed
    assert res.output.short_definition.citation_ids == ["E1"] and res.model == "m"
    assert res.evidence.get("E1") and res.search.source_kind == "demo_fixtures"


def test_explain_topic_without_gemini_still_returns_sources(settings):
    prepare(settings)
    res = ls.explain_topic(SearchService(settings), GeminiClient(None, "m"), settings, "asthma airway inflammation", mode="bm25")
    assert res.status == ls.NO_GEMINI and len(res.evidence) > 0 and res.output is None


def test_explain_topic_abstains_when_evidence_does_not_cover_query(settings):
    prepare(settings)
    backend = FakeBackend()
    res = ls.explain_topic(SearchService(settings), GeminiClient("k", "m", backend=backend), settings, "leukemia chemotherapy resistance", mode="bm25")
    assert res.status == ls.ABSTAINED and backend.calls == 0 and "enough evidence" in res.message


def test_explain_topic_safety_and_invalid_queries_never_call_gemini(settings):
    prepare(settings)
    backend = FakeBackend()
    gem = GeminiClient("k", "m", backend=backend)
    assert ls.explain_topic(SearchService(settings), gem, settings, "Do I have diabetes?").status == ls.REFUSED_SAFETY
    assert ls.explain_topic(SearchService(settings), gem, settings, "   ").status == ls.INVALID_QUERY
    assert backend.calls == 0


def test_explain_topic_gemini_errors_are_reported_with_evidence_kept(settings):
    from biomentor.exceptions import GeminiAuthError

    prepare(settings)
    gem = GeminiClient("k", "m", backend=FakeBackend(GeminiAuthError("Gemini rejected the API key.")))
    res = ls.explain_topic(SearchService(settings), gem, settings, "malaria parasite", mode="bm25")
    assert res.status == ls.ERROR and "API key" in res.message and len(res.evidence) > 0


def test_explain_topic_no_corpus_is_error_not_crash(settings):
    res = ls.explain_topic(SearchService(settings), GeminiClient(None, "m"), settings, "malaria", mode="bm25")
    assert res.status == ls.ERROR and "No local corpus" in res.message


# ---- PDF handling ----
def test_pdf_validation_rejects_bad_uploads():
    cases = [(None, "a.pdf"), (b"", "a.pdf"), (b"%PDF-1.4 data", "a.txt"), (b"plain text, not a pdf", "a.pdf"), (b"%PDF-" + b"0" * (2 * 1024 * 1024), "big.pdf")]
    for data, name in cases[:4]:
        try:
            ps.validate_pdf_bytes(data, name)
        except InvalidPDFError:
            continue
        raise AssertionError((data, name))
    try:
        ps.validate_pdf_bytes(cases[4][0], cases[4][1], max_mb=1)
    except InvalidPDFError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError
    ps.validate_pdf_bytes(b"%PDF-1.4 ...", "ok.PDF")  # does not raise


def test_extract_pdf_rejects_garbage_even_with_pdf_header():
    try:
        ps.extract_pdf(b"%PDF-1.4\nthis is not really a pdf", "x.pdf")
    except InvalidPDFError:
        return  # includes the 'PyMuPDF missing' case, which is also an InvalidPDFError
    raise AssertionError


def test_extract_pdf_real_document_and_scanned_detection():
    try:
        import pymupdf
    except ImportError:
        return  # PyMuPDF not installed in this environment; covered when it is
    doc = pymupdf.open()
    page = doc.new_page()
    body = "Abstract\n" + "Insulin resistance is a condition in which cells respond poorly to insulin. " * 6 + "\n\nIntroduction\nMore text follows here."
    page.insert_textbox(pymupdf.Rect(40, 40, 550, 780), body, fontsize=10)
    parsed = ps.extract_pdf(doc.tobytes(), "paper.pdf")
    assert parsed.n_pages == 1 and "Insulin resistance" in parsed.text and parsed.abstract and parsed.content_hash
    blank = pymupdf.open()
    blank.new_page()
    try:
        ps.extract_pdf(blank.tobytes(), "scan.pdf")
    except NoExtractableTextError as exc:
        assert "OCR" in str(exc)
    else:
        raise AssertionError


def test_detect_abstract():
    text = "A Title\n\nAbstract\n" + "Background text about the study design and findings. " * 8 + "\n\nKeywords: a, b\n\nIntroduction\nbody"
    abstract = ps.detect_abstract(text)
    assert abstract and abstract.startswith("Background") and "Keywords" not in abstract
    assert ps.detect_abstract("No such section here. " * 40) is None


def test_abstract_found_when_extractor_puts_heading_on_its_own_line():
    # Regression: PyMuPDF-style text ("Abstract" alone on a line) was flattened before detection.
    from biomentor.data.preprocessing import clean_pdf_lines

    raw = "Insulin resistance study\n\nAbstract\n" + "Insulin resistance is a condition in which cells respond poorly. \n" * 6 + "\nIntroduction\nMore text."
    abstract = ps.detect_abstract(clean_pdf_lines(raw))
    assert abstract and abstract.startswith("Insulin resistance is") and "More text" not in abstract
    assert [ln for ln in clean_pdf_lines(raw).splitlines() if 10 <= len(ln) <= 250][0] == "Insulin resistance study"


def test_select_context_passages_budget_and_coverage(passages):
    chosen, truncated = ps.select_context_passages(passages, 100000)
    assert not truncated and len(chosen) == len(passages)
    chosen, truncated = ps.select_context_passages(passages, 1500)
    assert truncated and sum(len(p.text) for p in chosen) <= 1500 and chosen == sorted(chosen, key=lambda p: passages.index(p))


def test_paper_pipeline_requires_consent_and_keeps_sources_separate(settings):
    parsed = ps.ParsedPaper("p.pdf", 1, "Abstract\n" + "Cells respond poorly to insulin in this uploaded study. " * 10, "abc123def456", "Uploaded study", None, 1, False)
    ctx = ps.prepare_paper(parsed, settings, FakeEmbedder())
    assert ctx.document.source_type == "uploaded_pdf" and ctx.passages and all(p.source_type == "uploaded_pdf" for p in ctx.passages)
    backend = FakeBackend()
    gem = GeminiClient("k", "m", backend=backend)
    try:
        ps.summarize_paper(ctx, gem, settings, "beginner", consent=False)
    except BioMentorError as exc:
        assert "agree" in str(exc) and backend.calls == 0  # nothing sent without consent
    else:
        raise AssertionError
    from biomentor.schemas import AnswerOutput

    payload = AnswerOutput(answer=[Claim(text="Cells respond poorly to insulin", basis="evidence", citation_ids=["E1", "E2"])], key_terms=[], limitations=[],
                           follow_up_questions=[], insufficient_evidence=False, insufficient_evidence_note="").model_dump_json()
    gem = GeminiClient("k", "m", backend=FakeBackend(payload))
    external = SearchService(settings)
    prepare(settings)
    ext = external.search("insulin resistance beta cells", mode="bm25", top_k=2).results
    res = ps.ask_paper(ctx, gem, "beginner", "What does the paper say about insulin?", True, external=ext, mode="bm25")
    origins = {i.origin for i in res.evidence}
    assert origins == {"uploaded", "external"}


# ---- comparison ----
def test_numbers_supported_check():
    assert _numbers_supported("n = 1,200 participants", "We enrolled 1200 participants.") == []
    assert _numbers_supported("500 patients", "We enrolled 1200 participants.") == ["500"]


def fv(value, ids):
    return FieldValue(value=value, citation_ids=ids)


def paper_cmp(doc_id, ids, sample="Not reported in the available text."):
    nr = fv("Not reported in the available text.", [])
    return PaperComparison(document_id=doc_id, research_question=fv("How cells respond to insulin", ids), study_type=nr, population_or_model=nr,
                           sample_size=fv(sample, ids if sample[0].isdigit() else []), intervention_or_exposure=nr, comparator=nr, outcomes=nr,
                           findings=fv("Reports reduced glucose uptake", ids), limitations=nr)


def test_comparison_pipeline_checks_numbers_and_ownership(settings, passages):
    a, b = [d for d in demo_documents() if d.document_id in ("syn-t2d-01", "syn-t2d-02")]
    from biomentor.data.chunking import chunk_document

    pbd = {a.document_id: chunk_document(a), b.document_id: chunk_document(b)}
    out = ComparisonOutput(
        papers=[paper_cmp(a.document_id, ["E1"], sample="5000 people"), paper_cmp(b.document_id, ["E1"])],  # b cites a's passage (E1)
        agreements=[Claim(text="Both discuss glucose", basis="evidence", citation_ids=["E1", "E2"])], disagreements=[], caveats=["abstracts only"])
    gem = GeminiClient("k", "m", backend=FakeBackend(out.model_dump_json()))
    res = compare_papers(gem, settings, [a, b], pbd)
    row_a, row_b = res.table
    assert row_a["Sample size"][0] == "Not reported in the available text." and res.unverified_numbers  # 5000 is not in the text
    assert row_a["Research question"][1] == ["E1"]
    assert row_b["Research question"][0] == "Not reported in the available text."  # cited another paper's passage -> rejected
    assert row_a["Publication date"] == "Not reported in the available text."  # fixtures have no date; never invented


def test_comparison_needs_two_papers(settings, passages):
    a = demo_documents()[0]
    gem = GeminiClient("k", "m", backend=FakeBackend())
    for docs in ([a], [a, demo_documents()[1]]):
        try:
            compare_papers(gem, settings, docs, {})
        except BioMentorError:
            continue
        raise AssertionError
