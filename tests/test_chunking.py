from biomentor.data.chunking import (
    ChunkConfig,
    chunk_document,
    chunk_text_to_passages,
    sentence_spans,
)
from biomentor.data.preprocessing import normalize_whitespace
from biomentor.schemas import Document


def make_doc(abstract, doc_id="d1", **kw):
    return Document(document_id=doc_id, title="A title", abstract=abstract, source_name="s",
                    source_type="synthetic_fixture", pmid=kw.get("pmid"), doi=kw.get("doi"),
                    journal="J", publication_year=2021, source_url="u")


LONG = " ".join(f"Sentence number {i} describes the behaviour of protein {i} in cells." for i in range(30))


def test_empty_and_none_and_whitespace_yield_no_passages():
    assert chunk_document(make_doc(None)) == []
    assert chunk_document(make_doc("")) == []
    assert chunk_document(make_doc("   \n  ")) == []


def test_short_text_is_one_passage():
    ps = chunk_document(make_doc("One short sentence."))
    assert len(ps) == 1 and ps[0].text == "One short sentence."


def test_offsets_reproduce_text_exactly():
    doc = make_doc(LONG)
    norm = normalize_whitespace(doc.abstract)
    for p in chunk_document(doc, ChunkConfig(300, 1)):
        assert norm[p.start_char:p.end_char] == p.text


def test_chunks_respect_max_chars_and_cover_everything():
    doc = make_doc(LONG)
    ps = chunk_document(doc, ChunkConfig(300, 1))
    assert len(ps) > 3 and all(len(p.text) <= 300 for p in ps)
    norm = normalize_whitespace(LONG)
    assert ps[0].start_char == 0 and ps[-1].end_char == len(norm)
    for a, b in zip(ps, ps[1:], strict=False):
        assert b.start_char <= a.end_char  # no gap


def test_overlap_repeats_boundary_sentence():
    ps = chunk_document(make_doc(LONG), ChunkConfig(300, 1))
    assert ps[0].end_char > ps[1].start_char  # overlap present
    none = chunk_document(make_doc(LONG), ChunkConfig(300, 0))
    assert none[0].end_char <= none[1].start_char + 1  # no overlap


def test_progress_guaranteed_when_overlap_large():
    ps = chunk_document(make_doc(LONG), ChunkConfig(300, 5))
    assert len(ps) > 1 and len({p.start_char for p in ps}) == len(ps)


def test_abbreviations_do_not_split_sentences():
    spans = sentence_spans("Smith et al. showed IL-6 rises (see Fig. 2). Levels fell afterwards.")
    assert len(spans) == 2


def test_very_long_sentence_is_split_on_words():
    text = "word " * 400
    ps = chunk_document(make_doc(text.strip() + "."), ChunkConfig(200, 0))
    assert len(ps) > 1 and all(len(p.text) <= 200 for p in ps)


def test_metadata_preserved_on_passages():
    ps = chunk_document(make_doc("Text here.", pmid="123", doi="10.1/x"))
    p = ps[0]
    assert (p.document_id, p.pmid, p.doi, p.journal, p.publication_year, p.source_url) == ("d1", "123", "10.1/x", "J", 2021, "u")
    assert p.title == "A title" and p.source_type == "synthetic_fixture"


def test_stable_ids_across_runs_and_unique():
    a = chunk_document(make_doc(LONG), ChunkConfig(300, 1))
    b = chunk_document(make_doc(LONG), ChunkConfig(300, 1))
    assert [p.passage_id for p in a] == [p.passage_id for p in b]
    assert len({p.passage_id for p in a}) == len(a)
    assert a[0].passage_id == "d1-p000"


def test_duplicate_chunks_are_removed():
    doc = make_doc("Same sentence repeated here. " * 1)
    ps = chunk_text_to_passages(doc, "Alpha beta gamma delta.\n\nAlpha beta gamma delta.", ChunkConfig(100, 0), keep_paragraphs=True)
    assert len(ps) == 1


def test_unicode_text_chunks_cleanly():
    ps = chunk_document(make_doc("Le récepteur β2 – activé par l’agoniste – détend le muscle. Ça marche très bien."))
    assert ps and "β2" in ps[0].text


def test_invalid_config_rejected():
    for bad in (dict(max_chars=10), dict(overlap_sentences=-1)):
        try:
            ChunkConfig(**bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
