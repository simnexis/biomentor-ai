import json

import requests

from biomentor.data.pubmed_client import PubMedClient, parse_pubmed_xml, pubmed_url
from biomentor.exceptions import PubMedError, PubMedParseError
from conftest import PUBMED_XML, FakeResponse, FakeSession


def make_client(*script, **kw):
    sleeps = []
    session = FakeSession(*script)
    client = PubMedClient(email="t@example.com", session=session, sleep=sleeps.append, backoff_s=0.5, **kw)
    return client, session, sleeps


def esearch_json(ids, count=None):
    return json.dumps({"esearchresult": {"count": str(count if count is not None else len(ids)), "idlist": ids}})


# ---- parsing ----
def test_parse_full_record():
    docs = parse_pubmed_xml(PUBMED_XML)
    d = docs[0]
    assert d.document_id == "pmid-99999901" and d.pmid == "99999901"
    assert d.title == "Synthetic title with italic markup"
    assert d.abstract == "BACKGROUND: First part of the abstract. RESULTS: Second part with H2O."
    assert d.authors == ["Tester A", "The Test Consortium"]
    assert d.journal == "Journal of Test Data"
    assert (d.publication_date, d.publication_year) == ("2020-03-05", 2020)
    assert d.doi == "10.0000/test.1"
    assert d.article_type == "Journal Article"
    assert d.source_url == pubmed_url("99999901") == "https://pubmed.ncbi.nlm.nih.gov/99999901/"
    assert d.source_type == "pubmed" and d.retrieved_at


def test_parse_missing_metadata_gives_none_and_skips_no_pmid():
    docs = parse_pubmed_xml(PUBMED_XML)
    assert len(docs) == 2  # the record without a PMID is skipped
    d = docs[1]
    assert d.abstract is None and d.authors is None and d.doi is None and d.journal is None
    assert d.publication_year == 2019 and d.publication_date == "2019"  # from MedlineDate


def test_parse_malformed_xml_raises():
    try:
        parse_pubmed_xml("<not xml")
    except PubMedParseError:
        return
    raise AssertionError("expected PubMedParseError")


# ---- requests ----
def test_search_returns_documents_and_sends_identification():
    client, session, _ = make_client(FakeResponse(200, esearch_json(["99999901", "99999902"])), FakeResponse(200, PUBMED_XML))
    docs = client.search("insulin", max_results=5)
    assert [d.pmid for d in docs] == ["99999901", "99999902"]
    params = session.requests[0][1]
    assert params["tool"] == "biomentor-ai" and params["email"] == "t@example.com" and params["db"] == "pubmed"


def test_transient_errors_are_retried_with_backoff():
    client, session, sleeps = make_client(FakeResponse(503), requests.Timeout(), FakeResponse(200, esearch_json(["1"])))
    assert client.search_ids("x").ids == ["1"]
    assert len(session.requests) == 3 and 0.5 in sleeps and 1.0 in sleeps


def test_gives_up_after_max_retries():
    client, _, _ = make_client(*[FakeResponse(500)] * 4)
    try:
        client.search_ids("x")
    except PubMedError as exc:
        assert "after 4 attempts" in str(exc)
        return
    raise AssertionError


def test_permanent_http_error_not_retried():
    client, session, _ = make_client(FakeResponse(400))
    try:
        client.search_ids("x")
    except PubMedError:
        assert len(session.requests) == 1
        return
    raise AssertionError


def test_ncbi_error_field_raises():
    client, _, _ = make_client(FakeResponse(200, json.dumps({"esearchresult": {"ERROR": "bad query"}})))
    try:
        client.search_ids("x")
    except PubMedError as exc:
        assert "bad query" in str(exc)
        return
    raise AssertionError


def test_empty_query_rejected():
    client, session, _ = make_client()
    try:
        client.search("   ")
    except PubMedError:
        assert not session.requests
        return
    raise AssertionError


def test_rate_limit_spacing_enforced():
    now = [0.0]
    sleeps = []
    session = FakeSession(FakeResponse(200, esearch_json([])), FakeResponse(200, esearch_json([])))
    client = PubMedClient(session=session, sleep=sleeps.append, clock=lambda: now[0], min_interval_s=0.34)
    client.search_ids("a")
    client.search_ids("b")  # clock did not advance -> must wait the full interval
    assert sleeps and abs(sleeps[0] - 0.34) < 1e-9


def test_pagination_and_bounded_results():
    client, session, _ = make_client(
        FakeResponse(200, esearch_json([str(i) for i in range(1, 101)], count=150)),
        FakeResponse(200, esearch_json([str(i) for i in range(101, 111)], count=150)),
        FakeResponse(200, "<PubmedArticleSet/>"),
        FakeResponse(200, "<PubmedArticleSet/>"),
    )
    client.search("x", max_results=110)
    esearch_calls = [p for u, p in session.requests if u.endswith("esearch.fcgi")]
    assert [c["retstart"] for c in esearch_calls] == [0, 100]
    assert esearch_calls[0]["retmax"] == 100 and esearch_calls[1]["retmax"] == 10


def test_results_cached_and_api_key_never_cached(tmp_path):
    session = FakeSession(FakeResponse(200, esearch_json(["5"])))
    client = PubMedClient(api_key="SECRETKEY123", session=session, sleep=lambda s: None, cache_dir=tmp_path)
    client.search_ids("insulin")
    client.search_ids("insulin")  # served from cache
    assert len(session.requests) == 1 and client.cache_hits == 1
    assert session.requests[0][1]["api_key"] == "SECRETKEY123"
    for f in tmp_path.rglob("*.json"):
        assert "SECRETKEY123" not in f.read_text()


def test_corrupt_cache_entry_is_ignored(tmp_path):
    session = FakeSession(FakeResponse(200, esearch_json(["5"])), FakeResponse(200, esearch_json(["5"])))
    client = PubMedClient(session=session, sleep=lambda s: None, cache_dir=tmp_path)
    client.search_ids("q")
    for f in tmp_path.rglob("*.json"):
        f.write_text("{not json")
    assert client.search_ids("q").ids == ["5"]
    assert len(session.requests) == 2


def test_fetch_deduplicates_ids_and_ignores_non_numeric():
    client, session, _ = make_client(FakeResponse(200, PUBMED_XML))
    docs = client.fetch_documents(["99999901", "99999901", "abc", "99999902"])
    assert [d.pmid for d in docs] == ["99999901", "99999902"]
    assert session.requests[0][1]["id"] == "99999901,99999902"


def test_parse_nlm_category_and_history_date():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999903</PMID>
      <Article>
        <ArticleTitle>Nlm Category and History Date Test</ArticleTitle>
        <Abstract>
          <AbstractText NlmCategory="OBJECTIVE">Investigate insulin.</AbstractText>
          <AbstractText NlmCategory="UNLABELLED">Plain text part.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Müller</LastName><Initials>H</Initials></Author>
          <Author><LastName>García</LastName><Initials>M</Initials></Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="pubmed">
          <Year>2021</Year><Month>7</Month><Day>15</Day>
        </PubMedPubDate>
      </History>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""
    docs = parse_pubmed_xml(xml)
    assert len(docs) == 1
    d = docs[0]
    assert d.title == "Nlm Category and History Date Test"
    assert d.abstract == "OBJECTIVE: Investigate insulin. Plain text part."
    assert d.authors == ["Müller H", "García M"]
    assert d.publication_year == 2021
    assert d.publication_date == "2021-07-15"


def test_parse_book_article():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedBookArticle>
    <BookDocument>
      <PMID>99999904</PMID>
      <Book>
        <BookTitle>NCBI Bookshelf Guide to Diabetes</BookTitle>
      </Book>
      <ArticleTitle>Insulin Resistance Mechanisms</ArticleTitle>
      <Abstract>
        <AbstractText>Comprehensive review of beta cell physiology.</AbstractText>
      </Abstract>
      <AuthorList>
        <Author><LastName>Smith</LastName><Initials>J</Initials></Author>
      </AuthorList>
    </BookDocument>
  </PubmedBookArticle>
</PubmedArticleSet>"""
    docs = parse_pubmed_xml(xml)
    assert len(docs) == 1
    d = docs[0]
    assert d.pmid == "99999904"
    assert d.title == "Insulin Resistance Mechanisms"
    assert d.journal == "NCBI Bookshelf Guide to Diabetes"
    assert d.abstract == "Comprehensive review of beta cell physiology."

