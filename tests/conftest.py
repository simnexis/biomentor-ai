"""Deterministic fixtures. Nothing here touches the network, Gemini, or real model downloads."""

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomentor.config import Settings  # noqa: E402
from biomentor.data.chunking import ChunkConfig, chunk_document  # noqa: E402
from biomentor.data.demo_fixtures import demo_documents  # noqa: E402
from biomentor.data.preprocessing import tokenize  # noqa: E402
from biomentor.retrieval.dense_retriever import l2_normalize  # noqa: E402

# Obviously fake identifiers: this is test data, not a real publication.
PUBMED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">99999901</PMID>
      <Article>
        <Journal><JournalIssue><PubDate><Year>2020</Year><Month>Mar</Month><Day>5</Day></PubDate></JournalIssue><Title>Journal of Test Data</Title></Journal>
        <ArticleTitle>Synthetic title with <i>italic</i> markup</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">First part of the abstract.</AbstractText>
          <AbstractText Label="RESULTS">Second part with H<sub>2</sub>O.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Tester</LastName><Initials>A</Initials></Author>
          <Author><CollectiveName>The Test Consortium</CollectiveName></Author>
        </AuthorList>
        <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="pubmed">99999901</ArticleId><ArticleId IdType="doi">10.0000/test.1</ArticleId></ArticleIdList></PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">99999902</PMID>
      <Article>
        <Journal><JournalIssue><PubDate><MedlineDate>2019 Dec-2020 Jan</MedlineDate></PubDate></JournalIssue></Journal>
        <ArticleTitle>Record with missing metadata</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation><Article><ArticleTitle>No PMID so it is skipped</ArticleTitle></Article></MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class FakeEmbedder:
    """Hashing bag-of-words embedder: deterministic, shares vocabulary signal, no model download."""

    model_name = "fake-hash-embedder"
    query_prefix = ""
    passage_prefix = ""
    dimension = 128

    def _enc(self, texts):
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, zlib.crc32(tok.encode()) % self.dimension] += 1.0
        return l2_normalize(out) if len(texts) else out

    def encode_passages(self, texts):
        return self._enc(list(texts))

    def encode_queries(self, texts):
        return self._enc(list(texts))


class FakeBackend:
    """Scripted Gemini backend: each call pops the next item; exceptions are raised."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_json(self, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeResponse:
    def __init__(self, status=200, text="", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}


class FakeSession:
    """Stands in for requests.Session; ``script`` items are FakeResponse or exceptions."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, dict(params or {})))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def docs():
    return demo_documents()


@pytest.fixture
def passages(docs):
    cfg = ChunkConfig(800, 1)
    return [p for d in docs for p in chunk_document(d, cfg)]


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")
