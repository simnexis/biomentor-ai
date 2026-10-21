"""Client for NCBI PubMed E-utilities (https://www.ncbi.nlm.nih.gov/books/NBK25501/).

* ``esearch`` turns a query into PubMed IDs; ``efetch`` returns title/abstract/metadata as XML.
* Etiquette: requests are spaced to <= 3/second (10/second with an API key), the ``tool`` and
  ``email`` parameters identify this application, and bounded result counts are enforced.
* Responses are cached on disk (never including the API key). Transient failures (HTTP 429/5xx,
  timeouts, connection errors) are retried with back-off; other HTTP errors fail immediately.
* PubMed supplies metadata and (usually) the abstract - *not* the full text. Abstract copyright
  stays with the publisher; this client stores records locally for research/education use only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element

import requests

from biomentor.data.preprocessing import normalize_whitespace
from biomentor.exceptions import PubMedError, PubMedParseError
from biomentor.schemas import SOURCE_PUBMED, Document

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
MAX_RESULTS_HARD_LIMIT = 500
ESEARCH_PAGE_SIZE = 100
EFETCH_BATCH_SIZE = 100
MAX_QUERY_CHARS = 400
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


@dataclass
class SearchResult:
    ids: list[str]
    total_count: int


def pubmed_url(pmid: str) -> str:
    return PUBMED_URL.format(pmid=pmid)


def _text(element: Element | None) -> str:
    """Text of an element including inline markup (<i>, <sup>, ...), whitespace-normalised."""
    if element is None:
        return ""
    return normalize_whitespace("".join(element.itertext()))


def _parse_pub_date(article: Element, node: Element | None = None) -> tuple[str | None, int | None]:
    """Return (``YYYY[-MM[-DD]]`` or None, year or None)."""
    pub_date = article.find("./Journal/JournalIssue/PubDate")
    year = month = day = None
    if pub_date is not None:
        year = _text(pub_date.find("Year")) or None
        month = _text(pub_date.find("Month")) or None
        day = _text(pub_date.find("Day")) or None
        if not year:
            medline = _text(pub_date.find("MedlineDate"))
            m = re.search(r"(19|20)\d{2}", medline)
            year = m.group(0) if m else None
            month = day = None
    if not year:
        art_date = article.find("./ArticleDate")
        if art_date is not None:
            year = _text(art_date.find("Year")) or None
            month = _text(art_date.find("Month")) or None
            day = _text(art_date.find("Day")) or None
    if not year and node is not None:
        for status in ("pubmed", "medline", "entrez"):
            p_date = node.find(f"./PubmedData/History/PubMedPubDate[@PubStatus='{status}']")
            if p_date is not None:
                year = _text(p_date.find("Year")) or None
                month = _text(p_date.find("Month")) or None
                day = _text(p_date.find("Day")) or None
                if year:
                    break
        if not year:
            date_completed = node.find("./MedlineCitation/DateCompleted")
            if date_completed is not None:
                year = _text(date_completed.find("Year")) or None
                month = _text(date_completed.find("Month")) or None
                day = _text(date_completed.find("Day")) or None
    if not year or not re.fullmatch(r"\d{4}", year):
        return None, None
    parts = [year]
    if month:
        mnum = int(month) if month.isdigit() else _MONTHS.get(month[:3].lower())
        if mnum and 1 <= mnum <= 12:
            parts.append(f"{mnum:02d}")
            if day and day.isdigit() and 1 <= int(day) <= 31:
                parts.append(f"{int(day):02d}")
    return "-".join(parts), int(year)


def _parse_article(node: Element, retrieved_at: str) -> Document | None:
    citation = node.find("MedlineCitation")
    if citation is None:
        citation = node.find("BookDocument")
    if citation is None:
        return None
    pmid = _text(citation.find("PMID"))
    if not pmid:
        return None
    article = citation.find("Article")
    if article is None:
        article = citation.find("Book")
    if article is None:
        article = citation

    title = _text(citation.find("ArticleTitle")) or _text(article.find("ArticleTitle")) or _text(article.find("BookTitle"))

    abstract_nodes = article.findall("./Abstract/AbstractText")
    if not abstract_nodes and citation is not article:
        abstract_nodes = citation.findall("./Abstract/AbstractText")
    abstract_parts = []
    for part in abstract_nodes:
        text = _text(part)
        if not text:
            continue
        label = part.attrib.get("Label") or part.attrib.get("NlmCategory")
        if label and label.strip().upper() not in {"UNLABELLED", "UNASSIGNED"}:
            abstract_parts.append(f"{label.strip().upper()}: {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts) or _text(article.find("Abstract")) or (
        _text(citation.find("Abstract")) if citation is not article else ""
    ) or None

    authors: list[str] = []
    author_list = article.findall("./AuthorList/Author")
    if not author_list and citation is not article:
        author_list = citation.findall("./AuthorList/Author")
    for author in author_list:
        collective = _text(author.find("CollectiveName"))
        last, initials = _text(author.find("LastName")), _text(author.find("Initials"))
        name = collective or " ".join(p for p in (last, initials) if p)
        if name:
            authors.append(name)

    date_str, year = _parse_pub_date(article, node)

    doi = None
    for aid in node.findall("./PubmedData/ArticleIdList/ArticleId"):
        if aid.attrib.get("IdType") == "doi" and _text(aid):
            doi = _text(aid)
            break
    if doi is None:
        for eloc in article.findall("ELocationID"):
            if eloc.attrib.get("EIdType") == "doi" and _text(eloc):
                doi = _text(eloc)
                break

    pub_types = [_text(p) for p in article.findall("./PublicationTypeList/PublicationType") if _text(p)]

    journal = (
        _text(article.find("./Journal/Title"))
        or _text(citation.find("./Book/BookTitle"))
        or _text(article.find("./BookTitle"))
        or None
    )

    return Document(
        document_id=f"pmid-{pmid}",
        title=title or f"PMID {pmid}",
        abstract=abstract,
        source_name="PubMed",
        source_type=SOURCE_PUBMED,
        pmid=pmid,
        doi=doi,
        authors=authors or None,
        journal=journal,
        publication_date=date_str,
        publication_year=year,
        article_type="; ".join(pub_types) or None,
        source_url=pubmed_url(pmid),
        license=None,  # PubMed metadata/abstract; reuse terms are set by each publisher
        retrieved_at=retrieved_at,
    )


def parse_pubmed_xml(xml_text: str | bytes, retrieved_at: str | None = None) -> list[Document]:
    """Parse an ``efetch`` PubMed XML payload. Missing fields become ``None``; records without a
    PMID are skipped. Raises :class:`PubMedParseError` on malformed XML."""
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(xml_text, str):
        xml_bytes = xml_text.encode("utf-8")
    else:
        xml_bytes = xml_text
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise PubMedParseError(f"Could not parse the PubMed response: {exc}") from exc
    documents = []
    for node in root.iter("PubmedArticle"):
        doc = _parse_article(node, retrieved_at)
        if doc is not None:
            documents.append(doc)
    for node in root.iter("PubmedBookArticle"):
        doc = _parse_article(node, retrieved_at)
        if doc is not None:
            documents.append(doc)
    return documents


class PubMedClient:
    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        tool: str = "biomentor-ai",
        cache_dir: Path | str | None = None,
        session: requests.Session | None = None,
        timeout: float = 20.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        min_interval_s: float | None = None,
        cache_ttl_hours: float = 24 * 7,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.email = email
        self._api_key = api_key
        self.tool = tool
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.min_interval_s = min_interval_s if min_interval_s is not None else (0.11 if api_key else 0.34)
        self.cache_ttl_s = cache_ttl_hours * 3600
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.network_requests = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ low level
    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        blob = json.dumps([endpoint, sorted(params.items())], sort_keys=True, default=str)
        return self.cache_dir / "pubmed" / f"{hashlib.sha256(blob.encode()).hexdigest()}.json"

    def _read_cache(self, path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - payload["fetched_at_epoch"] > self.cache_ttl_s:
                return None
            return payload["text"]
        except (OSError, ValueError, KeyError):
            return None  # corrupt cache entry -> refetch

    def _write_cache(self, path: Path | None, endpoint: str, params: dict[str, Any], text: str) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"endpoint": endpoint, "params": params, "fetched_at_epoch": time.time(), "text": text}),
                encoding="utf-8",
            )
        except OSError as exc:  # caching is best-effort
            logger.warning("Could not write PubMed cache: %s", exc)

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval_s - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _request(self, endpoint: str, params: dict[str, Any], use_cache: bool = True) -> str:
        base_params = {"db": "pubmed", "tool": self.tool, **params}
        if self.email:
            base_params["email"] = self.email
        cache_path = self._cache_path(endpoint, base_params)  # computed *before* adding the key
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                self.cache_hits += 1
                return cached
        request_params = dict(base_params)
        if self._api_key:
            request_params["api_key"] = self._api_key

        url = EUTILS_BASE + endpoint
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.network_requests += 1
            try:
                response = self.session.get(url, params=request_params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}"
            else:
                if response.status_code == 200:
                    self._write_cache(cache_path, endpoint, base_params, response.text)
                    return response.text
                if response.status_code not in _RETRY_STATUS:
                    raise PubMedError(f"NCBI returned HTTP {response.status_code} for {endpoint}.")
                last_error = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._sleep(min(int(retry_after), 30))
            if attempt < self.max_retries:
                self._sleep(self.backoff_s * (2**attempt))
        raise PubMedError(
            f"Could not reach NCBI PubMed after {self.max_retries + 1} attempts ({last_error}). "
            "Check your internet connection and try again."
        )

    # ------------------------------------------------------------------ public API
    @staticmethod
    def clean_query(term: str) -> str:
        term = normalize_whitespace(term)
        if not term:
            raise PubMedError("The search query is empty.")
        return term[:MAX_QUERY_CHARS]

    def search_ids(self, term: str, retmax: int = 20, retstart: int = 0, sort: str = "relevance") -> SearchResult:
        retmax = max(1, min(int(retmax), ESEARCH_PAGE_SIZE))
        text = self._request(
            "esearch.fcgi",
            {"term": self.clean_query(term), "retmax": retmax, "retstart": max(0, int(retstart)),
             "retmode": "json", "sort": sort},
        )
        try:
            result = json.loads(text)["esearchresult"]
        except (ValueError, KeyError) as exc:
            raise PubMedParseError("Unexpected esearch response from NCBI.") from exc
        if result.get("ERROR"):
            raise PubMedError(f"NCBI rejected the query: {result['ERROR']}")
        ids = [str(i) for i in result.get("idlist", []) if str(i).isdigit()]
        try:
            count = int(result.get("count", len(ids)))
        except (TypeError, ValueError):
            count = len(ids)
        return SearchResult(ids=ids, total_count=count)

    def fetch_documents(self, pmids: list[str]) -> list[Document]:
        unique = [p for p in dict.fromkeys(str(p).strip() for p in pmids) if p.isdigit()]
        documents: dict[str, Document] = {}
        for i in range(0, len(unique), EFETCH_BATCH_SIZE):
            batch = unique[i : i + EFETCH_BATCH_SIZE]
            xml = self._request("efetch.fcgi", {"id": ",".join(batch), "retmode": "xml", "rettype": "abstract"})
            for doc in parse_pubmed_xml(xml):
                documents.setdefault(doc.document_id, doc)
        # keep the order PubMed ranked them in
        return [documents[f"pmid-{p}"] for p in unique if f"pmid-{p}" in documents]

    def search(self, term: str, max_results: int = 20) -> list[Document]:
        """Search PubMed and return up to ``max_results`` (<= 500) de-duplicated documents."""
        max_results = max(1, min(int(max_results), MAX_RESULTS_HARD_LIMIT))
        ids: list[str] = []
        start = 0
        while len(ids) < max_results:
            page = self.search_ids(term, retmax=min(ESEARCH_PAGE_SIZE, max_results - len(ids)), retstart=start)
            if not page.ids:
                break
            ids.extend(i for i in page.ids if i not in ids)
            start += len(page.ids)
            if start >= page.total_count:
                break
        return self.fetch_documents(ids[:max_results])
