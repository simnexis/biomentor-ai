"""Pure HTML-formatting helpers (no Streamlit import, so they are unit-testable)."""

from __future__ import annotations

import html
import re

from biomentor.data.preprocessing import STOPWORDS, normalize_whitespace, stem, tokenize

SOURCE_LABELS = {
    "pubmed": ("PubMed record", "src-pubmed"),
    "synthetic_fixture": ("Synthetic demo text", "src-synth"),
    "uploaded_pdf": ("Your uploaded paper", "src-upload"),
}


def esc(text: object) -> str:
    return html.escape(str(text if text is not None else ""))


def highlight_html(text: str, query: str) -> str:
    """HTML-escape ``text`` and wrap words that match the query (by crude stem) in <mark>."""
    safe = html.escape(normalize_whitespace(text))
    words = {stem(t) for t in tokenize(query) if "-" not in t and len(t) >= 3 and t not in STOPWORDS}
    # Match on a shared prefix so "infection" also marks "infect", "infected", "infections".
    words = {(w if len(w) <= 5 else w[: max(5, len(w) - 3)]) for w in words if len(w) >= 3}
    if not words:
        return safe
    pattern = re.compile(r"(?<![\w])(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")(\w*)", re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(1)}{m.group(2)}</mark>", safe)


def chip(label: str, kind: str = "cite") -> str:
    return f'<span class="chip chip-{esc(kind)}">{esc(label)}</span>'


def cite_chips(ids: list[str]) -> str:
    return "".join(f'<sup class="cite">{esc(i)}</sup>' for i in ids)


def source_chip(source_type: str) -> str:
    label, css = SOURCE_LABELS.get(source_type, (source_type or "Source", "src-other"))
    return f'<span class="chip {css}">{esc(label)}</span>'


def claim_html(text: str, citation_ids: list[str] | None = None, basis: str = "evidence") -> str:
    tag = {"background": ' <span class="basis">general background</span>', "analogy": ' <span class="basis">analogy</span>'}.get(basis, "")
    return f"{esc(text)}{tag}{cite_chips(citation_ids or [])}"


def term_html(term: str, definition: str, citation_ids: list[str] | None = None) -> str:
    return f"<b>{esc(term)}</b> {esc(definition)}{cite_chips(citation_ids or [])}"


def meta_html(authors: list[str] | None, year: int | None, journal: str | None, pmid: str | None, doi: str | None) -> str:
    """Bibliographic line using only available fields, separated by commas."""
    parts: list[str] = []
    if authors:
        parts.append(esc(", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")))
    if journal:
        parts.append(f"<i>{esc(journal)}</i>")
    if year:
        parts.append(esc(year))
    if pmid:
        parts.append(f'<a href="https://pubmed.ncbi.nlm.nih.gov/{esc(pmid)}/" target="_blank" rel="noopener">PMID {esc(pmid)}</a>')
    if doi:
        parts.append(f'<a href="https://doi.org/{esc(doi)}" target="_blank" rel="noopener">DOI {esc(doi)}</a>')
    return ", ".join(parts) if parts else "No bibliographic details available"
