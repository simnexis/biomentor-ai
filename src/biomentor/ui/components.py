"""Rendering helpers: provenance rail, notices, cited prose, source cards, references."""

from __future__ import annotations

import streamlit as st

from biomentor.generation.citation_validator import CitationReport
from biomentor.generation.evidence import EvidenceSet
from biomentor.retrieval.index_manager import MODE_LABELS
from biomentor.schemas import Claim, Document, RetrievedPassage, Term
from biomentor.services.digest import DigestItem, build_references
from biomentor.ui.formatting import (
    claim_html,
    esc,
    highlight_html,
    meta_html,
    source_chip,
    term_html,
)

ORIGIN_LABELS = {"retrieved": "retrieved literature", "uploaded": "uploaded paper", "external": "external literature", "selected": "selected passage"}

__all__ = [
    "esc", "notice", "warn", "provenance_rail", "render_claim", "render_claims", "render_paragraphs", "render_lead",
    "render_mechanism", "render_terms", "render_citation_report", "render_evidence", "render_document_card",
    "render_digest", "render_references", "empty_state", "doc_meta_line",
]


def notice(message: str, kind: str = "warn") -> None:
    """Inline notice. kind: warn (amber), info (indigo), error (rose)."""
    st.markdown(f'<div class="note note-{esc(kind)}">{esc(message)}</div>', unsafe_allow_html=True)


def warn(message: str) -> None:
    notice(message, "warn")


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><b>{esc(title)}</b><br>{esc(body)}</div>', unsafe_allow_html=True)


def provenance_rail(search, extra: list[str] | None = None) -> None:
    """One quiet line saying where the evidence came from and how it was found."""
    if search is None:
        return
    items = [source_chip_for_kind(search.source_kind, search.source_label)]
    items.append(f"Method <b>{esc(MODE_LABELS.get(search.mode_used, search.mode_used))}</b>")
    items.append(f"<b>{search.n_passages_searched}</b> passages searched")
    if search.timing_ms.get("total"):
        items.append(f"{search.timing_ms['total'] / 1000:.1f} s")
    items += [esc(x) for x in (extra or [])]
    st.markdown('<div class="rail">' + "".join(f"<span>{i}</span>" for i in items) + "</div>", unsafe_allow_html=True)
    for w in search.warnings:
        notice(w, "warn")
    if search.source_kind == "demo_fixtures":
        notice("These sources are synthetic demo text written for testing. They are not real publications. Fetch real PubMed records with scripts/fetch_literature.py.", "warn")


def source_chip_for_kind(kind: str, label: str) -> str:
    css = {"live_pubmed": "src-pubmed", "local_corpus": "src-pubmed", "demo_fixtures": "src-synth", "local_mixed": "src-synth"}.get(kind, "")
    return f'<span class="chip {css}">{esc(label)}</span>'


# ------------------------------------------------------------------ cited prose
def _claim_markup(c: Claim | Term) -> str:
    if isinstance(c, Term):
        return term_html(c.term, c.definition, c.citation_ids)
    return claim_html(c.text, c.citation_ids, c.basis)


def render_claim(c: Claim | Term) -> str:
    return _claim_markup(c)


def render_lead(c: Claim) -> None:
    st.markdown(f'<div class="lead">{_claim_markup(c)}</div>', unsafe_allow_html=True)


def render_paragraphs(claims: list[Claim]) -> None:
    st.markdown("".join(f'<p class="prose">{_claim_markup(c)}</p>' for c in claims), unsafe_allow_html=True)


def render_claims(claims: list[Claim], numbered: bool = False) -> None:
    if numbered:
        render_mechanism(claims)
    else:
        render_paragraphs(claims)


def render_mechanism(claims: list[Claim]) -> None:
    st.markdown('<ol class="steps">' + "".join(f"<li><div>{_claim_markup(c)}</div></li>" for c in claims) + "</ol>", unsafe_allow_html=True)


def render_terms(terms: list[Term]) -> None:
    st.markdown("".join(f'<div class="term">{_claim_markup(t)}</div>' for t in terms), unsafe_allow_html=True)


# ------------------------------------------------------------------ sources
def doc_meta_line(doc: Document | None, passage=None) -> str:
    d = doc
    return meta_html(
        d.authors if d else None,
        (d.publication_year if d else None) or (passage.publication_year if passage else None),
        (d.journal if d else None) or (passage.journal if passage else None),
        (d.pmid if d else None) or (passage.pmid if passage else None),
        (d.doi if d else None) or (passage.doi if passage else None),
    )


def render_document_card(doc: Document | None, hits: list[RetrievedPassage], show_abstract: bool = False, query: str = "") -> None:
    first = hits[0].passage
    title = (doc.title if doc else first.title) or "(untitled)"
    st.markdown(
        f'<div class="card"><div class="title">{esc(title)}</div>{source_chip(first.source_type)}'
        f'<div class="bib">{doc_meta_line(doc, first)}</div></div>',
        unsafe_allow_html=True,
    )
    for h in hits:
        comps = ", ".join(f"{MODE_LABELS.get(k, k).split(' (')[0].lower()} rank {v}" for k, v in h.component_ranks.items())
        st.markdown(
            f'<div class="quote">{highlight_html(h.passage.text, query)}'
            f'<div class="meta">Result {h.rank}, score {h.score:.3f} ({esc(comps)}). Passage {esc(h.passage.passage_id)}</div></div>',
            unsafe_allow_html=True,
        )
    if show_abstract and doc and doc.abstract:
        with st.expander("Full abstract"):
            st.markdown(f'<p class="prose">{highlight_html(doc.abstract, query)}</p>', unsafe_allow_html=True)


def render_digest(items: list[DigestItem], query: str) -> None:
    """Extractive key sentences: exact quotes, each tied to its source."""
    for d in items:
        p = d.passage
        who = f"PMID {p.pmid}" if p.pmid else (p.title or p.passage_id)
        st.markdown(
            f'<div class="quote">{highlight_html(d.sentence, query)}<div class="meta">From {esc(who[:110])}</div></div>',
            unsafe_allow_html=True,
        )


def render_evidence(evidence: EvidenceSet, documents: dict[str, Document] | None = None, title: str = "Evidence used", query: str = "") -> None:
    if not len(evidence):
        return
    st.subheader(title)
    for item in evidence:
        p = item.passage
        doc = (documents or {}).get(p.document_id)
        with st.expander(f"{item.label}  {(p.title or p.passage_id)[:90]}"):
            st.markdown(
                f'{source_chip(p.source_type)}<span class="chip">{esc(ORIGIN_LABELS.get(item.origin, item.origin))}</span>'
                f'<div class="bib" style="font-size:.84rem;color:var(--muted)">{doc_meta_line(doc, p)}</div>'
                f'<div class="quote">{highlight_html(p.text, query)}'
                f'<div class="meta">Passage {esc(p.passage_id)}' + (f", retrieval score {item.score:.3f}" if item.score is not None else "") + "</div></div>",
                unsafe_allow_html=True,
            )


def render_references(evidence: EvidenceSet, documents: dict[str, Document] | None = None) -> None:
    refs = build_references(evidence, documents)
    if not refs:
        return
    st.subheader("References")
    rows = []
    for labels, ref, url in refs:
        chips = "".join(f'<sup class="cite">{esc(label)}</sup>' for label in labels)
        link = f' <a href="{esc(url)}" target="_blank" rel="noopener">Open on PubMed</a>' if url else ""
        rows.append(f"<li>{chips} {esc(ref)}{link}</li>")
    st.markdown('<ul class="refs">' + "".join(rows) + "</ul>", unsafe_allow_html=True)


def render_citation_report(report: CitationReport | None) -> None:
    if report is None:
        return
    with st.expander("How well are the citations checked?", expanded=not report.ok):
        st.caption(
            "Every citation id is checked against the evidence shown for this answer, and invalid ones are removed. "
            "A rough word-overlap check flags statements that share few words with their cited passage. "
            "A valid citation does not prove a statement is true, so read the passage."
        )
        st.write(report.summary())
        flagged = report.flagged()
        if flagged:
            st.dataframe(
                [{"statement": c.text[:160], "flags": ", ".join(c.flags), "word overlap": None if c.lexical_support is None else round(c.lexical_support, 2)} for c in flagged],
                hide_index=True,
            )
