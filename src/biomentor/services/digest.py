"""Features that work without an LLM: extractive evidence digest, reference list, Markdown export.

The *digest* picks the sentences of the retrieved passages that best match the query. Nothing is
rewritten, so every line is an exact quotation of its source - useful when no Gemini key is set, and
a transparent baseline to compare generated answers against.
"""

from __future__ import annotations

from dataclasses import dataclass

from biomentor.data.chunking import sentence_spans
from biomentor.data.preprocessing import content_stems, expand_biomedical_query
from biomentor.generation.evidence import EvidenceSet
from biomentor.schemas import Document, Passage, RetrievedPassage

GENERIC_QUERY_WORDS = {"explain", "what", "how", "tell", "about", "describe", "does", "work", "works", "role", "study", "studies"}


@dataclass(frozen=True)
class DigestItem:
    sentence: str
    passage: Passage
    score: float


def extractive_digest(
    query: str, results: list[RetrievedPassage], max_sentences: int = 5, max_per_document: int = 2, min_overlap: float = 0.15
) -> list[DigestItem]:
    """Best-matching sentences, exact quotes. Score = share of the query's content stems found in
    the sentence, plus a small bonus for sentences from highly ranked passages. Sentences with
    overlap below ``min_overlap`` are never returned (better empty than irrelevant)."""
    expanded = expand_biomedical_query(query)
    query_stems = {s for s in content_stems(expanded) if s not in GENERIC_QUERY_WORDS} or content_stems(expanded)
    if not query_stems or not results:
        return []
    candidates: list[tuple[float, int, int, DigestItem]] = []
    seen: set[str] = set()
    for pos, hit in enumerate(results):
        p = getattr(hit, "passage", hit)
        text = p.text
        for start, end in sentence_spans(text):
            sentence = text[start:end].strip()
            if len(sentence) < 25 or sentence.casefold() in seen:
                continue
            overlap = len(query_stems & content_stems(sentence)) / len(query_stems)
            if overlap < min_overlap:
                continue
            seen.add(sentence.casefold())
            score = overlap + 0.15 / (1 + pos)
            candidates.append((score, pos, start, DigestItem(sentence, p, round(score, 4))))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    chosen: list[DigestItem] = []
    per_doc: dict[str, int] = {}
    for _, _, _, item in candidates:
        doc = item.passage.document_id
        if per_doc.get(doc, 0) >= max_per_document:
            continue
        per_doc[doc] = per_doc.get(doc, 0) + 1
        chosen.append(item)
        if len(chosen) >= max_sentences:
            break
    return chosen


# ------------------------------------------------------------------ references
def format_reference(doc: Document | None, passage: Passage | None = None) -> str:
    """One-line reference using only fields that exist. Never invents metadata."""
    title = (doc.title if doc else None) or (passage.title if passage else "") or "(untitled)"
    parts: list[str] = []
    if doc and doc.authors:
        parts.append(", ".join(doc.authors[:3]) + (", et al." if len(doc.authors) > 3 else ""))
    parts.append(title.rstrip("."))
    journal = (doc.journal if doc else None) or (passage.journal if passage else None)
    year = (doc.publication_year if doc else None) or (passage.publication_year if passage else None)
    if journal:
        parts.append(journal)
    if year:
        parts.append(str(year))
    pmid = (doc.pmid if doc else None) or (passage.pmid if passage else None)
    doi = (doc.doi if doc else None) or (passage.doi if passage else None)
    if pmid:
        parts.append(f"PMID {pmid}")
    if doi:
        parts.append(f"doi:{doi}")
    kind = (doc.source_type if doc else None) or (passage.source_type if passage else "")
    line = ". ".join(parts)
    if kind == "synthetic_fixture":
        line += " [synthetic demo text, not a real publication]"
    elif kind == "uploaded_pdf":
        line += " [uploaded paper]"
    return line


def build_references(evidence: EvidenceSet, documents: dict[str, Document] | None = None) -> list[tuple[list[str], str, str | None]]:
    """Unique references: ``(evidence labels, formatted reference, url or None)`` in first-cited order."""
    by_doc: dict[str, tuple[list[str], str, str | None]] = {}
    for item in evidence:
        p = item.passage
        doc = (documents or {}).get(p.document_id)
        if p.document_id not in by_doc:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/" if p.pmid else None
            by_doc[p.document_id] = ([item.label], format_reference(doc, p), url)
        else:
            by_doc[p.document_id][0].append(item.label)
    return list(by_doc.values())


# ------------------------------------------------------------------ export
def _cite(ids: list[str]) -> str:
    return " " + "".join(f"[{i}]" for i in ids) if ids else ""


def _claim_line(c) -> str:
    text = getattr(c, "text", None)
    if text is None:  # Term
        text = f"**{c.term}**: {c.definition}"
    suffix = {"background": " (general background, not from the sources)", "analogy": " (analogy)"}.get(getattr(c, "basis", ""), "")
    return f"{text}{suffix}{_cite(c.citation_ids)}"


def result_to_markdown(result, documents: dict[str, Document] | None = None) -> str:
    """Export an :class:`ExplanationResult` (or its extractive fallback) as a Markdown study note."""
    lines = [f"# {result.query}", "", f"*Explanation level: {result.level}.*"]
    if result.search:
        lines.append(f"*Evidence: {result.search.source_label}; method: {result.search.mode_used}.*")
        if result.search.source_kind == "demo_fixtures":
            lines.append("*Warning: the sources are synthetic demo text, not real publications.*")
    lines.append("")
    out = result.output
    if out is not None:
        sections = [
            ("Short definition", [out.short_definition], False),
            ("Explanation", out.explanation, False),
            ("Mechanism", out.mechanism_steps, True),
            ("Key terms", out.key_terms, False),
            ("Important findings", out.key_findings, False),
            ("Limitations and uncertainty", out.limitations, False),
        ]
        for title, items, numbered in sections:
            if not items:
                continue
            lines += [f"## {title}", ""]
            for i, c in enumerate(items, start=1):
                lines.append(f"{i}. {_claim_line(c)}" if numbered else (f"- {_claim_line(c)}" if title in ("Key terms", "Important findings", "Limitations and uncertainty") else _claim_line(c) + "\n"))
            lines.append("")
        if out.follow_up_questions:
            lines += ["## Follow-up questions", ""] + [f"- {q}" for q in out.follow_up_questions] + [""]
    elif len(result.evidence):
        digest = extractive_digest(result.query, result.search.results if result.search else [])
        if digest:
            lines += ["## Most relevant sentences (extracted, not generated)", ""]
            lines += [f'- "{d.sentence}" (PMID {d.passage.pmid})' if d.passage.pmid else f'- "{d.sentence}"' for d in digest] + [""]
    if len(result.evidence):
        lines += ["## References", ""]
        for labels, ref, url in build_references(result.evidence, documents):
            lines.append(f"- [{', '.join(labels)}] {ref}" + (f" <{url}>" if url else ""))
        lines += ["", "## Evidence passages", ""]
        for item in result.evidence:
            lines += [f"**[{item.label}]** {item.passage.text}", ""]
    lines += ["---", "Educational use only. Not medical advice. Generated statements can contain errors even when citations are valid: check the passages."]
    return "\n".join(lines).strip() + "\n"
