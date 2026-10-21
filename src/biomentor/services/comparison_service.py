"""Compare two or more papers (abstract-level), with deterministic safeguards.

This is a reading aid, **not** a systematic review or evidence grading. Publication date comes from
metadata (never from the model); sample sizes and other numbers stated by the model are checked
against the cited text and replaced with "Not reported" if they cannot be found there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from biomentor.config import Settings
from biomentor.data.preprocessing import normalize_for_match
from biomentor.exceptions import BioMentorError
from biomentor.generation.citation_validator import CitationReport, validate_citations
from biomentor.generation.evidence import EvidenceItem, EvidenceSet
from biomentor.generation.gemini_client import GeminiClient
from biomentor.generation.prompts import comparison_prompt, system_prompt
from biomentor.schemas import NOT_REPORTED, ComparisonOutput, Document, FieldValue, Passage

FIELDS = [
    ("research_question", "Research question"),
    ("study_type", "Study type"),
    ("population_or_model", "Population or model studied"),
    ("sample_size", "Sample size"),
    ("intervention_or_exposure", "Intervention or exposure"),
    ("comparator", "Comparator"),
    ("outcomes", "Outcomes"),
    ("findings", "Findings"),
    ("limitations", "Limitations"),
]
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NOT_REPORTED_RE = re.compile(r"^\s*(?:not (?:reported|stated|specified|available|mentioned)|n/?a|unknown|none)\b", re.I)


@dataclass
class ComparisonResult:
    documents: list[Document]
    table: list[dict]  # one dict per paper: {"Paper": ..., "Publication date": ..., <field label>: (text, [labels])}
    agreements: list
    disagreements: list
    caveats: list[str]
    report: CitationReport
    evidence: EvidenceSet
    unverified_numbers: list[str]


def _numbers_supported(value: str, evidence_text: str) -> list[str]:
    """Numbers in ``value`` that do NOT appear in the evidence text."""
    haystack = normalize_for_match(evidence_text).replace(",", "")
    missing = []
    for number in _NUMBER_RE.findall(value):
        n = number.replace(",", "")
        if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", haystack):
            missing.append(number)
    return missing


def _doc_label(doc: Document) -> str:
    year = f" ({doc.publication_year})" if doc.publication_year else ""
    return f"{doc.title}{year}"


def compare_papers(
    gemini: GeminiClient,
    settings: Settings,
    documents: list[Document],
    passages_by_doc: dict[str, list[Passage]],
) -> ComparisonResult:
    if len(documents) < 2:
        raise BioMentorError("Select at least two papers to compare.")
    groups = []
    for doc in documents:
        ps = passages_by_doc.get(doc.document_id, [])
        if ps:
            groups.append((doc.document_id, ps))
    if len(groups) < 2:
        raise BioMentorError("At least two of the selected papers need abstract text to compare.")

    # one shared label space across papers so citations can be validated together
    per_doc = max(3, 12 // len(groups))
    items = []
    for doc_id, ps in groups:
        for p in ps[:per_doc]:
            items.append(EvidenceItem(f"E{len(items) + 1}", p, origin=doc_id))
    evidence = EvidenceSet(items)

    doc_ids = [d.document_id for d in documents if d.document_id in dict(groups)]
    raw = gemini.generate_structured(ComparisonOutput, comparison_prompt(doc_ids, evidence), system_prompt())
    output, report = validate_citations(raw, evidence)

    by_id = {p.document_id: p for p in output.papers if p.document_id in doc_ids}
    unverified: list[str] = []
    table: list[dict] = []
    for doc in documents:
        row: dict = {"Paper": _doc_label(doc), "Publication date": doc.publication_date or NOT_REPORTED, "_document_id": doc.document_id}
        entry = by_id.get(doc.document_id)
        for key, label in FIELDS:
            fv: FieldValue | None = getattr(entry, key) if entry else None
            if fv is None or not fv.value.strip() or _NOT_REPORTED_RE.match(fv.value):
                row[label] = (NOT_REPORTED, [])
                continue
            # Study-specific content must be tied to this paper's own passages, not another paper's.
            own = {i.label for i in evidence.items if i.origin == doc.document_id}
            cited = [c for c in fv.citation_ids if c in own]
            if not cited:
                row[label] = (NOT_REPORTED, [])
                continue
            if key == "sample_size":
                missing = _numbers_supported(fv.value, evidence.text_for(cited))
                if missing:
                    unverified.append(f"{_doc_label(doc)}: '{fv.value}' (could not find {', '.join(missing)} in the cited text)")
                    row[label] = (NOT_REPORTED, [])
                    continue
            row[label] = (fv.value.strip(), cited)
        table.append(row)

    return ComparisonResult(
        documents=documents, table=table, agreements=output.agreements, disagreements=output.disagreements,
        caveats=output.caveats, report=report, evidence=evidence, unverified_numbers=unverified,
    )
