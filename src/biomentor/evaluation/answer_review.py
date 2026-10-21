"""Human review of generated answers (the part automation cannot do).

Retrieval metrics say whether the *right documents* were found. They say nothing about whether a
generated answer is *true*. This workflow exports every statement of a saved answer with its cited
passages to a CSV; a person then judges each one and the script summarises the judgments.

Columns a reviewer fills in:
* ``citation_valid``   - does the cited passage exist / is it the one that was meant? (yes/no)
* ``claim_support``    - supported / partially / unsupported / not_checkable
* ``uncertainty_ok``   - are hedges ("may", "in mice") preserved? (yes/no)
* ``notes``
The ``auto_*`` columns come from the deterministic checks in ``citation_validator`` and are only hints.
"""

from __future__ import annotations

import csv
from pathlib import Path

from biomentor.generation.citation_validator import CitationReport, claim_text, iter_cited
from biomentor.generation.evidence import EvidenceSet

REVIEW_COLUMNS = [
    "answer_id", "statement_path", "statement", "basis", "cited_labels", "cited_passages",
    "auto_invalid_ids_removed", "auto_lexical_support", "auto_flags",
    "citation_valid", "claim_support", "uncertainty_ok", "notes",
]
SUPPORT_VALUES = {"supported", "partially", "unsupported", "not_checkable"}


def export_review_sheet(path: Path | str, answer_id: str, output_dict: dict, evidence: EvidenceSet, report: CitationReport | None = None) -> int:
    auto = {c.path: c for c in (report.checks if report else [])}
    rows = []
    for stmt_path, item in iter_cited(output_dict):
        text = claim_text(item)
        if not text:
            continue
        labels = item.get("citation_ids", [])
        check = auto.get(stmt_path)
        rows.append({
            "answer_id": answer_id, "statement_path": stmt_path, "statement": text, "basis": item.get("basis", ""),
            "cited_labels": ",".join(labels),
            "cited_passages": " || ".join(f"[{label}] {evidence.get(label).passage.text}" for label in labels if evidence.get(label)),
            "auto_invalid_ids_removed": ",".join(check.invalid_ids) if check else "",
            "auto_lexical_support": "" if not check or check.lexical_support is None else f"{check.lexical_support:.2f}",
            "auto_flags": ",".join(check.flags) if check else "",
            "citation_valid": "", "claim_support": "", "uncertainty_ok": "", "notes": "",
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize_reviews(path: Path | str) -> dict:
    """Aggregate a reviewed sheet. Unreviewed rows are counted but excluded from the rates."""
    total = reviewed = 0
    support: dict[str, int] = {v: 0 for v in SUPPORT_VALUES}
    citation_yes = citation_total = uncertainty_yes = uncertainty_total = 0
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            s = (row.get("claim_support") or "").strip().lower()
            if s not in SUPPORT_VALUES:
                continue
            reviewed += 1
            support[s] += 1
            if (row.get("citation_valid") or "").strip().lower() in {"yes", "no"}:
                citation_total += 1
                citation_yes += row["citation_valid"].strip().lower() == "yes"
            if (row.get("uncertainty_ok") or "").strip().lower() in {"yes", "no"}:
                uncertainty_total += 1
                uncertainty_yes += row["uncertainty_ok"].strip().lower() == "yes"
    checkable = reviewed - support["not_checkable"]
    return {
        "statements_total": total, "statements_reviewed": reviewed, "claim_support_counts": support,
        "supported_rate": (support["supported"] / checkable) if checkable else None,
        "unsupported_rate": (support["unsupported"] / checkable) if checkable else None,
        "citation_valid_rate": (citation_yes / citation_total) if citation_total else None,
        "uncertainty_preserved_rate": (uncertainty_yes / uncertainty_total) if uncertainty_total else None,
    }
