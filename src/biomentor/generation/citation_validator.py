"""Citation checks for structured model output.

Two *different* questions are answered here, and the difference matters:

1. **Citation validity (deterministic).** Does every cited label exist in the current evidence set?
   Invalid labels are stripped from the output and reported. An evidence-based claim left with no
   valid citation is flagged.
2. **Lexical support (heuristic).** What fraction of a claim's content words also occur in the
   passages it cites? A low value is a *warning that the claim may not be supported*; a high value
   does NOT prove it is. A valid citation is not the same as factual support - the real check is
   the human review workflow in ``biomentor.evaluation.answer_review``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from biomentor.data.preprocessing import content_stems
from biomentor.generation.evidence import EvidenceSet
from biomentor.schemas import NOT_REPORTED

T = TypeVar("T", bound=BaseModel)

_TEXT_KEYS = ("text", "definition", "value", "model_answer", "explanation")
MIN_STEMS_FOR_SUPPORT = 3


@dataclass
class ClaimCheck:
    path: str
    text: str
    basis: str
    cited_ids: list[str]
    invalid_ids: list[str]
    lexical_support: float | None
    flags: list[str] = field(default_factory=list)


@dataclass
class CitationReport:
    checks: list[ClaimCheck] = field(default_factory=list)

    @property
    def n_claims(self) -> int:
        return len(self.checks)

    @property
    def n_invalid_ids(self) -> int:
        return sum(len(c.invalid_ids) for c in self.checks)

    @property
    def n_uncited_evidence_claims(self) -> int:
        return sum(1 for c in self.checks if "no_valid_citation" in c.flags)

    @property
    def n_weak_support(self) -> int:
        return sum(1 for c in self.checks if "weak_lexical_support" in c.flags)

    @property
    def ok(self) -> bool:
        return self.n_invalid_ids == 0 and self.n_uncited_evidence_claims == 0

    def flagged(self) -> list[ClaimCheck]:
        return [c for c in self.checks if c.flags]

    def summary(self) -> str:
        return (
            f"{self.n_claims} statements checked: {self.n_invalid_ids} invalid citation id(s) removed, "
            f"{self.n_uncited_evidence_claims} evidence-based statement(s) without a valid citation, "
            f"{self.n_weak_support} with weak word overlap against their cited passage (heuristic)."
        )


def iter_cited(data: Any, path: str = "") -> Iterator[tuple[str, dict]]:
    if isinstance(data, dict):
        if "citation_ids" in data:
            yield path, data
        for key, value in data.items():
            yield from iter_cited(value, f"{path}.{key}" if path else key)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            yield from iter_cited(value, f"{path}[{i}]")


def claim_text(item: dict) -> str:
    for key in _TEXT_KEYS:
        if isinstance(item.get(key), str) and item[key].strip():
            return item[key].strip()
    return ""


def validate_citations(
    output: T, evidence: EvidenceSet, weak_threshold: float = 0.5, check_support: bool = True
) -> tuple[T, CitationReport]:
    """Return ``(sanitised_output, report)``. ``sanitised_output`` has invalid ids removed."""
    data = output.model_dump()
    valid = set(evidence.labels)
    report = CitationReport()

    for path, item in iter_cited(data):
        cited = [c for c in item["citation_ids"]]
        invalid = [c for c in cited if c not in valid]
        good = [c for c in cited if c in valid]
        item["citation_ids"] = good
        text = claim_text(item)
        if not text or text == NOT_REPORTED:
            if invalid:  # still report removals
                report.checks.append(ClaimCheck(path, text, "n/a", good, invalid, None, ["invalid_citation_removed"]))
            continue
        # Terms (has 'term') are general vocabulary: background unless cited. Items with an
        # explicit basis use it. Everything else (findings, field values, quiz items) is evidence.
        basis = item.get("basis") or ("background" if "term" in item and not good else "evidence")
        flags: list[str] = []
        if invalid:
            flags.append("invalid_citation_removed")
        if basis == "evidence" and not good:
            flags.append("no_valid_citation")
        support = None
        if check_support and good:
            claim_stems = content_stems(text)
            if len(claim_stems) >= MIN_STEMS_FOR_SUPPORT:
                ev_stems = content_stems(evidence.text_for(good))
                support = len(claim_stems & ev_stems) / len(claim_stems)
                if basis == "evidence" and support < weak_threshold:
                    flags.append("weak_lexical_support")
        report.checks.append(ClaimCheck(path, text, basis, good, invalid, support, flags))

    return type(output).model_validate(data), report
