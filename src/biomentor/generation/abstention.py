"""Query validation, safety routing and insufficient-evidence ("abstention") decisions.

These are deterministic, explainable rules - not a classifier - and they are intentionally
conservative. Evidence sufficiency uses retrieval *coverage* (how many of the query's content
words appear in the retrieved passages) rather than raw scores, because BM25 scores, cosine
similarities and RRF values are not calibrated probabilities of relevance.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from biomentor.data.preprocessing import content_stems, normalize_whitespace
from biomentor.exceptions import QueryValidationError
from biomentor.schemas import RetrievedPassage

MAX_QUERY_CHARS = 500

SAFETY_MESSAGE = (
    "I can't diagnose conditions, interpret personal symptoms or test results, or advise on "
    "medication, doses or treatment for a specific person. Please talk to a qualified healthcare "
    "professional - and if this may be an emergency, contact your local emergency services. "
    "I'm happy to explain the underlying science in general terms: try rephrasing as a general "
    "question, such as \"How does <condition> affect the body?\" or \"What do studies say about <topic>?\""
)

_PERSONAL_MEDICAL_PATTERNS = [
    r"\bdo i (?:have|got)\b(?! to\b)",
    r"\bam i (?:sick|dying|infected|pregnant|having|going to die|at risk)\b",
    r"\bshould i (?:take|stop|start|use|get|see a doctor|worry|go to)\b",
    r"\b(?:what|which) (?:dose|dosage|medication|medicine|drug|pill|antibiotic)s? (?:should|do|can) i\b",
    r"\bhow (?:much|many) .{0,40}\bshould i (?:take|use)\b",
    r"\bcan i (?:take|stop taking|mix|combine)\b",
    r"\b(?:diagnose|prescribe|treat) (?:me|my)\b",
    r"\b(?:my|our) (?:symptoms?|test results?|blood (?:test|work|results?)|lab results?|scan|mri|x-?ray|biopsy|prescription|medication)\b",
    r"\b(?:i|we) (?:have|am having|feel|am feeling|keep getting) (?:a |an )?(?:pain|fever|cough|rash|headache|lump|chest|dizz)",
]
_PERSONAL_RE = re.compile("|".join(_PERSONAL_MEDICAL_PATTERNS), re.IGNORECASE)

# Words that appear in many queries without describing the topic.
_GENERIC = {"explain", "what", "how", "tell", "about", "describe", "mechanism", "cause", "causes",
            "disease", "work", "works", "role", "does", "study", "studies", "research", "paper", "evidence"}


@dataclass
class SafetyAssessment:
    is_personal_medical: bool
    message: str = ""


@dataclass
class EvidenceAssessment:
    sufficient: bool
    reason: str
    coverage: float
    n_passages: int


def validate_query(query: str | None, min_content_stems: int = 1) -> str:
    """Return the cleaned query or raise :class:`QueryValidationError` with a friendly message."""
    cleaned = normalize_whitespace(query)
    if not cleaned:
        raise QueryValidationError("Please enter a disease, concept, or question to search for.")
    if len(cleaned) > MAX_QUERY_CHARS:
        raise QueryValidationError(f"That query is too long ({len(cleaned)} characters). Please keep it under {MAX_QUERY_CHARS}.")
    if len(content_stems(cleaned)) < min_content_stems:
        raise QueryValidationError("Please include at least one specific term (for example a disease, molecule or process).")
    return cleaned


def check_query_safety(query: str) -> SafetyAssessment:
    if _PERSONAL_RE.search(query or ""):
        return SafetyAssessment(True, SAFETY_MESSAGE)
    return SafetyAssessment(False)


def assess_evidence(
    query: str,
    results: Sequence[RetrievedPassage],
    min_passages: int = 1,
    min_coverage: float = 0.34,
    top_n: int = 5,
) -> EvidenceAssessment:
    """Decide whether the retrieved passages plausibly address the query.

    ``coverage`` = fraction of the query's (non-generic) content stems that occur somewhere in the
    top ``top_n`` passages. If the query has no specific terms the coverage test is skipped.
    """
    if len(results) < min_passages:
        return EvidenceAssessment(False, "No relevant passages were retrieved for this query.", 0.0, len(results))
    query_stems = {s for s in content_stems(query) if s not in _GENERIC}
    if not query_stems:
        return EvidenceAssessment(True, "ok", 1.0, len(results))
    haystack: set[str] = set()
    for r in results[:top_n]:
        haystack |= content_stems(r.passage.retrieval_text())
    coverage = len(query_stems & haystack) / len(query_stems)
    if coverage < min_coverage:
        return EvidenceAssessment(
            False,
            f"The retrieved passages mention only {coverage:.0%} of your query's key terms, so they probably do not address it.",
            coverage,
            len(results),
        )
    return EvidenceAssessment(True, "ok", coverage, len(results))
