"""Disease / concept explanation: retrieve -> filter -> bound -> generate -> validate -> display."""

from __future__ import annotations

from dataclasses import dataclass, field

from biomentor.config import Settings
from biomentor.exceptions import (
    BioMentorError,
    GeminiError,
    GeminiNotConfiguredError,
    QueryValidationError,
)
from biomentor.generation.abstention import (
    SAFETY_MESSAGE,
    assess_evidence,
    check_query_safety,
    validate_query,
)
from biomentor.generation.citation_validator import CitationReport
from biomentor.generation.evidence import EvidenceSet
from biomentor.generation.gemini_client import GeminiClient
from biomentor.generation.prompts import explanation_prompt, system_prompt
from biomentor.generation.rag_pipeline import friendly_error, run_grounded_task
from biomentor.schemas import ExplanationOutput
from biomentor.services.research_service import SearchResponse, SearchService

# status values
OK = "ok"
NO_GEMINI = "no_gemini"
ABSTAINED = "abstained"
REFUSED_SAFETY = "refused_safety"
INVALID_QUERY = "invalid_query"
ERROR = "error"


@dataclass
class ExplanationResult:
    query: str
    level: str
    status: str
    message: str = ""
    output: ExplanationOutput | None = None
    report: CitationReport | None = None
    evidence: EvidenceSet = field(default_factory=lambda: EvidenceSet([]))
    search: SearchResponse | None = None
    model: str | None = None


def explain_topic(
    search_service: SearchService,
    gemini: GeminiClient | None,
    settings: Settings,
    query: str,
    level: str = "biology student",
    mode: str = "hybrid",
    source: str = "local",
    top_k: int = 8,
) -> ExplanationResult:
    """Never raises for expected failures: the status/message explain what happened, and any
    retrieved evidence is still returned so the user can read the sources."""
    # 1. validate
    try:
        cleaned = validate_query(query)
    except QueryValidationError as exc:
        return ExplanationResult(query or "", level, INVALID_QUERY, str(exc))

    # safety routing (personal diagnosis / treatment questions)
    if check_query_safety(cleaned).is_personal_medical:
        return ExplanationResult(cleaned, level, REFUSED_SAFETY, SAFETY_MESSAGE)

    # 2. retrieve
    try:
        search = search_service.search(cleaned, mode=mode, top_k=top_k, source=source)
    except BioMentorError as exc:
        return ExplanationResult(cleaned, level, ERROR, str(exc))

    # 3-5. filter, de-duplicate, bound (EvidenceSet does de-dup and budgets) and keep metadata
    assessment = assess_evidence(cleaned, search.results)
    evidence = EvidenceSet.build([("retrieved", search.results)], max_passages=top_k, max_chars=settings.max_context_chars)
    if not assessment.sufficient or len(evidence) == 0:
        return ExplanationResult(
            cleaned, level, ABSTAINED,
            f"I don't have enough evidence to answer this reliably. {assessment.reason}",
            evidence=evidence, search=search,
        )

    # 6. generate (optional)
    if gemini is None or not gemini.is_configured:
        return ExplanationResult(
            cleaned, level, NO_GEMINI,
            "Gemini is not configured, so no AI explanation was generated. The retrieved sources are shown below; "
            "add GEMINI_API_KEY to enable explanations.",
            evidence=evidence, search=search,
        )
    try:
        grounded = run_grounded_task(gemini, ExplanationOutput, system_prompt(level), explanation_prompt(cleaned, evidence, level), evidence)
    except GeminiNotConfiguredError as exc:
        return ExplanationResult(cleaned, level, NO_GEMINI, str(exc), evidence=evidence, search=search)
    except GeminiError as exc:
        return ExplanationResult(cleaned, level, ERROR, friendly_error(exc), evidence=evidence, search=search)

    # 10. the model itself may report insufficient evidence
    out = grounded.output
    status = ABSTAINED if out.insufficient_evidence and not out.key_findings and not out.explanation else OK
    return ExplanationResult(
        cleaned, level, status,
        out.insufficient_evidence_note if out.insufficient_evidence else "",
        output=out, report=grounded.report, evidence=evidence, search=search, model=gemini.model,
    )
