"""The shared generate -> parse -> validate-citations step used by every Gemini-backed feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from biomentor.exceptions import GeminiError
from biomentor.generation.citation_validator import CitationReport, validate_citations
from biomentor.generation.evidence import EvidenceSet
from biomentor.generation.gemini_client import GeminiClient

T = TypeVar("T", bound=BaseModel)


@dataclass
class GroundedResult(Generic[T]):
    output: T
    report: CitationReport
    evidence: EvidenceSet


def run_grounded_task(
    gemini: GeminiClient,
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
    evidence: EvidenceSet,
) -> GroundedResult[T]:
    """Call Gemini, validate the structure, then strip/flag invalid citations.

    Raises a :class:`GeminiError` subclass on failure - callers show the (secret-free) message and
    keep the evidence visible.
    """
    raw = gemini.generate_structured(schema, user_prompt, system_prompt)
    sanitised, report = validate_citations(raw, evidence)
    return GroundedResult(sanitised, report, evidence)


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, GeminiError):
        return str(exc)
    return f"Something went wrong while generating the answer ({type(exc).__name__})."
