"""Pydantic models.

Part 1 - domain objects (documents, passages, retrieval results).
Part 2 - structured outputs requested from Gemini. These deliberately use only required fields
and plain ``str`` / ``list`` types so they convert cleanly to a response schema; leniency is
handled by validators here rather than by the schema.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

SOURCE_PUBMED = "pubmed"
SOURCE_SYNTHETIC = "synthetic_fixture"
SOURCE_UPLOAD = "uploaded_pdf"

NOT_REPORTED = "Not reported in the available text."


# =============================================================================================
# Part 1: domain objects
# =============================================================================================
class Document(BaseModel):
    document_id: str
    title: str
    abstract: str | None = None
    source_name: str
    source_type: str
    pmid: str | None = None
    doi: str | None = None
    authors: list[str] | None = None
    journal: str | None = None
    publication_date: str | None = None
    publication_year: int | None = None
    article_type: str | None = None
    source_url: str | None = None
    license: str | None = None
    retrieved_at: str | None = None


class Passage(BaseModel):
    """A chunk of a document. ``text`` is the original text used for citations; the offsets
    index into the whitespace-normalised abstract/full text of the parent document."""

    passage_id: str
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    title: str = ""
    source_name: str = ""
    source_type: str = ""
    pmid: str | None = None
    doi: str | None = None
    journal: str | None = None
    publication_year: int | None = None
    source_url: str | None = None

    def retrieval_text(self) -> str:
        """Text used for indexing: the title gives each chunk its topical context."""
        return f"{self.title}. {self.text}" if self.title else self.text


class RetrievedPassage(BaseModel):
    passage: Passage
    score: float
    rank: int  # 1-based
    method: str
    component_ranks: dict[str, int] = Field(default_factory=dict)
    component_scores: dict[str, float] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.passage.text

    @property
    def passage_id(self) -> str:
        return self.passage.passage_id

    @property
    def document_id(self) -> str:
        return self.passage.document_id

    @property
    def title(self) -> str:
        return self.passage.title

    @property
    def pmid(self) -> str | None:
        return self.passage.pmid

    @property
    def publication_year(self) -> int | None:
        return self.passage.publication_year

    @property
    def journal(self) -> str | None:
        return self.passage.journal

    def retrieval_text(self) -> str:
        return self.passage.retrieval_text()


# =============================================================================================
# Part 2: structured LLM output
# =============================================================================================
_BASIS_VALUES = ("evidence", "background", "analogy")


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    seen: list[str] = []
    for item in value:
        s = str(item).strip()
        if s and s not in seen:
            seen.append(s)
    return seen


class Claim(BaseModel):
    text: str = Field(description="One self-contained statement.")
    basis: str = Field(
        description="'evidence' if it is stated in the supplied passages, 'background' for general "
        "textbook knowledge NOT in the passages, 'analogy' for an explanatory analogy."
    )
    citation_ids: list[str] = Field(
        description="Evidence labels (e.g. 'E1') that support the statement. Empty for background/analogy."
    )

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("basis")
    @classmethod
    def _basis(cls, v: str) -> str:
        v = (v or "").strip().lower()
        return v if v in _BASIS_VALUES else "evidence"  # unknown -> strictest treatment

    @field_validator("citation_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class Term(BaseModel):
    term: str
    definition: str
    citation_ids: list[str] = Field(description="Evidence labels, or empty for general background.")

    @field_validator("citation_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class ExplanationOutput(BaseModel):
    short_definition: Claim
    explanation: list[Claim] = Field(description="Level-appropriate explanation, 2-5 short paragraphs.")
    mechanism_steps: list[Claim] = Field(
        description="Numbered mechanism steps in order. Empty list if the evidence does not describe a mechanism."
    )
    key_terms: list[Term]
    key_findings: list[Claim] = Field(description="Findings reported in the evidence only.")
    limitations: list[Claim] = Field(description="Limitations, caveats and uncertainty.")
    follow_up_questions: list[str]
    insufficient_evidence: bool
    insufficient_evidence_note: str = Field(description="Empty string unless evidence is insufficient.")

    @field_validator("follow_up_questions", mode="before")
    @classmethod
    def _qs(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class AnswerOutput(BaseModel):
    answer: list[Claim]
    key_terms: list[Term]
    limitations: list[Claim]
    follow_up_questions: list[str]
    insufficient_evidence: bool
    insufficient_evidence_note: str

    @field_validator("follow_up_questions", mode="before")
    @classmethod
    def _qs(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class PaperSummaryOutput(BaseModel):
    abstract_explanation: list[Claim] = Field(description="Plain-language explanation of the abstract, if present.")
    summary: list[Claim]
    key_findings: list[Claim]
    limitations: list[Claim] = Field(description="Only limitations stated or clearly implied by the text.")
    key_terms: list[Term]
    insufficient_evidence: bool
    insufficient_evidence_note: str


class FieldValue(BaseModel):
    value: str = Field(description=f"The reported information, or exactly '{NOT_REPORTED}'.")
    citation_ids: list[str]

    @field_validator("citation_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class PaperComparison(BaseModel):
    document_id: str
    research_question: FieldValue
    study_type: FieldValue
    population_or_model: FieldValue
    sample_size: FieldValue
    intervention_or_exposure: FieldValue
    comparator: FieldValue
    outcomes: FieldValue
    findings: FieldValue
    limitations: FieldValue


class ComparisonOutput(BaseModel):
    papers: list[PaperComparison]
    agreements: list[Claim]
    disagreements: list[Claim]
    caveats: list[str]

    @field_validator("caveats", mode="before")
    @classmethod
    def _cv(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class MCQItem(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    supporting_quote: str = Field(description="A short VERBATIM quote from the cited passage that proves the answer.")
    citation_ids: list[str]

    @field_validator("citation_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class ShortAnswerItem(BaseModel):
    question: str
    model_answer: str
    key_points: list[str]
    supporting_quote: str
    citation_ids: list[str]

    @field_validator("citation_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[str]:
        return _as_str_list(v)

    @field_validator("key_points", mode="before")
    @classmethod
    def _kp(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class FlashcardItem(BaseModel):
    front: str
    back: str
    supporting_quote: str
    citation_ids: list[str]

    @field_validator("citation_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[str]:
        return _as_str_list(v)


class QuizOutput(BaseModel):
    mcq: list[MCQItem]
    short_answer: list[ShortAnswerItem]
    flashcards: list[FlashcardItem]
