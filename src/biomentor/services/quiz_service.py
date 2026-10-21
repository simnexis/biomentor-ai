"""Study Lab: quiz generation grounded in selected passages, plus validation and scoring.

Every generated item must (a) have a valid structure, (b) cite passages that are in the supplied
evidence, and (c) include a ``supporting_quote`` that appears *verbatim* in a cited passage. Items
failing any check are dropped and reported. The verbatim-quote check is a real, deterministic test
that the item is anchored in the source; it does not prove the *question* is well posed, so answers
remain worth reviewing.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from biomentor.data.preprocessing import content_stems, normalize_for_match
from biomentor.exceptions import BioMentorError
from biomentor.generation.evidence import EvidenceSet
from biomentor.generation.gemini_client import GeminiClient
from biomentor.generation.prompts import quiz_prompt, system_prompt
from biomentor.schemas import FlashcardItem, MCQItem, Passage, QuizOutput, ShortAnswerItem

DIFFICULTIES = ("easy", "medium", "hard")
KINDS = ("mcq", "short_answer", "flashcards")
MIN_QUOTE_CHARS = 12


@dataclass
class DroppedItem:
    kind: str
    question: str
    reason: str


@dataclass
class Quiz:
    mcq: list[MCQItem] = field(default_factory=list)
    short_answer: list[ShortAnswerItem] = field(default_factory=list)
    flashcards: list[FlashcardItem] = field(default_factory=list)
    dropped: list[DroppedItem] = field(default_factory=list)
    evidence: EvidenceSet = field(default_factory=lambda: EvidenceSet([]))

    def is_empty(self) -> bool:
        return not (self.mcq or self.short_answer or self.flashcards)


def _quote_in_passages(quote: str, labels: list[str], evidence: EvidenceSet) -> bool:
    q = normalize_for_match(quote).strip(" .\"'")
    if len(q) < MIN_QUOTE_CHARS:
        return False
    return q in normalize_for_match(evidence.text_for(labels))


def _check_common(quote: str, citation_ids: list[str], evidence: EvidenceSet) -> str | None:
    valid = [c for c in citation_ids if evidence.get(c)]
    if not valid:
        return "cites no passage from the supplied evidence"
    if not _quote_in_passages(quote, valid, evidence):
        return "supporting quote is not found verbatim in the cited passage"
    return None


def validate_quiz(output: QuizOutput, evidence: EvidenceSet, rng: random.Random | None = None) -> Quiz:
    """Validate structure + grounding; shuffle MCQ options (models tend to favour early positions)."""
    rng = rng or random.Random(0)
    quiz = Quiz(evidence=evidence)

    for item in output.mcq:
        opts = [o.strip() for o in item.options]
        if not item.question.strip():
            quiz.dropped.append(DroppedItem("mcq", "(empty)", "empty question"))
        elif len(opts) != 4 or len({normalize_for_match(o) for o in opts}) != 4 or any(not o for o in opts):
            quiz.dropped.append(DroppedItem("mcq", item.question, "needs exactly 4 distinct non-empty options"))
        elif not 0 <= item.correct_index < 4:
            quiz.dropped.append(DroppedItem("mcq", item.question, "correct_index out of range"))
        elif (problem := _check_common(item.supporting_quote, item.citation_ids, evidence)):
            quiz.dropped.append(DroppedItem("mcq", item.question, problem))
        else:
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [opts[i] for i in order]
            quiz.mcq.append(item.model_copy(update={
                "options": shuffled, "correct_index": order.index(item.correct_index),
                "citation_ids": [c for c in item.citation_ids if evidence.get(c)],
            }))

    for item in output.short_answer:
        if not item.question.strip() or not item.model_answer.strip():
            quiz.dropped.append(DroppedItem("short_answer", item.question or "(empty)", "empty question or answer"))
        elif (problem := _check_common(item.supporting_quote, item.citation_ids, evidence)):
            quiz.dropped.append(DroppedItem("short_answer", item.question, problem))
        else:
            quiz.short_answer.append(item.model_copy(update={"citation_ids": [c for c in item.citation_ids if evidence.get(c)]}))

    for item in output.flashcards:
        if not item.front.strip() or not item.back.strip():
            quiz.dropped.append(DroppedItem("flashcards", item.front or "(empty)", "empty card side"))
        elif (problem := _check_common(item.supporting_quote, item.citation_ids, evidence)):
            quiz.dropped.append(DroppedItem("flashcards", item.front, problem))
        else:
            quiz.flashcards.append(item.model_copy(update={"citation_ids": [c for c in item.citation_ids if evidence.get(c)]}))
    return quiz


def generate_quiz(
    gemini: GeminiClient, passages: list[Passage], kind: str = "mcq", n: int = 5, difficulty: str = "medium",
    max_chars: int = 9000, seed: int = 0,
) -> Quiz:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if not passages:
        raise BioMentorError("Select at least one source passage (or an uploaded paper) first.")
    n = max(1, min(int(n), 10))
    evidence = EvidenceSet.build([("selected", passages)], max_passages=min(len(passages), 12), max_chars=max_chars)
    output = gemini.generate_structured(QuizOutput, quiz_prompt(evidence, kind, n, difficulty), system_prompt())
    return validate_quiz(output, evidence, random.Random(seed))


# ------------------------------------------------------------------ scoring
@dataclass
class McqResult:
    index: int
    question: str
    chosen: int | None
    correct: int
    is_correct: bool


@dataclass
class QuizScore:
    correct: int
    total: int
    results: list[McqResult]

    @property
    def percent(self) -> float:
        return round(100.0 * self.correct / self.total, 1) if self.total else 0.0

    def incorrect(self) -> list[McqResult]:
        return [r for r in self.results if not r.is_correct]


def score_mcq(items: list[MCQItem], answers: dict[int, int | None]) -> QuizScore:
    """``answers`` maps question index -> chosen option index (``None``/missing = unanswered = wrong)."""
    results = []
    for i, item in enumerate(items):
        chosen = answers.get(i)
        results.append(McqResult(i, item.question, chosen, item.correct_index, chosen == item.correct_index))
    return QuizScore(sum(r.is_correct for r in results), len(items), results)


def short_answer_overlap(user_answer: str, key_points: list[str]) -> float:
    """Heuristic only: fraction of key points whose content words mostly appear in the user's answer."""
    if not key_points:
        return 0.0
    answer_stems = content_stems(user_answer)
    hit = 0
    for point in key_points:
        stems = content_stems(point)
        if stems and len(stems & answer_stems) / len(stems) >= 0.5:
            hit += 1
    return hit / len(key_points)


def quote_in_passage(quote: str, passage: Passage) -> bool:
    """Used by the UI to highlight the supporting quote inside its passage."""
    return bool(re.search(re.escape(normalize_for_match(quote)), normalize_for_match(passage.text)))
