import random

from biomentor.exceptions import BioMentorError
from biomentor.generation.evidence import EvidenceSet
from biomentor.generation.gemini_client import GeminiClient
from biomentor.schemas import FlashcardItem, MCQItem, QuizOutput, ShortAnswerItem
from biomentor.services.quiz_service import (
    generate_quiz,
    score_mcq,
    short_answer_overlap,
    validate_quiz,
)
from conftest import FakeBackend

QUOTE = "Insulin lowers blood glucose".lower()


def evidence(passages):
    ps = [p for p in passages if p.document_id == "syn-t2d-01"]
    return EvidenceSet.build([("selected", ps)], max_passages=3)


def mcq(**over):
    base = dict(question="What happens when cells respond poorly to insulin?",
                options=["Less glucose is taken up from the blood", "More glucose is taken up", "Insulin is destroyed", "Nothing changes"],
                correct_index=0, explanation="Cells take up less glucose.",
                supporting_quote="less glucose is taken up from the blood", citation_ids=["E1"])
    base.update(over)
    return MCQItem(**base)


def test_valid_mcq_kept_options_shuffled_and_correct_index_follows(passages):
    quiz = validate_quiz(QuizOutput(mcq=[mcq()], short_answer=[], flashcards=[]), evidence(passages), random.Random(3))
    item = quiz.mcq[0]
    assert len(quiz.mcq) == 1 and not quiz.dropped
    assert item.options[item.correct_index] == "Less glucose is taken up from the blood"
    assert sorted(item.options) == sorted(mcq().options)


def test_quote_not_in_source_is_dropped(passages):
    quiz = validate_quiz(QuizOutput(mcq=[mcq(supporting_quote="Gene therapy cures diabetes completely")], short_answer=[], flashcards=[]), evidence(passages))
    assert quiz.is_empty() and "verbatim" in quiz.dropped[0].reason


def test_quote_is_matched_case_and_whitespace_insensitively(passages):
    quiz = validate_quiz(QuizOutput(mcq=[mcq(supporting_quote="LESS  glucose is taken up\nfrom the blood")], short_answer=[], flashcards=[]), evidence(passages))
    assert len(quiz.mcq) == 1


def test_invalid_citation_or_tiny_quote_dropped(passages):
    ev = evidence(passages)
    for bad in (mcq(citation_ids=["E9"]), mcq(supporting_quote="glucose")):
        assert validate_quiz(QuizOutput(mcq=[bad], short_answer=[], flashcards=[]), ev).is_empty()


def test_structural_problems_dropped(passages):
    ev = evidence(passages)
    cases = [mcq(options=["a", "b", "c"]), mcq(options=["a", "a", "b", "c"]), mcq(correct_index=4), mcq(correct_index=-1), mcq(question="  ")]
    quiz = validate_quiz(QuizOutput(mcq=cases, short_answer=[], flashcards=[]), ev)
    assert quiz.is_empty() and len(quiz.dropped) == len(cases)


def test_short_answer_and_flashcards_validated(passages):
    ev = evidence(passages)
    sa_ok = ShortAnswerItem(question="Why does glucose rise?", model_answer="Cells take up less.", key_points=["less uptake"],
                            supporting_quote="the liver continues to release glucose", citation_ids=["E1"])
    sa_bad = sa_ok.model_copy(update={"supporting_quote": "invented sentence about nothing at all"})
    fc_ok = FlashcardItem(front="Beta cells", back="Secrete insulin", supporting_quote="pancreatic beta cells to secrete enough insulin", citation_ids=["E1"])
    fc_bad = fc_ok.model_copy(update={"back": " "})
    quiz = validate_quiz(QuizOutput(mcq=[], short_answer=[sa_ok, sa_bad], flashcards=[fc_ok, fc_bad]), ev)
    assert len(quiz.short_answer) == 1 and len(quiz.flashcards) == 1 and len(quiz.dropped) == 2


def test_scoring_counts_unanswered_as_wrong():
    items = [mcq(), mcq(question="Second?")]
    score = score_mcq(items, {0: 0})  # q0 correct (index 0), q1 unanswered
    assert (score.correct, score.total, score.percent) == (1, 2, 50.0)
    assert [r.index for r in score.incorrect()] == [1]
    wrong = score_mcq(items, {0: 2, 1: 0})
    assert wrong.correct == 1 and wrong.incorrect()[0].chosen == 2


def test_scoring_empty_quiz():
    s = score_mcq([], {})
    assert (s.correct, s.total, s.percent) == (0, 0, 0.0)


def test_short_answer_overlap_heuristic():
    pts = ["cells take up less glucose", "liver releases glucose"]
    assert short_answer_overlap("Cells take up less glucose and the liver releases glucose", pts) == 1.0
    assert short_answer_overlap("Bananas are yellow", pts) == 0.0 and short_answer_overlap("x", []) == 0.0


def test_generate_quiz_end_to_end_with_scripted_backend(passages):
    ps = [p for p in passages if p.document_id == "syn-t2d-01"]
    payload = QuizOutput(mcq=[mcq()], short_answer=[], flashcards=[]).model_dump_json()
    client = GeminiClient("k", "m", backend=FakeBackend(payload), sleep=lambda s: None)
    quiz = generate_quiz(client, ps, "mcq", 3, "easy")
    assert len(quiz.mcq) == 1 and quiz.evidence.labels == ["E1"]


def test_generate_quiz_guards(passages):
    client = GeminiClient("k", "m", backend=FakeBackend())
    for args in (([], "mcq", 3, "easy"), (passages, "bogus", 3, "easy"), (passages, "mcq", 3, "impossible")):
        try:
            generate_quiz(client, *args)
        except (BioMentorError, ValueError):
            continue
        raise AssertionError(args[1:])
    no_key = GeminiClient(None, "m")
    try:
        generate_quiz(no_key, passages[:1], "mcq", 1, "easy")
    except BioMentorError as exc:
        assert "Gemini is not configured" in str(exc)
    else:
        raise AssertionError
