"""Study Lab - quizzes and flashcards grounded in selected passages or an uploaded paper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from biomentor.exceptions import BioMentorError  # noqa: E402
from biomentor.services.paper_service import select_context_passages  # noqa: E402
from biomentor.services.quiz_service import (  # noqa: E402
    DIFFICULTIES,
    generate_quiz,
    score_mcq,
    short_answer_overlap,
)
from biomentor.ui import components as ui  # noqa: E402
from biomentor.ui.runtime import (  # noqa: E402
    get_gemini,
    get_search_service,
    get_settings,
    setup_page,
    show_error,
)

setup_page("Study Lab")
st.markdown('<span class="tech-tag">Active Recall Engine • Verified Citation Grounding</span>', unsafe_allow_html=True)
st.title("Study Lab & Active Recall")
st.caption("Questions are built only from passages you choose, and every item must include a quote that appears word-for-word in its source.")
settings, svc = get_settings(), get_search_service()

# ------------------------------------------------------------------ choose the source material
options = {}
last = st.session_state.get("last_results")
if last:
    options["Results of my last Research Search"] = "search"
if st.session_state.get("paper_ctx") is not None:
    options["My uploaded paper"] = "paper"
if svc.has_local_corpus():
    options["Search the local corpus for a topic"] = "topic"

if not options:
    st.info("Nothing to study from yet. Run a Research Search, upload a paper, or prepare a local corpus first.")
    st.stop()

choice = st.radio("Source material", list(options), horizontal=False)
kind_of_source = options[choice]
passages, consent_needed = [], False

if kind_of_source == "search":
    passages = [r.passage for r in last]
    st.caption(f"{len(passages)} passages from your last search.")
elif kind_of_source == "paper":
    ctx = st.session_state["paper_ctx"]
    passages, truncated = select_context_passages(ctx.passages, 9000, ctx.paper.abstract)
    consent_needed = True
    st.caption(f"Uploaded paper: {ctx.paper.title_guess or ctx.paper.filename} ({len(passages)} passages{' - a sample, the paper is long' if truncated else ''}).")
else:
    topic = st.text_input("Topic", placeholder="e.g. malaria")
    if topic.strip():
        try:
            resp = svc.search(topic, mode="bm25", top_k=6, source="local")
            passages = [r.passage for r in resp.results]
            st.caption(f"{len(passages)} passages found." + (" (SYNTHETIC demo fixtures)" if resp.source_kind == "demo_fixtures" else ""))
        except BioMentorError as exc:
            show_error(exc)

c1, c2, c3 = st.columns(3)
kind = c1.selectbox("Question type", ["mcq", "short_answer", "flashcards"], format_func=lambda k: {"mcq": "Multiple choice", "short_answer": "Short answer", "flashcards": "Flashcards"}[k])
difficulty = c2.selectbox("Difficulty", DIFFICULTIES, index=1)
n = c3.number_input("How many", 1, 10, 5)

if passages:
    with st.expander("Source passages that will be used"):
        for p in passages[:12]:
            st.markdown(f'<div class="bm-snippet">{ui.esc(p.text)}<div class="bm-meta">{ui.esc(p.title)} · <code>{ui.esc(p.passage_id)}</code></div></div>', unsafe_allow_html=True)

if not settings.has_gemini:
    st.info("Generating questions needs a Gemini API key. The source passages above remain available for reading.")
    st.stop()

consent = True
if consent_needed:
    consent = st.checkbox("I understand the paper's text will be sent to Google Gemini to generate questions.")

if st.button("Generate", type="primary", disabled=not passages or not consent):
    try:
        with st.spinner("Writing questions from your sources…"):
            st.session_state["quiz"] = generate_quiz(get_gemini(), passages, kind, int(n), difficulty, max_chars=9000)
            st.session_state["quiz_kind"] = kind
            st.session_state.pop("quiz_scored", None)
            st.session_state["quiz_nonce"] = st.session_state.get("quiz_nonce", 0) + 1
    except BioMentorError as exc:
        show_error(exc)

quiz = st.session_state.get("quiz")
if quiz is None:
    st.stop()

if quiz.dropped:
    with st.expander(f"{len(quiz.dropped)} generated item(s) were rejected by validation"):
        for d in quiz.dropped:
            st.write(f"- {d.question[:120]} - *{d.reason}*")
if quiz.is_empty():
    st.warning("No valid questions could be produced from these passages. Try other passages or a different type.")
    st.stop()


def source_box(item, reveal_key):
    labels = item.citation_ids
    st.markdown(f"Supporting quote: *“{item.supporting_quote}”*")
    for label in labels:
        ev = quiz.evidence.get(label)
        if ev:
            st.caption(f"Source [{label}]: {ev.passage.title or ev.passage.passage_id} · {ev.describe()} · `{ev.passage.passage_id}`")


nonce = st.session_state.get("quiz_nonce", 0)
if quiz.mcq and st.session_state.get("quiz_kind") == "mcq":
    st.subheader("Multiple choice")
    scored = st.session_state.get("quiz_scored")
    with st.form(f"mcq_{nonce}"):
        for i, item in enumerate(quiz.mcq):
            st.markdown(f"**{i + 1}. {item.question}**")
            st.radio("Answer", item.options, index=None, key=f"mcq_{nonce}_{i}", label_visibility="collapsed")
        submit = st.form_submit_button("Submit answers", type="primary")
    if submit:
        answers = {}
        for i, item in enumerate(quiz.mcq):
            chosen = st.session_state.get(f"mcq_{nonce}_{i}")
            answers[i] = item.options.index(chosen) if chosen in item.options else None
        st.session_state["quiz_scored"] = score_mcq(quiz.mcq, answers)
        scored = st.session_state["quiz_scored"]
    if scored:
        st.success(f"Score: {scored.correct}/{scored.total} ({scored.percent}%)")
        wrong = scored.incorrect()
        if wrong:
            st.subheader("Review of incorrect or unanswered questions")
            for r in wrong:
                item = quiz.mcq[r.index]
                with st.expander(f"{r.index + 1}. {item.question}", expanded=True):
                    st.write("Your answer: " + (item.options[r.chosen] if r.chosen is not None else "*(no answer)*"))
                    st.write(f"Correct answer: **{item.options[r.correct]}**")
                    st.write(item.explanation)
                    source_box(item, r.index)
        with st.expander("Explanations for all questions"):
            for i, item in enumerate(quiz.mcq):
                st.markdown(f"**{i + 1}.** {item.options[item.correct_index]} - {item.explanation}")
                source_box(item, i)

if quiz.short_answer and st.session_state.get("quiz_kind") == "short_answer":
    st.subheader("Short answer")
    for i, item in enumerate(quiz.short_answer):
        st.markdown(f"**{i + 1}. {item.question}**")
        mine = st.text_area("Your answer", key=f"sa_{nonce}_{i}", height=80)
        if st.checkbox("Show model answer", key=f"sa_show_{nonce}_{i}"):
            st.write(item.model_answer)
            st.markdown("Key points: " + "; ".join(item.key_points))
            if mine.strip():
                st.caption(f"Rough word-overlap with the key points: {short_answer_overlap(mine, item.key_points):.0%} (a heuristic - judge for yourself).")
            source_box(item, i)

if quiz.flashcards and st.session_state.get("quiz_kind") == "flashcards":
    st.subheader("Flashcards")
    for i, item in enumerate(quiz.flashcards):
        with st.container(border=True):
            st.markdown(f"**{item.front}**")
            if st.toggle("Flip", key=f"fc_{nonce}_{i}"):
                st.write(item.back)
                source_box(item, i)
