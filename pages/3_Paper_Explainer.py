"""Paper Explainer - upload a PDF, read it, and ask questions (uploaded evidence kept separate)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from biomentor.exceptions import BioMentorError, InvalidPDFError, NoExtractableTextError  # noqa: E402
from biomentor.generation.prompts import LEVELS  # noqa: E402
from biomentor.services import paper_service as ps  # noqa: E402
from biomentor.ui import components as ui  # noqa: E402
from biomentor.ui.runtime import (  # noqa: E402
    get_embedder,
    get_gemini,
    get_search_service,
    get_settings,
    setup_page,
    show_error,
)

setup_page("Paper Explainer")
st.title("Paper Explainer")
settings, svc = get_settings(), get_search_service()

st.caption(
    f"PDFs up to {settings.max_pdf_mb} MB. The file is read in memory and is not saved to disk; it disappears when you close the session or press 'Clear paper'."
)
upload = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if upload is not None:
    data = upload.getvalue()
    key = (upload.name, len(data))
    if st.session_state.get("paper_key") != key:
        for k in ("paper_ctx", "paper_summary", "paper_answer", "paper_sel", "paper_external", "paper_error"):
            st.session_state.pop(k, None)
        st.session_state["paper_key"] = key
        try:
            with st.spinner("Extracting text…"):
                parsed = ps.extract_pdf(data, upload.name, settings.max_pdf_mb, settings.max_pdf_pages)
                st.session_state["paper_ctx"] = ps.prepare_paper(parsed, settings, get_embedder())
        except NoExtractableTextError as exc:
            st.session_state["paper_error"] = ("warning", str(exc))
        except InvalidPDFError as exc:
            st.session_state["paper_error"] = ("error", str(exc))

err = st.session_state.get("paper_error")
if err:
    (st.warning if err[0] == "warning" else st.error)(err[1])
ctx = st.session_state.get("paper_ctx")

if st.session_state.get("paper_ctx") is not None and st.button("Clear paper"):
    for k in ("paper_ctx", "paper_key", "paper_summary", "paper_answer", "paper_sel", "paper_external", "paper_error"):
        st.session_state.pop(k, None)
    st.rerun()

if ctx is None:
    if not err:
        st.info("Upload a PDF to start. Scanned PDFs (images of text) can't be read without OCR.")
    st.stop()

paper = ctx.paper
st.success(f"**{paper.title_guess or paper.filename}** - {paper.n_pages} pages, {len(paper.text):,} characters, {len(ctx.passages)} passages"
           + (f" (only the first {paper.pages_read} pages were read)" if paper.truncated else ""))

with st.expander("Extracted text preview", expanded=False):
    st.text(paper.text[:4000] + ("\n…" if len(paper.text) > 4000 else ""))
if paper.abstract:
    with st.expander("Abstract (detected)"):
        st.write(paper.abstract)
else:
    st.caption("No clearly labelled abstract was detected, so none is shown.")

# ------------------------------------------------------------------ consent + settings
st.subheader("Explain with Gemini")
if not settings.has_gemini:
    st.info("Explanations, summaries and Q&A need a Gemini API key. The extracted text above stays available without one.")
    st.stop()

level = st.selectbox("Explanation level", LEVELS, index=1)
consent = st.checkbox(
    "I understand that this paper's text (or excerpts of it) will be sent to Google Gemini when I click a button below.",
    help="Nothing is sent until you click a generate button and tick this box.",
)
if not consent:
    st.caption("Tick the box above to enable the buttons.")
gemini = get_gemini()


def run(label, fn, state_key):
    try:
        with st.spinner(label):
            st.session_state[state_key] = fn()
    except BioMentorError as exc:
        show_error(exc)


tab_sum, tab_sel, tab_qa = st.tabs(["Summary", "Explain a passage", "Ask a question"])

with tab_sum:
    if st.button("Summarise this paper", disabled=not consent):
        run("Summarising…", lambda: ps.summarize_paper(ctx, gemini, settings, level, consent), "paper_summary")
    res = st.session_state.get("paper_summary")
    if res:
        grounded, excerpted = res
        out = grounded.output
        if excerpted:
            ui.warn("The paper is longer than the context budget, so a representative sample of passages was used - the summary may miss sections.")
        if out.insufficient_evidence:
            ui.warn(out.insufficient_evidence_note or "Some parts could not be supported by the text.")
        if out.abstract_explanation:
            st.markdown("**Abstract, explained**")
            ui.render_paragraphs(out.abstract_explanation)
        st.markdown("**Summary**")
        ui.render_paragraphs(out.summary)
        for title, claims in (("Important findings", out.key_findings), ("Limitations (as supported by the text)", out.limitations)):
            if claims:
                st.markdown(f"**{title}**")
                ui.render_claims(claims)
        if out.key_terms:
            st.markdown("**Terminology**")
            ui.render_terms(out.key_terms)
        ui.render_citation_report(grounded.report)
        ui.render_evidence(grounded.evidence, None, "Passages from your paper that were used")

with tab_sel:
    selected_text = st.text_area("Paste a passage from the paper", height=140, placeholder="Copy a sentence or paragraph from the paper…")
    if st.button("Explain this passage", disabled=not consent or not selected_text.strip()):
        run("Explaining…", lambda: ps.explain_selection(ctx, gemini, level, selected_text, consent), "paper_sel")
    res = st.session_state.get("paper_sel")
    if res:
        ui.render_paragraphs(res.output.answer)
        ui.render_terms(res.output.key_terms)
        ui.render_citation_report(res.report)
        ui.render_evidence(res.evidence, None, "Context used")

with tab_qa:
    question = st.text_input("Ask about this paper", placeholder="What was the main finding? What are the limitations?")
    use_external = st.checkbox("Also bring in external literature (kept separate from the paper)", help="Searches the local corpus or live PubMed and labels those passages as 'external'.")
    source = st.radio("External source", ["local", "live"], horizontal=True, disabled=not use_external,
                      format_func=lambda x: {"local": "Local corpus", "live": "Live PubMed"}[x]) if use_external else "local"
    if st.button("Answer", disabled=not consent or not question.strip()):
        def _answer():
            external = None
            if use_external:
                external = [r for r in svc.search(question, mode="hybrid", top_k=4, source=source).results]
            return ps.ask_paper(ctx, gemini, level, question, consent, external=external)
        run("Thinking…", _answer, "paper_answer")
    res = st.session_state.get("paper_answer")
    if res:
        if res.output.insufficient_evidence:
            ui.warn(res.output.insufficient_evidence_note or "The paper does not appear to answer this.")
        ui.render_paragraphs(res.output.answer)
        if res.output.limitations:
            st.markdown("**Limitations**")
            ui.render_claims(res.output.limitations)
        ui.render_citation_report(res.report)
        uploaded = [i for i in res.evidence if i.origin == "uploaded"]
        external = [i for i in res.evidence if i.origin == "external"]
        for title, items in (("From your uploaded paper", uploaded), ("From external literature", external)):
            if items:
                st.markdown(f"**{title}**")
                for item in items:
                    with st.expander(f"[{item.label}] {item.passage.title[:80] or item.passage.passage_id}"):
                        st.write(item.passage.text)
                        st.caption(item.describe())
