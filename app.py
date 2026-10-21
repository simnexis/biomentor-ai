"""BioMentor AI - landing page.   Run with:  streamlit run app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from biomentor.ui import components as ui  # noqa: E402
from biomentor.ui.runtime import get_search_service, get_settings, setup_page  # noqa: E402

setup_page("Home")
settings, svc = get_settings(), get_search_service()

st.markdown(
    '<div class="hero"><h1>Ask a biomedical question. See the evidence.</h1>'
    "<p>BioMentor searches the literature, shows you the exact passages it found, and explains them at your level. "
    "Every statement points back to a source you can read.</p></div>",
    unsafe_allow_html=True,
)

with st.form("home_ask"):
    q = st.text_input("Your question", placeholder="e.g. How does insulin resistance lead to type 2 diabetes?", label_visibility="collapsed")
    go = st.form_submit_button("Search the literature", type="primary")
if go and q.strip():
    st.session_state["de_query"] = q.strip()
    st.session_state["de_autorun"] = True
    st.switch_page("pages/1_Disease_Explorer.py")
elif go:
    ui.notice("Type a question or a topic first.", "info")

st.markdown("**Or start from an example**")
examples = [
    "How does insulin resistance lead to type 2 diabetes?",
    "What is the life cycle of the malaria parasite?",
    "What happens in the airways during an asthma attack?",
    "Why does tuberculosis treatment need several antibiotics?",
]
cols = st.columns(2)
for i, example in enumerate(examples):
    if cols[i % 2].button(example, key=f"ex{i}", use_container_width=True):
        st.session_state["de_query"] = example
        st.session_state["de_autorun"] = True
        st.switch_page("pages/1_Disease_Explorer.py")

st.markdown("## What you can do here")
a, b = st.columns(2)
with a:
    st.page_link("pages/1_Disease_Explorer.py", label="Disease Explorer", icon="🔬")
    st.caption("Explain any topic from the literature, with sources, references and a Markdown export.")
    st.page_link("pages/3_Paper_Explainer.py", label="Paper Explainer", icon="📄")
    st.caption("Upload a PDF, get a summary, and ask questions answered only from that paper.")
    st.page_link("pages/5_Model_Evaluation.py", label="Model Evaluation", icon="📊")
    st.caption("Measure retrieval quality with real metrics. Nothing is shown until you run it.")
with b:
    st.page_link("pages/2_Research_Search.py", label="Research Search", icon="🔎")
    st.caption("Keyword, semantic and hybrid search. Select papers to compare side by side.")
    st.page_link("pages/4_Study_Lab.py", label="Study Lab", icon="🎓")
    st.caption("Quizzes and flashcards built from real passages, each with a supporting quote.")

if not settings.has_gemini:
    ui.notice("No Gemini key is set. Search, key sentences, references and evaluation all work without one. Add GEMINI_API_KEY to also get written explanations.", "info")
if not svc.has_local_corpus():
    ui.notice("No local corpus yet, so searches use live PubMed (needs internet). For offline use run: python scripts/prepare_demo_data.py", "info")

with st.expander("Setup status"):
    st.write(f"- Local corpus prepared: **{'yes' if svc.has_local_corpus() else 'no'}**")
    st.write(f"- Gemini configured: **{'yes' if settings.has_gemini else 'no (optional)'}**")
    st.write(f"- Embedding model: `{settings.embedding_model}` (downloaded on first semantic search, about 100 MB)")
