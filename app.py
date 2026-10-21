"""BioMentor AI - landing page.   Run with:  streamlit run app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from biomentor.evaluation.results import aggregate_frame, load_results  # noqa: E402
from biomentor.ui import components as ui  # noqa: E402
from biomentor.ui.runtime import get_search_service, get_settings, setup_page  # noqa: E402

setup_page("Home")
settings, svc = get_settings(), get_search_service()

st.markdown('<span class="tech-tag">Biomedical RAG Platform • NCBI PubMed Integrated</span>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>Evidence-Grounded Biomedical AI</h1>'
    "<p>Dual-stage hybrid retrieval combining lexical BM25 with dense semantic vector embeddings and "
    "neural cross-encoder reranking. Synthesizes clinical explanations with 100% verified citation attribution.</p></div>",
    unsafe_allow_html=True,
)

with st.form("home_ask"):
    q = st.text_input("Your question", placeholder="e.g. How does insulin resistance lead to type 2 diabetes?", label_visibility="collapsed")
    go = st.form_submit_button("Search Evidence", type="primary")
if go and q.strip():
    st.session_state["de_query"] = q.strip()
    st.session_state["de_autorun"] = True
    st.switch_page("pages/1_Disease_Explorer.py")
elif go:
    ui.notice("Type a question or select a benchmark example below.", "info")

st.markdown("**Benchmark Case Studies**")
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

st.markdown("## Platform Capabilities")
a, b = st.columns(2)
with a:
    st.page_link("pages/1_Disease_Explorer.py", label="Disease Explorer")
    st.caption("Literature-grounded medical synthesis, structured clinical summaries, and evidence cards.")
    st.page_link("pages/3_Paper_Explainer.py", label="Paper Explainer")
    st.caption("Upload PDF papers, extract in-memory passages, and question-answer with verbatim quote validation.")
    st.page_link("pages/5_Model_Evaluation.py", label="Model Evaluation")
    st.caption("Comprehensive retrieval benchmarking with paired statistical tests (Wilcoxon) and bootstrap CIs.")
with b:
    st.page_link("pages/2_Research_Search.py", label="Research Search")
    st.caption("Compare BM25, dense semantic, hybrid RRF, and cross-encoder neural reranked retrieval.")
    st.page_link("pages/4_Study_Lab.py", label="Study Lab")
    st.caption("Active-recall clinical quizzes and flashcards grounded by exact word-for-word citations.")

# Architecture & Live Benchmark Highlights
with st.expander("System Architecture & Engineering Specifications", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**1. Lexical Search**")
        st.caption("BM25 ($k_1=1.5, b=0.75$)\\n\\nBiomedical tokenization with hyphen preservation (e.g. `SARS-CoV-2`).")
    with c2:
        st.markdown("**2. Dense Semantic**")
        st.caption("Bi-Encoder (`all-MiniLM-L6-v2`)\\n\\n384-dim normalized embeddings with fast inner-product search.")
    with c3:
        st.markdown("**3. Rank Fusion**")
        st.caption("Reciprocal Rank Fusion\\n\\n$RRF(d) = \\sum \\frac{1}{60 + r(d)}$\\n\\nCombines disparate score distributions.")
    with c4:
        st.markdown("**4. Neural Reranker**")
        st.caption("Cross-Encoder (`ms-marco-MiniLM`)\\n\\nFull self-attention between query and candidate passages.")
eval_results = load_results(settings.eval_dir)
if eval_results:
    st.markdown("## Live Retrieval Benchmark")
    df_agg = aggregate_frame(eval_results)
    rename_map = {
        "recall_at_5": "Recall@5",
        "recall_at_10": "Recall@10",
        "precision_at_5": "Precision@5",
        "mrr": "MRR",
        "ndcg_at_10": "nDCG@10",
        "Latency mean (ms)": "Latency (ms)",
    }
    df_display = df_agg.rename(columns=rename_map)
    display_cols = ["Method", "Recall@5", "Recall@10", "MRR", "nDCG@10", "Latency (ms)"]
    available_cols = [c for c in display_cols if c in df_display.columns]
    st.dataframe(df_display[available_cols], hide_index=True, use_container_width=True)

if not settings.has_gemini:
    ui.notice("No Gemini API key set. Lexical, dense, hybrid retrieval, references, and benchmarks run locally. Add GEMINI_API_KEY in the sidebar to activate LLM explanations.", "info")
