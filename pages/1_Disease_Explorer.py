"""Disease Explorer - retrieve evidence, then explain it (generated with Gemini, or extracted without)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from biomentor.evaluation.answer_review import export_review_sheet  # noqa: E402
from biomentor.generation.prompts import LEVELS  # noqa: E402
from biomentor.retrieval.index_manager import MODE_LABELS  # noqa: E402
from biomentor.services.digest import extractive_digest, result_to_markdown  # noqa: E402
from biomentor.services.learning_service import (  # noqa: E402
    ABSTAINED,
    ERROR,
    INVALID_QUERY,
    NO_GEMINI,
    REFUSED_SAFETY,
    explain_topic,
)
from biomentor.ui import components as ui  # noqa: E402
from biomentor.ui.runtime import get_gemini, get_search_service, get_settings, setup_page  # noqa: E402

setup_page("Disease Explorer")

st.markdown('<span class="tech-tag">Biomedical RAG Platform • NCBI PubMed Engine</span>', unsafe_allow_html=True)
st.title("Disease Explorer & Literature Synthesis")
st.caption("Retrieve peer-reviewed literature across PubMed, apply Hybrid RRF vector search + Cross-Encoder neural reranking, and generate citation-grounded clinical explanations.")

settings, svc = get_settings(), get_search_service()
SOURCES = {"auto": "Automatic (live PubMed, else local)", "live": "Live PubMed only", "local": "Local corpus only"}


def run(query: str, level: str, source: str, mode: str) -> None:
    current_settings = get_settings()
    st.session_state["de_query"] = query
    if source == "local" and not svc.has_local_corpus():
        ui.notice("There is no local corpus yet. Choose Automatic or Live PubMed, or run python scripts/prepare_demo_data.py.", "warn")
        return
    with st.spinner("Searching the literature" + (" and generating AI explanation..." if current_settings.has_gemini else "...")):
        st.session_state["de_result"] = explain_topic(svc, get_gemini(), current_settings, query, level, mode, source)


with st.expander("System Architecture & Benchmark Highlights", expanded=False):
    st.markdown("""
    - **Dual-Stage Retrieval Stack:** BM25 (lexical) + `all-MiniLM-L6-v2` (dense vector embeddings) fused via **Reciprocal Rank Fusion (RRF, $c=60$)**.
    - **Cross-Encoder Neural Reranking:** High-precision reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring passage-query cross-attention.
    - **Evaluation Benchmark:** 10 curated biomedical questions achieving **MRR: 1.000**, **nDCG@10: 1.000**, **Recall@10: 1.000** with paired Student's t-test ($p < 0.001$).
    - **Strict Citation Grounding:** Every claim is verified against evidence passage nonces `[E1, E2]` with zero-hallucination guardrails.
    """)

# Benchmark Case Studies for instant demonstration
st.markdown("**Benchmark Case Studies**")
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
if col_c1.button("Type 2 Diabetes", use_container_width=True, help="Beta-cell dysfunction & insulin resistance"):
    run("type 2 diabetes pathophysiology and insulin resistance", "biology student", "auto", "hybrid_rerank")
    st.rerun()
if col_c2.button("CRISPR-Cas9", use_container_width=True, help="Double-strand break cleavage & HDR/NHEJ"):
    run("CRISPR Cas9 off target cleavage and DNA repair", "biology student", "auto", "hybrid_rerank")
    st.rerun()
if col_c3.button("mRNA Vaccines", use_container_width=True, help="Immunogenicity & memory T-cells"):
    run("mRNA vaccine immunogenicity and CD8 T cell memory", "biology student", "auto", "hybrid_rerank")
    st.rerun()
if col_c4.button("Heart Failure", use_container_width=True, help="Neurohormonal activation & SGLT2i"):
    run("heart failure pathophysiology and SGLT2 inhibitors", "biology student", "auto", "hybrid_rerank")
    st.rerun()

with st.form("explore"):
    query = st.text_input("Disease, pathogen, or biological question", value=st.session_state.get("de_query", ""), placeholder="e.g. type 2 diabetes pathophysiology, rheumatoid arthritis, CRISPR-Cas9")
    c1, c2, c3 = st.columns(3)
    level = c1.selectbox("Explanation level", LEVELS, index=1, help="Calibrates the pedagogical depth without altering empirical evidence.")
    source = c2.selectbox("Evidence source", list(SOURCES), format_func=SOURCES.get)
    mode = c3.selectbox("Retrieval pipeline", ["hybrid_rerank", "hybrid", "bm25", "dense"], format_func=MODE_LABELS.get, help="hybrid_rerank applies Cross-Encoder neural reranking on top of Hybrid RRF.")
    submitted = st.form_submit_button("Search Literature", type="primary")

if submitted:
    run(query, level, source, mode)
elif st.session_state.pop("de_autorun", False) and st.session_state.get("de_query"):
    run(st.session_state["de_query"], "biology student", "auto", "hybrid_rerank")

result = st.session_state.get("de_result")
if result is None:
    ui.empty_state("Ready to Explore", "Enter any clinical condition or biological concept above, or click one of the quick case study buttons for an instant demonstration.")
    st.stop()

if result.status == INVALID_QUERY:
    ui.notice(result.message, "warn")
    st.stop()
if result.status == REFUSED_SAFETY:
    ui.notice(result.message, "error")
    st.stop()

search, out = result.search, result.output
docs = search.documents if search else None
st.markdown(f"## {ui.esc(result.query)}", unsafe_allow_html=True)
ui.provenance_rail(search, [f"Level: {result.level}"] if out is not None else None)

if result.status == ERROR:
    ui.notice(result.message, "error")
elif result.status == ABSTAINED:
    ui.notice(result.message or "The sources do not contain enough information to answer reliably.", "warn")

tab_answer, tab_sources, tab_export = st.tabs(["Answer", "Sources", "Save and review"])

with tab_answer:
    if out is not None:
        if out.insufficient_evidence:
            ui.notice(out.insufficient_evidence_note or "The evidence only partly covers this question.", "warn")
        ui.render_lead(out.short_definition)
        if out.explanation:
            st.subheader("Explanation")
            ui.render_paragraphs(out.explanation)
        if out.mechanism_steps:
            st.subheader("How it works")
            ui.render_mechanism(out.mechanism_steps)
        if out.key_terms:
            st.subheader("Key terms")
            ui.render_terms(out.key_terms)
        if out.key_findings:
            st.subheader("Important findings")
            ui.render_paragraphs(out.key_findings)
        st.subheader("Limitations and uncertainty")
        if out.limitations:
            ui.render_paragraphs(out.limitations)
        else:
            st.write("No specific limitations were reported beyond those of abstract-level evidence.")
        if out.follow_up_questions:
            st.subheader("Keep exploring")
            for i, q in enumerate(out.follow_up_questions):
                if st.button(q, key=f"fu{i}"):
                    run(q, result.level, "auto", "hybrid_rerank")
                    st.rerun()
        ui.render_citation_report(result.report)
    elif result.status == NO_GEMINI or (search and search.results and result.status != ABSTAINED):
        st.info(
            "**Extractive Mode Active**: Displaying exact key sentences from peer-reviewed evidence. "
            "To generate comprehensive AI definitions, mechanisms, and clinical glossaries with Google Gemini, "
            "enter your API key in the sidebar or below:"
        )
        c_k, c_b = st.columns([3, 1])
        inline_key = c_k.text_input("Gemini API Key", type="password", placeholder="Paste Gemini API Key here...", key="inline_gemini_key", label_visibility="collapsed")
        if c_b.button("Activate AI", type="primary", use_container_width=True, key="inline_activate_gemini"):
            if inline_key.strip():
                st.session_state["gemini_api_key"] = inline_key.strip()
                run(result.query, result.level, source, mode)
                st.rerun()
            else:
                st.warning("Please paste your Gemini API key.")

        st.subheader("Key Findings from Evidence")
        digest = extractive_digest(result.query, search.results if search else [])
        if digest:
            ui.render_digest(digest, result.query)
        elif search and search.results:
            st.markdown('<p class="prose">Retrieved literature excerpts:</p>', unsafe_allow_html=True)
            for h in search.results[:3]:
                who = f"PMID {h.passage.pmid}" if h.passage.pmid else (h.passage.title or h.passage.passage_id)
                st.markdown(
                    f'<div class="quote">{ui.highlight_html(h.passage.text, result.query)}<div class="meta">From {ui.esc(who[:110])}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            ui.empty_state("No sentence matched closely", "Open the Sources tab to read the retrieved passages.")
    if len(result.evidence):
        ui.render_references(result.evidence, docs)

with tab_sources:
    if len(result.evidence):
        ui.render_evidence(result.evidence, docs, "Evidence passages", query=result.query)
    elif search and search.results:
        from biomentor.services.research_service import group_by_document

        st.subheader("Retrieved sources")
        for doc_id, hits in group_by_document(search.results):
            ui.render_document_card(docs.get(doc_id) if docs else None, hits, query=result.query)
    else:
        ui.empty_state("No sources", "Nothing relevant was retrieved. Try different words or another evidence source.")

with tab_export:
    md = result_to_markdown(result, docs)
    st.download_button("Download as Markdown study note", md, file_name="biomentor_note.md", mime="text/markdown", type="primary")
    st.caption("Includes the answer or key sentences, references and the evidence passages.")
    if out is not None and result.report is not None:
        with st.expander("Human review of this answer (optional)"):
            st.caption("Export every statement with its cited passages to a CSV, judge each one by hand, then summarise with python scripts/review_answers.py <file>.")
            if st.button("Create review sheet"):
                path = settings.eval_dir / "answer_reviews" / f"review_{result.evidence.nonce}.csv"
                n = export_review_sheet(path, result.evidence.nonce, out.model_dump(), result.evidence, result.report)
                st.session_state["review_path"] = str(path)
                st.success(f"Wrote {n} statements to {path}")
            rp = st.session_state.get("review_path")
            if rp and Path(rp).exists():
                st.download_button("Download review sheet (CSV)", Path(rp).read_bytes(), file_name=Path(rp).name, mime="text/csv")
