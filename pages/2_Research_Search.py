"""Research Search - keyword / semantic / hybrid retrieval, source inspection, paper comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from biomentor.data.chunking import ChunkConfig, chunk_document  # noqa: E402
from biomentor.exceptions import BioMentorError  # noqa: E402
from biomentor.retrieval.index_manager import MODE_LABELS  # noqa: E402
from biomentor.services.comparison_service import FIELDS, compare_papers  # noqa: E402
from biomentor.services.research_service import group_by_document  # noqa: E402
from biomentor.ui import components as ui  # noqa: E402
from biomentor.ui.runtime import (  # noqa: E402
    get_gemini,
    get_reranker,
    get_search_service,
    get_settings,
    setup_page,
    show_error,
)

setup_page("Research Search")
st.title("Research Search")
settings, svc = get_settings(), get_search_service()

with st.form("search"):
    query = st.text_input("Search biomedical literature", placeholder="e.g. gut microbiome and obesity")
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    source = c1.radio("Source", ["auto", "local", "live"], format_func=lambda x: {"auto": "Automatic (live, else local)", "local": "Local indexed corpus", "live": "Live PubMed (NCBI)"}[x])
    mode = c2.selectbox("Method", ["hybrid", "bm25", "dense", "hybrid_rerank"], format_func=lambda m: MODE_LABELS[m],
                        help="Cross-encoder reranking is optional and downloads a second model on first use.")
    top_k = c3.number_input("Results", 1, 50, 10)
    live_n = c4.number_input("PubMed fetch", 10, 100, 30, step=10, help="Records fetched from PubMed when the source is live.")
    go = st.form_submit_button("Search", type="primary")

if go:
    if source == "local" and not svc.has_local_corpus():
        st.warning("No local corpus yet. Run `python scripts/prepare_demo_data.py` (synthetic demo) or `python scripts/fetch_literature.py --query ...` then `python scripts/build_index.py`, or use live PubMed.")
    else:
        if mode == "hybrid_rerank":
            svc.reranker = get_reranker()
        else:
            svc.reranker = None
        try:
            with st.spinner("Searching…"):
                st.session_state["rs_resp"] = svc.search(query, mode=mode, top_k=int(top_k), source=source, live_fetch=int(live_n))
                st.session_state["rs_docs_index"] = None
        except BioMentorError as exc:
            st.session_state.pop("rs_resp", None)
            show_error(exc)

resp = st.session_state.get("rs_resp")
if resp is None:
    ui.empty_state("No search yet", "Run a search to see ranked passages with their sources.")
    st.stop()

ui.provenance_rail(resp)

if not resp.results:
    st.warning("No passages matched. Try different keywords, or another method.")
    st.stop()

# Keep passages of the displayed results available to the Study Lab / comparison.
st.session_state["last_results"] = resp.results
st.session_state["last_documents"] = resp.documents
if resp.source_kind == "live_pubmed":
    with st.expander("Keep these records"):
        st.caption("Save the fetched PubMed records into your local corpus (data/raw/pubmed_live.jsonl). Rebuild the index to make them searchable offline.")
        c1, c2 = st.columns(2)
        if c1.button("Save records to local corpus"):
            n = svc.save_live_documents(list(resp.documents.values()))
            st.success(f"Added {n} new record(s).")
        if c2.button("Rebuild local index"):
            with st.spinner("Rebuilding (the embedding model downloads on first use)…"):
                try:
                    stats, manifest = svc.rebuild_local_index()
                    st.success(f"{stats.documents} documents, {stats.passages} passages. Dense index built: {manifest['dense']['built']}.")
                    if manifest["dense"]["error"] and not manifest["dense"]["built"]:
                        st.warning(manifest["dense"]["error"])
                except BioMentorError as exc:
                    show_error(exc)

if "compare_ids" not in st.session_state:
    st.session_state["compare_ids"] = {}

st.subheader(f"{len(resp.results)} passages from {len(group_by_document(resp.results))} documents")
for doc_id, hits in group_by_document(resp.results):
    doc = resp.documents.get(doc_id)
    ui.render_document_card(doc, hits, show_abstract=True, query=resp.query)
    if doc and doc.abstract:
        checked = st.checkbox("Select for comparison", key=f"cmp_{doc_id}", value=doc_id in st.session_state["compare_ids"])
        if checked:
            st.session_state["compare_ids"][doc_id] = doc
        else:
            st.session_state["compare_ids"].pop(doc_id, None)

# ------------------------------------------------------------------ comparison
selected = list(st.session_state["compare_ids"].values())
st.divider()
st.subheader(f"Compare papers ({len(selected)} selected)")
st.caption("Abstract-level comparison to help you read - not a systematic review or evidence grading.")
if len(selected) < 2:
    st.write("Select at least two papers above.")
elif not settings.has_gemini:
    st.info("Comparison needs a Gemini API key (add GEMINI_API_KEY to .env or .streamlit/secrets.toml).")
elif st.button("Compare selected papers", type="primary"):
    # Chunk each selected paper's whole abstract (selections can outlive the current result list).
    cfg = ChunkConfig(settings.chunk_max_chars, settings.chunk_overlap_sentences)
    passages_by_doc = {doc.document_id: chunk_document(doc, cfg) for doc in selected}
    try:
        with st.spinner("Comparing…"):
            st.session_state["cmp_result"] = compare_papers(get_gemini(), settings, selected, passages_by_doc)
    except BioMentorError as exc:
        show_error(exc)

cmp = st.session_state.get("cmp_result")
if cmp is not None:
    import pandas as pd

    rows = []
    for row in cmp.table:
        flat = {"Paper": row["Paper"], "Publication date": row["Publication date"]}
        for _, label in FIELDS:
            text, cites = row[label]
            flat[label] = text + (f" [{', '.join(cites)}]" if cites else "")
        rows.append(flat)
    st.dataframe(pd.DataFrame(rows).set_index("Paper").T)
    for title, claims in (("Where the papers agree", cmp.agreements), ("Apparent disagreements", cmp.disagreements)):
        if claims:
            st.markdown(f"**{title}**")
            ui.render_claims(claims)
    for c in cmp.caveats:
        st.caption(f"Caveat: {c}")
    if cmp.unverified_numbers:
        ui.warn("Some sample sizes could not be verified in the cited text and were replaced with 'Not reported': " + "; ".join(cmp.unverified_numbers))
    ui.render_citation_report(cmp.report)
    ui.render_evidence(cmp.evidence, resp.documents, "Evidence behind the comparison")
