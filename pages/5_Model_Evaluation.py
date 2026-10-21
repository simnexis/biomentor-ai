"""Model Evaluation - shows saved results of scripts/run_evaluation.py. Never fabricates numbers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from biomentor.evaluation.evaluate_retrieval import METRIC_KEYS  # noqa: E402
from biomentor.evaluation.results import (  # noqa: E402
    aggregate_frame,
    load_results,
    query_frame,
    significance_frame,
    skipped_methods,
)
from biomentor.evaluation.runner import find_benchmark, run_and_save
from biomentor.exceptions import BioMentorError
from biomentor.ui import components as ui  # noqa: E402
from biomentor.ui.runtime import get_embedder, get_reranker, get_settings, setup_page, show_error  # noqa: E402

setup_page("Model Evaluation")
st.title("Model Evaluation")
settings = get_settings()

bench_path = find_benchmark(settings)
with st.container(border=True):
    st.markdown("**Run the evaluation here**")
    if bench_path is None:
        st.caption("No benchmark found. Create data/evaluation/benchmark.jsonl (see docs/EVALUATION.md) or run python scripts/prepare_demo_data.py for the synthetic demo benchmark.")
    else:
        st.caption(f"Benchmark: {bench_path.name}. BM25 always runs. Semantic and hybrid also run when the embedding model is available (first run downloads it).")
        with_rerank = st.checkbox("Include Cross-Encoder Reranking (ms-marco-MiniLM)", value=True)
        if st.button("Run evaluation now", type="primary"):
            try:
                with st.spinner("Evaluating retrieval methods..."):
                    run_and_save(settings, get_embedder(), reranker=get_reranker() if with_rerank else None)
                st.success("Done. Results are shown below.")
            except BioMentorError as exc:
                show_error(exc)
results = load_results(settings.eval_dir)

if results is None:
    st.subheader("Not evaluated yet.")
    st.write("No evaluation results exist. Use the button above, or run python scripts/run_evaluation.py. A built index and a benchmark are required.")
    st.stop()

bench, corpus, cfg = results["benchmark"], results["corpus"], results["config"]
if bench.get("synthetic"):
    ui.notice("This run used the SYNTHETIC demo benchmark on SYNTHETIC fixtures. It only demonstrates that the evaluation code works; "
            "with so few documents the scores are not evidence of real-world retrieval quality.")

st.caption(f"Evaluated at {results['evaluated_at']} · metrics at {results['metric_level']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Benchmark queries", bench["n_queries"])
c2.metric("Corpus documents", corpus["n_documents"])
c3.metric("Corpus passages", corpus["n_passages"])
c4.metric("Graded relevance", "yes" if bench.get("graded") else "no (binary)")

st.subheader("Aggregate results")
df = aggregate_frame(results)
if df.empty:
    st.warning("No retrieval method completed in this run.")
else:
    st.dataframe(df.set_index("Method").style.format({**{k: "{:.3f}" for k in METRIC_KEYS}, "Latency mean (ms)": "{:.1f}", "Latency p95 (ms)": "{:.1f}"}))
    metric = st.selectbox("Chart metric", METRIC_KEYS, index=3)
    st.bar_chart(df.set_index("Method")[metric])
    sig_df = significance_frame(results, baseline_mode="bm25", metric=metric)
    if not sig_df.empty:
        st.markdown(f"**Statistical Significance vs. BM25 Baseline ({metric.upper()}, Paired t-test)**")
        st.dataframe(sig_df.set_index("Method"))
    import json
    st.download_button(
        "Download Full Evaluation Report (JSON)",
        data=json.dumps(results, indent=2),
        file_name="biomentor_retrieval_evaluation.json",
        mime="application/json",
    )
for label, reason in skipped_methods(results).items():
    st.info(f"**{label}** was not evaluated: {reason}")

with st.expander("Configuration"):
    st.json({"benchmark": bench, "corpus": corpus, "config": cfg})

st.subheader("Query-level results")
modes = [m for m, v in results["methods"].items() if v.get("status") == "ok"]
if modes:
    mode = st.selectbox("Method", modes, format_func=lambda m: results["methods"][m]["label"])
    st.dataframe(query_frame(results, mode).drop(columns=["relevant_documents"], errors="ignore"))

    st.subheader("Error analysis")
    misses = results["methods"][mode].get("error_analysis", [])
    if not misses:
        st.write("Every query had a relevant document in the top 5 for this method.")
    for q in misses:
        rank = q["first_relevant_rank"]
        st.markdown(f"- **{q['query']}** - first relevant document at rank {rank if rank else 'not retrieved'}. Relevant: `{', '.join(q['relevant_documents'])}`; top-5 returned: `{', '.join(q['top5_documents'])}`")

st.divider()
st.caption("A small benchmark has wide uncertainty: one query can move a mean by several points. Automated retrieval metrics measure whether labelled documents were found - "
           "not whether a generated answer is scientifically correct. See docs/EVALUATION.md and the answer-review workflow.")
