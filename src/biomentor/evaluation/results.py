"""Persist and reload evaluation results (JSON for the full record, CSVs for tables)."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from biomentor.data.storage import read_json, write_json
from biomentor.evaluation.evaluate_retrieval import METRIC_KEYS

RESULTS_FILE = "latest_results.json"


def results_dir(eval_dir: Path) -> Path:
    return Path(eval_dir) / "results"


def save_results(results: dict, eval_dir: Path) -> Path:
    out = results_dir(eval_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / RESULTS_FILE, results)

    rows = [
        {"method": m["label"], "mode": mode, **{k: round(m["aggregate"][k], 4) for k in METRIC_KEYS},
         "latency_ms_mean": round(m["aggregate"]["latency_ms_mean"], 2)}
        for mode, m in results["methods"].items() if m.get("status") == "ok"
    ]
    if rows:
        with open(out / "aggregate_metrics.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    query_rows = [
        {"mode": mode, **{k: v for k, v in q.items() if k not in ("top5_documents", "relevant_documents")},
         "top5_documents": "; ".join(q["top5_documents"]), "relevant_documents": "; ".join(q["relevant_documents"])}
        for mode, m in results["methods"].items() if m.get("status") == "ok" for q in m["queries"]
    ]
    if query_rows:
        pd.DataFrame(query_rows).to_csv(out / "query_level_metrics.csv", index=False)
    return out / RESULTS_FILE


def load_results(eval_dir: Path) -> dict | None:
    """``None`` if no evaluation has been run (the UI then shows 'Not evaluated yet')."""
    path = results_dir(eval_dir) / RESULTS_FILE
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "methods" in data else None


def aggregate_frame(results: dict) -> pd.DataFrame:
    rows = []
    for m in results["methods"].values():
        if m.get("status") == "ok":
            rows.append({"Method": m["label"], **{k: m["aggregate"][k] for k in METRIC_KEYS},
                         "Latency mean (ms)": m["aggregate"]["latency_ms_mean"], "Latency p95 (ms)": m["aggregate"]["latency_ms_p95"]})
    return pd.DataFrame(rows)


def query_frame(results: dict, mode: str) -> pd.DataFrame:
    m = results["methods"].get(mode, {})
    if m.get("status") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame(m["queries"])
    df["top5_documents"] = df["top5_documents"].map(lambda x: ", ".join(x))
    df["relevant_documents"] = df["relevant_documents"].map(lambda x: ", ".join(x))
    return df


def skipped_methods(results: dict) -> dict[str, str]:
    return {m["label"]: m.get("reason", "") for m in results["methods"].values() if m.get("status") != "ok"}


def significance_frame(results: dict, baseline_mode: str = "bm25", metric: str = "mrr") -> pd.DataFrame:
    """Compare each retrieval method against baseline with paired Student's t-test p-values."""
    from biomentor.evaluation.metrics import paired_significance_test

    methods = results.get("methods", {})
    if baseline_mode not in methods or methods[baseline_mode].get("status") != "ok":
        return pd.DataFrame()
    base_queries = {q["question_id"]: q[metric] for q in methods[baseline_mode].get("queries", [])}
    rows = []
    for mode, m in methods.items():
        if mode == baseline_mode or m.get("status") != "ok":
            continue
        treat_queries = {q["question_id"]: q[metric] for q in m.get("queries", [])}
        common_ids = [qid for qid in base_queries if qid in treat_queries]
        if not common_ids:
            continue
        base_vals = [base_queries[qid] for qid in common_ids]
        treat_vals = [treat_queries[qid] for qid in common_ids]
        sig = paired_significance_test(base_vals, treat_vals)
        sig_label = "Significant (p < 0.05)" if sig["p_value"] < 0.05 else "p >= 0.05"
        rows.append({
            "Method": m["label"],
            "Metric": metric.upper(),
            "Baseline (BM25)": round(sum(base_vals) / len(base_vals), 3),
            "Method Mean": round(sum(treat_vals) / len(treat_vals), 3),
            "Δ Mean": round(sig["mean_diff"], 3),
            "t-stat": sig["t_statistic"],
            "p-value": sig["p_value"],
            "Significance": sig_label,
        })
    return pd.DataFrame(rows)

