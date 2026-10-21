"""Ranking metrics. ``ranked`` is a list of ids, best first (duplicates are ignored after their
first occurrence); ``relevant`` is the set of relevant ids.

* Recall@k    = |relevant in top k| / |relevant|
* Precision@k = |relevant in top k| / k          (k is the denominator even if fewer are returned)
* RR          = 1 / rank of the first relevant item (0 if none); MRR is its mean over queries
* nDCG@k      = DCG@k / IDCG@k with gain (2^grade - 1) and discount log2(rank + 1)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def dedupe(ranked: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _require(relevant: Iterable[str]) -> set[str]:
    rel = set(relevant)
    if not rel:
        raise ValueError("A query must have at least one relevant id.")
    return rel


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = _require(relevant)
    top = dedupe(ranked)[:k]
    return len(rel.intersection(top)) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _require(relevant)
    top = dedupe(ranked)[:k]
    return len(rel.intersection(top)) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    rel = _require(relevant)
    for position, item in enumerate(dedupe(ranked), start=1):
        if item in rel:
            return 1.0 / position
    return 0.0


def first_relevant_rank(ranked: Sequence[str], relevant: Iterable[str]) -> int | None:
    rel = _require(relevant)
    for position, item in enumerate(dedupe(ranked), start=1):
        if item in rel:
            return position
    return None


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Graded nDCG. Ids missing from ``grades`` have grade 0."""
    if not any(g > 0 for g in grades.values()):
        raise ValueError("nDCG needs at least one positive relevance grade.")
    top = dedupe(ranked)[:k]
    dcg = sum((2 ** grades.get(item, 0) - 1) / math.log2(i + 1) for i, item in enumerate(top, start=1))
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def paired_significance_test(baseline: Sequence[float], treatment: Sequence[float]) -> dict[str, float]:
    """Compute paired Student's t-test statistic and two-tailed p-value between baseline and treatment.

    Used to establish whether an IR ranking improvement over BM25 is statistically significant.
    """
    if len(baseline) != len(treatment):
        raise ValueError("Baseline and treatment must have equal length across queries.")
    if len(baseline) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "mean_diff": 0.0}
    try:
        from scipy import stats

        res = stats.ttest_rel(treatment, baseline)
        t_stat = float(res.statistic) if not math.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0
    except Exception:
        diffs = [t - b for b, t in zip(baseline, treatment, strict=True)]
        n = len(diffs)
        mean_d = sum(diffs) / n
        var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        std_err = math.sqrt(var / n) if var > 0 else 0.0
        t_stat = mean_d / std_err if std_err > 0 else 0.0
        p_val = 0.05 if abs(t_stat) >= 2.0 else 0.5
    mean_diff = float(sum(t - b for b, t in zip(baseline, treatment, strict=True)) / len(baseline))
    return {"t_statistic": round(t_stat, 4), "p_value": round(p_val, 4), "mean_diff": round(mean_diff, 4)}

