"""Evaluate BM25 / dense / hybrid (/ hybrid+rerank) on a benchmark and save the results.

    python scripts/run_evaluation.py                                  # data/evaluation/benchmark.jsonl, else demo_benchmark.jsonl
    python scripts/run_evaluation.py --benchmark data/evaluation/benchmark.jsonl --rerank

Results go to data/evaluation/results/ and appear on the "Model Evaluation" page. Methods that
cannot run (e.g. no embedding model offline) are recorded as skipped - no numbers are invented.
"""

import argparse
from pathlib import Path

from _common import get_embedder, get_settings
from biomentor.evaluation.runner import find_benchmark, run_and_save
from biomentor.exceptions import BioMentorError
from biomentor.retrieval.reranker import CrossEncoderReranker


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark")
    p.add_argument("--no-dense", action="store_true", help="skip dense/hybrid (BM25 only)")
    p.add_argument("--rerank", action="store_true", help="also evaluate hybrid + cross-encoder reranking")
    p.add_argument("--passage-depth", type=int, default=50)
    args = p.parse_args()

    s = get_settings()
    bench = Path(args.benchmark) if args.benchmark else find_benchmark(s)
    try:
        results = run_and_save(
            s,
            None if args.no_dense else get_embedder(s),
            bench,
            CrossEncoderReranker(s.reranker_model) if args.rerank and not args.no_dense else None,
            args.passage_depth,
        )
    except BioMentorError as exc:
        print(exc)
        return 1

    print(f"Benchmark: {bench.name} ({results['benchmark']['n_queries']} queries){'  [SYNTHETIC - smoke test only]' if results['benchmark']['synthetic'] else ''}")
    print(f"Corpus: {results['corpus']['n_documents']} documents, {results['corpus']['n_passages']} passages\n")
    print(f"{'method':34s} {'R@5':>6s} {'R@10':>6s} {'P@5':>6s} {'MRR':>6s} {'nDCG@10':>8s} {'ms/q':>7s}")
    for m in results["methods"].values():
        if m["status"] != "ok":
            print(f"{m['label']:34s} skipped: {m['reason'][:90]}")
            continue
        a = m["aggregate"]
        print(f"{m['label']:34s} {a['recall@5']:6.3f} {a['recall@10']:6.3f} {a['precision@5']:6.3f} {a['mrr']:6.3f} {a['ndcg@10']:8.3f} {a['latency_ms_mean']:7.1f}")
    print(f"\nSaved to {s.eval_dir / 'results' / 'latest_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
