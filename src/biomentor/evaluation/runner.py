"""One function that runs the whole evaluation (used by the CLI script and the Model Evaluation page)."""

from __future__ import annotations

from pathlib import Path

from biomentor.config import Settings
from biomentor.evaluation.benchmark import load_benchmark, validate_against_corpus
from biomentor.evaluation.evaluate_retrieval import run_evaluation
from biomentor.evaluation.results import save_results
from biomentor.exceptions import BenchmarkError, BioMentorError
from biomentor.retrieval.dense_retriever import Embedder
from biomentor.retrieval.index_manager import IndexManager
from biomentor.retrieval.reranker import Reranker

BENCHMARK_NAMES = ("benchmark.jsonl", "demo_benchmark.jsonl")


def find_benchmark(settings: Settings) -> Path | None:
    return next((settings.eval_dir / n for n in BENCHMARK_NAMES if (settings.eval_dir / n).exists()), None)


def run_and_save(
    settings: Settings,
    embedder: Embedder | None,
    benchmark_path: Path | None = None,
    reranker: Reranker | None = None,
    passage_depth: int = 50,
) -> dict:
    """Evaluate BM25 (+ dense/hybrid if ``embedder`` is given, + reranking if ``reranker`` is given).

    Raises :class:`BenchmarkError` / :class:`BioMentorError` with a readable message when there is no
    benchmark, no corpus, or the benchmark's ids do not exist in the corpus."""
    bench = Path(benchmark_path) if benchmark_path else find_benchmark(settings)
    if bench is None:
        raise BenchmarkError("No benchmark found. Create data/evaluation/benchmark.jsonl (see docs/EVALUATION.md) or run scripts/prepare_demo_data.py.")
    records = load_benchmark(bench)
    manager = IndexManager(settings)
    index = manager.load(embedder=embedder, reranker=reranker)  # raises IndexNotFoundError if no corpus
    problems = validate_against_corpus(records, {p.document_id for p in index.passages})
    if problems:
        raise BioMentorError(
            "The benchmark does not match the indexed corpus (" + "; ".join(problems[:3]) + (" ..." if len(problems) > 3 else "") + "). "
            "Rebuild the index with the corpus the benchmark was written for."
        )
    modes = ["bm25"] + (["dense", "hybrid"] if embedder is not None else []) + (["hybrid_rerank"] if reranker is not None and embedder is not None else [])
    config = {
        "embedding_model": embedder.model_name if embedder is not None else None,
        "reranker_model": getattr(reranker, "model_name", None),
        "chunk_config": manager.chunk_config.to_dict(),
        "rrf_c": settings.rrf_c,
        "candidate_k": settings.candidate_k,
        "passage_depth": passage_depth,
        "bm25": {"k1": 1.5, "b": 0.75},
        "vector_backend": None,
    }
    results = run_evaluation(index, records, bench, modes, config, reranker, passage_depth)
    if index._dense is not None:  # noqa: SLF001 - report which backend actually ran
        config["vector_backend"] = index._dense.backend
    save_results(results, settings.eval_dir)
    return results
