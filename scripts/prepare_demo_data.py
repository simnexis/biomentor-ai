"""Write the SYNTHETIC demo corpus + benchmark into data/ (offline, no network).

    python scripts/prepare_demo_data.py            # writes raw docs, processed passages, demo benchmark
    python scripts/prepare_demo_data.py --no-index # skip index build

The demo documents are labelled "[Synthetic fixture]" and are NOT real publications.
"""

import argparse

from _common import get_embedder, get_settings
from biomentor.data.chunking import ChunkConfig
from biomentor.data.demo_fixtures import demo_benchmark_records, demo_documents
from biomentor.data.ingestion import process_corpus
from biomentor.data.storage import save_documents
from biomentor.evaluation.benchmark import save_benchmark
from biomentor.retrieval.index_manager import IndexManager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-index", action="store_true", help="only write corpus files; skip the index manifest")
    parser.add_argument("--dense", action="store_true", help="also embed passages now (downloads the embedding model)")
    args = parser.parse_args()

    s = get_settings()
    raw_path = s.raw_dir / "demo_synthetic.jsonl"
    save_documents(raw_path, demo_documents())
    docs, passages, stats = process_corpus([raw_path], s.processed_dir, ChunkConfig(s.chunk_max_chars, s.chunk_overlap_sentences))
    bench_path = s.eval_dir / "demo_benchmark.jsonl"
    n = save_benchmark(bench_path, demo_benchmark_records())
    print(f"Wrote {len(docs)} synthetic documents -> {len(passages)} passages ({raw_path.name}); {n} synthetic benchmark queries -> {bench_path}")
    if not args.no_index:
        manifest = IndexManager(s).build(passages, len(docs), stats.corpus_kind, embedder=get_embedder(s) if args.dense else None)
        print(f"Index manifest written (dense built: {manifest['dense']['built']}).")
    print("NOTE: this is synthetic demo data, not real literature. Use scripts/fetch_literature.py for PubMed records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
