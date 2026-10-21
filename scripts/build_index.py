"""Process data/raw/*.jsonl into passages and build the search index.

    python scripts/build_index.py                # BM25 + dense (downloads the embedding model on first run)
    python scripts/build_index.py --no-dense     # BM25 only
    python scripts/build_index.py --raw pubmed_corpus.jsonl   # use only this raw file (default: all)
"""

import argparse

from _common import get_embedder, get_settings
from biomentor.data.chunking import ChunkConfig
from biomentor.data.ingestion import find_raw_files, process_corpus
from biomentor.retrieval.index_manager import IndexManager


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--raw", action="append", help="raw file name(s) in data/raw (default: every *.jsonl)")
    args = p.parse_args()

    s = get_settings()
    raw = [s.raw_dir / n for n in args.raw] if args.raw else find_raw_files(s.raw_dir)
    if not raw:
        print("No raw data in data/raw. Run scripts/prepare_demo_data.py or scripts/fetch_literature.py first.")
        return 1
    docs, passages, stats = process_corpus(raw, s.processed_dir, ChunkConfig(s.chunk_max_chars, s.chunk_overlap_sentences))
    print(f"{stats.raw_documents} raw -> {stats.documents} unique documents ({stats.duplicates_removed} duplicates, "
          f"{stats.documents_without_abstract} without abstract) -> {stats.passages} passages [{stats.corpus_kind}]")
    manifest = IndexManager(s).build(passages, len(docs), stats.corpus_kind, embedder=None if args.no_dense else get_embedder(s))
    d = manifest["dense"]
    print("Dense index:", f"built ({d['model_name']}, dim {d['dimension']}, backend {d['backend']})" if d["built"] else f"NOT built - {d['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
