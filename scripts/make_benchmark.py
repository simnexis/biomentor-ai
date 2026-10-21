"""Build a REAL benchmark over your fetched corpus with human labels (pooled annotation).

    # 1) pool candidates from BM25/dense/hybrid for your queries -> a CSV to label
    python scripts/make_benchmark.py candidates --queries data/evaluation/queries.txt
    #    queries.txt: one "topic | question" per line

    # 2) open data/evaluation/annotation_sheet.csv, read each title/abstract, fill `relevance`:
    #    0 = not relevant, 1 = partially, 2 = highly relevant

    # 3) compile the labelled sheet into data/evaluation/benchmark.jsonl
    python scripts/make_benchmark.py compile --annotator "your name"
"""

import argparse
from datetime import date

from _common import get_embedder, get_settings
from biomentor.data.storage import load_documents
from biomentor.evaluation.benchmark import (
    compile_annotations,
    save_benchmark,
    write_annotation_sheet,
)
from biomentor.retrieval.index_manager import IndexManager


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--queries", required=True)
    c.add_argument("--top-k", type=int, default=10)
    c.add_argument("--no-dense", action="store_true")
    k = sub.add_parser("compile")
    k.add_argument("--annotator", required=True, help="who judged relevance (recorded as provenance)")
    k.add_argument("--sheet")
    args = p.parse_args()

    s = get_settings()
    sheet = s.eval_dir / "annotation_sheet.csv"
    if args.cmd == "candidates":
        queries = []
        with open(args.queries, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    topic, _, q = line.partition("|")
                    queries.append((topic.strip(), q.strip()) if q else ("general", topic.strip()))
        queries = [(q, t) for t, q in queries]
        index = IndexManager(s).load(embedder=None if args.no_dense else get_embedder(s))
        docs = {d.document_id: d for d in load_documents(s.processed_dir / "documents.jsonl")}
        n = write_annotation_sheet(sheet, queries, index, docs, args.top_k)
        print(f"Wrote {n} candidate rows for {len(queries)} queries -> {sheet}\nNow fill the 'relevance' column (0/1/2), then run: make_benchmark.py compile")
        return 0

    path = args.sheet or sheet
    prov = f"Human-annotated by {args.annotator} on {date.today().isoformat()} using pooled candidates from BM25/dense/hybrid over the local PubMed corpus."
    records, warnings = compile_annotations(path, prov)
    for w in warnings:
        print("warning:", w)
    out = s.eval_dir / "benchmark.jsonl"
    save_benchmark(out, records)
    print(f"Wrote {len(records)} benchmark records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
