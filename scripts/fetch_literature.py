"""Fetch real PubMed records (title, abstract, metadata) via NCBI E-utilities into data/raw/.

    python scripts/fetch_literature.py --query "type 2 diabetes insulin resistance" --max 40
    python scripts/fetch_literature.py --query-file my_queries.txt --max 30 --output pubmed_corpus.jsonl

Set NCBI_EMAIL (and optionally NCBI_API_KEY) in .env. Requests are rate-limited and cached.
Only titles/abstracts and metadata are stored - not full texts.
"""

import argparse

from _common import get_settings
from biomentor.data.ingestion import append_raw_documents
from biomentor.data.pubmed_client import PubMedClient
from biomentor.exceptions import PubMedError


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", action="append", default=[], help="PubMed query (repeatable)")
    p.add_argument("--query-file", help="text file with one query per line")
    p.add_argument("--max", type=int, default=30, help="max records per query (<= 500)")
    p.add_argument("--output", default="pubmed_corpus.jsonl", help="file name inside data/raw/")
    args = p.parse_args()

    queries = list(args.query)
    if args.query_file:
        with open(args.query_file, encoding="utf-8") as fh:
            queries += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if not queries:
        p.error("provide --query or --query-file")

    s = get_settings()
    if not s.ncbi_email:
        print("Tip: set NCBI_EMAIL in .env - NCBI asks API users to identify themselves.")
    client = PubMedClient(s.ncbi_email, s.ncbi_api_key, s.ncbi_tool, cache_dir=s.cache_dir)
    out = s.raw_dir / args.output
    failures = 0
    for q in queries:
        try:
            docs = client.search(q, args.max)
        except PubMedError as exc:
            print(f"[FAILED] {q!r}: {exc}")
            failures += 1
            continue
        added = append_raw_documents(out, docs)
        with_abs = sum(1 for d in docs if d.abstract)
        print(f"{q!r}: {len(docs)} records ({with_abs} with abstracts), {added} new -> {out.name}")
    print(f"Done. {client.network_requests} network requests, {client.cache_hits} cache hits. Next: python scripts/build_index.py")
    return 1 if failures == len(queries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
