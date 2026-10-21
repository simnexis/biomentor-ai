# Evaluation

## What is measured
Retrieval quality of BM25, dense, hybrid (RRF) and, optionally, hybrid + cross-encoder reranking: Recall@5, Recall@10, Precision@5, MRR, nDCG@10, and per-query latency. All methods use the same benchmark and procedure.

**Procedure** (`src/biomentor/evaluation/evaluate_retrieval.py`): retrieve the top 50 passages → collapse to documents (a document's rank is its best passage's rank) → score against human relevance labels → average over queries. A warm-up query runs first so one-off model-loading time is excluded from latency. Methods that cannot run are recorded as *skipped with a reason*; no numbers are ever filled in.

## Benchmark format
`data/evaluation/benchmark.jsonl`, one record per line: `question_id, query, topic, relevant_document_ids, relevant_passage_ids (optional), relevance_grades (optional: 1 partial, 2 high), annotation_notes, provenance`.

## Building a real benchmark (human-labelled)
1. Fetch and index a PubMed corpus (README).
2. Write `data/evaluation/queries.txt`, one `topic | question` per line (aim for 30+ questions across several topics).
3. `python scripts/make_benchmark.py candidates --queries data/evaluation/queries.txt` pools the top results of BM25, dense and hybrid into `annotation_sheet.csv`.
4. Read each title and abstract and fill `relevance` with 0 / 1 / 2. Judge relevance to the *question*, not topical similarity.
5. `python scripts/make_benchmark.py compile --annotator "Your Name"` writes `benchmark.jsonl`.
6. `python scripts/run_evaluation.py --rerank`.

Pooling means every method's picks were judged, which avoids rewarding only the system that produced the candidates, but documents no system retrieved are unjudged (counted as non-relevant).

## Current state of results in this repository
- **Real-literature benchmark: none yet.** As required by scientific integrity, relevance labels must be produced by human review of retrieved records (workflow in step 1-6 above). No fabricated real-world relevance labels or scores are claimed.
- **Synthetic demo benchmark (12 documents, 14 queries):** Evaluated for real across all 4 methods using `scripts/run_evaluation.py --rerank`:
  - **Keyword (BM25):** Recall@5: 1.000, Recall@10: 1.000, Precision@5: 0.229, MRR: 0.917, nDCG@10: 0.938, Latency: 0.1 ms/q
  - **Semantic (embeddings, `all-MiniLM-L6-v2`):** Recall@5: 1.000, Recall@10: 1.000, Precision@5: 0.229, MRR: 1.000, nDCG@10: 1.000, Latency: 17.7 ms/q
  - **Hybrid (BM25 + semantic, RRF):** Recall@5: 1.000, Recall@10: 1.000, Precision@5: 0.229, MRR: 0.964, nDCG@10: 0.974, Latency: 0.6 ms/q
  - **Hybrid + cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`):** Recall@5: 1.000, Recall@10: 1.000, Precision@5: 0.229, MRR: 1.000, nDCG@10: 1.000, Latency: 471.2 ms/q
  *Note: Because this fixture has only 12 documents, "top 5" covers 40% of the corpus; these numbers demonstrate code correctness and pipeline latency rather than real-world retrieval power.*

## Limits of a small benchmark
- With 30 queries, one query moves a mean by ~3 points; differences of a few points between methods are noise. Report per-query results and, ideally, bootstrap confidence intervals.
- Labels reflect one annotator's judgement.
- Tuning `k1`, `b`, `c`, chunk size on the same benchmark you report overfits it; hold out queries.
- A domain-specific embedding model is *not* assumed to be better - compare it on your benchmark.

## Retrieval metrics vs. scientific correctness
Retrieval metrics only say whether labelled documents were found. They do not tell you whether a generated answer is true. Two further layers exist:
1. **Automatic checks** - citation ids exist; word-overlap warning; quote-verbatim check for quiz items.
2. **Human review** - the Disease Explorer page exports every statement with its cited passages to `data/evaluation/answer_reviews/*.csv`. A reviewer fills `citation_valid`, `claim_support` (supported / partially / unsupported / not_checkable), `uncertainty_ok`; `python scripts/review_answers.py <csv>` summarises them. Only this step measures scientific correctness, and only as well as the reviewer's expertise.
