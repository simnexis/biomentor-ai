# Architecture

BioMentor is a single Python process: Streamlit renders pages and calls service functions directly. There is no API server and no database server; state lives in local files.

```
app.py, pages/*.py                 UI only: layout, widgets, rendering
src/biomentor/ui/                  cached resources (settings, embedder, search service), shared rendering
src/biomentor/services/            use-cases: search, explain, paper Q&A, comparison, quiz
src/biomentor/generation/          evidence set, prompts, Gemini client, parser, citation validator, abstention
src/biomentor/retrieval/           BM25, dense (+FAISS), RRF hybrid, reranker, index manager
src/biomentor/data/                PubMed client, preprocessing, chunking, ingestion, storage, demo fixtures
src/biomentor/evaluation/          metrics, benchmark, experiment runner, results, answer review
scripts/                           command-line pipeline (fetch → build index → evaluate)
```

## Request flow: "Search and explain"

1. `validate_query` - non-empty, bounded length, at least one content term.
2. `check_query_safety` - personal diagnosis/treatment patterns are declined before any retrieval or LLM call.
3. `SearchService.search` - local index or live PubMed (live results are chunked and indexed in memory); falls back from hybrid/dense to BM25 with a visible warning if the embedding model is unavailable.
4. `assess_evidence` - abstains if the retrieved passages cover too little of the query's key terms.
5. `EvidenceSet.build` - de-duplicates and bounds passages by count and characters, assigns labels `E1…En`, keeps metadata.
6. Prompt = rules (system) + labelled evidence fenced with a random per-request marker (retrieved text is *data*, not instructions).
7. `GeminiClient.generate_structured` - JSON constrained by a Pydantic schema, validated; retry policy per error class.
8. `validate_citations` - every cited label must exist in the evidence set (invalid ones are removed and reported); evidence-based statements without a valid citation are flagged; a separate word-overlap heuristic warns about weak support.
9. UI shows the answer with citations, a citation report, and the evidence passages with source links.

Without a Gemini key the flow stops after step 5 and shows ranked sources.

## Design decisions

- **Document-level and passage-level IDs are stable** so citations, benchmarks and indexes can be tied together.
- **Index compatibility** (embedding model, prefixes, chunking, corpus fingerprint, vector count) is checked on load; mismatches raise a clear error and the app falls back to BM25.
- **Dense vectors are always saved as `vectors.npy`**; `faiss.index` is an optional accelerator. Without FAISS a numpy exact search gives identical results (and is fast at this scale).
- **Gemini failures never remove functionality**: the evidence stays on screen.
- **Uploads are processed in memory**, so there is nothing to clean up on disk.
