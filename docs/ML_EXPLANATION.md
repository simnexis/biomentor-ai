# The ML and NLP behind BioMentor

Written for someone who knows basic machine learning.

## Tokenisation and chunking
Text is lower-cased and split into word tokens; stop-words are dropped. Hyphenated terms are kept whole *and* split ("SARS-CoV-2" → `sars-cov-2`, `sars`, `cov`) so exact and partial queries both match. Abstracts are cut into **sentence-aware chunks** (default 800 characters, 1 sentence overlap). Smaller chunks give precise evidence but less context; larger ones give context but blurrier embeddings (MiniLM truncates long inputs); overlap protects facts that straddle a boundary at the cost of some duplication. Each chunk stores exact character offsets into the normalised source, so a citation is a literal quote.

## BM25 (keyword retrieval)
A bag-of-words ranking function. For each query term it multiplies **IDF** (rare terms matter more) by a **saturating term-frequency** factor (the 10th occurrence adds less than the 2nd) normalised by passage length:

`score = Σ IDF(q) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`, with `k1=1.5`, `b=0.75`.

Strength: exact terms, gene names, acronyms. Weakness: no understanding of synonyms ("heart attack" vs "myocardial infarction").

## Embeddings, cosine similarity (semantic retrieval)
A Sentence-Transformer maps text to a vector so that texts with similar meaning are close. We L2-normalise every vector, so the dot product equals **cosine similarity** (the cosine of the angle between vectors, in [−1, 1]). Strength: paraphrases. Weakness: can miss exact rare terms and returns *something* for every query, so a score is not a probability of relevance.

## FAISS
A library for nearest-neighbour search over vectors. We use `IndexFlatIP` (exact inner-product search, no approximation). At thousands of passages exact search is instant; approximate indexes (IVF, HNSW) matter at millions. If FAISS is not installed, a numpy matrix product gives the same ranking.

## Hybrid retrieval and Reciprocal Rank Fusion
Keyword and semantic search fail in different ways, so we combine them. BM25 scores are unbounded and cosine scores live in [−1, 1]; adding them would let one scale dominate. **RRF** uses only ranks:

`RRF(d) = Σ over systems 1 / (c + rank_system(d))`, `c = 60`

A passage ranked highly by both systems wins; one found by only one system still counts. `c` softens the gap between rank 1 and rank 2. We keep each system's rank and score for inspection in the UI.

## Optional cross-encoder reranking
A bi-encoder embeds query and passage separately; a **cross-encoder** reads them together and scores relevance directly - usually more accurate, but too slow for a whole corpus, so it re-orders only the top candidates. Whether it helps *this* corpus is measured, not assumed.

## RAG (retrieval-augmented generation)
The LLM answers from supplied passages instead of memory. BioMentor adds: bounded context, labelled passages, a prompt that separates findings from interpretation, structured JSON output validated with Pydantic, and abstention when evidence is thin.

## Citation validation vs. claim support
Two different things:
1. **Citation validity** (deterministic): does `E3` exist in this answer's evidence? Invalid ids are removed.
2. **Claim support** (hard): is the statement actually true according to the passage? We compute only a rough **word-overlap heuristic** as a warning flag. It cannot detect negation or subtle distortion. Real verification is human review (exported CSV workflow).

## Retrieval metrics
- **Recall@k** - fraction of relevant documents found in the top k.
- **Precision@k** - fraction of the top k that are relevant.
- **MRR** - mean of 1/rank of the first relevant result.
- **nDCG@k** - rewards putting highly relevant results higher, using graded labels and a logarithmic position discount.
