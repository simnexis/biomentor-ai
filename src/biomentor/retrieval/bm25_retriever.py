"""Okapi BM25, implemented directly with an inverted index (transparent and dependency-light).

For a query term ``q`` and passage ``d``::

    score(d, q) = IDF(q) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    IDF(q)      = ln( (N - n_q + 0.5) / (n_q + 0.5) + 1 )

``tf`` is the term's count in the passage, ``|d|`` the passage length in tokens, ``N`` the number of
passages and ``n_q`` the number containing the term. This is the same formula family as the
``rank-bm25`` package's BM25Okapi, with the always-positive "+1" IDF used by Lucene/Elasticsearch.
Defaults ``k1=1.5, b=0.75``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from biomentor.data.preprocessing import tokenize
from biomentor.schemas import Passage, RetrievedPassage


class BM25Retriever:
    name = "bm25"

    def __init__(self, passages: Sequence[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = list(passages)
        self.k1, self.b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for idx, passage in enumerate(self.passages):
            tokens = tokenize(passage.retrieval_text())
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((idx, tf))
        self._lengths = np.asarray(lengths, dtype=np.float64)
        n = len(self.passages)
        self._avgdl = float(self._lengths.mean()) if n else 0.0
        self._idf = {
            term: math.log((n - len(post) + 0.5) / (len(post) + 0.5) + 1.0) for term, post in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self.passages)

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.passages), dtype=np.float64)
        if not self.passages or self._avgdl == 0:
            return scores
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for idx, tf in postings:
                norm = self.k1 * (1.0 - self.b + self.b * self._lengths[idx] / self._avgdl)
                scores[idx] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return scores

    def search(self, query: str, k: int = 10) -> list[RetrievedPassage]:
        """Top-``k`` passages with score > 0 (empty query or corpus -> ``[]``). Ties keep corpus order."""
        if k <= 0 or not query or not query.strip() or not self.passages:
            return []
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")
        results: list[RetrievedPassage] = []
        for idx in order[:k]:
            if scores[idx] <= 0:
                break
            results.append(
                RetrievedPassage(
                    passage=self.passages[int(idx)],
                    score=float(scores[idx]),
                    rank=len(results) + 1,
                    method=self.name,
                    component_ranks={self.name: len(results) + 1},
                    component_scores={self.name: float(scores[idx])},
                )
            )
        return results
