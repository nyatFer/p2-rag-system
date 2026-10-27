"""Reciprocal rank fusion, shared by hybrid and its variants.

It is rrf() from the 13 lab's pipeline.py: each item scores the sum of 1 / (k + rank) over the
rankings, with k = 60 and ranks from 1. It fuses full rankings, not top-k lists, because an item
that is 11th in both lists can belong in the fused top 10.
"""

from __future__ import annotations

from p2 import retrievers
from p2.runfile import order

K = 60


def rrf(rankings, k: int = K) -> list[tuple[str, float]]:
    """Fuse rankings (lists of (id, score), best first) into one, best first, ties in id order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (item, _score) in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return order(scores.items())


class Fusion:
    """The RRF fusion of named systems, at the document level (search) and the chunk level (search_chunks)."""

    def __init__(self, names, corpus, cfg, k: int = K):
        self.systems = [retrievers.build(name, corpus, cfg) for name in names]
        self.n_docs = len(corpus.docs)
        self.k = k

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        return rrf([s.search(text, self.n_docs) for s in self.systems], self.k)[:k]

    def search_chunks(self, text: str, k: int) -> list[tuple[str, float]]:
        return rrf([s.search_chunks(text, len(s.chunks)) for s in self.systems], self.k)[:k]
