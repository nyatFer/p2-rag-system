"""The dense retriever (stage 1): embed every chunk once, embed the query, and rank chunks by cosine similarity.

It is the embedding arm of the 12 lab's retrieve.py (dense_rankings) with one change: it scores
chunks (the [chunking] settings of p2.toml) instead of whole articles, and ChunkScorer lifts them to
documents by each document's best chunk.

The model is the [dense] model of p2.toml, bge-small-en-v1.5 by default. The chunk vectors are
cached in .cache/vectors/ by p2/embed.py, so only the first run encodes the corpus.
"""

from __future__ import annotations

from p2 import embed
from p2.retrievers import ChunkScorer

NEEDS_CLAUDE = False


class Dense(ChunkScorer):
    def __init__(self, corpus, cfg, model: str | None = None):
        super().__init__(corpus, cfg)
        self.model = model or cfg.table("dense").get("model", embed.BGE)
        self.embedder = embed.load(self.model)
        self.vectors = embed.doc_vectors(self.embedder, [c.text for c in self.chunks], cfg.root)

    def score_chunks(self, text: str) -> list[float]:
        if not self.chunks:
            return []
        # Every vector has length 1, so the dot product is the cosine similarity.
        return (self.vectors @ self.embedder.encode([text])[0]).tolist()


def build(corpus, cfg):
    return Dense(corpus, cfg)
