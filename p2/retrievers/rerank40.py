"""The ablation: rerank with Claude seeing hybrid's top 40 instead of its top 20; everything else as in rerank."""

from __future__ import annotations

from p2.retrievers.rerank import Rerank

NEEDS_CLAUDE = True
BASE = "rerank"
CANDIDATES = 40


class Rerank40(Rerank):
    def __init__(self, corpus, cfg):
        super().__init__(corpus, cfg)
        self.candidates = CANDIDATES


def build(corpus, cfg):
    return Rerank40(corpus, cfg)
