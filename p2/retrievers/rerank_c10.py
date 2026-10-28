"""Stretch option 3, the cheaper variant: rerank with Claude seeing hybrid's top 10 instead of its top 20; everything else as in rerank."""

from __future__ import annotations

from p2.retrievers.rerank import Rerank

NEEDS_CLAUDE = True
BASE = "rerank"
CANDIDATES = 10


class RerankC10(Rerank):
    def __init__(self, corpus, cfg):
        super().__init__(corpus, cfg)
        self.candidates = CANDIDATES


def build(corpus, cfg):
    return RerankC10(corpus, cfg)
