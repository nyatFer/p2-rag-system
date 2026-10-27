"""hybrid with two embedders instead of bm25 and an embedder: RRF of dense (bge-small) and dense_potion.

The alternative fusion the brief allows, built to compare against hybrid on the practice queries.
"""

from __future__ import annotations

from p2.retrievers._rrf import Fusion

NEEDS_CLAUDE = False
BASE = "hybrid"


def build(corpus, cfg):
    return Fusion(["dense", "dense_potion"], corpus, cfg)
