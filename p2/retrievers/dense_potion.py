"""dense with potion-retrieval-32M instead of the [dense] model: the second embedder for hybrid_dd."""

from __future__ import annotations

from p2 import embed
from p2.retrievers.dense import Dense

NEEDS_CLAUDE = False
BASE = "dense"


def build(corpus, cfg):
    return Dense(corpus, cfg, model=embed.POTION)
