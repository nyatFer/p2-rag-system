"""The hybrid retriever (stage 1): reciprocal rank fusion of bm25 and dense.

bm25 matches exact words and codes such as 75.403, and dense matches meaning, so on the shared
practice queries each one finds what the other misses. The fusion is in _rrf.py (k = 60).
"""

from __future__ import annotations

from p2.retrievers._rrf import Fusion

NEEDS_CLAUDE = False


def build(corpus, cfg):
    return Fusion(["bm25", "dense"], corpus, cfg)
