"""Your hybrid retriever (stage 1): fuse the rankings of two systems into one.

This file is yours to write. Until you write it, build() raises NotImplementedError, and p2 skips
the system and lists it as to do.

Where the idea is:
- the 13 lab's pipeline.py in the course repo (class/13-rag-pipeline/lab/starter/pipeline.py):
  rrf() is reciprocal rank fusion with k = 60, where each document scores the sum of 1 / (60 + rank)
  over the rankings it appears in, and command_fuse() fuses bm25 with an embedder;
- get the two systems you fuse with p2.retrievers.build("bm25", corpus, cfg) and
  build("dense", corpus, cfg), so each index is built once per run;
- fuse full rankings, not top-10 lists (ask each system for len(corpus.docs) documents), because a
  document that is 11th in both lists can belong in the fused top 10;
- order the fused documents with p2.runfile.order(), so equal scores fall in document id order.

If you want `p2 answer` to work with hybrid, also give it search_chunks(text, k): the same fusion over chunk rankings.
"""

NEEDS_CLAUDE = False


def build(corpus, cfg):
    raise NotImplementedError(
        "hybrid is not built yet: write p2/retrievers/hybrid.py (the idea is rrf() in the 13 lab's pipeline.py)."
    )
