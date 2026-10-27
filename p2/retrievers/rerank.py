"""Your reranker (stage 1): ask Claude to reorder a first-stage system's top 20.

This file is yours to write. Until you write it, build() raises NotImplementedError, and p2 skips
the system and lists it as to do. NEEDS_CLAUDE = True tells p2 that searching calls claude -p, so
CI never regenerates this system's runs; you produce them on your machine and commit them.

Where the idea is:
- the 13 lab's pipeline.py in the course repo (class/13-rag-pipeline/lab/starter/pipeline.py):
  rerank_one() shows the top 20 as numbered passages, RERANK_SCHEMA forces the reply into
  {"order": [20 numbers]}, a reply that is not each number once keeps the original order, and the
  documents below the 20 keep their order underneath;
- when you port RERANK_SCHEMA, also give its items "minimum": 1 and "maximum": 20 (the number of
  passages), so Claude cannot answer with a number that names no passage: the lab's schema only
  fixes how many numbers come back, and a reply such as [12, 2, 60, ...] silently keeps the first-stage order;
- make every call through p2.claude.call(prompt, schema, model=cfg.model,
  cache_dir=p2.claude.cache_folder(cfg)), which launches claude -p the way the lab does, caches
  replies, and records the tokens in the run's trace (traces/, which you commit with the run file:
  p2 check reads it, because CI never calls Claude to run this system again);
- get the first stage with p2.retrievers.build("hybrid", corpus, cfg).

`p2 answer --system rerank` also needs search_chunks(text, k): the same reranking over the first
stage's top 20 chunks (hybrid's search_chunks gives them).

A reranker over 20 candidates is one call per query: 20 calls for the practice queries, 40 for the
test queries. Try two queries first, and look at what Claude sees, before the full run.
"""

NEEDS_CLAUDE = True


def build(corpus, cfg):
    raise NotImplementedError(
        "rerank is not built yet: write p2/retrievers/rerank.py (the idea is rerank_one() in the 13 lab's pipeline.py)."
    )
