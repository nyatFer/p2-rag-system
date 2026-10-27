"""Your dense retriever (stage 1): embed every chunk once, embed the query, and rank chunks by cosine similarity.

This file is yours to write, and the system is called "dense" because the file is dense.py.
Until you write it, build() raises NotImplementedError, and p2 skips the system and lists it as to do.

Where the idea is:
- the 12 lab's retrieve.py in the course repo (class/12-embeddings-and-retrieval/lab/starter/retrieve.py):
  load_bge, cached_doc_vectors and dense_rankings embed the articles and rank them with one matrix product;
- p2/embed.py has those pieces ready for chunks: embed.load(embed.BGE) loads bge-small-en-v1.5,
  embed.doc_vectors(model, texts, cfg.root) encodes and caches the chunk vectors, and
  model.encode([query]) gives the query vector;
- p2/retrievers/bm25.py shows the shape p2 expects: a class built on ChunkScorer whose
  score_chunks(text) returns one score per chunk in self.chunks, and ChunkScorer then gives you both
  search (documents) and search_chunks (chunks, which `p2 answer` needs).

Pick the embedder you can defend in DECISIONS.md; bge-small and potion-retrieval-32M are the two the lab measured.
"""

NEEDS_CLAUDE = False


def build(corpus, cfg):
    raise NotImplementedError(
        "dense is not built yet: write p2/retrievers/dense.py (the idea is in the 12 lab's retrieve.py, dense_rankings, and p2/embed.py has the helpers)."
    )
