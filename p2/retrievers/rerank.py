"""The reranker (stage 1): ask Claude to reorder a first-stage system's top 20.

It is rerank_one() from the 13 lab's pipeline.py: the top 20 go to Claude as numbered passages, each
cut to its first 600 characters, the reply is forced into {"order": [20 numbers from 1 to 20]}, a
reply that is not each number once (or a failed call) keeps the first-stage order, and the items
below the 20 keep their order underneath. One call per query; search_chunks makes its own call over
the first stage's top 20 chunks, for `p2 answer --system rerank`.

The first stage and the number of candidates are the [rerank] table of p2.toml (hybrid and 20 by
default). Every call goes through p2.claude.call, which caches replies and records the tokens in the
run's trace, so commit traces/ with the run file.
"""

from __future__ import annotations

import sys

from p2 import claude, retrievers

NEEDS_CLAUDE = True
PASSAGE_CHARS = 600

PROMPT = """Below are {n} passages from a document collection, numbered 1 to {n}, and a search query.
Order the passages from the most useful for answering the query to the least useful.
Reply with all {n} passage numbers, each exactly once, best first.

Query: {query}

{passages}"""


def schema(n: int) -> dict:
    numbers = {"type": "integer", "minimum": 1, "maximum": n}
    return {
        "type": "object",
        "properties": {"order": {"type": "array", "items": numbers, "minItems": n, "maxItems": n}},
        "required": ["order"],
    }


class Rerank:
    def __init__(self, corpus, cfg):
        settings = cfg.table("rerank")
        self.cfg = cfg
        self.model = cfg.model
        self.first = retrievers.build(settings.get("first", "hybrid"), corpus, cfg)
        self.candidates = int(settings.get("candidates", 20))
        self.doc_text = {docid: doc.text for docid, doc in corpus.docs.items()}
        self.chunk_text = {c.id: c.text for c in corpus.chunks(cfg.chunk_words, cfg.chunk_overlap)}

    def rerank(self, query: str, ids: list[str], texts: dict[str, str]) -> list[str]:
        """The candidate ids in Claude's order, or in their first-stage order if the reply is unusable."""
        n = len(ids)
        if n < 2:
            return ids
        passages = "\n\n".join(f"[{i}] {texts[c][:PASSAGE_CHARS]}" for i, c in enumerate(ids, 1))
        prompt = PROMPT.format(n=n, query=query, passages=passages)
        reply = claude.call(prompt, schema(n), model=self.cfg.model, cache_dir=claude.cache_folder(self.cfg))
        order = (reply.output or {}).get("order")
        if reply.error or not isinstance(order, list) or sorted(order) != list(range(1, n + 1)):
            problem = reply.error or f"the reply was not each of 1 to {n} once: {order}"
            print(f"  rerank kept the first-stage order for {query[:50]!r}: {problem}", file=sys.stderr)
            return ids
        return [ids[i - 1] for i in order]

    def _search(self, query: str, k: int, ranking: list[tuple[str, float]], texts: dict[str, str]) -> list[tuple[str, float]]:
        ids = [i for i, _ in ranking]
        ranked = self.rerank(query, ids[: self.candidates], texts) + ids[self.candidates :]
        # Scores 1 / rank keep Claude's order in the run file, where equal scores would be sorted by id.
        return [(i, 1 / r) for r, i in enumerate(ranked, 1)][:k]

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        ranking = self.first.search(text, max(k, self.candidates))
        return self._search(text, k, ranking, self.doc_text)

    def search_chunks(self, text: str, k: int) -> list[tuple[str, float]]:
        ranking = self.first.search_chunks(text, max(k, self.candidates))
        return self._search(text, k, ranking, self.chunk_text)


def build(corpus, cfg):
    return Rerank(corpus, cfg)
