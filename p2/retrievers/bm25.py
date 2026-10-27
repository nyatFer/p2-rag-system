"""BM25, the provided baseline: keyword scoring over chunks, lifted to documents by each document's best chunk.

It is the 12 lab's BM25 (k1 0.9, b 0.4, calibrated on the lab corpus) with three changes:
- it scores chunks (the [chunking] settings of p2.toml); with words = 0 every document is one
  chunk and the scores are the lab's exactly;
- it uses the tokenizer named in [bm25] of p2.toml ("codes" by default; "lab" is the lab's);
- it builds an inverted index once, so a query touches only the chunks that share a word with it.

For one query word w in chunk c:
    idf(w)  = ln(1 + (N - df(w) + 0.5) / (df(w) + 0.5))      N chunks, df(w) chunks containing w
    part    = idf(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(c) / avg_len))
and a chunk's score is the sum of the parts over the distinct query words.

A query that shares words with fewer than k documents still gets k lines: the rest score 0 and come
in document id order, as every ChunkScorer ranks ties. Those lines are padding, not matches; a padded
document counts in the scores only when it happens to be relevant.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from p2 import textproc
from p2.retrievers import ChunkScorer

NEEDS_CLAUDE = False


class BM25(ChunkScorer):
    def __init__(self, corpus, cfg):
        super().__init__(corpus, cfg)
        self.tokenizer = cfg.tokenizer
        self.k1, self.b = cfg.bm25_k1, cfg.bm25_b
        token_lists = [textproc.tokens(c.text, self.tokenizer) for c in self.chunks]
        self.lengths = [len(t) for t in token_lists]
        self.n = len(token_lists)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # word -> [(chunk index, tf)]
        for i, toks in enumerate(token_lists):
            for word, tf in Counter(toks).items():
                self.postings[word].append((i, tf))

    def idf(self, word: str) -> float:
        df = len(self.postings.get(word, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score_chunks(self, text: str) -> list[float]:
        scores = [0.0] * self.n
        if not self.n or self.avg_len == 0:
            return scores
        k1, b = self.k1, self.b
        for word in dict.fromkeys(textproc.tokens(text, self.tokenizer)):  # distinct words, first-seen order
            posting = self.postings.get(word)
            if not posting:
                continue
            idf = self.idf(word)
            for i, tf in posting:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.lengths[i] / self.avg_len))
        return scores


def build(corpus, cfg):
    return BM25(corpus, cfg)
