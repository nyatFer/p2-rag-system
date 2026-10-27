"""The registry of retrieval systems, and the interface every system follows.

A system is a module in this folder, and its file name is its name: bm25.py is the system "bm25".
To add a system (a variant for an ablation, for example), add a file here; nothing else needs
registering. Files whose names start with an underscore are helpers, not systems.

A system module defines two things:

    NEEDS_CLAUDE = False          True if searching calls claude -p (CI never runs those)

    def build(corpus, cfg):       returns an object with
        .search(text, k)          -> [(docid, score), ...], best first, at most k, each docid once
        .search_chunks(text, k)   -> [(chunk_id, score), ...]   optional; `p2 answer` needs it

`corpus` is a p2.corpus.Corpus (corpus.docs, corpus.chunks(words, overlap)); `cfg` is the
p2.config.Config from p2.toml (cfg.chunk_words, cfg.k, cfg.table("dense"), cfg.root, ...).

The easy way to get both methods is to subclass ChunkScorer below and write one method,
score_chunks(text), that returns a score for every chunk: ChunkScorer then ranks the chunks for
search_chunks and lifts them to documents for search (a document scores its best chunk). bm25.py
is the worked example.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Sequence
from types import ModuleType

from p2.corpus import Chunk, Corpus, doc_level
from p2.runfile import order

CANONICAL = ("bm25", "dense", "hybrid", "rerank")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class UnknownSystem(LookupError):
    pass


def names() -> list[str]:
    """Every system: the four of the brief first, then the others in name order."""
    found = {m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_") and NAME_RE.match(m.name)}
    return [n for n in CANONICAL if n in found] + sorted(found - set(CANONICAL))


def module(name: str) -> ModuleType:
    """The module of a system."""
    if name not in names():
        raise UnknownSystem(f"There is no system called {name!r}; the systems are {', '.join(names())} (one file each in p2/retrievers/).")
    mod = importlib.import_module(f"{__name__}.{name}")
    if not callable(getattr(mod, "build", None)):
        raise UnknownSystem(f"p2/retrievers/{name}.py has no build(corpus, cfg) function; add one (see bm25.py).")
    return mod


def needs_claude(name: str) -> bool:
    return bool(getattr(module(name), "NEEDS_CLAUDE", False))


_built: dict[tuple, object] = {}


def build(name: str, corpus: Corpus, cfg) -> object:
    """Build a system for a corpus, once per process for the same corpus and settings.

    A system that combines others (a fusion, a reranker) should get them through this function,
    so a run of several systems builds each index only once.
    """
    key = (name, id(corpus), cfg.fingerprint())
    if key not in _built:
        _built[key] = module(name).build(corpus, cfg)
    return _built[key]


class ChunkScorer:
    """Base class for a system that gives every chunk a score.

    Write __init__ (call super().__init__(corpus, cfg) first, then build your index over
    self.chunks) and score_chunks(text). You get search_chunks and search for free.
    """

    model: str | None = None  # the model name to record in traces, if the system uses one

    def __init__(self, corpus: Corpus, cfg):
        self.corpus = corpus
        self.cfg = cfg
        self.chunks: list[Chunk] = corpus.chunks(cfg.chunk_words, cfg.chunk_overlap)

    def score_chunks(self, text: str) -> Sequence[float]:
        """One score per chunk, in the order of self.chunks; higher means more relevant."""
        raise NotImplementedError

    def search_chunks(self, text: str, k: int) -> list[tuple[str, float]]:
        """The k best chunks as (chunk id, score); equal scores in chunk id order."""
        scores = self.score_chunks(text)
        return order((c.id, float(s)) for c, s in zip(self.chunks, scores))[:k]

    def search(self, text: str, k: int) -> list[tuple[str, float]]:
        """The k best documents as (docid, score); a document scores its best chunk."""
        scores = self.score_chunks(text)
        return doc_level((c.id, float(s)) for c, s in zip(self.chunks, scores))[:k]
