"""The provided BM25 equals the 12 lab's BM25 when documents are whole and the tokenizer is the lab's."""

import math
import re
from collections import Counter

import pytest
from core_repo import make_repo

from p2 import config, corpus, retrievers
from p2.retrievers.bm25 import BM25

STOPWORDS = set("a an and are as at be but by do does for from how i if in is it my of on or so that the this to was what when where which who why with you your".split())


def lab_tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def lab_bm25_scores(docs, query, k1=0.9, b=0.4):
    """Copied from the 12 lab's retrieve.py."""
    doc_tokens = {d: lab_tokens(t) for d, t in docs.items()}
    avg_len = sum(len(t) for t in doc_tokens.values()) / len(docs)
    df = Counter(w for t in doc_tokens.values() for w in set(t))
    scores = {}
    for d, toks in doc_tokens.items():
        tf = Counter(toks)
        score = 0.0
        for w in set(lab_tokens(query)):
            if tf[w]:
                idf = math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(toks) / avg_len))
        scores[d] = score
    return scores


@pytest.mark.parametrize("query", ["rock dust incombustible", "seat belts haulage trucks", "75.403", "nothing matches zzz"])
def test_scores_equal_the_lab_with_whole_documents(tmp_path, query):
    root = make_repo(tmp_path)
    cfg = config.load(root).replace(chunk_words=0, chunk_overlap=0, tokenizer="lab")
    c = corpus.load(root, "shared")
    system = BM25(c, cfg)
    ours = dict(system.search(query, len(c.docs)))
    lab = lab_bm25_scores({d: doc.text for d, doc in c.docs.items()}, query)
    assert set(ours) == set(lab)
    for d in lab:
        assert ours[d] == pytest.approx(lab[d], abs=1e-12)
    # Same order as the lab: score descending, ties by id.
    assert [d for d, _ in system.search(query, len(c.docs))] == sorted(lab, key=lambda d: (-lab[d], d))


def test_chunk_level_search_and_registry(tmp_path):
    root = make_repo(tmp_path)
    cfg = config.load(root)
    c = corpus.load(root, "shared")
    system = retrievers.build("bm25", c, cfg)
    assert retrievers.build("bm25", c, cfg) is system  # built once per corpus and settings
    top = system.search("haulage trucks seat belts", 3)
    assert {d for d, _ in top[:2]} == {"cfr30-56.14131", "cfr30-57.14131"}
    chunks = system.search_chunks("rock dust", 2)
    assert all("#" in cid for cid, _ in chunks)
    assert retrievers.names()[:4] == ["bm25", "dense", "hybrid", "rerank"]
    assert retrievers.needs_claude("rerank") and not retrievers.needs_claude("bm25")
    with pytest.raises(NotImplementedError):
        retrievers.module("dense").build(c, cfg)
    with pytest.raises(retrievers.UnknownSystem):
        retrievers.module("nope")
