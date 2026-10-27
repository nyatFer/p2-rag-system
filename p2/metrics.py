"""recall@10, MRR@10 and nDCG@10, per query and averaged, with trec_eval's semantics.

How a query is scored:
- The run's documents are put in order by score, highest first; equal scores go in document id
  order (ascending, our rule; trec_eval itself orders ties by descending id). Scores are compared at
  6 decimals, the precision of a run file, so a ranking in memory scores the same as the file it
  is written to. The rank column of a run file is not used, as in trec_eval.
- A document is relevant when its qrels label is above 0. R is the number of relevant documents.
- recall@k: relevant documents in the top k, divided by R. When R is larger than k, a perfect run
  scores k / R, not 1.
- MRR@k: 1 divided by the rank of the first relevant document in the top k, else 0.
- nDCG@k: the sum over the top k of gain / log2(rank + 1), divided by the same sum for the ideal
  order of the judged documents. The gain is the qrels label (trec_eval's default), so 0/1 labels
  give binary nDCG and graded labels (2, 3, ...) give graded nDCG; ndcg(..., graded=False) treats
  every relevant label as 1.
- A query with no relevant document in the qrels is not scored. A judged query the run does not
  cover scores 0 on every metric (trec_eval -c).
- A run that lists a document twice for one query is rejected (ValueError): deduplicate first, for
  example by keeping a document's best chunk (corpus.doc_level).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from p2.runfile import order

K = 10
METRICS = ("recall@10", "mrr@10", "ndcg@10")


def ranked_ids(pairs: Iterable[tuple[str, float]], k: int | None = None) -> list[str]:
    """Document ids in metric order (score descending, then id ascending), cut at k."""
    pairs = list(pairs)
    ids = [d for d, _ in pairs]
    if len(set(ids)) != len(ids):
        dupes = sorted({d for d in ids if ids.count(d) > 1})
        raise ValueError(f"The run lists {', '.join(dupes)} more than once for one query; keep one line per document.")
    ordered = [d for d, _ in order(pairs)]
    return ordered if k is None else ordered[:k]


def recall(pairs, rels: dict[str, int], k: int = K) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked_ids(pairs, k))) / len(relevant)


def rr(pairs, rels: dict[str, int], k: int = K) -> float:
    for rank, d in enumerate(ranked_ids(pairs, k), 1):
        if rels.get(d, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg(pairs, rels: dict[str, int], k: int = K, graded: bool = True) -> float:
    def gain(r: int) -> float:
        return float(r) if graded else 1.0

    dcg = sum(gain(rels[d]) / math.log2(rank + 1) for rank, d in enumerate(ranked_ids(pairs, k), 1) if rels.get(d, 0) > 0)
    ideal = sorted((gain(r) for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def query_metrics(pairs, rels: dict[str, int], k: int = K) -> dict[str, float]:
    """recall@k, mrr@k and ndcg@k of one query. The names say @10 because 10 is the contract's k."""
    pairs = list(pairs)
    return {"recall@10": recall(pairs, rels, k), "mrr@10": rr(pairs, rels, k), "ndcg@10": ndcg(pairs, rels, k)}


def judged_queries(qrels: dict[str, dict[str, int]], qids: Iterable[str] | None = None) -> list[str]:
    """The queries that get scored: those with at least one relevant document (in `qids` order if given)."""
    has_relevant = {q for q, rels in qrels.items() if any(r > 0 for r in rels.values())}
    if qids is None:
        return sorted(has_relevant)
    return [q for q in qids if q in has_relevant]


def evaluate(run: dict[str, list[tuple[str, float]]], qrels: dict[str, dict[str, int]], qids: Iterable[str] | None = None, k: int = K) -> dict[str, dict[str, float]]:
    """{qid: {metric: value}} for every judged query; a judged query missing from the run scores 0."""
    return {q: query_metrics(run.get(q, []), qrels[q], k) for q in judged_queries(qrels, qids)}


def mean(per_query: dict[str, dict[str, float]], metric: str, qids: Iterable[str] | None = None) -> float:
    """The mean of one metric over the given queries (default: all of them); 0 when there are none."""
    keys = list(per_query) if qids is None else [q for q in qids if q in per_query]
    return sum(per_query[q][metric] for q in keys) / len(keys) if keys else 0.0
