"""Metric tests with hand-checked values: trec_eval's semantics, with ties broken by our document id rule."""

import math

import pytest

from p2 import metrics

LOG2_3 = math.log2(3)


def test_baseline_matches_trec_eval():
    # pytrec-eval-terrier gave recip_rank 1.0, recall_10 1.0, ndcg_cut_10 0.9197 for this case (research notes C 1.3).
    run = [("d1", 3.0), ("d3", 2.0), ("d2", 1.0)]
    rels = {"d1": 1, "d2": 1}
    m = metrics.query_metrics(run, rels)
    assert m["mrr@10"] == 1.0
    assert m["recall@10"] == 1.0
    # DCG = 1/log2(2) + 1/log2(4) = 1.5; ideal = 1 + 1/log2(3)
    assert m["ndcg@10"] == pytest.approx(1.5 / (1 + 1 / LOG2_3))
    assert round(m["ndcg@10"], 4) == 0.9197


def test_tie_is_broken_by_document_id_ascending():
    # d3 is listed first, but d1 and d3 tie, so d1 ranks first under our rule.
    run = [("d3", 1.0), ("d1", 1.0), ("d2", 0.5)]
    assert metrics.rr(run, {"d1": 1}) == 1.0
    assert metrics.rr(run, {"d3": 1}) == 0.5
    # Insertion order inside a tie never matters.
    assert metrics.rr(list(reversed(run)), {"d3": 1}) == 0.5
    # trec_eval would rank zz first (descending ids); we rank d1 first, so the relevant zz is second.
    assert metrics.rr([("zz", 1.0), ("d1", 1.0)], {"zz": 1}) == 0.5


def test_equal_scores_are_ordered_by_document_id():
    run = [("c", 2.0), ("b", 2.0), ("a", 2.0), ("z", 3.0)]
    assert metrics.ranked_ids(run) == ["z", "a", "b", "c"]


def test_k_greater_than_r():
    # R = 2 relevant documents, k = 10, found at ranks 3 and 7: recall is the full 1.0.
    run = [(f"x{i:02d}", 100.0 - i) for i in range(12)]
    rels = {"x02": 1, "x06": 1}
    m = metrics.query_metrics(run, rels)
    assert m["recall@10"] == 1.0
    assert m["mrr@10"] == pytest.approx(1 / 3)
    assert m["ndcg@10"] == pytest.approx((1 / math.log2(4) + 1 / math.log2(8)) / (1 + 1 / LOG2_3))
    assert round(m["ndcg@10"], 6) == 0.510956


def test_r_greater_than_k_caps_recall_at_k_over_r():
    # 20 relevant documents and a perfect top 10: recall@10 is 10/20, nDCG and RR are 1.
    run = [(f"r{i:02d}", 50.0 - i) for i in range(20)]
    rels = {f"r{i:02d}": 1 for i in range(20)}
    m = metrics.query_metrics(run, rels)
    assert m["recall@10"] == 0.5
    assert m["ndcg@10"] == pytest.approx(1.0)
    assert m["mrr@10"] == 1.0
    # The research case: k = 5 with 20 relevant gives 0.25 for a perfect system.
    assert metrics.recall(run, rels, k=5) == 0.25


def test_graded_and_binary_ndcg():
    # Graded qrels d1 = 3, d2 = 1, run puts d2 first: pytrec-eval-terrier gave 0.7967.
    run = [("d2", 2.0), ("d1", 1.0)]
    rels = {"d1": 3, "d2": 1}
    graded = metrics.ndcg(run, rels)
    assert graded == pytest.approx((1 + 3 / LOG2_3) / (3 + 1 / LOG2_3))
    assert round(graded, 4) == 0.7967
    assert metrics.ndcg(run, rels, graded=False) == pytest.approx(1.0)


def test_relevant_document_below_k_scores_zero():
    run = [(f"n{i:02d}", 20.0 - i) for i in range(12)]
    rels = {"n10": 1}  # rank 11
    m = metrics.query_metrics(run, rels)
    assert m == {"recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}


def test_duplicate_document_is_rejected():
    with pytest.raises(ValueError, match="more than once"):
        metrics.query_metrics([("d1", 2.0), ("d1", 1.0)], {"d1": 1})


def test_evaluate_scores_judged_queries_and_zero_for_missing_runs():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 0}, "q3": {"d3": 1}}
    run = {"q1": [("d1", 1.0)], "q2": [("d2", 1.0)], "q9": [("d9", 1.0)]}
    per_query = metrics.evaluate(run, qrels)
    assert set(per_query) == {"q1", "q3"}  # q2 has no relevant document, q9 is not judged
    assert per_query["q3"]["mrr@10"] == 0.0  # judged but missing from the run
    assert metrics.mean(per_query, "mrr@10") == 0.5
    assert metrics.judged_queries(qrels, ["q3", "q2", "q1"]) == ["q3", "q1"]
