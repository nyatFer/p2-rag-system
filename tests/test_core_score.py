"""p2 score: results, the EVAL.md block rewrite, and the comparisons p2 check makes."""

import json

from core_repo import EVAL_MD, make_repo

from p2 import cli, config, score


def run(root, *argv):
    return cli.main(["--root", str(root), *argv])


def test_rewrite_fills_known_blocks_and_leaves_the_rest(tmp_path):
    root = make_repo(tmp_path)
    assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25") == 0
    assert run(root, "score") == 0
    text = (root / "EVAL.md").read_text(encoding="utf-8")
    assert text.startswith("# EVAL\n\nProse that the student writes. TODO: write it.\n")
    blocks = score.blocks(text)
    assert "| bm25 |" in blocks["shared-practice"]
    assert "3 judged queries" in blocks["shared-practice"]
    assert "Only one system" in blocks["shared-practice-pairs-mrr"]
    assert "No answers files yet" in blocks["answers"]
    assert blocks["someone-elses-block"] == "\nleft alone\n"
    # Running score again changes nothing.
    assert run(root, "score") == 0
    assert (root / "EVAL.md").read_text(encoding="utf-8") == text
    results = json.loads((root / "results" / "results.json").read_text(encoding="utf-8"))
    bm25 = results["sets"]["shared/practice"]["systems"]["bm25"]
    assert set(bm25["per_query"]) == {"p01", "p02", "p03"}
    assert set(bm25["by_class"]) == {"identifier", "paraphrase", "mixed"}


def test_pairs_with_two_systems(tmp_path):
    root = make_repo(tmp_path)
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25")
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25", "--label", "other")
    results, notes = score.compute(root, config.load(root))
    assert notes == []
    pairs = results["sets"]["shared/practice"]["pairs"]
    assert [(p["a"], p["b"], p["metric"]) for p in pairs] == [("other", "bm25", m) for m in ("recall@10", "mrr@10", "ndcg@10")]
    assert all(p["mean_diff"] == 0 and p["reading"] == "identical on every query" and p["mdd80"] is None for p in pairs)
    pairs_text = score.render("shared-practice-pairs-mrr", results)
    assert "other minus bm25 | MRR@10 | +0.000 | [+0.000, +0.000] | - | identical on every query |" in pairs_text


def test_block_names():
    results = {"sets": {}, "answers": {}}
    for name in ("shared-practice", "shared-practice-classes", "shared-practice-pairs-ndcg", "own", "own-pairs", "own-ablation", "answers", "repeats"):
        assert score.render(name, results) is not None, name
    for name in ("shared-practice-bogus", "own-pairs-xyz", "shared-practice-ablation", "notes"):
        assert score.render(name, results) is None, name


def test_marker_problems_and_tolerant_comparison():
    assert score.marker_problems(EVAL_MD) == []
    broken = EVAL_MD.replace("<!-- p2:end answers -->", "")
    assert score.marker_problems(broken) == ["the answers block is missing its end marker"]
    assert score.same_text("| bm25 | 0.512 |", "| bm25 | 0.513 |")
    assert not score.same_text("| bm25 | 0.512 |", "| bm25 | 0.515 |")
    assert not score.same_text("| bm25 | 0.512 |", "| dense | 0.512 |")
    assert score.same_results({"a": [0.1234, "x"]}, {"a": [0.1236, "x"]}) is None
    assert score.same_results({"a": [0.1234]}, {"a": [0.125]}) == "results.a[0]"


def test_repeated_runs_and_answers_are_reported_with_spread_and_wilson(tmp_path, monkeypatch):
    from test_core_answer import fake_call

    from p2 import answer

    root = make_repo(tmp_path)
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25")
    assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25", "--label", "again", "--repeat", "2") == 0
    monkeypatch.setattr(answer.claude, "call", fake_call)
    assert run(root, "answer", "--system", "bm25", "--label", "rep", "--repeat", "2") == 2  # 6 calls: p2 asks first
    assert run(root, "answer", "--system", "bm25", "--label", "rep", "--repeat", "2", "--yes") == 0
    results, _ = score.compute(root, config.load(root))
    entry = results["sets"]["shared/practice"]["repeats"]["again"]
    assert entry["repeat"] == [1, 2] and entry["spread"]["mrr@10"]["range"] == 0.0
    assert [c["reading"] for c in entry["vs"]["bm25"]["mrr@10"]] == ["not distinguishable"] * 2
    group = results["answer_repeats"]["shared/rep"]
    assert group["n_repeats"] == 2 and group["verified_share"] == [1.0, 1.0] and group["verified_share_range"] == 0.0
    assert not any(k.startswith("pooled") for k in group)  # the repeats answer the same questions, so nothing is pooled
    text = score.render("repeats", results)
    assert "again.r1" in text and "| range (max minus min) | 0.000 |" in text and "| pooled |" not in text
    # Repeats stay out of the main tables and pairs.
    assert list(results["sets"]["shared/practice"]["systems"]) == ["bm25"]
