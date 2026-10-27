"""Run files, qrels, queries and questions: the format contract and its validation."""

from pathlib import Path

from p2.runfile import format_run, order, read_qrels, read_queries, read_questions, read_run, validate_run, write_run


def put(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_order_breaks_ties_by_id_and_rounds_to_six_decimals():
    assert order([("b", 1.0), ("a", 1.0), ("c", 2.0)]) == [("c", 2.0), ("a", 1.0), ("b", 1.0)]
    # Equal at 6 decimals counts as a tie, so the file order and the in-memory order agree.
    assert [d for d, _ in order([("b", 1.0000004), ("a", 1.0000001)])] == ["a", "b"]


def test_write_and_read_round_trip(tmp_path):
    path = tmp_path / "runs" / "bm25.practice.trec"
    write_run(path, {"q1": [("d2", 0.5), ("d1", 0.5), ("d3", 2.0)], "q2": [("d1", 1.0)]}, tag="bm25", k=2)
    assert path.read_text(encoding="utf-8") == "q1 Q0 d3 1 2.000000 bm25\nq1 Q0 d1 2 0.500000 bm25\nq2 Q0 d1 1 1.000000 bm25\n"
    assert read_run(path) == {"q1": [("d3", 2.0), ("d1", 0.5)], "q2": [("d1", 1.0)]}
    assert validate_run(path, k=2, docids={"d1", "d2", "d3"}, qids=["q1", "q2"]) == []


def test_format_run_with_no_queries_is_empty():
    assert format_run({}, "bm25", 10) == ""


def test_validation_rejects_a_duplicate_document(tmp_path):
    path = put(tmp_path, "r.trec", "q1 Q0 d1 1 2.000000 x\nq1 Q0 d1 2 1.000000 x\n")
    assert any("more than once" in p for p in validate_run(path))


def test_validation_rejects_equal_scores_out_of_id_order(tmp_path):
    path = put(tmp_path, "r.trec", "q1 Q0 d2 1 1.000000 x\nq1 Q0 d1 2 1.000000 x\n")
    assert any("document id order" in p for p in validate_run(path))


def test_validation_catches_every_contract_break(tmp_path):
    cases = {
        "q1 Q0 d1 1 1.000000\n": "fields, not 6",
        "q1 Q1 d1 1 1.000000 x\n": "not Q0",
        "q1 Q0 D1 1 1.000000 x\n": "not allowed",
        "q1 Q0 d1 1 1.00000 x\n": "6 decimals",
        "q1 Q0 d1 2 1.000000 x\n": "expected 1",
        "q1 Q0 d1 1 1.000000 x\nq1 Q0 d2 2 3.000000 x\n": "goes up",
        "q1 Q0 d1 1 1.000000 x\nq1 Q0 d2 2 0.500000 y\n": "more than one tag",
        "q1 Q0 d1 1 1.000000 x\nq2 Q0 d1 1 1.000000 x\nq1 Q0 d2 2 0.500000 x\n": "not all together",
    }
    for text, expected in cases.items():
        problems = validate_run(put(tmp_path, "r.trec", text))
        assert any(expected in p for p in problems), (text, problems)


def test_validation_checks_k_corpus_and_coverage(tmp_path):
    path = put(tmp_path, "r.trec", "q1 Q0 d1 1 2.000000 x\nq1 Q0 d9 2 1.000000 x\nq7 Q0 d1 1 1.000000 x\n")
    problems = validate_run(path, k=1, docids={"d1"}, qids=["q1", "q2"])
    text = " | ".join(problems)
    assert "more than k = 1" in text
    assert "not in the corpus" in text
    assert "not in the queries file" in text
    assert "covers 1 of 2 queries" in text


def test_qrels_parsing_and_problems(tmp_path):
    path = put(tmp_path, "qrels.txt", "# comment\nq1 0 d1 1\nq1 0 d2 2\nq1 0 d1 1\nq2 0 d3\nq3 0 d4 x\n")
    qrels, problems = read_qrels(path)
    assert qrels == {"q1": {"d1": 1, "d2": 2}}
    assert len(problems) == 3


def test_queries_need_origin_in_the_own_gold_set(tmp_path):
    path = put(tmp_path, "q.tsv", "# header\nq1\tparaphrase\tsome text\thand\nq2\tparaphrase\tother text\n\nq2\tx\tdup\n")
    queries, problems = read_queries(path, require_origin=True)
    assert [q.qid for q in queries] == ["q1", "q2"]
    assert queries[0].origin == "hand"
    assert any("origin" in p for p in problems)
    assert any("second time" in p for p in problems)
    assert read_queries(path)[1] == [p for p in read_queries(path)[1] if "origin" not in p]


def test_questions(tmp_path):
    path = put(tmp_path, "questions.tsv", "a01\tin\tHow?\td1,d2\na02\tout\tWhy?\t-\na03\tin\tWhat?\t-\n")
    questions, problems = read_questions(path)
    assert questions[0].gold == ("d1", "d2")
    assert questions[1].gold == ()
    assert problems == ["line 3 (a03) is an in question with no gold document"]
