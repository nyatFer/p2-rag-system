"""The p2 command: run files, traces, --all, --extra-corpus, and the commands other modules provide."""

import json

from core_repo import make_repo, write

from p2 import cli, trace


def run(root, *argv):
    return cli.main(["--root", str(root), *argv])


def test_run_writes_run_file_and_trace(tmp_path, capsys):
    root = make_repo(tmp_path)
    assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25") == 0
    lines = (root / "runs" / "shared" / "bm25.practice.trec").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("p01 Q0 cfr30-75.403 1 ")
    assert all(line.endswith(" bm25") for line in lines)
    spans = trace.read(root / "traces" / "retrieval" / "shared-practice-bm25.jsonl")
    assert [s["p2.qid"] for s in spans] == ["p01", "p02", "p03"]
    assert spans[0]["semconv"] == trace.SEMCONV and spans[0]["gen_ai.operation.name"] == "retrieval"
    assert spans[0]["p2.top_ids"][0] == "cfr30-75.403"
    assert "MRR@10" in capsys.readouterr().out


def test_run_all_skips_stubs_and_claude_systems(tmp_path, capsys):
    root = make_repo(tmp_path)
    assert run(root, "run", "--all") == 0
    out = capsys.readouterr().out
    assert "dense on shared practice: not built yet" in out
    assert "rerank on shared practice: skipped, it calls Claude" in out
    assert "Skipping the own corpus" in out
    assert sorted(p.name for p in (root / "runs" / "shared").iterdir()) == ["bm25.practice.trec", "bm25.test.trec"]


def test_extra_corpus_never_writes_into_runs(tmp_path):
    root = make_repo(tmp_path)
    extra = tmp_path / "canaries"
    write(extra / "docs" / "canary-0001.md", "# Canary\n\nThe zorblat valve must be inspected weekly.\n")
    write(extra / "queries.tsv", "c01\tidentifier\tzorblat valve\n")
    out = tmp_path / "canary.trec"
    assert run(root, "run", "--corpus", "shared", "--queries", str(extra / "queries.tsv"), "--system", "bm25", "--extra-corpus", str(extra), "--out", str(out)) == 0
    assert out.read_text(encoding="utf-8").startswith("c01 Q0 canary-0001 1 ")
    assert not (root / "runs").exists()


def test_bad_arguments_explain_themselves(tmp_path, capsys):
    root = make_repo(tmp_path)
    assert run(root, "run", "--corpus", "shared", "--system", "nope") == 2
    assert run(root, "run", "--corpus", "shared", "--system", "bm25", "--label", "Bad.Label") == 2
    assert run(root, "run", "--corpus", "shared", "--system", "bm25", "--ablation") == 2
    out = capsys.readouterr().out
    assert "There is no system called 'nope'" in out and "--ablation goes with --corpus own" in out


def test_ablation_run_goes_to_its_folder_and_is_rerun_by_all(tmp_path):
    root = make_repo(tmp_path)
    for i in range(3):
        write(root / "corpora" / "own" / "docs" / f"doc-{i}.md", f"# Doc {i}\n\nText about topic {i} and pumps.\n")
    write(root / "eval" / "own" / "queries.tsv", "o01\tparaphrase\tpumps topic 1\thand\n")
    write(root / "eval" / "own" / "qrels.txt", "o01 0 doc-1 1\n")
    assert run(root, "run", "--corpus", "own", "--system", "bm25", "--label", "chunks-small", "--ablation") == 0
    path = root / "runs" / "own" / "ablation" / "chunks-small.trec"
    assert path.read_text(encoding="utf-8").startswith("o01 Q0 doc-1 1 ")
    path.unlink()
    path.parent.joinpath("chunks-small.trec").write_text("o01 Q0 doc-0 1 0.100000 bm25\n", encoding="utf-8", newline="\n")
    assert run(root, "run", "--all") == 0
    assert path.read_text(encoding="utf-8").startswith("o01 Q0 doc-1 1 ")
    assert (root / "runs" / "own" / "bm25.trec").is_file()


def test_lazy_modules_say_when_they_are_missing(capsys):
    assert cli.lazy("no_such_module_here", "thing", "hint") is None
    assert "p2 thing is not built yet" in capsys.readouterr().out


def test_score_with_test_qrels_writes_only_the_out_file(tmp_path):
    root = make_repo(tmp_path)
    run(root, "run", "--corpus", "shared", "--queries", "test", "--system", "bm25")
    qrels = tmp_path / "private.qrels.txt"
    qrels.write_text("t01 0 cfr30-56.14131 1\nt02 0 cfr30-56.14107 1\n", encoding="utf-8", newline="\n")
    out = tmp_path / "graded.json"
    assert run(root, "score", "--test-qrels", str(qrels), "--out", str(out)) == 0
    graded = json.loads(out.read_text(encoding="utf-8"))
    assert graded["sets"]["shared/test"]["systems"]["bm25"]["mean"]["mrr@10"] == 1.0
    assert not (root / "results").exists()


def test_a_queries_file_of_your_own_writes_outside_runs(tmp_path):
    root = make_repo(tmp_path)
    queries = tmp_path / "mine.queries.tsv"
    queries.write_text("m01\tparaphrase\trock dust\n", encoding="utf-8", newline="\n")
    assert run(root, "run", "--corpus", "shared", "--queries", str(queries), "--system", "bm25") == 0
    assert (root / ".cache" / "extra" / "bm25.mine.trec").is_file()
    assert not (root / "runs").exists()
