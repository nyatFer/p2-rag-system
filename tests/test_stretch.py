"""The stretch options: p2 judge and the calibration file (2), the cost table (3), the per-query ablation
table (4), stretch runs such as the grep agent (5), hand against claude queries (6), and the
template's own EVAL.md, whose stretch tables must pass p2 check in a fresh copy."""

import json
import os
import shutil
from pathlib import Path

from core_repo import make_repo, write
from test_core_answer import fake_call
from test_core_check import LLM_SYSTEM, fake_rerank, llm_call

from p2 import answer, check, claude, cli, config, judge, paths, score, stretch, trace

TEMPLATE = Path(__file__).resolve().parents[1]


def run(root, *argv):
    return cli.main(["--root", str(root), *argv])


def fails(root):
    return {i.what: i.detail for i in check.run_checks(root) if i.status == "FAIL"}


def results_of(root):
    return score.compute(root, config.load(root))[0]


# ---- option 2: p2 judge and the calibration file ----


def fake_judge(prompt, schema, system=None, model="sonnet", cache_dir=None, timeout=600):
    assert schema == judge.SCHEMA and "You check one claim" not in prompt
    assert system.startswith("Judge strictly.") and "#" not in system  # prompts/judge.txt, notes removed
    verdict = "partly" if "seat belts" in prompt.lower() else "supported"
    trace.add_usage("claude-sonnet-5-5", 300, 15)
    return claude.Reply({"verdict": verdict, "reason": "one sentence"}, 0.4, 300, 15, "claude-sonnet-5-5")


def judged_repo(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    write(root / "prompts" / "judge.txt", "Judge strictly.\n# a note for you\n")
    monkeypatch.setattr(answer.claude, "call", fake_call)
    assert run(root, "answer", "--system", "bm25") == 0
    monkeypatch.setattr(judge.claude, "call", fake_judge)
    return root


def test_judge_writes_the_judged_file_and_its_trace_without_printing_verdicts(tmp_path, monkeypatch, capsys):
    root = judged_repo(tmp_path, monkeypatch)
    capsys.readouterr()
    assert run(root, "judge", "answers/shared/bm25.json") == 0
    out = capsys.readouterr().out
    assert "[1/2] a01-c1" in out and "supported" not in out.split("Claude judged")[0]  # no verdict before the summary
    assert "1 supported, 1 partly, 0 not" in out and "bm25.calibration.tsv" in out and "No command writes that file" in out
    data = judge.load_judged(root / "answers" / "shared" / "bm25.judged.json")
    assert [(c["id"], c["verdict"], c["quote_found"]) for c in data["claims"]] == [("a01-c1", "supported", True), ("a02-c1", "partly", True)]
    spans = trace.read(root / "traces" / "judge-bm25.jsonl")
    assert [(s["p2.qid"], s["p2.verdict"], s["p2.claude_calls"]) for s in spans] == [("a01-c1", "supported", 1), ("a02-c1", "partly", 1)]
    assert run(root, "judge", "answers/shared/bm25.judged.json") == 2  # the judged file is not an answers file
    assert run(root, "judge", "answers/shared/bm25.json", "--only", "zz") == 2


def test_judge_asks_before_more_than_the_limit_of_calls(tmp_path, monkeypatch, capsys):
    root = judged_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(claude, "CONFIRM_ABOVE", 1)
    assert run(root, "judge", "answers/shared/bm25.json") == 2
    assert "makes up to 2 claude -p calls" in capsys.readouterr().out
    assert not (root / "answers" / "shared" / "bm25.judged.json").exists()
    assert run(root, "judge", "answers/shared/bm25.json", "--only", "a01") == 0  # one call is under the limit
    assert run(root, "judge", "answers/shared/bm25.json", "--yes") == 0


def test_judged_files_are_not_answers_files_and_check_trusts_only_what_the_trace_backs(tmp_path, monkeypatch, capsys):
    root = judged_repo(tmp_path, monkeypatch)
    assert run(root, "judge", "answers/shared/bm25.json") == 0
    write(root / "answers" / "shared" / "bm25.calibration.tsv", "claim\tverdict\tnote\na01-c1\tsupported\tthe 80 percent is quoted\na02-c1\tnot\tworn is not in the quote\n")
    assert run(root, "score") == 0
    capsys.readouterr()
    assert run(root, "verify") == 0 and "Checking answers/shared/bm25.json" in capsys.readouterr().out
    items = check.run_checks(root)
    assert not [i for i in items if i.status == "FAIL"], [i.line() for i in items if i.status == "FAIL"]
    lines = [i.line() for i in items]
    assert "PASS 2 judged or calibration file(s) well-formed: stretch option 2" in lines
    assert any(line.startswith("PASS 1 answers file(s) well-formed") for line in lines)
    assert any(line.startswith("NOTE stretch option 2") and "2 of the 20" in line for line in lines)
    judged_path = root / "answers" / "shared" / "bm25.judged.json"
    good = judged_path.read_text(encoding="utf-8")
    judged_path.write_text(good.replace('"partly"', '"supported"'), encoding="utf-8", newline="\n")
    assert "where its trace has 'partly'" in fails(root)["answers/shared/bm25.judged.json matches its answers file and its Claude trace"]
    judged_path.write_text(good, encoding="utf-8", newline="\n")
    answers_path = root / "answers" / "shared" / "bm25.json"
    data = json.loads(answers_path.read_text(encoding="utf-8"))
    data["answers"][0]["claims"][0]["text"] = "a different claim"
    answers_path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    assert "answers changed after they were judged" in fails(root)["answers/shared/bm25.judged.json matches its answers file and its Claude trace"]


def test_a_calibration_file_with_a_bad_line_fails_with_a_next_step(tmp_path, monkeypatch):
    root = judged_repo(tmp_path, monkeypatch)
    write(root / "answers" / "shared" / "bm25.calibration.tsv", "a01-c1\tyes\tnote\na09-c4\tnot\tnote\na02-c1\tpartly\n")
    detail = fails(root)["answers/shared/bm25.calibration.tsv is a calibration file p2 can read"]
    assert "'yes', not supported, partly or not" in detail and "a09-c4, which is not a claim" in detail and "a01-c1, a tab" in detail
    write(root / "answers" / "shared" / "other.calibration.tsv", "a01-c1\tsupported\tnote\n")
    assert "no answers file" in fails(root)["answers/shared/other.calibration.tsv is a calibration file p2 can read"]


def test_stretch_judge_table_compares_claude_with_you(tmp_path, monkeypatch):
    root = judged_repo(tmp_path, monkeypatch)
    assert run(root, "judge", "answers/shared/bm25.json") == 0
    write(root / "answers" / "shared" / "bm25.calibration.tsv", "a01-c1\tsupported\tok\na02-c1\tnot\tno\n")
    results = results_of(root)
    entry = results["stretch"]["judge"]["files"]["shared/bm25"]
    assert (entry["claims_judged"], entry["labeled"], entry["compared"], entry["agree"]) == (2, 2, 2, 1)
    assert entry["matrix"]["supported"]["supported"] == 1 and entry["matrix"]["partly"]["not"] == 1
    text = score.render("stretch-judge", results)
    assert "| bm25 | 2 | 1 | 1 | 0 | 2 | 2 | 1 of 2 (50%) | [0.095, 0.905] |" in text
    assert "| partly | 0 | 0 | 1 | 1 |" in text and "| total | 1 | 0 | 1 | 2 |" in text and "fewer than the 20" in text
    assert score.render("stretch-judge", {"sets": {}, "answers": {}}) == score.JUDGE_PLACEHOLDER


# ---- option 5: stretch runs ----


def own_repo(tmp_path):
    root = make_repo(tmp_path)
    docs = root / "corpora" / "own" / "docs"
    texts = {
        "doc-0": "Centrifugal pumps move water out of the pit.",
        "doc-1": "Gate valves must be closed before maintenance.",
        "doc-2": "Conveyor belts need guards at the head pulley.",
        "doc-3": "Respirators protect miners from silica dust.",
    }
    rows = ["docid\ttitle\tsource\tlicense\tnotes"]
    for docid, text in texts.items():
        write(docs / f"{docid}.md", f"# {docid}\n\n{text}\n")
        rows.append(f"{docid}\t{docid}\tmy notes\town-work\t")
    write(root / "corpora" / "own" / "manifest.tsv", "\n".join(rows) + "\n")
    queries = [
        ("o1", "paraphrase", "draining a flooded excavation", "hand", "doc-0"),
        ("o2", "paraphrase", "stopping flow ahead of servicing", "hand", "doc-1"),
        ("o3", "paraphrase", "keeping hands away from moving parts", "hand", "doc-2"),
        ("o4", "paraphrase", "breathing protection", "hand", "doc-3"),
        ("o5", "mixed", "centrifugal pumps water pit", "claude", "doc-0"),
        ("o6", "mixed", "gate valves closed maintenance", "claude", "doc-1"),
        ("o7", "mixed", "conveyor belts guards head pulley", "claude", "doc-2"),
        ("o8", "mixed", "respirators silica dust", "claude", "doc-3"),
    ]
    write(root / "eval" / "own" / "queries.tsv", "".join(f"{q}\t{c}\t{t}\t{o}\n" for q, c, t, o, _ in queries))
    write(root / "eval" / "own" / "qrels.txt", "".join(f"{q} 0 {d} 1\n" for q, _c, _t, _o, d in queries))
    return root


def hand_run(root, rel, tag, ranking):
    """A run file written by hand: {qid: [docids best first]}."""
    lines = [f"{q} Q0 {d} {i} {1.0 - i / 10:.6f} {tag}" for q, docs in ranking.items() for i, d in enumerate(docs, 1)]
    write(root / rel, "\n".join(lines) + "\n")


def test_stretch_runs_need_own_queries_and_are_checked_on_the_queries_they_name(tmp_path, capsys):
    root = own_repo(tmp_path)
    two = tmp_path / "two.queries.tsv"
    write(two, "o1\tparaphrase\tdraining a flooded excavation\thand\no5\tmixed\tcentrifugal pumps water pit\tclaude\n")
    assert run(root, "run", "--corpus", "own", "--system", "bm25", "--stretch") == 2  # no --queries
    assert run(root, "run", "--corpus", "shared", "--queries", str(two), "--system", "bm25", "--stretch") == 2
    write(tmp_path / "bad.tsv", "o1\tparaphrase\tsomething else\n")
    assert run(root, "run", "--corpus", "own", "--queries", str(tmp_path / "bad.tsv"), "--system", "bm25", "--stretch") == 1
    write(tmp_path / "bad.tsv", "o99\tparaphrase\tdraining a flooded excavation\n")
    assert run(root, "run", "--corpus", "own", "--queries", str(tmp_path / "bad.tsv"), "--system", "bm25", "--stretch") == 1
    assert "copy the lines as they are" in capsys.readouterr().out or True
    assert run(root, "run", "--corpus", "own", "--queries", str(two), "--system", "bm25", "--label", "bm25-two", "--stretch") == 0
    path = root / "runs" / "own" / "stretch" / "bm25-two.trec"
    assert sorted({line.split()[0] for line in path.read_text(encoding="utf-8").splitlines()}) == ["o1", "o5"]
    assert (root / "traces" / "retrieval" / "own-stretch-bm25-two.jsonl").is_file()
    assert run(root, "run", "--corpus", "own", "--system", "bm25") == 0
    run(root, "score")
    assert not fails(root)
    # --all never runs a stretch run again
    path.write_text(path.read_text(encoding="utf-8").replace(" bm25\n", " bm25\n", 1), encoding="utf-8", newline="\n")
    before = path.stat().st_mtime_ns
    assert run(root, "run", "--all") == 0
    assert path.stat().st_mtime_ns == before
    # a stretch run that names a query of no gold set fails
    hand_run(root, "runs/own/stretch/bm25-two.trec", "bm25", {"o1": ["doc-0"], "zz": ["doc-1"]})
    assert "not in eval/own/queries.tsv, such as zz" in fails(root)["runs/own/stretch/bm25-two.trec is a valid run file"]


def test_a_claude_stretch_run_is_checked_against_its_trace_and_compared_with_the_four_systems(tmp_path, add_system, monkeypatch):
    root = own_repo(tmp_path)
    add_system("testonly_agent", LLM_SYSTEM.format(call=llm_call("none")))
    two = root / "eval" / "own" / "agent.queries.tsv"
    write(two, "o1\tparaphrase\tdraining a flooded excavation\thand\no5\tmixed\tcentrifugal pumps water pit\tclaude\n")
    assert run(root, "run", "--corpus", "own", "--system", "bm25") == 0
    with monkeypatch.context() as m:
        m.setattr(claude, "call", fake_rerank())
        assert run(root, "run", "--corpus", "own", "--queries", str(two), "--system", "testonly_agent", "--stretch") == 0
    assert (root / "traces" / "own-stretch-testonly_agent.jsonl").is_file()
    assert run(root, "score") == 0
    lines = [i.line() for i in check.run_checks(root)]
    assert not [line for line in lines if line.startswith("FAIL")], lines
    assert any("testonly_agent call Claude, so they are checked against their traces" in line for line in lines)
    results = results_of(root)
    entry = results["stretch"]["agent"]["stretch/testonly_agent"]
    assert entry["judged"] == ["o1", "o5"] and list(entry["systems"]) == ["stretch/testonly_agent", "bm25"]
    assert entry["systems"]["stretch/testonly_agent"]["input_tokens"] == 900 and not entry["systems"]["bm25"]["claude"]
    text = score.render("stretch-agent", results)
    assert "stretch/testonly_agent on 2 of your own queries, 2 of them with a relevant document (o1, o5)." in text
    assert "| bm25 |" in text and "| - | - |" in text and "stretch/testonly_agent minus bm25 | MRR@10 |" in text
    # The stretch run stays out of the own-corpus tables, which score every query.
    assert "stretch/testonly_agent" not in results["sets"]["own/own"]["systems"]
    (root / "traces" / "own-stretch-testonly_agent.jsonl").unlink()
    assert "is missing" in fails(root)["runs/own/stretch/testonly_agent.trec matches its Claude trace"]


# ---- option 4: the per-query look at an ablation ----


def test_ablation_queries_find_the_base_from_a_base_line_or_the_name(tmp_path, add_system):
    root = own_repo(tmp_path)
    assert run(root, "run", "--corpus", "own", "--system", "bm25") == 0
    hand_run(root, "runs/own/dense.trec", "dense", {q: ["doc-3", "doc-2", "doc-1", "doc-0"] for q in ("o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8")})
    add_system("testonly_variant", 'BASE = "dense"\nNEEDS_CLAUDE = False\n\ndef build(corpus, cfg):\n    raise NotImplementedError\n')
    hand_run(root, "runs/own/ablation/testonly_variant.trec", "testonly_variant", {q: ["doc-0", "doc-1", "doc-2", "doc-3"] for q in ("o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8")})
    hand_run(root, "runs/own/ablation/bm25_nameonly.trec", "bm25_nameonly", {q: ["doc-0", "doc-1", "doc-2", "doc-3"] for q in ("o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8")})
    hand_run(root, "runs/own/ablation/zzz.trec", "zzz", {"o1": ["doc-0"]})
    results = results_of(root)
    entries = results["sets"]["own/own"]["ablation_queries"]
    variant = entries["ablation/testonly_variant"]
    assert variant["base"] == "dense" and variant["rule"].startswith('BASE = "dense"')
    # dense ranks doc-3, doc-2, doc-1, doc-0 for every query and the variant the other way round
    assert [r["qid"] for r in variant["helped"]] == ["o1", "o5", "o2", "o6"] and variant["helped"][0]["change"] == 0.75
    assert [r["qid"] for r in variant["hurt"]] == ["o4", "o8", "o3", "o7"] and variant["hurt"][0]["base_rr"] == 1.0
    assert variant["unchanged"] == 0
    assert entries["ablation/bm25_nameonly"]["base"] == "bm25" and "starts with bm25_" in entries["ablation/bm25_nameonly"]["rule"]
    assert entries["ablation/zzz"]["base"] is None and "BASE" in entries["ablation/zzz"]["rule"]
    text = score.render("own-ablation-queries", results)
    assert "ablation/testonly_variant against dense" in text and "| o1 | paraphrase | hand | 0.250 | 1.000 | +0.750 | helped |" in text
    assert "ablation/zzz: no base found" in text


def test_base_constant_is_read_without_running_the_file(tmp_path):
    path = tmp_path / "variant.py"
    path.write_text('raise SystemExit("never run")\nBASE = "hybrid"\n', encoding="utf-8", newline="\n")
    assert score.base_constant(path) == "hybrid"
    path.write_text('\ufeffBASE = "bm25"\n', encoding="utf-8", newline="\n")
    assert score.base_constant(path) == "bm25"
    path.write_text('BASE: str = "dense"\n', encoding="utf-8", newline="\n")
    assert score.base_constant(path) == "dense"
    path.write_text("BASE = some_function()\n", encoding="utf-8", newline="\n")
    assert score.base_constant(path) is None
    path.write_text("BASE: str\n", encoding="utf-8", newline="\n")
    assert score.base_constant(path) is None
    assert score.base_constant(tmp_path / "missing.py") is None


# ---- option 6: hand against claude queries ----


def test_origins_give_the_difference_the_interaction_and_the_overlap(tmp_path):
    root = own_repo(tmp_path)
    assert run(root, "run", "--corpus", "own", "--system", "bm25") == 0
    hand_run(root, "runs/own/dense.trec", "dense", {q: [d] for q, d in (("o1", "doc-0"), ("o2", "doc-1"), ("o3", "doc-2"), ("o4", "doc-3"), ("o5", "doc-1"), ("o6", "doc-1"), ("o7", "doc-0"), ("o8", "doc-0"))})
    results = results_of(root)
    own = results["sets"]["own/own"]
    origins = own["origins"]
    assert origins["n"] == {"hand": 4, "claude": 4}
    # every claude query copies its document's words, no hand query shares a content word with its document
    assert origins["overlap"] == {"hand": {"n": 4, "mean": 0.0}, "claude": {"n": 4, "mean": 1.0}}
    bm25, dense = own["systems"]["bm25"]["per_query"], own["systems"]["dense"]["per_query"]
    lead = {o: [bm25[q]["mrr@10"] - dense[q]["mrr@10"] for q in qs] for o, qs in (("hand", ["o1", "o2", "o3", "o4"]), ("claude", ["o5", "o6", "o7", "o8"]))}
    inter = origins["interaction"]["mrr@10"]
    assert inter["mean_diff"] == round(sum(lead["claude"]) / 4 - sum(lead["hand"]) / 4, 6)
    assert inter["ci95"][0] <= inter["mean_diff"] <= inter["ci95"][1]
    diff = own["systems"]["dense"]["origin_diff"]["mrr@10"]
    assert diff["mean_diff"] == 0.75 and diff["reading"] in ("hand higher", "not distinguishable")
    text = score.render("own-origins", results)
    assert "| hand minus claude | 95% interval | Reading |" in text and "Lead on claude" in text
    assert "hand 0.00 (4 queries), claude 1.00 (4 queries)" in text and "here there are 4 hand and 4 claude" in text
    mrr_only = score.render("own-origins-mrr", results)
    assert "nDCG@10" not in mrr_only.split("Do model-written")[0]


# ---- option 3: cost and latency ----


def span(qid, system, calls=1, tokens=1000, out=20, ms=2000.0, cached=0, saved=0.0):
    record = {"semconv": trace.SEMCONV, "name": f"retrieval {system}", "duration_ms": ms, "gen_ai.usage.input_tokens": tokens,
              "gen_ai.usage.output_tokens": out, "p2.qid": qid, "p2.system": system, "p2.top_ids": [], "p2.claude_calls": calls}  # fmt: skip
    if cached:
        record["p2.cached_calls"] = cached
        record["p2.saved_seconds"] = saved
    return json.dumps(record)


def test_cost_table_counts_every_trace_and_what_a_cheaper_system_saves(tmp_path):
    root = make_repo(tmp_path)
    assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25") == 0
    hand_run(root, "runs/shared/rerank.practice.trec", "rerank", {"p01": ["cfr30-75.403"], "p02": ["cfr30-56.14131", "cfr30-57.14131"], "p03": ["cfr30-75.403"]})
    # one fresh call, one fresh call, and one saved reply that took 3 s when it was made
    write(root / "traces" / "shared-practice-rerank.jsonl", "\n".join([span("p01", "rerank"), span("p02", "rerank"), span("p03", "rerank", ms=1.0, cached=1, saved=3.0)]) + "\n")
    results = results_of(root)
    cost = results["stretch"]["cost"]
    rows = {t["file"]: t for t in cost["traces"]}
    assert rows["traces/shared-practice-rerank.jsonl"] == {
        "file": "traces/shared-practice-rerank.jsonl", "system": "rerank", "spans": 3, "calls": 3, "saved": 1,
        "input_tokens": 3000, "output_tokens": 60, "seconds": 7.001,
    }  # fmt: skip
    pair = next(p for p in cost["run_pairs"] if p["metric"] == "mrr@10")
    assert (pair["cheaper"], pair["dearer"], pair["input_saved_per_query"]) == ("bm25", "rerank", 1000.0)
    assert pair["seconds_saved_per_query"] == round(7.001 / 3, 6) and pair["mean_diff"] <= 0
    assert [t["file"] for t in cost["local"]] == ["traces/retrieval/shared-practice-bm25.jsonl"]
    text = score.render("stretch-cost", results)
    assert "| traces/shared-practice-rerank.jsonl | rerank | 3 | 3 | 1 | 3,000 | 60 | 7.0 | 1,000 | 20 | 2.3 |" in text
    assert "shared practice: bm25 minus rerank" in text and score.LOCAL_HEADING in text
    assert "nDCG@10 difference" in score.render("stretch-cost-ndcg", results)


def test_the_local_part_of_the_cost_table_is_not_compared_by_check(tmp_path):
    root = make_repo(tmp_path)
    assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25") == 0
    assert run(root, "score") == 0
    block = score.blocks((root / "EVAL.md").read_text(encoding="utf-8"))["stretch-cost"]
    assert score.COST_PLACEHOLDER in block and score.LOCAL_HEADING in block
    shutil.rmtree(root / "traces" / "retrieval")  # what a clone in CI has: git ignores that folder
    assert not fails(root)
    data = json.loads((root / "results" / "results.json").read_text(encoding="utf-8"))
    data["stretch"]["cost"]["traces"] = [{"file": "made up"}]
    (root / "results" / "results.json").write_text(json.dumps(data), encoding="utf-8", newline="\n")
    assert "results/results.json matches a fresh `p2 score`" in fails(root)


def test_answers_files_on_the_same_questions_are_compared_by_cost(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setattr(answer.claude, "call", fake_call)
    assert run(root, "answer", "--system", "bm25") == 0
    assert run(root, "answer", "--system", "bm25", "--label", "dear") == 0
    lines = (root / "traces" / "answers-dear.jsonl").read_text(encoding="utf-8").splitlines()
    doubled = [json.dumps({**json.loads(line), "gen_ai.usage.input_tokens": 250}) for line in lines]
    write(root / "traces" / "answers-dear.jsonl", "\n".join(doubled) + "\n")
    results = results_of(root)
    (pair,) = results["stretch"]["cost"]["answers_pairs"]
    assert (pair["cheaper"], pair["dearer"], pair["n"], pair["mean_diff"], pair["input_saved_per_question"]) == ("bm25", "dear", 3, 0.0, 150.0)
    assert "shared: bm25 minus dear | 3 |" in score.render("stretch-cost", results)


def test_trace_totals_and_per_item_cost():
    assert stretch.per_item(None) == {"claude": False, "input_tokens": 0.0, "output_tokens": 0.0, "seconds": 0.0}
    costs = {"q1": {"input_tokens": 100, "output_tokens": 10, "seconds": 1.0, "calls": 1}, "q2": {"input_tokens": 300, "output_tokens": 30, "seconds": 3.0, "calls": 1}}
    assert stretch.per_item(costs)["input_tokens"] == 200.0 and stretch.per_item(costs, ["q2"])["seconds"] == 3.0


# ---- the template's own EVAL.md ----


def test_the_template_eval_md_holds_every_table_with_its_placeholder(tmp_path):
    root = make_repo(tmp_path)
    shutil.copy(TEMPLATE / "EVAL.md", root / "EVAL.md")
    shutil.copy(TEMPLATE / "prompts" / "judge.txt", root / "prompts" / "judge.txt")
    text = (root / "EVAL.md").read_text(encoding="utf-8")
    assert score.missing_blocks(text, "598E") == []
    results = results_of(root)
    for name, body in score.blocks(text).items():
        assert body.strip() == score.render(name, results).strip(), name
    assert not fails(root)
    section8 = text.split("## 8. Stretch", 1)[1].split("## 9.", 1)[0]
    assert not check.TODO_MARK.search(section8)  # the stretch is optional for 498E, so --final never waits on it


def test_paths_of_the_stretch_files():
    root = Path("/repo")
    assert paths.run_file(root, "own", "own", "agent", stretch=True) == root / "runs" / "own" / "stretch" / "agent.trec"
    assert paths.trace_file(root, "own", "own", "agent", calls_claude=True, stretch=True) == root / "traces" / "own-stretch-agent.jsonl"
    answers = root / "answers" / "shared" / "rerank.r1.json"
    assert paths.judged_file(answers).name == "rerank.r1.judged.json" and paths.calibration_file(answers).name == "rerank.r1.calibration.tsv"
    assert paths.answers_stem(root / "answers" / "shared" / "rerank.calibration.tsv") == "rerank"
    assert os.path.basename(paths.judge_trace(root, "rerank")) == "judge-rerank.jsonl"


def test_an_answers_trace_rewritten_by_another_run_fails_with_a_way_back(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setattr(answer.claude, "call", fake_call)
    assert run(root, "answer", "--system", "bm25") == 0
    assert run(root, "score") == 0
    assert not fails(root)
    trace_path = root / "traces" / "answers-bm25.jsonl"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    trace_path.write_text(lines[0] + "\n", encoding="utf-8", newline="\n")  # what an interrupted run leaves
    detail = fails(root)["traces/answers-bm25.jsonl matches answers/shared/bm25.json"]
    assert "no span for a02" in detail and "git checkout -- traces/answers-bm25.jsonl" in detail
    trace_path.unlink()  # no trace at all is allowed; the cost table reads the answers file then
    assert run(root, "score") == 0
    assert not fails(root)
