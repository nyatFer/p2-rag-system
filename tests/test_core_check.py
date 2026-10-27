"""p2 check: a fresh repository passes with TODO lines, and broken or stale files FAIL with a next step."""

import json
import re
import shutil
from pathlib import Path

from core_repo import EVAL_MD, PRACTICE_QRELS, PRACTICE_QUERIES, make_repo

from p2 import check, claude, cli


def run(root, *argv):
    return cli.main(["--root", str(root), *argv])


def statuses(root, final=False):
    return check.run_checks(root, final=final)


def fails(items):
    return [i for i in items if i.status == "FAIL"]


def test_fresh_repository_passes_with_todo_lines(tmp_path, capsys):
    root = make_repo(tmp_path)
    assert run(root, "check") == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "TODO runs of bm25, dense, hybrid and rerank on the shared practice queries" in out
    assert "TODO own corpus has at least 200 documents" in out
    assert run(root, "check", "--final") == 1


def test_scored_runs_regenerate_and_match(tmp_path):
    root = make_repo(tmp_path)
    run(root, "run", "--all")
    run(root, "score")
    items = statuses(root)
    assert fails(items) == [], [i.line() for i in fails(items)]
    lines = [i.line() for i in items]
    assert any(line.startswith("PASS committed runs match a fresh run: 2 run(s) of bm25") for line in lines), lines
    assert "PASS results/results.json matches a fresh `p2 score`" in lines
    assert "PASS EVAL.md tables match a fresh `p2 score`: 13 table(s)" in lines


def test_corrupted_run_fails_with_one_next_step(tmp_path):
    root = make_repo(tmp_path)
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25")
    path = root / "runs" / "shared" / "bm25.practice.trec"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]  # a duplicate document with the wrong rank
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    bad = fails(statuses(root))
    assert bad and bad[0].what == "runs/shared/bm25.practice.trec is a valid run file"
    assert bad[0].detail.endswith("Run `uv run p2 run` for it again (or remove it), then commit.")


def test_tampered_run_and_hand_edited_results_fail(tmp_path):
    root = make_repo(tmp_path)
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25")
    run(root, "score")
    results_path = root / "results" / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    data["sets"]["shared/practice"]["systems"]["bm25"]["mean"]["mrr@10"] = 0.99
    results_path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    path = root / "runs" / "shared" / "bm25.practice.trec"
    text = path.read_text(encoding="utf-8").replace("cfr30-75.403", "cfr30-75.400", 1)
    path.write_text(text, encoding="utf-8", newline="\n")
    whats = {i.what for i in fails(statuses(root))}
    assert "results/results.json matches a fresh `p2 score`" in whats
    assert any(w.endswith("matches a fresh run") or w.endswith("is a valid run file") for w in whats)


def test_carriage_returns_and_stub_runs_fail(tmp_path):
    root = make_repo(tmp_path)
    doc = root / "corpora" / "shared" / "docs" / "cfr30-75.400.md"
    doc.write_bytes(doc.read_bytes().replace(b"\n", b"\r\n"))
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25", "--label", "tmp")
    text = (root / "runs" / "shared" / "tmp.practice.trec").read_text(encoding="utf-8").replace(" bm25\n", " dense\n")
    (root / "runs" / "shared" / "tmp.practice.trec").unlink()
    (root / "runs" / "shared" / "dense.practice.trec").write_text(text, encoding="utf-8", newline="\n")
    whats = {i.what: i.detail for i in fails(statuses(root))}
    assert "no carriage returns under corpora/" in whats
    assert "NotImplementedError" in whats["runs/shared/dense.practice.trec matches a fresh run"]


def test_compare_lists_allows_near_ties_only():
    committed = [("a", 1.0), ("b", 0.9995), ("c", 0.5)]
    assert check.compare_lists(committed, [("b", 0.9996), ("a", 0.9999), ("c", 0.5)]) is None
    assert check.compare_lists(committed, [("a", 1.0), ("c", 0.9995), ("b", 0.5)]) is not None
    assert check.compare_lists(committed, committed[:2]) is not None
    # A near-tie at the cut-off may swap a document out of the list.
    assert check.compare_lists(committed, [("a", 1.0), ("b", 0.9995), ("d", 0.5004)]) is None


def git(root, *args):
    import subprocess

    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args], cwd=root, check=True, capture_output=True)


TEMPLATE_PREREG = (Path(__file__).resolve().parents[1] / "PREREG.md").read_text(encoding="utf-8")


def filled(text, sections=(1, 2, 3, 4), words="dense beats bm25 on my corpus"):
    """PREREG.md with the TODO line of each named section replaced by a sentence of the student's own."""
    out, current = [], None
    for line in text.split("\n"):
        heading = re.match(r"^## (\d+)\.", line)
        if heading:
            current = int(heading.group(1))
        if current in sections and line.startswith("TODO:"):
            line = f"Section {current}: {words}."
        out.append(line)
    return "\n".join(out)


def rider_item(root, start="598E: PREREG.md sections"):
    return next(i for i in statuses(root) if i.what.startswith(start))


def make_rider_repo(tmp_path):
    root = make_repo(tmp_path)
    toml = root / "p2.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace('section = "498E"', 'section = "598E"'), encoding="utf-8", newline="\n")
    (root / "PREREG.md").write_text(TEMPLATE_PREREG, encoding="utf-8", newline="\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "template")
    return root


def commit_qrels(root, path="eval/own/qrels.txt"):
    (root / path).write_text("# qid 0 docid rel\no01 0 doc-1 1\n", encoding="utf-8", newline="\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "judgments")


def commit_prereg(root, text, message="prereg"):
    (root / "PREREG.md").write_text(text, encoding="utf-8", newline="\n")
    git(root, "commit", "-q", "-am", message)


def test_prereg_committed_before_own_qrels_passes_and_outcome_comes_after(tmp_path):
    root = make_rider_repo(tmp_path)
    assert rider_item(root).status == "TODO"
    commit_prereg(root, filled(TEMPLATE_PREREG))
    item = rider_item(root)
    assert item.status == "TODO" and "not committed yet" in item.detail  # sections 1 to 4 done, judgments still to come
    commit_qrels(root)
    assert rider_item(root).status == "PASS"
    assert rider_item(root, "598E: PREREG.md outcome").status == "TODO"
    commit_prereg(root, filled(filled(TEMPLATE_PREREG), sections=(5,), words="the interval was above zero"))
    assert rider_item(root, "598E: PREREG.md outcome").status == "PASS"
    assert rider_item(root).status == "PASS"


def test_prereg_committed_after_own_qrels_fails(tmp_path):
    root = make_rider_repo(tmp_path)
    commit_qrels(root)
    item = rider_item(root)
    assert item.status == "FAIL" and "tell the instructor" in item.detail
    commit_prereg(root, filled(TEMPLATE_PREREG))
    assert rider_item(root).status == "FAIL"


def test_prereg_one_character_edit_or_empty_file_does_not_count(tmp_path):
    root = make_rider_repo(tmp_path)
    commit_prereg(root, TEMPLATE_PREREG + "\n")  # a one-character change before the judgments
    commit_qrels(root)
    commit_prereg(root, filled(TEMPLATE_PREREG))
    assert rider_item(root).status == "FAIL"
    root2 = make_rider_repo(tmp_path / "second")
    commit_prereg(root2, "")
    commit_qrels(root2)
    assert rider_item(root2).status == "FAIL"


def test_prereg_sections_changed_after_the_judgments_fail(tmp_path):
    root = make_rider_repo(tmp_path)
    commit_prereg(root, filled(TEMPLATE_PREREG))
    commit_qrels(root)
    commit_prereg(root, filled(TEMPLATE_PREREG, words="rerank beats hybrid after all"))
    item = rider_item(root)
    assert item.status == "FAIL" and "section 1 changed after commit" in item.detail and "section 6" in item.detail


def test_prereg_judgments_in_another_file_first_still_count(tmp_path):
    root = make_rider_repo(tmp_path)
    commit_qrels(root, "eval/own/judgments.txt")
    commit_prereg(root, filled(TEMPLATE_PREREG))
    assert rider_item(root).status == "FAIL"


def test_rider_answers_must_cover_every_question_in_every_repeat(tmp_path, monkeypatch):
    from test_core_answer import fake_call

    from p2 import answer

    root = make_rider_repo(tmp_path)
    monkeypatch.setattr(answer.claude, "call", fake_call)
    assert run(root, "answer", "--system", "bm25", "--label", "rep", "--only", "a01", "--repeat", "3") == 0
    run(root, "score")
    item = rider_item(root, "598E: three repeated answers files")
    assert item.status == "TODO" and "rep.r1 answers 1 of 3" in item.detail
    assert run(root, "answer", "--system", "bm25", "--label", "rep", "--repeat", "3", "--yes") == 0
    run(root, "score")
    assert rider_item(root, "598E: three repeated answers files").status == "PASS"


def test_rider_prose_lines_in_eval_md_count_as_todo(tmp_path):
    root = make_rider_repo(tmp_path)
    (root / "EVAL.md").write_text(EVAL_MD.replace("TODO: write it.", "Written.") + "\n598E: write the repeats result here.\n", encoding="utf-8", newline="\n")
    item = next(i for i in statuses(root) if i.what == "EVAL.md has no TODO markers left")
    assert item.status == "TODO" and "598E: write" in item.detail


# ---- runs: names, tags, flags and the separate process ----


def test_a_run_named_after_a_system_must_be_written_by_it(tmp_path):
    root = make_repo(tmp_path)
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25")
    shutil.copy(root / "runs" / "shared" / "bm25.practice.trec", root / "runs" / "shared" / "dense.practice.trec")
    items = statuses(root)
    bad = {i.what: i.detail for i in fails(items)}
    assert "written by bm25" in bad["runs/shared/dense.practice.trec is a valid run file"]
    complete = next(i for i in items if i.what.startswith("runs of bm25, dense, hybrid and rerank on the shared practice"))
    assert "dense" in complete.detail


def test_a_label_that_names_another_system_is_refused(tmp_path, capsys):
    root = make_repo(tmp_path)
    assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25", "--label", "rerank") == 2
    assert "name of another system" in capsys.readouterr().out
    assert not (root / "runs").exists()


def oracle_run(root, name, query_set="practice", tag=None):
    """A hand-made run that puts each practice query's relevant documents first."""
    lines = []
    for qid, _cls, _text in PRACTICE_QUERIES:
        docs = [d for q, d, _r in PRACTICE_QRELS if q == qid]
        for rank, docid in enumerate(docs, 1):
            lines.append(f"{qid} Q0 {docid} {rank} {1.0 - rank / 10:.6f} {tag or name}")
    path = root / "runs" / "shared" / f"{name}.{query_set}.trec"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


BM25_SYSTEM = """
from p2.retrievers import bm25
NEEDS_CLAUDE = {flag}
{extra}
def build(corpus, cfg):
    return bm25.build(corpus, cfg)
"""


def test_needs_claude_is_verified_not_trusted(tmp_path, add_system):
    root = make_repo(tmp_path)
    add_system("testonly_flag", BM25_SYSTEM.format(flag=True, extra=""))
    oracle_run(root, "testonly_flag")
    bad = {i.what: i.detail for i in fails(statuses(root))}
    assert "for p0" in bad["runs/shared/testonly_flag.practice.trec matches a fresh run"]


def test_a_retriever_cannot_patch_the_checker(tmp_path, add_system):
    root = make_repo(tmp_path)
    patch = "import sys\nfor m in list(sys.modules.values()):\n    if getattr(m, '__name__', '') == 'p2.check':\n        m.compare_lists = lambda *a, **k: None"
    add_system("testonly_patcher", BM25_SYSTEM.format(flag=False, extra=patch))
    oracle_run(root, "testonly_patcher")
    bad = {i.what for i in fails(statuses(root))}
    assert "runs/shared/testonly_patcher.practice.trec matches a fresh run" in bad


def test_a_broken_retriever_fails_alone_and_the_others_are_still_checked(tmp_path, add_system):
    root = make_repo(tmp_path)
    add_system("testonly_broken", "import not_a_package_anyone_has\nNEEDS_CLAUDE = False\ndef build(corpus, cfg):\n    return None\n")
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25")
    oracle_run(root, "testonly_broken")
    items = statuses(root)
    bad = {i.what: i.detail for i in fails(items)}
    assert "ModuleNotFoundError" in bad["runs/shared/testonly_broken.practice.trec matches a fresh run"]
    assert any(i.line().startswith("PASS committed runs match a fresh run: 1 run(s) of bm25") for i in items)
    assert not any(i.what == "p2 check itself" for i in items)


LLM_SYSTEM = """
from p2 import claude
from p2.retrievers import bm25

NEEDS_CLAUDE = True
SCHEMA = {{"type": "object", "properties": {{"order": {{"type": "array", "items": {{"type": "integer"}}}}}}, "required": ["order"]}}


class Reranker:
    def __init__(self, corpus, cfg):
        self.first = bm25.build(corpus, cfg)
        self.cfg = cfg

    def search(self, text, k):
        hits = self.first.search(text, k)
        {call}
        order = reply.output["order"] if reply.output else list(range(1, len(hits) + 1))
        return [(hits[i - 1][0], 1.0 / (r + 1)) for r, i in enumerate(order)]


def build(corpus, cfg):
    return Reranker(corpus, cfg)
"""


def fake_rerank(reverse=True):
    def call(prompt, schema, system=None, model="sonnet", cache_dir=None, timeout=600):
        from p2 import trace

        trace.add_usage("claude-sonnet-5-5", 900, 10)
        n = int(prompt.split()[-1])
        order = list(range(n, 0, -1)) if reverse else list(range(1, n + 1))
        return claude.Reply({"order": order}, 0.1, 900, 10, "claude-sonnet-5-5")

    return call


def llm_call(guard):
    if guard == "none":
        return "reply = claude.call(f'{text} {len(hits)}', SCHEMA, model=self.cfg.model)"
    return "try:\n            reply = claude.call(f'{text} {len(hits)}', SCHEMA, model=self.cfg.model)\n        except:  # noqa: E722\n            reply = claude.Reply(None, 0, 0, 0, 'x', error='failed')"


def test_a_claude_run_is_checked_against_its_trace(tmp_path, add_system, monkeypatch):
    root = make_repo(tmp_path)
    add_system("testonly_llm", LLM_SYSTEM.format(call=llm_call("none")))
    with monkeypatch.context() as m:
        m.setattr(claude, "call", fake_rerank())
        assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "testonly_llm") == 0
    trace_path = root / "traces" / "shared-practice-testonly_llm.jsonl"
    assert trace_path.is_file()
    lines = [i.line() for i in statuses(root)]
    assert any(line.startswith("PASS committed runs match a fresh run") and "testonly_llm call Claude" in line for line in lines), lines
    spans = trace_path.read_text(encoding="utf-8").splitlines()
    trace_path.write_text("\n".join(spans[:-1]) + "\n", encoding="utf-8", newline="\n")
    bad = {i.what: i.detail for i in fails(statuses(root))}
    assert "has no span for p03" in bad["runs/shared/testonly_llm.practice.trec matches its Claude trace"]
    trace_path.unlink()
    bad = {i.what: i.detail for i in fails(statuses(root))}
    assert "is missing" in bad["runs/shared/testonly_llm.practice.trec matches its Claude trace"]


def test_a_claude_run_that_repeats_its_first_stage_fails(tmp_path, add_system, monkeypatch):
    root = make_repo(tmp_path)
    add_system("testonly_llm", LLM_SYSTEM.format(call=llm_call("none")))
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25")
    with monkeypatch.context() as m:
        m.setattr(claude, "call", fake_rerank(reverse=False))
        run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "testonly_llm")
    # the scores differ from bm25's, but the documents and their order are the same for every query
    bad = {i.what: i.detail for i in fails(statuses(root))}
    assert "bm25.practice.trec for every query" in bad["runs/shared/testonly_llm.practice.trec is not a copy of another run"]


def test_catching_every_error_around_the_call_does_not_hide_it(tmp_path, add_system, monkeypatch):
    root = make_repo(tmp_path)
    add_system("testonly_llm", LLM_SYSTEM.format(call=llm_call("bare")))
    with monkeypatch.context() as m:
        m.setattr(claude, "call", fake_rerank())
        run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "testonly_llm")
    lines = [i.line() for i in statuses(root)]
    assert not [line for line in lines if line.startswith("FAIL") and "testonly_llm" in line], lines
    assert any("testonly_llm call Claude" in line for line in lines)


def test_a_claude_system_asks_before_many_calls(tmp_path, add_system, monkeypatch, capsys):
    root = make_repo(tmp_path)
    add_system("testonly_llm", LLM_SYSTEM.format(call=llm_call("none")))
    monkeypatch.setattr(claude, "call", fake_rerank())
    assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "testonly_llm", "--repeat", "2") == 2
    out = capsys.readouterr().out
    assert "makes up to 6 claude -p calls" in out and "--yes" in out
    assert not (root / "runs").exists()
    assert run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "testonly_llm", "--repeat", "2", "--yes") == 0


# ---- the rest of integrity ----


def test_a_changed_shared_file_fails_against_the_pins(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    pins = check.compute_pins(root)
    monkeypatch.setattr(check, "load_pins", lambda: pins)
    assert not fails(statuses(root))
    with open(root / "eval" / "shared" / "practice.qrels.txt", "a", encoding="utf-8") as fh:
        fh.write("p01 0 cfr30-75.400 1\n")
    bad = {i.what: i.detail for i in fails(statuses(root))}
    detail = bad["shared corpus, queries, judgments and questions are the course's copy"]
    assert "eval/shared/practice.qrels.txt" in detail and "git checkout" in detail


def test_a_deleted_eval_table_fails(tmp_path):
    root = make_repo(tmp_path)
    run(root, "run", "--corpus", "shared", "--queries", "practice", "--system", "bm25")
    run(root, "score")
    path = root / "EVAL.md"
    text = path.read_text(encoding="utf-8")
    start, end = text.index("<!-- p2:begin shared-practice -->"), text.index("<!-- p2:end shared-practice -->")
    path.write_text(text[:start] + "| bm25 | 0.999 |" + text[end + len("<!-- p2:end shared-practice -->") :], encoding="utf-8", newline="\n")
    bad = {i.what: i.detail for i in fails(statuses(root))}
    assert "1 table(s) are missing, such as shared-practice" in bad["EVAL.md tables match a fresh `p2 score`"]


def test_compare_lists_looks_up_a_swapped_document_in_the_deeper_ranking():
    fresh = [(f"d{i}", 1.0 - i / 10) for i in range(10)]
    deep = fresh + [(f"e{i}", 0.05 - i / 1000) for i in range(20)]
    assert check.compare_lists(fresh, fresh, deep) is None
    swapped = fresh[:9] + [("zz", fresh[9][1])]  # the last document replaced, its score kept
    assert "does not rank in its top 30" in check.compare_lists(swapped, fresh, deep)
    below = fresh[:9] + [("e0", fresh[9][1])]  # a real document with a made-up score
    assert "where a fresh run gives it" in check.compare_lists(below, fresh, deep)
    # a system whose scores depend on k: the deeper ranking is not used, and the old near-tie rule applies
    other = [(d, s + 0.01) for d, s in fresh] + deep[10:]
    assert check.compare_lists(swapped, fresh, other) is None


def test_a_manifest_saved_with_a_byte_order_mark_is_read(tmp_path):
    root = make_repo(tmp_path)
    docs = root / "corpora" / "own" / "docs"
    rows = ["docid\ttitle\tsource\tlicense\tnotes"]
    for i in range(3):
        (docs / f"doc-{i}.md").write_text(f"# Doc {i}\n\nPumps and valves, part {i}.\n", encoding="utf-8", newline="\n")
        rows.append(f"doc-{i}\tDoc {i}\tmy notes\town-work\t")
    (root / "corpora" / "own" / "manifest.tsv").write_text("﻿" + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    items = {i.what: i for i in statuses(root)}
    assert items["own corpus documents and manifest"].status == "PASS"
    assert items["own corpus licenses (p2 license, offline)"].status == "PASS"


def test_corpus_over_the_token_limit_fails_and_a_stale_ingest_report_is_a_todo(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    docs = root / "corpora" / "own" / "docs"
    rows = ["docid\ttitle\tsource\tlicense\tnotes"]
    for i in range(3):
        (docs / f"doc-{i}.md").write_text(f"# Doc {i}\n\nPumps and valves, part {i}.\n", encoding="utf-8", newline="\n")
        rows.append(f"doc-{i}\tDoc {i}\tmy notes\town-work\t")
    (root / "corpora" / "own" / "manifest.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    (root / "corpora" / "own" / "INGEST.md").write_text("# Ingest report\n\n- Documents: 4\n\n| `doc-0` | 6 | - | - | - |\n| `doc-9` | 6 | - | - | - |\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(check, "OWN_TOKEN_LIMIT", 10)
    items = {i.what: i for i in statuses(root)}
    size = items[check.OWN_SIZE_ITEM]
    assert size.status == "FAIL" and "above the limit of 10" in size.detail and "45 minutes" in size.detail
    report = items["corpora/own/INGEST.md describes the documents in docs/"]
    assert report.status == "TODO" and "doc-9" in report.detail and "p2 ingest --report" in report.detail


def test_a_corpus_near_the_token_limit_passes_with_a_note_and_a_small_one_gets_the_part_pages_hint(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    docs = root / "corpora" / "own" / "docs"
    rows = ["docid\ttitle\tsource\tlicense\tnotes"]
    for i in range(3):
        (docs / f"doc-{i}.md").write_text(f"# Doc {i}\n\nPumps and valves, part {i}.\n", encoding="utf-8", newline="\n")
        rows.append(f"doc-{i}\tDoc {i}\tmy notes\town-work\t")
    (root / "corpora" / "own" / "manifest.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(check, "OWN_TOKEN_WARNING", 10)
    items = {i.what: i for i in statuses(root)}
    size = items[check.OWN_SIZE_ITEM]
    assert size.status == "PASS" and "near the limit of 1,000,000" in size.detail and "15 minutes, once" in size.detail
    floor = items["own corpus has at least 200 documents"]
    assert floor.status == "TODO" and floor.detail.startswith("it has 3;") and "--part-pages 5" in floor.detail


def test_an_answers_file_with_a_failed_call_is_not_complete(tmp_path, monkeypatch):
    from test_core_answer import fake_call

    from p2 import answer

    def failing(prompt, *args, **kwargs):
        if "cranes" in prompt:
            return claude.Reply(None, 1.0, 0, 0, "sonnet", error="no reply within 600 s")
        return fake_call(prompt, *args, **kwargs)

    root = make_repo(tmp_path)
    monkeypatch.setattr(answer.claude, "call", failing)
    assert run(root, "answer", "--system", "bm25") == 0
    item = next(i for i in statuses(root) if i.what == "an answers file on all the shared questions")
    assert item.status == "TODO" and "the call for a03 failed" in item.detail


def test_notes_point_at_a_hollow_gold_set_without_failing(tmp_path):
    root = make_repo(tmp_path)
    docs = root / "corpora" / "own" / "docs"
    rows = ["docid\ttitle\tsource\tlicense\tnotes"]
    for i in range(5):
        (docs / f"doc-{i}.md").write_text(f"# Doc {i}\n\nPumps and valves, part {i}.\n", encoding="utf-8", newline="\n")
        rows.append(f"doc-{i}\tDoc {i}\tmy notes\town-work\t")
    (root / "corpora" / "own" / "manifest.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    patterned = "".join(f"o{n}\tmixed\tvalve requirement number {n} for pumps\tclaude\n" for n in range(3, 6))
    (root / "eval" / "own" / "queries.tsv").write_text("o01\tmixed\tpumps and valves\thand\no02\tmixed\tPumps, and valves?\tclaude\n" + patterned, encoding="utf-8", newline="\n")
    (root / "eval" / "own" / "qrels.txt").write_text("".join(f"{q} 0 doc-{i} 1\n" for q in ("o01", "o02", "o3", "o4", "o5") for i in range(5)), encoding="utf-8", newline="\n")
    run(root, "run", "--corpus", "own", "--system", "bm25")
    run(root, "score")
    notes = {i.what: i.detail for i in statuses(root) if i.status == "NOTE"}
    assert "own gold set: relevant documents per query" in notes
    assert "o02 reads the same as o01" in notes["own gold set: repeated queries"]
    assert "own gold set: room to tell systems apart" in notes
    assert "3 queries read alike apart from their numbers" in notes["own gold set: queries made from one pattern"]
    assert run(root, "check") == 0
