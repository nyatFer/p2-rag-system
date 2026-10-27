"""The answer harness with Claude replaced by a stand-in: the answers file, the trace, and the quote check."""

import json

from core_repo import make_repo

from p2 import answer, cli, claude, trace


def fake_call(prompt, schema, system=None, model="sonnet", cache_dir=None, timeout=600):
    assert system == "Answer only from the chunks."  # the # note line is removed
    if "cranes" in prompt:
        out = {"answer": "", "not_found": True, "claims": []}
    else:
        start = prompt.index('<chunk id="') + len('<chunk id="')
        cid = prompt[start : prompt.index('"', start)]
        body = prompt[prompt.index(">", start) + 1 : prompt.index("</chunk>")].split()
        out = {"answer": "yes", "not_found": False, "claims": [{"text": "t", "chunk_id": cid, "quote": " ".join(body[-4:])}]}
    trace.add_usage("claude-sonnet-4-5", 100, 20)
    return claude.Reply(out, 0.5, 100, 20, "claude-sonnet-4-5")


def test_answers_file_shape_and_totals(tmp_path, monkeypatch, capsys):
    root = make_repo(tmp_path)
    monkeypatch.setattr(answer.claude, "call", fake_call)
    assert cli.main(["--root", str(root), "answer", "--corpus", "shared", "--system", "bm25", "--label", "bm25-top3"]) == 0
    data = json.loads((root / "answers" / "shared" / "bm25-top3.json").read_text(encoding="utf-8"))
    assert data["chunking"] == {"words": 20, "overlap": 5} and data["top"] == 3
    record = data["answers"][0]
    for field in ("qid", "question", "kind", "system", "retrieved", "answer", "not_found", "claims", "seconds", "input_tokens", "output_tokens", "model"):
        assert field in record
    assert len(record["retrieved"]) == 3 and all("#" in c for c in record["retrieved"])
    out = capsys.readouterr().out
    assert "Answers bm25-top3 | verified 2 of 2 (100%) | not_found in 0 of 2 (0%) | declined out 1 of 1 (100%)" in out
    spans = trace.read(root / "traces" / "answers-bm25-top3.jsonl")
    assert [s["gen_ai.operation.name"] for s in spans] == ["chat"] * 3
    assert spans[0]["gen_ai.usage.output_tokens"] == 20
    assert cli.main(["--root", str(root), "verify", "answers/shared/bm25-top3.json"]) == 0


def test_repeat_and_only(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setattr(answer.claude, "call", fake_call)
    assert cli.main(["--root", str(root), "answer", "--system", "bm25", "--only", "a01,a03", "--repeat", "2"]) == 0
    for r in (1, 2):
        data = json.loads((root / "answers" / "shared" / f"bm25.r{r}.json").read_text(encoding="utf-8"))
        assert [a["qid"] for a in data["answers"]] == ["a01", "a03"] and data["repeat"] == r
    assert cli.main(["--root", str(root), "answer", "--system", "bm25", "--only", "zz"]) == 2
