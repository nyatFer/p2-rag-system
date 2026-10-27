"""The claude -p helper: the command it builds, how it reads replies, the cache, the trace, and a fake claude end to end,
with no tools and as an agent with Grep, Glob and Read (stretch option 5)."""

import json
import os
import stat
import sys

import pytest

from p2 import claude, trace

SCHEMA = {"type": "object", "properties": {"order": {"type": "array"}}, "required": ["order"]}
REPLY = {
    "type": "result", "subtype": "success", "is_error": False, "result": "",
    "structured_output": {"order": [2, 1]},
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5, "output_tokens": 7},
    "modelUsage": {"claude-sonnet-4-5": {"outputTokens": 7}}, "total_cost_usd": 0.01,
}  # fmt: skip


def test_command_flags():
    cmd = claude.command("/bin/claude", SCHEMA, "sonnet", "be brief")
    assert cmd[:2] == ["/bin/claude", "-p"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--setting-sources") + 1] == "project,local"
    assert cmd[-2:] == ["--system-prompt", "be brief"]
    assert "--system-prompt" not in claude.command("/bin/claude", SCHEMA, "sonnet", None)


def test_parse_success_error_and_garbage():
    ok = claude.parse(json.dumps(REPLY), "", 1.5, "sonnet")
    assert ok.output == {"order": [2, 1]} and ok.error is None
    assert (ok.input_tokens, ok.output_tokens, ok.model) == (115, 7, "claude-sonnet-4-5")
    failed = claude.parse(json.dumps({**REPLY, "is_error": True, "result": "Not logged in"}), "", 1.0, "sonnet")
    assert failed.output is None and failed.error == "Not logged in"
    garbage = claude.parse("oops", "boom", 1.0, "sonnet")
    assert garbage.error.startswith("unparseable output: boom")


def test_child_env_drops_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    env = claude.child_env()
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env


@pytest.mark.skipif(os.name == "nt", reason="the fake claude is a POSIX script")
def test_fake_claude_end_to_end_with_cache_and_trace(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    fake = bin_dir / "claude"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "prompt = sys.stdin.read()\n"
        f"open({str(log)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'prompt': prompt, 'cwd': os.getcwd(), 'key': os.environ.get('ANTHROPIC_API_KEY')}}) + '\\n')\n"
        f"sys.stdout.write({json.dumps(REPLY)!r})\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    cache = tmp_path / "cache"
    with trace.Tracer(tmp_path / "t.jsonl") as tracer:
        with tracer.span("retrieval rerank", "retrieval", "rerank", "q1"):
            first = claude.call("Rank these.", SCHEMA, model="sonnet", cache_dir=cache)
            second = claude.call("Rank these.", SCHEMA, model="sonnet", cache_dir=cache)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 1  # the second call came from the cache
    assert calls[0]["prompt"] == "Rank these." and calls[0]["key"] is None
    assert calls[0]["cwd"] != os.getcwd()
    assert first.output == second.output == {"order": [2, 1]} and second.cached
    span = trace.read(tmp_path / "t.jsonl")[0]
    assert span["p2.claude_calls"] == 2 and span["p2.cached_calls"] == 1
    assert span["gen_ai.usage.input_tokens"] == 230 and span["gen_ai.response.model"] == "claude-sonnet-4-5"


def test_command_with_tools_allows_only_grep_glob_and_read():
    cmd = claude.command("/bin/claude", SCHEMA, "sonnet", None, claude.AGENT_TOOLS)
    assert cmd[cmd.index("--tools") + 1] == "Grep,Glob,Read"
    assert cmd[cmd.index("--allowedTools") + 1] == "Grep(./**),Glob(./**),Read(./**)"
    for flag, value in (("--model", "sonnet"), ("--output-format", "json"), ("--setting-sources", "project,local")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--no-session-persistence" in cmd and "--allowedTools" not in claude.command("/bin/claude", SCHEMA, "sonnet", None)
    assert claude.check_tools("Grep, Read") == ("Grep", "Read")
    for bad in (("Bash",), ("Read", "Edit"), ("Read", "Read")):
        with pytest.raises(ValueError):
            claude.command("/bin/claude", SCHEMA, "sonnet", None, bad)


def test_a_call_with_tools_needs_a_folder_and_claude_off_still_blocks_it(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="needs cwd"):
        claude.call("find it", SCHEMA, tools=claude.AGENT_TOOLS)
    monkeypatch.setenv(claude.NO_CLAUDE_VAR, "1")
    with pytest.raises(claude.ClaudeBlocked):
        claude.call("find it", SCHEMA, tools=claude.AGENT_TOOLS, cwd=tmp_path / "missing")


def test_the_cache_key_of_a_call_without_tools_is_unchanged():
    # Replies saved before agents existed must still be found.
    import hashlib

    blob = json.dumps(["sonnet", "sys", SCHEMA, "prompt"], sort_keys=True, ensure_ascii=False)
    assert claude.cache_key("prompt", SCHEMA, "sonnet", "sys") == hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    assert claude.cache_key("prompt", SCHEMA, "sonnet", "sys", ("Grep",), "abc") != claude.cache_key("prompt", SCHEMA, "sonnet", "sys")


AGENT_REPLY = {**REPLY, "structured_output": {"doc_ids": ["doc-1"]}, "num_turns": 5}


@pytest.mark.skipif(os.name == "nt", reason="the fake claude is a POSIX script")
def test_fake_agent_runs_in_its_folder_with_its_tools_and_a_cache_that_follows_the_documents(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    fake = bin_dir / "claude"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "prompt = sys.stdin.read()\n"
        f"open({str(log)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'prompt': prompt, 'cwd': os.getcwd(), 'key': os.environ.get('ANTHROPIC_API_KEY')}}) + '\\n')\n"
        f"sys.stdout.write({json.dumps(AGENT_REPLY)!r})\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc-1.md").write_text("# One\n\nPumps.\n", encoding="utf-8", newline="\n")
    cache = tmp_path / "cache"
    with trace.Tracer(tmp_path / "t.jsonl") as tracer:
        with tracer.span("retrieval agent", "retrieval", "agent", "q1"):
            first = claude.call("Find pumps.", SCHEMA, model="sonnet", cache_dir=cache, tools=claude.AGENT_TOOLS, cwd=docs)
            again = claude.call("Find pumps.", SCHEMA, model="sonnet", cache_dir=cache, tools=claude.AGENT_TOOLS, cwd=docs)
        (docs / "doc-2.md").write_text("# Two\n\nValves.\n", encoding="utf-8", newline="\n")
        with tracer.span("retrieval agent", "retrieval", "agent", "q2"):
            changed = claude.call("Find pumps.", SCHEMA, model="sonnet", cache_dir=cache, tools=claude.AGENT_TOOLS, cwd=docs)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 2  # the second call came from the cache; a new document asked again
    # The agent ran on a copy of the documents outside the repo, never in the docs folder itself.
    assert os.path.realpath(calls[0]["cwd"]) != os.path.realpath(docs) and calls[0]["key"] is None
    assert os.path.basename(calls[0]["cwd"]) == "docs" and not os.path.realpath(calls[0]["cwd"]).startswith(os.path.realpath(tmp_path))
    argv = calls[0]["argv"]
    assert argv[argv.index("--tools") + 1] == "Grep,Glob,Read" and argv[argv.index("--allowedTools") + 1] == "Grep(./**),Glob(./**),Read(./**)"
    assert calls[0]["prompt"] == "Find pumps."
    assert first.output == {"doc_ids": ["doc-1"]} and first.turns == 5 and again.cached and again.turns == 5 and not changed.cached
    spans = trace.read(tmp_path / "t.jsonl")
    assert spans[0]["p2.claude_calls"] == 2 and spans[0]["p2.turns"] == 10 and spans[0]["p2.cached_calls"] == 1
    assert spans[0].get("p2.saved_seconds", 0.0) == round(first.seconds, 1)  # the seconds the saved call took when it was made
