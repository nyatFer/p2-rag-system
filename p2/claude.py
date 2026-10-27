"""One `claude -p` call, the way the 13 lab makes it, for the reranker, the answers and the judge.

- It launches the claude command by its full path, on the model from p2.toml (default Sonnet),
  with --tools "" (no tools: the model sees only what the prompt gives it), --json-schema (the reply
  is forced into that shape), --output-format json (so the token counts come back), and, when a
  system prompt is given, --system-prompt (it replaces Claude Code's own instructions).
- It runs in an empty temporary folder with --setting-sources project,local, so no CLAUDE.md or
  user setting changes the call, and --no-session-persistence, so calls do not fill your history.
- The one exception is an agent that searches with tools, as in the 12 lab's agent_search.py
  (stretch option 5): call(..., tools=AGENT_TOOLS, cwd=<a docs folder>) gives Claude the Grep, Glob
  and Read tools only (--tools and --allowedTools), and runs it in a fresh copy of that folder in
  the system's temporary space, each tool allowed only inside that copy (Read(./**) and the like, so
  even an absolute path elsewhere is denied): the agent searches your documents and nothing else, cannot
  reach the rest of your repo (your gold set above all), and loads no CLAUDE.md, which also keeps
  every call cheaper than it was in the lab. Every other rule below still holds.
- It uses your Claude Code sign-in: ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are removed from the
  child's environment, so a leftover key is never billed by accident.
- The prompt goes in on standard input, which has no length limit (a command line on Windows stops
  near 32,000 characters).
- With a cache folder, a reply is saved under a hash of everything sent (for an agent, also its tools
  and the contents of its folder), and the same call later returns the saved reply at no cost; pass
  no cache folder (or --fresh) to ask again.
- The call's model, token counts and seconds are added to the open trace span (see p2/trace.py).
- `p2 check` runs your systems again with Claude switched off (the P2_NO_CLAUDE environment
  variable): a call then raises ClaudeBlocked, which is how the check learns that a system really
  calls Claude. ClaudeBlocked is not an Exception, so an `except Exception` in your code lets it through.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from p2 import trace

KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
AGENT_TOOLS = ("Grep", "Glob", "Read")  # the only tools a call may have: they read and search, and change nothing
NO_CLAUDE_VAR = "P2_NO_CLAUDE"
TIMEOUT_S = 600
_warned_keys = False
blocked_calls = 0  # how many calls ClaudeBlocked stopped in this process (p2 check reads it)


class ClaudeNotFound(RuntimeError):
    pass


class ClaudeBlocked(BaseException):
    """Raised by call() while Claude is switched off (P2_NO_CLAUDE is set), as it is inside p2 check.

    It derives from BaseException, like KeyboardInterrupt, so a reranker that catches Exception to
    keep its first-stage order on a failed call does not hide it."""


@dataclass
class Reply:
    output: dict | None  # the structured reply, or None on an error
    seconds: float
    input_tokens: int
    output_tokens: int
    model: str
    error: str | None = None
    cost_usd: float | None = None
    cached: bool = False
    turns: int | None = None  # how many turns an agent with tools took (None for a call without tools)


def find_claude() -> str:
    """The full path of the claude command."""
    path = shutil.which("claude")
    if not path:
        raise ClaudeNotFound("Could not find the claude command; install Claude Code (see setup.md in the course repo) and open a new terminal.")
    if Path(path).suffix.lower() in (".cmd", ".bat", ".ps1"):
        # Windows runs a .cmd or .bat file through cmd.exe, which cuts a command line at its first
        # newline and treats & | < > % as commands, so a multi-line prompt or a JSON schema would arrive broken.
        raise ClaudeNotFound(
            f"The claude command here is {path}, a Windows script that cannot pass a multi-line prompt safely. "
            "Install Claude Code with the official installer from setup.md in the course repo, open a new terminal, and check that `where claude` lists a claude.exe first."
        )
    return path


def child_env() -> dict[str, str]:
    """A copy of the environment without API keys, so the call uses your sign-in."""
    global _warned_keys
    removed = [k for k in KEY_VARS if os.environ.get(k)]
    if removed and not _warned_keys:
        print(f"Note: {', '.join(removed)} is set in your environment; claude -p calls ignore it and use your sign-in.", file=sys.stderr)
        _warned_keys = True
    return {k: v for k, v in os.environ.items() if k not in KEY_VARS}


def check_tools(tools) -> tuple[str, ...]:
    """The tools of a call, checked: none, or some of Grep, Glob and Read (the order is kept)."""
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",") if t.strip()]
    tools = tuple(tools or ())
    other = [t for t in tools if t not in AGENT_TOOLS]
    if other or len(set(tools)) != len(tools):
        raise ValueError(f"A claude -p call in p2 may use only the tools {', '.join(AGENT_TOOLS)}, each once; not {', '.join(other) or 'a tool twice'}.")
    return tools


def command(claude: str, schema: dict, model: str, system: str | None, tools: tuple[str, ...] = ()) -> list[str]:
    """The claude -p command line; the prompt itself goes in on standard input."""
    tools = check_tools(tools)
    # Each tool is allowed only inside the working folder (the fresh copy of the documents): a path
    # outside it, even an absolute one, is denied, so the agent cannot open the gold set.
    allowed = ["--tools", ",".join(tools), "--allowedTools", ",".join(f"{t}(./**)" for t in tools)] if tools else ["--tools", ""]
    cmd = [
        claude, "-p",
        "--model", model,
        *allowed,
        "--output-format", "json",
        "--json-schema", json.dumps(schema),
        "--setting-sources", "project,local",
        "--no-session-persistence",
    ]  # fmt: skip
    if system is not None:
        cmd += ["--system-prompt", system]
    return cmd


def parse(stdout: str, stderr: str, seconds: float, model: str) -> Reply:
    """A Reply from claude's --output-format json output."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        detail = (stderr.strip() or stdout.strip() or "no output")[:300]
        return Reply(None, seconds, 0, 0, model, error=f"unparseable output: {detail}")
    if not isinstance(data, dict):
        return Reply(None, seconds, 0, 0, model, error="unexpected output shape")
    usage = data.get("usage") or {}
    tokens_in = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    tokens_out = int(usage.get("output_tokens") or 0)
    used = list((data.get("modelUsage") or {}).keys())
    resolved = max(used, key=lambda m: (data["modelUsage"][m] or {}).get("outputTokens", 0)) if used else model
    cost = data.get("total_cost_usd")
    turns = data.get("num_turns") if isinstance(data.get("num_turns"), int) else None
    output = data.get("structured_output")
    if data.get("is_error") or not isinstance(output, dict):
        reason = data.get("result") if isinstance(data.get("result"), str) and data.get("result") else data.get("subtype")
        return Reply(None, seconds, tokens_in, tokens_out, resolved, error=str(reason or "no structured_output in the reply")[:300], cost_usd=cost, turns=turns)
    return Reply(output, seconds, tokens_in, tokens_out, resolved, cost_usd=cost, turns=turns)


_fingerprints: dict[tuple, str] = {}


def folder_fingerprint(folder: Path) -> str:
    """A short hash of every file name and file content under a folder (dot files left out), so an
    agent's saved reply is used again only while its documents are the same."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts))
    stamp = (str(folder.resolve()), tuple((p.as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in files))
    if stamp not in _fingerprints:
        h = hashlib.sha256()
        for p in files:
            h.update(p.relative_to(folder).as_posix().encode("utf-8") + b"\0" + p.read_bytes() + b"\0")
        _fingerprints[stamp] = h.hexdigest()[:24]
    return _fingerprints[stamp]


def cache_key(prompt: str, schema: dict, model: str, system: str | None, tools: tuple[str, ...] = (), folder: str | None = None) -> str:
    """The name a reply is saved under; a call without tools keeps the key it always had."""
    parts = [model, system, schema, prompt] + ([list(tools), folder] if tools else [])
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def call(
    prompt: str, schema: dict, system: str | None = None, model: str = "sonnet", cache_dir: Path | None = None,
    timeout: int = TIMEOUT_S, tools=(), cwd: Path | None = None,
) -> Reply:
    """One claude -p call with a reply forced into `schema`.

    With no tools (the default) Claude sees only the prompt. With tools (some of AGENT_TOOLS) it runs
    as an agent on a fresh copy of `cwd` (a folder that must exist) in the system's temporary space, and
    can search and read those files and nothing else."""
    global blocked_calls
    if os.environ.get(NO_CLAUDE_VAR):
        blocked_calls += 1
        raise ClaudeBlocked("Claude is switched off here (p2 check never calls Claude)")
    tools = check_tools(tools)
    if tools and (cwd is None or not Path(cwd).is_dir()):
        raise ValueError("A claude -p call with tools needs cwd, the folder it searches (for example your corpus's docs folder).")
    folder = folder_fingerprint(Path(cwd)) if tools else None
    cache_file = Path(cache_dir) / f"{cache_key(prompt, schema, model, system, tools, folder)}.json" if cache_dir else None
    if cache_file and cache_file.is_file():
        saved = json.loads(cache_file.read_text(encoding="utf-8"))
        reply = Reply(saved["output"], saved["seconds"], saved["input_tokens"], saved["output_tokens"], saved["model"], cost_usd=saved.get("cost_usd"), cached=True, turns=saved.get("turns"))
        trace.add_usage(reply.model, reply.input_tokens, reply.output_tokens, reply.cost_usd, cached=True, seconds=reply.seconds, turns=reply.turns)
        return reply
    cmd = command(find_claude(), schema, model, system, tools)
    start = time.perf_counter()
    try:
        if tools:
            # A fresh copy of the documents, outside the repo: the agent can read only them, cannot reach
            # the gold set or anything else around them, and loads no CLAUDE.md from the folders above.
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                work = Path(tmp) / "docs"
                shutil.copytree(Path(cwd), work, ignore=shutil.ignore_patterns(".*"))
                done = subprocess.run(
                    cmd, cwd=work, env=child_env(), input=prompt, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=timeout,
                )  # fmt: skip
        else:
            # An empty folder, so no CLAUDE.md is picked up; on Windows a virus scanner can hold it for a
            # moment after the call, and a folder left behind must not lose the reply.
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as empty:
                done = subprocess.run(
                    cmd, cwd=empty, env=child_env(), input=prompt, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=timeout,
                )  # fmt: skip
    except subprocess.TimeoutExpired:
        seconds = time.perf_counter() - start
        trace.add_usage(model, 0, 0)
        return Reply(None, seconds, 0, 0, model, error=f"no reply within {timeout} s")
    seconds = time.perf_counter() - start
    reply = parse(done.stdout, done.stderr, seconds, model)
    if not tools:
        reply.turns = None  # a call without tools always takes the same turns; only an agent's are worth keeping
    trace.add_usage(reply.model, reply.input_tokens, reply.output_tokens, reply.cost_usd, turns=reply.turns)
    if cache_file and reply.error is None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        saved = {"output": reply.output, "seconds": round(seconds, 1), "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens, "model": reply.model, "cost_usd": reply.cost_usd}
        if reply.turns is not None:
            saved["turns"] = reply.turns
        cache_file.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8", newline="\n")
    return reply


CONFIRM_ABOVE = 5  # more claude -p calls than this, and p2 asks first


def confirm_calls(calls: int, what: str, yes: bool) -> bool:
    """True when a command may make `calls` claude -p calls: few enough, --yes, or a yes at the prompt.

    Without a terminal to ask in (Claude Code running the command, say), p2 stops and says how to go
    ahead, so whoever runs it has to bring the number back to the student first."""
    if calls <= CONFIRM_ABOVE or yes:
        return True
    message = f"{what} makes up to {calls} claude -p calls, which draw on your Claude plan (replies saved from an earlier run cost nothing)."
    stdin = sys.stdin
    if stdin is not None and stdin.isatty():
        try:
            answer = input(message + " Go ahead? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() in ("y", "yes"):
            return True
        print("Nothing was run.")
        return False
    print(message)
    print("Nothing was run. To go ahead, run the same command again with --yes; if Claude Code is running it for you, it should tell you this number and ask you first.")
    return False


def cache_folder(cfg) -> Path | None:
    """The reply cache for these settings: .cache/claude/, or None when replies must be fresh."""
    return cfg.root / ".cache" / "claude" if cfg.cache_claude else None
