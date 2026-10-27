"""Traces: one JSON line (a span) per query of a run, or per question of an answers file.

Field names follow the OpenTelemetry GenAI semantic conventions, which are still marked
"Development" and may rename attributes, so every span pins the version it follows in "semconv".
A Claude call made while a span is open (the reranker, an answer, a judged claim, an agent's search)
adds its model, token counts and seconds to that span, so the trace of a reranked run, an answers file
or a judged file records what the calls cost.

To look at a trace, open the .jsonl file: one line per query, question or claim. `p2 score` adds up the
calls, tokens and seconds of every trace in traces/ in the stretch-cost table of EVAL.md.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SEMCONV = "gen_ai, as of 2026-10"
_open_spans: list[dict] = []


class Tracer:
    """Writes spans to one .jsonl file, replacing what the file held before."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8", newline="\n")

    @contextmanager
    def span(self, name: str, operation: str, system: str, qid: str, model: str | None = None):
        """Times the block and writes one span; the block can add fields to the yielded dict."""
        record = {
            "semconv": SEMCONV,
            "name": name,
            "start": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "duration_ms": 0.0,
            "gen_ai.operation.name": operation,
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": 0,
            "gen_ai.usage.output_tokens": 0,
            "p2.qid": qid,
            "p2.system": system,
            "p2.top_ids": [],
            "p2.claude_calls": 0,
        }
        started = time.perf_counter()
        _open_spans.append(record)
        try:
            yield record
        finally:
            _open_spans.remove(record)
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def add_usage(
    model: str | None, input_tokens: int, output_tokens: int, cost_usd: float | None = None, cached: bool = False,
    seconds: float | None = None, turns: int | None = None,
) -> None:
    """Add one Claude call to the innermost open span (does nothing when no span is open).

    A saved reply (cached) costs no time now, so the seconds the original call took are kept in
    p2.saved_seconds: the span's duration plus those seconds is what its calls cost when they were made.
    turns is how many turns an agent with tools took (p2.turns)."""
    if not _open_spans:
        return
    span = _open_spans[-1]
    span["p2.claude_calls"] += 1
    span["gen_ai.usage.input_tokens"] += int(input_tokens or 0)
    span["gen_ai.usage.output_tokens"] += int(output_tokens or 0)
    if model:
        span["gen_ai.response.model"] = model
        if span["gen_ai.request.model"] is None:
            span["gen_ai.request.model"] = model
    if cost_usd is not None:
        span["p2.cost_usd"] = round(span.get("p2.cost_usd", 0.0) + float(cost_usd), 6)
    if cached:
        span["p2.cached_calls"] = span.get("p2.cached_calls", 0) + 1
        if seconds:
            span["p2.saved_seconds"] = round(span.get("p2.saved_seconds", 0.0) + float(seconds), 1)
    if turns:
        span["p2.turns"] = span.get("p2.turns", 0) + int(turns)


def read(path: Path) -> list[dict]:
    """The spans of a trace file."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
