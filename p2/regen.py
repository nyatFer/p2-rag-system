"""Run committed systems again for `p2 check`, in a process of their own, with Claude switched off.

    python -m p2.regen ROOT PLAN.json OUT.json

`p2 check` writes the plan (which systems to run on which corpus and query set), starts this module
with P2_NO_CLAUDE set, and reads OUT.json. The check itself never imports your retrievers, so nothing
in them can change how the check judges their runs; this process is the only place they run.

For every job OUT.json holds a status:
- "ok": the fresh rankings at k, and deeper ones (DEEP documents) that let the check look up the
  score of a document just below the cut-off;
- "claude": the system tried to call Claude, so its runs are checked against their committed trace;
- "not_built": the system still raises NotImplementedError;
- "error": anything else went wrong, with the error.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

DEEP = 50


def _validate(name: str, hits, docs: dict, k: int) -> list[list]:
    hits = [(str(d), round(float(s), 6)) for d, s in hits]
    ids = [d for d, _ in hits]
    if len(hits) > k:
        raise ValueError(f"{name}.search returned {len(hits)} documents for k = {k}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}.search returned a document twice")
    unknown = [d for d in ids if d not in docs]
    if unknown:
        raise ValueError(f"{name}.search returned {unknown[0]}, which is not a document id of the corpus")
    return [list(h) for h in hits]


def _short(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text[:300]}" if text else type(error).__name__


def main(argv: list[str]) -> int:
    os.environ["P2_NO_CLAUDE"] = "1"
    root, plan_path, out_path = Path(argv[0]), Path(argv[1]), Path(argv[2])
    from p2 import claude, config, paths, retrievers
    from p2 import corpus as corpus_mod
    from p2.runfile import read_queries

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    try:
        cfg = config.load(root)
    except config.ConfigError:
        cfg = config.Config(root=root)  # the check reports the settings problem; run with the defaults
    k = int(plan.get("k") or cfg.k)
    deep = max(DEEP, 3 * k)
    corpora: dict[str, corpus_mod.Corpus] = {}
    out: dict[str, dict] = {}
    groups: dict[tuple[str, str], list[dict]] = {}
    for job in plan["jobs"]:
        groups.setdefault((job["corpus"], job["system"]), []).append(job)
    for (corpus_name, name), jobs in groups.items():
        before = claude.blocked_calls
        flag = None
        results: dict[str, dict] = {}
        try:
            module = retrievers.module(name)
            flag = bool(getattr(module, "NEEDS_CLAUDE", False))
            if corpus_name not in corpora:
                corpora[corpus_name] = corpus_mod.load(root, corpus_name)
            corpus = corpora[corpus_name]
            system = retrievers.build(name, corpus, cfg)
            for job in jobs:
                queries, _ = read_queries(paths.queries(root, corpus_name, job["query_set"]))
                fresh, deeper = {}, {}
                for q in queries:
                    fresh[q.qid] = _validate(name, system.search(q.text, k), corpus.docs, k)
                    deeper[q.qid] = _validate(name, system.search(q.text, deep), corpus.docs, deep)
                results[job["id"]] = {"status": "ok", "fresh": fresh, "deep": deeper}
            status, detail = "ok", ""
        except claude.ClaudeBlocked:
            status, detail = "claude", ""
        except NotImplementedError as error:
            status, detail = "not_built", _short(error)
        except Exception as error:  # noqa: BLE001 - a student's retriever can fail in any way; report it against that system
            status, detail = "error", _short(error)
            print(f"p2 check: running {name} again raised this:", file=sys.stderr)
            traceback.print_exc(limit=4, file=sys.stderr)
        if status == "ok" and claude.blocked_calls > before:
            status = "claude"  # it tried to call Claude and caught what stopped it
        for job in jobs:
            entry = results.get(job["id"]) if status == "ok" else None
            out[job["id"]] = {**(entry or {}), "status": status, "detail": detail, "needs_claude": flag}
    out_path.write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
