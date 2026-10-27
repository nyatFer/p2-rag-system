"""`p2 check`: is everything committed well-formed, honest and reproducible, and what is left to do?

It prints one line per item, PASS, FAIL, TODO or NOTE, then a summary, and exits 1 on any FAIL.
Every FAIL says what to do next. TODO is work you have not done yet, and it does not fail the check,
so a fresh copy of the template passes; `p2 check --final` (the submission bar, and what the
autograder runs) also exits 1 on any TODO. A NOTE is something worth a second look that never fails.

Integrity, the part CI runs on every push:
- the shared corpus and its queries, judgments and questions are the course's own copy (SHA-256);
- the formats of every corpus, manifest, queries, qrels, run and answers file; every document id in
  runs, qrels and answers exists in its corpus; no carriage return under corpora/;
- a run file named after a system (dense.practice.trec) was written by that system (its tag);
- every committed run is run again in a separate process with Claude switched off, and compared:
  every document's score within 0.001 of a fresh run's, so documents may trade places only on
  near-ties; a system that calls Claude is checked against its committed trace instead (one span
  per query with a Claude call, listing the same documents), and must not repeat another system's run;
- results/results.json and the EVAL.md tables equal a fresh `p2 score` (to 3 decimals), and none of
  the EVAL.md tables is missing; the one part left out is the seconds of the systems that do not call
  Claude in the stretch-cost table, which come from traces/retrieval/, a folder git ignores;
- every answers file passes the quote check's mechanics (its retrieved ids are real chunks, its
  claims have a chunk id and a quote), and its trace, when there is one, still has a span with the
  same chunks for every question; how many quotes verify is a result, reported by p2 score;
- stretch option 2: every judged file matches its answers file and its committed trace, and every
  calibration file (your own verdicts) is well-formed;
- stretch runs (runs/own/stretch/) use queries of your own gold set and are checked like the others;
- `p2 license` (offline) on your own corpus once it has documents, and the corpus size limits.
Completeness, reported as TODO until done:
- runs for bm25, dense, hybrid and rerank on shared practice, shared test and your own corpus;
- your own corpus (at least 200 documents) and gold set (at least 30 queries, at least 10 written by
  hand, each with a relevant document), one ablation run, an answers file on all 12 shared questions,
  and no TODO markers left in EVAL.md and DECISIONS.md;
- 598E: three repeated reranker runs and three complete answers files, PREREG.md sections 1 to 4
  committed before your first judgment and unchanged since, and its outcome written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from p2 import config, judge, paths, retrievers, score, trace, verify
from p2 import corpus as corpus_mod
from p2.limits import CI_MINUTES, OWN_MAX_BYTES, OWN_MAX_FILE_BYTES, OWN_MIN_DOCS, OWN_TOKEN_LIMIT, OWN_TOKEN_WARNING, SUGGESTED_PART_PAGES, TOKENS_PER_WORD
from p2.runfile import read_qrels, read_queries, read_questions, read_run, run_tag, summarize, validate_run

SCORE_TOLERANCE = 1e-3
# The own corpus's size rules live in p2/limits.py, which also says why they are what they are.
OWN_SIZE_ITEM = (
    f"own corpus size (at most {OWN_MAX_BYTES // 2**20} MB, {OWN_MAX_FILE_BYTES // 2**20} MB per file"
    f" and {OWN_TOKEN_LIMIT:,} estimated tokens)"
)
GOLD_MIN_QUERIES = 30
GOLD_MIN_HAND = 10
REPEATS = 3
RELEVANT_SHARE_NOTE = 0.10
TOO_EASY = 0.95
NO_CLAUDE_SYSTEMS = ("bm25", "dense", "hybrid")
TODO_MARK = re.compile(r"\bTODO\b")
RIDER_MARK = re.compile(r"^598E: write\b", re.M)
PINS_FILE = Path(__file__).with_name("shared.sha256")
PINNED_DIRS = ("corpora/shared", "eval/shared")
RESTORE_SHARED = "Restore it from the template with `git checkout $(git rev-list --max-parents=0 HEAD) -- corpora/shared eval/shared`, then commit."
SKILL_FILE = ".claude/skills/license-check/SKILL.md"


@dataclass
class Item:
    status: str  # PASS, FAIL, TODO or NOTE
    what: str
    detail: str = ""

    def line(self) -> str:
        return f"{self.status} {self.what}" + (f": {self.detail}" if self.detail else "")


class Report:
    def __init__(self):
        self.items: list[Item] = []

    def ok(self, what: str, detail: str = "") -> None:
        self.items.append(Item("PASS", what, detail))

    def fail(self, what: str, problem: str, fix: str) -> None:
        self.items.append(Item("FAIL", what, f"{problem}. {fix}"))

    def todo(self, what: str, detail: str) -> None:
        self.items.append(Item("TODO", what, detail))

    def note(self, what: str, detail: str) -> None:
        self.items.append(Item("NOTE", what, detail))


class Context:
    """What several checks share: the settings, the corpora, the queries and the run files."""

    def __init__(self, root: Path, cfg: config.Config):
        self.root = root
        self.cfg = cfg
        self._corpora: dict[str, corpus_mod.Corpus] = {}
        self.refs, self.odd = score.discover(root)
        self.good: list[score.RunRef] = []  # the run files that passed the format checks

    def corpus(self, name: str) -> corpus_mod.Corpus:
        if name not in self._corpora:
            self._corpora[name] = corpus_mod.load(self.root, name)
        return self._corpora[name]

    def doc_ids(self, name: str) -> set[str]:
        return {p.stem for p in corpus_mod.doc_files(paths.docs_dir(self.root, name))}

    def queries(self, corpus: str, query_set: str):
        path = paths.queries(self.root, corpus, query_set)
        return read_queries(path, require_origin=corpus == "own")[0] if path.is_file() else []

    def qrels(self, corpus: str, query_set: str) -> dict:
        path = paths.qrels(self.root, corpus, query_set)
        return read_qrels(path)[0] if path.is_file() else {}

    def chunk_texts(self, data: dict) -> dict[str, str]:
        """The chunks an answers file was made from (cut once per corpus and setting)."""
        chunking = data.get("chunking") or {}
        words = int(chunking.get("words", self.cfg.chunk_words))
        overlap = int(chunking.get("overlap", self.cfg.chunk_overlap))
        return self.corpus(data.get("corpus", "shared")).chunk_texts(words, overlap)


def rel(ctx: Context, path: Path) -> str:
    return paths.rel(ctx.root, path)


# ---- integrity ----


def load_pins() -> dict[str, str] | None:
    """{relative path: sha256} of every course-owned shared file, or None when this copy has no list."""
    if not PINS_FILE.is_file():
        return None
    pins = {}
    for line in PINS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(None, 1)
            pins[name.strip()] = digest
    return pins


def compute_pins(root: Path) -> dict[str, str]:
    """{relative path: sha256} of every file under corpora/shared/ and eval/shared/ (dot files left out)."""
    found = {}
    for folder in PINNED_DIRS:
        base = root / folder
        for path in sorted(base.rglob("*")) if base.is_dir() else []:
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
                found[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


PINS_HEADER = """\
# SHA-256 of every course-owned file under corpora/shared/ and eval/shared/.
# p2 check compares those folders with this list, so every shared score is computed against the course's copy.
# The template repo rebuilds it after a deliberate change to the shared corpus or its gold set:
#   uv run python -m p2.check --write-pins
"""


def render_pins(pins: dict[str, str]) -> str:
    return PINS_HEADER + "".join(f"{digest}  {name}\n" for name, digest in sorted(pins.items()))


def check_pinned(ctx: Context, r: Report) -> None:
    pins = load_pins()
    if pins is None:
        return
    what = "shared corpus, queries, judgments and questions are the course's copy"
    found = compute_pins(ctx.root)
    changed = sorted(name for name in pins if name in found and found[name] != pins[name])
    missing = sorted(set(pins) - set(found))
    extra = sorted(set(found) - set(pins))
    problems = [f"{len(changed)} file(s) differ from the template, such as {changed[0]}"] if changed else []
    problems += [f"{len(missing)} file(s) are missing, such as {missing[0]}"] if missing else []
    problems += [f"{len(extra)} file(s) were added, such as {extra[0]}"] if extra else []
    if problems:
        r.fail(what, "; ".join(problems) + " (these folders belong to the course, and every score is computed against them)", RESTORE_SHARED)
    else:
        r.ok(what, f"{len(pins):,} files")


def check_line_endings(ctx: Context, r: Report) -> None:
    base = ctx.root / "corpora"
    bad = [p for p in sorted(base.rglob("*")) if p.is_file() and b"\r" in p.read_bytes()] if base.is_dir() else []
    if bad:
        r.fail("no carriage returns under corpora/", f"{len(bad)} file(s) have Windows line endings, such as {rel(ctx, bad[0])}",
               "Convert them to LF line endings (your editor's line-ending setting, or run p2 ingest again) and commit.")  # fmt: skip
    else:
        r.ok("no carriage returns under corpora/")


def check_corpus(ctx: Context, r: Report, name: str) -> bool:
    """Format checks for one corpus; True when it has documents."""
    docs = corpus_mod.doc_files(paths.docs_dir(ctx.root, name))
    what = f"{name} corpus documents and manifest"
    if not docs:
        if name == "own":
            r.todo(what, "your own corpus has no documents yet; stage 2 starts with `uv run p2 ingest SRC_DIR`")
        else:
            r.todo(what, "the shared corpus has no documents in this copy")
        return False
    problems = corpus_mod.check_docs(paths.docs_dir(ctx.root, name)) + corpus_mod.check_manifest(ctx.root, name)
    if problems:
        fix = "Fix the files named (or run `uv run p2 ingest` again), then commit." if name == "own" else RESTORE_SHARED
        r.fail(what, summarize(problems), fix)
    else:
        r.ok(what, f"{len(docs):,} documents, one manifest row each")
    return True


def check_own_size(ctx: Context, r: Report) -> None:
    base = paths.corpus_dir(ctx.root, "own")
    files = [p for p in base.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    big = [p for p in files if p.stat().st_size > OWN_MAX_FILE_BYTES]
    what = OWN_SIZE_ITEM
    if big:
        r.fail(what, f"{rel(ctx, big[0])} is {big[0].stat().st_size / 1e6:.1f} MB", "Split it into parts or drop it, then commit.")
        return
    if total > OWN_MAX_BYTES:
        r.fail(what, f"corpora/own/ holds {total / 1e6:.1f} MB", f"Remove documents until it is under {OWN_MAX_BYTES // 2**20} MB, then commit.")
        return
    words = sum(len(d.text.split()) for d in ctx.corpus("own").docs.values())
    tokens = int(words * TOKENS_PER_WORD)
    note = f"{total / 1e6:.1f} MB, about {tokens:,} tokens"
    if tokens > OWN_TOKEN_LIMIT:
        r.fail(what, f"the corpus is about {tokens:,} tokens (words times {TOKENS_PER_WORD}), above the limit of {OWN_TOKEN_LIMIT:,}, so CI could not encode it within its {CI_MINUTES} minutes",
               "Remove documents (or cut long ones to the parts your queries need) until `uv run p2 check` says it is under the limit, then commit.")  # fmt: skip
        return
    if tokens > OWN_TOKEN_WARNING:
        note += (f"; that is above {OWN_TOKEN_WARNING:,} and near the limit of {OWN_TOKEN_LIMIT:,}, so the first CI run that encodes it"
                 " can take up to about 15 minutes, once, and later runs reuse the cache")  # fmt: skip
    r.ok(what, note)


def check_shared_eval(ctx: Context, r: Report, has_docs: bool) -> None:
    docs = ctx.doc_ids("shared")
    files = {
        "practice queries": paths.queries(ctx.root, "shared", "practice"),
        "practice qrels": paths.qrels(ctx.root, "shared", "practice"),
        "test queries": paths.queries(ctx.root, "shared", "test"),
        "questions": paths.questions(ctx.root, "shared"),
    }
    problems = []
    missing = [rel(ctx, p) for p in files.values() if not p.is_file()]
    for label, path in files.items():
        if not path.is_file():
            continue
        if label.endswith("queries"):
            problems += [f"{rel(ctx, path)} {p}" for p in read_queries(path)[1]]
        elif label.endswith("qrels"):
            qrels, found = read_qrels(path)
            problems += [f"{rel(ctx, path)} {p}" for p in found]
            unknown = sorted({d for rels in qrels.values() for d in rels if d not in docs})
            if docs and unknown:
                problems.append(f"{rel(ctx, path)} names {unknown[0]}, which is not in the shared corpus")
        else:
            questions, found = read_questions(path)
            problems += [f"{rel(ctx, path)} {p}" for p in found]
            unknown = sorted({g for q in questions for g in q.gold if g not in docs})
            if docs and unknown:
                problems.append(f"{rel(ctx, path)} names {unknown[0]}, which is not in the shared corpus")
    what = "shared queries, qrels and questions"
    if missing and not has_docs:
        r.todo(what, f"{missing[0]} is not in this copy yet")
    elif missing or problems:
        r.fail(what, summarize([f"{m} is missing" for m in missing] + problems), RESTORE_SHARED)
    else:
        r.ok(what)


def check_own_eval(ctx: Context, r: Report) -> None:
    qpath, rpath = paths.queries(ctx.root, "own", "own"), paths.qrels(ctx.root, "own", "own")
    what = "own gold set format (eval/own/queries.tsv and qrels.txt)"
    queries, problems = read_queries(qpath, require_origin=True) if qpath.is_file() else ([], [])
    problems = [f"queries.tsv {p}" for p in problems]
    qrels, found = read_qrels(rpath) if rpath.is_file() else ({}, [])
    problems += [f"qrels.txt {p}" for p in found]
    if not queries and not qrels and not problems:
        r.todo(what, "no queries yet; write them in eval/own/queries.tsv (stage 2)")
        return
    docs = ctx.doc_ids("own")
    unknown = sorted({d for rels in qrels.values() for d in rels if d not in docs})
    if unknown:
        problems.append(f"qrels.txt names {unknown[0]}, which is not in corpora/own/docs/")
    stray = sorted(set(qrels) - {q.qid for q in queries})
    if stray:
        problems.append(f"qrels.txt judges {stray[0]}, which is not in queries.tsv")
    if problems:
        r.fail(what, summarize(problems), "Fix those lines in eval/own/, then commit.")
    else:
        r.ok(what, f"{len(queries)} queries, {sum(len(v) for v in qrels.values())} judgments")


def _field(finding, name: str):
    return finding.get(name) if isinstance(finding, dict) else getattr(finding, name, None)


def check_license(ctx: Context, r: Report) -> None:
    from p2 import license as license_mod

    what = "own corpus licenses (p2 license, offline)"
    findings = list(license_mod.offline_report(paths.corpus_dir(ctx.root, "own")))
    bad = [f for f in findings if _field(f, "status") in ("fail", "flag")]
    if not bad:
        r.ok(what, f"{len(findings):,} documents checked")
        return
    fails = sum(1 for f in bad if _field(f, "status") == "fail")
    flags = len(bad) - fails
    first = bad[0]
    hits = _field(first, "flags") or []
    if _field(first, "status") == "flag" and hits:
        why = f"{hits[0]['kind']} on line {hits[0]['line_no']}" if hits[0].get("line_no") else hits[0]["kind"]
    else:
        why = str(_field(first, "reason") or "").split(";")[0]
    counts = " and ".join(x for x in (f"{fails} document(s) fail" if fails else "", f"{flags} flag(s) are unresolved" if flags else "") if x)
    r.fail(what, f"{counts} (first: {_field(first, 'docid')}, {why})",
           f"Run `uv run p2 license`, then have Claude Code read {SKILL_FILE} and walk you through each one.")  # fmt: skip


def check_ingest_report(ctx: Context, r: Report) -> None:
    path = paths.corpus_dir(ctx.root, "own") / "INGEST.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    listed = set(re.findall(r"^\| `([a-z0-9][a-z0-9._-]*)` \|", text, re.M))
    present = ctx.doc_ids("own")
    gone = sorted(listed - present)
    count = re.search(r"^- Documents: ([\d,]+)", text, re.M)
    what = "corpora/own/INGEST.md describes the documents in docs/"
    if gone or (count and int(count.group(1).replace(",", "")) != len(present)):
        detail = f"it lists {len(gone)} document(s) that are gone, such as {gone[0]}" if gone else f"it counts {count.group(1)} documents and docs/ has {len(present)}"
        r.todo(what, detail + "; `uv run p2 ingest --report` rewrites it for the documents there now")
    else:
        r.ok(what)


def check_runs(ctx: Context, r: Report) -> list[score.RunRef]:
    """Validate every run file; returns the ones that are well-formed."""
    for path in ctx.odd:
        r.fail(f"{rel(ctx, path)} is a run file p2 can read", "its name does not follow runs/shared/<name>.<practice|test>.trec, runs/own/<name>.trec, runs/own/ablation/<name>.trec or runs/own/stretch/<name>.trec",
               "Rename or remove it, then commit.")  # fmt: skip
    systems = retrievers.names()
    good = []
    for ref in ctx.refs:
        queries = ctx.queries(ref.corpus, ref.query_set)
        qids = [q.qid for q in queries] if queries and not ref.stretch else None
        problems = validate_run(ref.path, k=ctx.cfg.k, docids=ctx.doc_ids(ref.corpus), qids=qids)
        if ref.stretch and not problems:
            # a stretch run covers some of your own queries, not all of them
            stray = sorted(set(read_run(ref.path)) - {q.qid for q in queries})
            if stray:
                problems.append(f"it has queries that are not in eval/own/queries.tsv, such as {stray[0]}")
        tag = run_tag(ref.path)
        if tag and tag not in systems:
            problems.append(f"its tag {tag} names no system in p2/retrievers/")
        elif tag and not ref.ablation and ref.name in systems and tag != ref.name:
            problems.append(f"it is named after {ref.name} but was written by {tag} (its tag), and a run named after a system must come from that system")
        if problems:
            r.fail(f"{rel(ctx, ref.path)} is a valid run file", summarize(problems), "Run `uv run p2 run` for it again (or remove it), then commit.")
        else:
            good.append(ref)
    if not ctx.refs and not ctx.odd:
        r.todo("run files", "no runs yet; start with `uv run p2 run --all`")
    elif good:
        r.ok(f"{len(good)} run file(s) well-formed")
    ctx.good = good
    return good


def _consistent(fresh: list, deep: list) -> bool:
    """True when the deeper ranking starts with the same documents and scores as the top k, so it can
    be used to look up documents below the cut-off (a system whose scores depend on k fails this)."""
    if len(deep) < len(fresh):
        return False
    deep_scores = dict(deep)
    fresh_ids = {d for d, _ in fresh}
    if any(d not in deep_scores or abs(deep_scores[d] - s) > 1e-6 for d, s in fresh):
        return False
    floor = min((s for _, s in fresh), default=float("inf"))
    return all(s <= floor + 1e-6 for d, s in deep if d not in fresh_ids)


def compare_lists(committed: list, fresh: list, deep: list | None = None, tol: float = SCORE_TOLERANCE) -> str | None:
    """None when a committed ranking agrees with a fresh one, else the first difference.

    Every committed document must have its fresh score (within tol), looked up in the fresh top k or,
    when the system ranks consistently at any depth, in a deeper fresh ranking; documents may trade
    places only where their scores are that close. A fresh document scoring clearly above the
    committed cut-off must be in the committed list."""
    n = len(committed)
    if n != len(fresh):
        return f"lists {n} documents where a fresh run lists {len(fresh)}"
    usable = deep is not None and _consistent(fresh, deep)
    lookup = dict(fresh)
    if usable:
        for d, s in deep:
            lookup.setdefault(d, s)
    for i, (d, s) in enumerate(committed, 1):
        if d in lookup:
            if abs(s - lookup[d]) >= tol:
                return f"gives {d} (rank {i}) the score {s:.6f} where a fresh run gives it {lookup[d]:.6f}"
        elif usable:
            return f"lists {d} at rank {i}, which a fresh run does not rank in its top {len(deep)}"
        elif not fresh or abs(s - fresh[-1][1]) >= tol:
            return f"lists {d} at rank {i}, which a fresh run does not list, and its score is not a near-tie at the cut-off"
    listed = {d for d, _ in committed}
    cut = committed[-1][1] if committed else float("-inf")
    for j, (d, s) in enumerate(fresh, 1):
        if d not in listed and s - cut >= tol:
            return f"leaves out {d}, which a fresh run ranks at {j} with score {s:.6f}"
    return None


def regenerate(ctx: Context, refs: list[score.RunRef]) -> dict[str, dict] | str:
    """Run the systems of these run files again in a separate process with Claude switched off.

    Returns {run file: status} (see p2/regen.py), or a sentence saying why it could not run."""
    jobs = []
    for ref in refs:
        tag = run_tag(ref.path)
        if tag:
            jobs.append({"id": rel(ctx, ref.path), "corpus": ref.corpus, "query_set": ref.query_set, "system": tag})
    if not jobs:
        return {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        plan, out = Path(tmp) / "plan.json", Path(tmp) / "out.json"
        plan.write_text(json.dumps({"k": ctx.cfg.k, "jobs": jobs}), encoding="utf-8")
        env = {**os.environ, "P2_NO_CLAUDE": "1", "PYTHONUTF8": "1"}
        try:
            done = subprocess.run([sys.executable, "-m", "p2.regen", str(ctx.root), str(plan), str(out)], env=env, cwd=ctx.root,
                                  stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")  # fmt: skip
        except OSError as error:
            return f"p2 check could not start a second Python process ({error})"
        if not out.is_file():
            return f"running your systems again stopped with exit code {done.returncode} before it finished (the lines above say why)"
        return json.loads(out.read_text(encoding="utf-8"))


def check_regenerate(ctx: Context, r: Report, refs: list[score.RunRef]) -> None:
    statuses = regenerate(ctx, refs)
    if isinstance(statuses, str):
        r.fail("committed runs match a fresh run", statuses, "Fix that, then run `uv run p2 check` again.")
        return
    regenerated, claude_runs = [], []
    for ref in refs:
        tag = run_tag(ref.path)
        st = statuses.get(rel(ctx, ref.path))
        if not tag or st is None:
            continue
        what = f"{rel(ctx, ref.path)} matches a fresh run"
        if st["status"] == "not_built":
            r.fail(what, f"p2/retrievers/{tag}.py still raises NotImplementedError", f"Build {tag} first, or remove this run file, then commit.")
        elif st["status"] == "error":
            r.fail(what, f"running p2/retrievers/{tag}.py again raised {st['detail']}",
                   "Fix it (the lines above show where it happened; a package that is missing goes in with `uv add`), then run `uv run p2 check` again.")  # fmt: skip
        elif st["status"] == "claude":
            if tag in NO_CLAUDE_SYSTEMS:
                r.fail(what, f"{tag} calls Claude, and bm25, dense and hybrid must not, because CI runs them again",
                       "Move the Claude step into rerank (or a system of your own with NEEDS_CLAUDE = True), then run and commit again.")  # fmt: skip
            elif not st["needs_claude"]:
                r.fail(what, f"{tag} calls Claude but p2/retrievers/{tag}.py says NEEDS_CLAUDE = False", f"Set NEEDS_CLAUDE = True in p2/retrievers/{tag}.py, then run `uv run p2 check` again.")
            else:
                claude_runs.append(ref)
        else:
            committed = read_run(ref.path)
            problem = None
            for q in ctx.queries(ref.corpus, ref.query_set):
                if ref.stretch and q.qid not in committed:
                    continue  # a stretch run covers only the queries it names
                fresh = [tuple(x) for x in st["fresh"].get(q.qid, [])]
                deep = [tuple(x) for x in st["deep"].get(q.qid, [])]
                diff = compare_lists(committed.get(q.qid, []), fresh, deep)
                if diff:
                    problem = f"for {q.qid} it {diff}"
                    break
            if problem:
                r.fail(what, problem, "Run `uv run p2 run --all` (did p2.toml or a retriever change since?) and commit the new runs.")
            else:
                regenerated.append(ref)
    passed_claude = check_claude_runs(ctx, r, claude_runs)
    if regenerated or passed_claude:
        names = ", ".join(sorted({run_tag(x.path) for x in regenerated})) or "none"
        note = f"{len(regenerated)} run(s) of {names} run again and compared"
        if passed_claude:
            note += f"; {len(passed_claude)} run(s) of {', '.join(sorted({run_tag(x.path) for x in passed_claude}))} call Claude, so they are checked against their traces instead"
        r.ok("committed runs match a fresh run", note)


def claude_trace_problem(ctx: Context, ref: score.RunRef, committed: dict) -> str | None:
    """Why the committed trace of a Claude run does not back up its run file, or None."""
    path = paths.trace_file(ctx.root, ref.corpus, ref.query_set, ref.name, ref.repeat, ref.ablation, calls_claude=True, stretch=ref.stretch)
    if not path.is_file():
        return f"{rel(ctx, path)} is missing; `uv run p2 run` writes it with the run file as the record of the Claude calls"
    try:
        spans = trace.read(path)
    except (OSError, ValueError):
        return f"{rel(ctx, path)} cannot be read"
    by_qid: dict[str, dict] = {}
    for span in spans:
        by_qid.setdefault(str(span.get("p2.qid")), span)
    tag = run_tag(ref.path)
    for qid, ranked in committed.items():
        span = by_qid.get(qid)
        if span is None:
            return f"{rel(ctx, path)} has no span for {qid}"
        if span.get("p2.system") != tag:
            return f"{rel(ctx, path)} was written by {span.get('p2.system')}, not {tag}"
        if int(span.get("p2.claude_calls") or 0) < 1:
            return f"{rel(ctx, path)} records no Claude call for {qid}"
        if list(span.get("p2.top_ids") or []) != [d for d, _ in ranked]:
            return f"{rel(ctx, path)} lists other documents for {qid} than the run file"
    if sum(int(by_qid[q].get("gen_ai.usage.input_tokens") or 0) for q in committed) <= 0:
        return f"{rel(ctx, path)} records no input tokens at all"
    return None


def check_claude_runs(ctx: Context, r: Report, refs: list[score.RunRef]) -> list[score.RunRef]:
    """Runs of systems that call Claude: backed by their committed trace, and not a copy of another system's run."""
    passed = []
    for ref in refs:
        committed = read_run(ref.path)
        tag = run_tag(ref.path)
        what = f"{rel(ctx, ref.path)} matches its Claude trace"
        problem = claude_trace_problem(ctx, ref, committed)
        if problem:
            again = f"`uv run p2 run --corpus own --queries FILE --system {tag} --stretch`" if ref.stretch else "`uv run p2 run`"
            r.fail(what, problem, f"Run it again with {again} and commit the run file and its trace in traces/ together.")
            continue
        lists = {q: [d for d, _ in v] for q, v in committed.items()}
        twin = None
        for other in ctx.good:
            if other.path == ref.path or other.corpus != ref.corpus or other.query_set != ref.query_set or run_tag(other.path) == tag:
                continue
            theirs = {q: [d for d, _ in v] for q, v in read_run(other.path).items()}
            if ref.stretch:
                theirs = {q: theirs.get(q) for q in lists}
            if theirs == lists:
                twin = other
                break
        if twin is not None:
            r.fail(f"{rel(ctx, ref.path)} is not a copy of another run", f"it lists the same documents as {rel(ctx, twin.path)} for every query, so Claude's order was never used",
                   "Check that your system uses Claude's reply (a reply that is not each number once keeps the first-stage order), then run it again and commit.")  # fmt: skip
            continue
        passed.append(ref)
    return passed


def check_results(ctx: Context, r: Report) -> dict:
    fresh, _notes = score.compute(ctx.root, ctx.cfg)
    has_any = bool(fresh["sets"] or fresh["answers"])
    path = paths.results_file(ctx.root)
    what = "results/results.json matches a fresh `p2 score`"
    if not path.is_file():
        if has_any:
            r.fail(what, "results/results.json is missing", "Run `uv run p2 score` and commit results/results.json and EVAL.md.")
        else:
            r.todo(what, "nothing to score yet; run `uv run p2 score` once you have runs")
    else:
        try:
            committed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            committed = None
        where = "the file is not valid JSON" if committed is None else score.same_results(score.comparable(committed), score.comparable(fresh))
        if where:
            r.fail(what, f"they differ at {where}", "Run `uv run p2 score` and commit what it writes.")
        else:
            r.ok(what)
    eval_path = ctx.root / score.EVAL_FILE
    what = "EVAL.md tables match a fresh `p2 score`"
    if not eval_path.is_file():
        r.fail(what, "EVAL.md is missing", "Restore EVAL.md from the template, then run `uv run p2 score` and commit.")
        return fresh
    text = eval_path.read_text(encoding="utf-8")
    problems = score.marker_problems(text)
    missing = score.missing_blocks(text, ctx.cfg.section)
    stale = []
    known = 0
    for name, body in score.blocks(text).items():
        expected = score.render(name, fresh)
        if expected is None:
            continue
        known += 1
        if not body.strip() and expected.startswith("_"):
            continue  # an empty block, and nothing to put in it yet
        if not score.same_text(score.checked_part(name, body), score.checked_part(name, expected)):
            stale.append(name)
    if problems:
        r.fail(what, summarize(problems), "Put the p2 markers back as in the template, then run `uv run p2 score`.")
    elif missing:
        shown = missing[0]
        r.fail(what, f"{len(missing)} table(s) are missing, such as {shown}",
               f"Put its two marker lines back where the template has them, `<!-- p2:begin {shown} -->` and `<!-- p2:end {shown} -->`, then run `uv run p2 score` and commit.")  # fmt: skip
    elif stale:
        r.fail(what, f"{len(stale)} table(s) are out of date or edited by hand, such as {stale[0]}", "Run `uv run p2 score` and commit EVAL.md.")
    elif known:
        r.ok(what, f"{known} table(s)")
    return fresh


def check_answers(ctx: Context, r: Report) -> None:
    files = paths.answers_files(ctx.root)
    if not files:
        return
    questions_cache: dict[str, dict] = {}
    good = 0
    for path in files:
        what = f"{rel(ctx, path)} is a valid answers file"
        try:
            data = verify.load_answers(path)
        except verify.AnswersError as error:
            r.fail(what, f"it {error}", "Run `uv run p2 answer` again to rewrite it (or remove it), then commit.")
            continue
        corpus = data.get("corpus", path.parent.name)
        problems = verify.mechanics(data, ctx.chunk_texts(data))
        if corpus not in questions_cache:
            qpath = paths.questions(ctx.root, corpus)
            questions_cache[corpus] = {q.qid: q for q in read_questions(qpath)[0]} if qpath.is_file() else {}
        known = questions_cache[corpus]
        for record in data["answers"]:
            q = known.get(record["qid"])
            if q is None:
                problems.append(f"{record['qid']} is not a question in eval/{corpus}/questions.tsv")
            elif q.kind != record["kind"] or q.text != record["question"]:
                problems.append(f"{record['qid']} does not match its line in eval/{corpus}/questions.tsv")
        if problems:
            r.fail(what, summarize(problems), "Run `uv run p2 answer` again to rewrite it, then commit.")
            continue
        stale = answers_trace_problem(ctx, path, data)
        if stale:
            trace_path = rel(ctx, paths.answers_trace(ctx.root, path.stem))
            r.fail(f"{trace_path} matches {rel(ctx, path)}", stale,
                   f"Restore the committed trace with `git checkout -- {trace_path}`, or run `uv run p2 answer` for {path.stem} again (saved replies cost nothing) and commit both.")  # fmt: skip
        else:
            good += 1
    if good:
        r.ok(f"{good} answers file(s) well-formed", "how many quotes verify is in results.json and EVAL.md")


def answers_trace_problem(ctx: Context, path: Path, data: dict) -> str | None:
    """Why an answers file's trace no longer describes it (an interrupted run or a trial rewrote the
    trace), or None; an answers file without a trace is fine, and the cost table then reads the file."""
    trace_path = paths.answers_trace(ctx.root, path.stem)
    if not trace_path.is_file():
        return None
    try:
        spans = {str(s.get("p2.qid")): s for s in trace.read(trace_path)}
    except (OSError, ValueError):
        return "the trace cannot be read"
    for record in data["answers"]:
        span = spans.get(str(record["qid"]))
        if span is None:
            return f"the trace has no span for {record['qid']}, so it was written by another run of p2 answer"
        if list(span.get("p2.top_ids") or []) != list(record["retrieved"]):
            return f"the trace lists other chunks for {record['qid']} than the answers file"
    return None


def check_judged(ctx: Context, r: Report) -> None:
    """Stretch option 2: judged files backed by their answers file and trace, and well-formed calibration files."""
    base = ctx.root / "answers"
    judged = sorted(base.glob(f"*/*{paths.JUDGED_SUFFIX}")) if base.is_dir() else []
    calibrations = sorted(base.glob(f"*/*{paths.CALIBRATION_SUFFIX}")) if base.is_dir() else []
    good = 0
    for path in judged:
        problems = judge.integrity(ctx.root, path)
        if problems:
            r.fail(f"{rel(ctx, path)} matches its answers file and its Claude trace", summarize(problems),
                   f"Run `uv run p2 judge {rel(ctx, path.with_name(paths.answers_stem(path) + '.json'))}` again and commit the judged file and its trace in traces/ together.")  # fmt: skip
        else:
            good += 1
    for path in calibrations:
        answers = path.with_name(paths.answers_stem(path) + ".json")
        what = f"{rel(ctx, path)} is a calibration file p2 can read"
        if not answers.is_file():
            r.fail(what, f"there is no answers file {rel(ctx, answers)} beside it", "Name it after the answers file whose claims you labeled (rerank.calibration.tsv for rerank.json), then commit.")
            continue
        try:
            known = {c["id"] for c in judge.claims_of(verify.load_answers(answers))}
        except verify.AnswersError:
            continue  # check_answers reports the answers file
        labels, problems = judge.read_calibration(path, known)
        if problems:
            r.fail(what, summarize(problems), "Fix those lines (a claim id such as a01-c1, a tab, supported, partly or not, a tab, a note), then commit.")
        else:
            good += 1
    if good:
        r.ok(f"{good} judged or calibration file(s) well-formed", "stretch option 2")


# ---- completeness ----


def _how_to_run(ctx: Context, corpus: str, query_set: str, missing: list[str]) -> str:
    """The commands that write the missing canonical runs, with the cost of the Claude one."""
    steps = []
    plain = [s for s in missing if s != "rerank"]
    if plain:
        steps.append(f"`uv run p2 run --all` writes {', '.join(plain)}")
    if "rerank" in missing:
        n = len(ctx.queries(corpus, query_set))
        where = "--corpus own" if corpus == "own" else f"--corpus shared --queries {query_set}"
        cost = f"{n} claude -p calls" if n else "one claude -p call per query"
        steps.append(f"`uv run p2 run {where} --system rerank` writes rerank ({cost})")
    return ", and ".join(steps)


def check_complete_runs(ctx: Context, r: Report) -> None:
    present = {(x.corpus, x.query_set, x.name) for x in ctx.good if x.repeat is None and not x.ablation and not x.stretch and run_tag(x.path) == x.name}
    for corpus, query_set in (("shared", "practice"), ("shared", "test"), ("own", "own")):
        where = "own corpus" if corpus == "own" else f"shared {query_set} queries"
        missing = [s for s in retrievers.CANONICAL if (corpus, query_set, s) not in present]
        what = f"runs of bm25, dense, hybrid and rerank on the {where}"
        if missing:
            r.todo(what, f"missing {', '.join(missing)}; {_how_to_run(ctx, corpus, query_set, missing)}")
        else:
            r.ok(what)


def check_complete_own(ctx: Context, r: Report) -> None:
    n = len(ctx.doc_ids("own"))
    what = f"own corpus has at least {OWN_MIN_DOCS} documents"
    if n >= OWN_MIN_DOCS:
        r.ok(what, f"{n:,}")
    elif n:
        r.todo(what, f"it has {n}; when long PDFs keep you under {OWN_MIN_DOCS}, `p2 ingest --part-pages {SUGGESTED_PART_PAGES}` splits them into parts that each count (README, size rules)")
    else:
        r.todo(what, f"it has {n}")
    queries = ctx.queries("own", "own")
    rels = {q for q, judged in ctx.qrels("own", "own").items() if any(v > 0 for v in judged.values())}
    hand = sum(1 for q in queries if q.origin == "hand")
    without = [q.qid for q in queries if q.qid not in rels]
    gaps = []
    if len(queries) < GOLD_MIN_QUERIES:
        gaps.append(f"{len(queries)} queries of at least {GOLD_MIN_QUERIES}")
    if hand < GOLD_MIN_HAND:
        gaps.append(f"{hand} written by hand of at least {GOLD_MIN_HAND}")
    if without:
        gaps.append(f"{len(without)} without a relevant document, such as {without[0]}")
    what = "own gold set: 30 queries, 10 by hand, each with a relevant document"
    if gaps:
        r.todo(what, "; ".join(gaps))
    else:
        r.ok(what, f"{len(queries)} queries, {hand} by hand")
    ablations = [x for x in ctx.good if x.ablation]
    if ablations:
        r.ok("at least one ablation run", ", ".join(x.name for x in ablations))
    else:
        r.todo("at least one ablation run", "`uv run p2 run --corpus own --system NAME --ablation` writes one to runs/own/ablation/")


def _normal_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def check_notes(ctx: Context, r: Report, results: dict) -> None:
    """Things worth a second look in stage 2 that a program cannot judge; they never fail."""
    queries = ctx.queries("own", "own")
    qrels = ctx.qrels("own", "own")
    n_docs = len(ctx.doc_ids("own"))
    judged = [q for q, rels in qrels.items() if any(v > 0 for v in rels.values())]
    if judged and n_docs:
        mean_rel = sum(sum(1 for v in qrels[q].values() if v > 0) for q in judged) / len(judged)
        if mean_rel > RELEVANT_SHARE_NOTE * n_docs:
            r.note("own gold set: relevant documents per query",
                   f"on average {mean_rel:.0f} of your {n_docs} documents are relevant to each query, more than {RELEVANT_SHARE_NOTE:.0%}; every system finds something relevant then, so check your relevance rule in EVAL.md")  # fmt: skip
    seen: dict[str, str] = {}
    for q in queries:
        key = _normal_query(q.text)
        if key and key in seen:
            r.note("own gold set: repeated queries", f"{q.qid} reads the same as {seen[key]}; a repeated query counts twice in every mean")
            break
        seen.setdefault(key, q.qid)
    alike: dict[str, list[str]] = {}
    for q in queries:
        key = re.sub(r"\s*\b\d+\b\s*", " ", _normal_query(q.text)).strip()
        if len(key.split()) >= 3:
            alike.setdefault(key, []).append(q.qid)
    group = max(alike.values(), key=len, default=[])
    if len(group) >= 3:
        r.note("own gold set: queries made from one pattern",
               f"{len(group)} queries read alike apart from their numbers, such as {group[0]} and {group[1]}; queries from one pattern test one thing many times")  # fmt: skip
    pooled = ((results.get("stretch") or {}).get("judge") or {}).get("pooled")
    labeled = sum(e["labeled"] for e in (((results.get("stretch") or {}).get("judge") or {}).get("files") or {}).values())
    if labeled and (pooled or {}).get("compared", 0) < judge.CALIBRATION_MIN:
        r.note("stretch option 2: claims you labeled that Claude judged",
               f"{(pooled or {}).get('compared', 0)} of the {judge.CALIBRATION_MIN} the stretch asks for; label more claims, or judge a second answers file and label claims from it too")  # fmt: skip
    own = results.get("sets", {}).get("own/own")
    if own and own["systems"]:
        best = max(v["mean"]["mrr@10"] for v in own["systems"].values())
        bm25_recall = own["systems"].get("bm25", {}).get("mean", {}).get("recall@10", 0.0)
        if best >= TOO_EASY or bm25_recall >= TOO_EASY:
            r.note("own gold set: room to tell systems apart",
                   f"the best MRR@10 is {best:.3f} and bm25's recall@10 is {bm25_recall:.3f}; scores of 0.95 or more leave little room between systems, and harder queries make the comparison mean more")  # fmt: skip
        runs = {x.system: read_run(x.path) for x in ctx.good if x.corpus == "own" and x.repeat is None}
        for name, run in runs.items():
            if not name.startswith("ablation/"):
                continue
            ranked = {q: [d for d, _ in v] for q, v in run.items()}
            twin = next((other for other, o in runs.items() if other != name and {q: [d for d, _ in v] for q, v in o.items()} == ranked), None)
            if twin:
                r.note(f"{name} differs from the other runs", f"it lists the same documents as {twin} for every query, so the change you made did not change any ranking; say so in EVAL.md section 5")


def check_complete_answers(ctx: Context, r: Report) -> None:
    qpath = paths.questions(ctx.root, "shared")
    wanted = {q.qid for q in read_questions(qpath)[0]} if qpath.is_file() else set()
    what = "an answers file on all the shared questions"
    complete, partial = [], []
    for path in [p for p in paths.answers_files(ctx.root) if p.parent.name == "shared"]:
        if "." in path.stem and paths.REPEAT_RE.match(path.stem.rsplit(".", 1)[-1]):
            continue
        try:
            data = verify.load_answers(path)
        except verify.AnswersError:
            continue
        answered = {a["qid"] for a in data["answers"] if not a.get("error")}
        errors = sorted(a["qid"] for a in data["answers"] if a.get("error"))
        if wanted and wanted <= answered:
            complete.append(path.stem)
        else:
            partial.append((path.stem, len(wanted & answered), errors))
    if complete:
        r.ok(what, ", ".join(complete))
    elif partial:
        stem, n, errors = partial[0]
        why = f"the call for {errors[0]} failed" if errors else "it was made with --only"
        r.todo(what, f"answers/shared/{stem}.json answers {n} of {len(wanted)} ({why}); run `uv run p2 answer --corpus shared --system NAME --label {stem}` without --only")
    else:
        r.todo(what, f"`uv run p2 answer --corpus shared --system NAME` answers all {len(wanted) or 12}")


def check_complete_writing(ctx: Context, r: Report) -> None:
    for name in ("EVAL.md", "DECISIONS.md"):
        path = ctx.root / name
        what = f"{name} has no TODO markers left"
        if not path.is_file():
            r.todo(what, f"{name} is missing; restore it from the template")
            continue
        text = path.read_text(encoding="utf-8")
        count = len(TODO_MARK.findall(text))
        rider = len(RIDER_MARK.findall(text)) if name == "EVAL.md" and ctx.cfg.section == "598E" else 0
        if count or rider:
            r.todo(what, f"{count} left" + (f", and {rider} '598E: write ...' line(s) in section 9" if rider else ""))
        else:
            r.ok(what)


# ---- 598E ----


def git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


SECTION_RE = re.compile(r"^## (\d+)\.", re.M)
QRELS_LINE = re.compile(r"^\S+\s+\S+\s+[a-z0-9][a-z0-9._-]*\s+-?\d+$")
MIN_WORDS = 3


def prereg_sections(text: str) -> dict[int, str]:
    """{section number: its text} of a PREREG.md, from the `## N.` headings."""
    found = list(SECTION_RE.finditer(text))
    return {int(m.group(1)): text[m.end() : found[i + 1].start() if i + 1 < len(found) else len(text)] for i, m in enumerate(found)}


def _lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip() and line.strip() != "---"]


def own_words(section: str, template_section: str) -> list[str]:
    """The lines of a section that are not the template's own lines."""
    template = set(_lines(template_section))
    return [line for line in _lines(section) if line not in template]


def unfilled(text: str, template: str, numbers=(1, 2, 3, 4)) -> list[int]:
    """The sections that still have a TODO marker or fewer than MIN_WORDS words of the student's own."""
    sections, base = prereg_sections(text), prereg_sections(template)
    out = []
    for n in numbers:
        section = sections.get(n)
        if section is None or TODO_MARK.search(section) or sum(len(line.split()) for line in own_words(section, base.get(n, ""))) < MIN_WORDS:
            out.append(n)
    return out


def first_judgment_commit(root: Path) -> str | None:
    """The oldest commit in which any file under eval/own/ holds a qrels line (renamed files included)."""
    for sha in (git(root, "log", "--reverse", "--topo-order", "--format=%H", "--", "eval/own") or "").split():
        for name in (git(root, "ls-tree", "-r", "--name-only", sha, "--", "eval/own") or "").split("\n"):
            content = git(root, "show", f"{sha}:{name}") if name else None
            if content and any(QRELS_LINE.match(line.strip()) for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#")):
                return sha
    return None


def check_rider(ctx: Context, r: Report) -> None:
    present = {(x.name, x.repeat) for x in ctx.good if x.corpus == "shared" and x.query_set == "practice" and x.repeat}
    missing = [f"r{i}" for i in range(1, REPEATS + 1) if ("rerank", i) not in present]
    what = "598E: rerank.practice.r1 to r3"
    if missing:
        n = len(ctx.queries("shared", "practice"))
        r.todo(what, f"missing {', '.join(missing)}; `uv run p2 run --corpus shared --queries practice --system rerank --repeat 3` ({REPEATS * n} claude -p calls)")
    else:
        r.ok(what)
    check_rider_answers(ctx, r)
    check_prereg(ctx, r)


def check_rider_answers(ctx: Context, r: Report) -> None:
    qpath = paths.questions(ctx.root, "shared")
    wanted = {q.qid for q in read_questions(qpath)[0]} if qpath.is_file() else set()
    groups: dict[str, dict[int, Path]] = {}
    for path in [p for p in paths.answers_files(ctx.root) if p.parent.name == "shared"]:
        parts = path.stem.split(".")
        if len(parts) == 2 and paths.REPEAT_RE.match(parts[1]):
            groups.setdefault(parts[0], {})[int(parts[1][1:])] = path
    what = "598E: three repeated answers files on all the shared questions, scored with Wilson intervals"
    complete, gaps = [], []
    for label, reps in sorted(groups.items()):
        if not set(range(1, REPEATS + 1)) <= set(reps):
            gaps.append(f"{label} has {len(reps)} of {REPEATS} repeats")
            continue
        short = []
        for i in range(1, REPEATS + 1):
            try:
                data = verify.load_answers(reps[i])
            except verify.AnswersError:
                short.append(f"{label}.r{i} cannot be read")
                continue
            answered = {a["qid"] for a in data["answers"] if not a.get("error")}
            if wanted and not wanted <= answered:
                short.append(f"{label}.r{i} answers {len(wanted & answered)} of {len(wanted)}")
        if short:
            gaps.append(short[0])
        else:
            complete.append(label)
    scored = False
    results_path = paths.results_file(ctx.root)
    if complete and results_path.is_file():
        try:
            scored = bool(json.loads(results_path.read_text(encoding="utf-8")).get("answer_repeats"))
        except ValueError:
            scored = False
    if not complete:
        hint = "`uv run p2 answer --corpus shared --system NAME --repeat 3` (no --only), then `uv run p2 score`"
        r.todo(what, (f"{gaps[0]}; " if gaps else "") + hint)
    elif not scored:
        r.todo(what, "run `uv run p2 score` so results.json has the Wilson intervals")
    else:
        r.ok(what, ", ".join(complete))


def check_prereg(ctx: Context, r: Report) -> None:
    prereg = ctx.root / "PREREG.md"
    order_what = "598E: PREREG.md sections 1 to 4 committed before your first judgment, and unchanged since"
    outcome_what = "598E: PREREG.md outcome (section 5) written"
    if not prereg.is_file():
        r.todo(order_what, "PREREG.md is missing; restore it from the template")
        return
    text = prereg.read_text(encoding="utf-8")
    shallow = (git(ctx.root, "rev-parse", "--is-shallow-repository") or "").strip()
    log = (git(ctx.root, "log", "--reverse", "--topo-order", "--format=%H", "--", "PREREG.md") or "").split()
    template = git(ctx.root, "show", f"{log[0]}:PREREG.md") if log else None
    if shallow not in ("true", "false"):
        r.todo(order_what, "this folder has no git history to read")
    elif shallow == "true":
        r.todo(order_what, "the git history here is shallow; `git fetch --unshallow`, or run `uv run p2 check --final` on your machine")
    elif template is None:
        r.todo(order_what, "PREREG.md has never been committed; fill in sections 1 to 4 and commit it before your first judgment in eval/own/qrels.txt")
    elif not unfilled(template, ""):
        r.todo(order_what, "the first commit of PREREG.md is not the template's, so p2 cannot tell when you filled it in; tell the instructor")
    else:
        order = (git(ctx.root, "rev-list", "--reverse", "--topo-order", "HEAD") or "").split()
        position = {sha: i for i, sha in enumerate(order)}
        prereg_at = next((sha for sha in log if not unfilled(git(ctx.root, "show", f"{sha}:PREREG.md") or "", template)), None)
        judged_at = first_judgment_commit(ctx.root)
        left = unfilled(text, template)
        if prereg_at is None and judged_at is not None:
            r.fail(order_what, f"eval/own/ has judgments from commit {judged_at[:7]}, and no earlier commit has sections 1 to 4 of PREREG.md filled in",
                   "History cannot be rewritten fairly, so tell the instructor, and explain the order in section 6 of PREREG.md.")  # fmt: skip
        elif prereg_at is None:
            todo = f"section {left[0]} still has a TODO marker or no words of yours" if left else "sections 1 to 4 are filled in here but not committed yet"
            r.todo(order_what, f"{todo}; commit PREREG.md before your first judgment in eval/own/qrels.txt")
        elif judged_at is None:
            r.todo(order_what, f"PREREG.md was filled in at commit {prereg_at[:7]}; your own judgments are not committed yet")
        elif position.get(prereg_at, 1 << 30) >= position.get(judged_at, -1):
            r.fail(order_what, f"eval/own/ had judgments in commit {judged_at[:7]}, before PREREG.md sections 1 to 4 were filled in at {prereg_at[:7]}",
                   "History cannot be rewritten fairly, so tell the instructor, and explain the order in section 6 of PREREG.md.")  # fmt: skip
        else:
            then = prereg_sections(git(ctx.root, "show", f"{prereg_at}:PREREG.md") or "")
            now = prereg_sections(text)
            changed = [n for n in (1, 2, 3, 4) if _lines(then.get(n, "")) != _lines(now.get(n, ""))]
            if changed:
                r.fail(order_what, f"section {changed[0]} changed after commit {prereg_at[:7]}, where you pre-registered it",
                       f"Put sections 1 to 4 back as they were (`git show {prereg_at[:7]}:PREREG.md` shows them) and write the change as a dated entry in section 6.")  # fmt: skip
            else:
                r.ok(order_what, f"filled in at {prereg_at[:7]}, first judgment at {judged_at[:7]}")
    if template is not None and not unfilled(text, template, numbers=(5,)):
        r.ok(outcome_what)
    else:
        r.todo(outcome_what, "write section 5 of PREREG.md once your own corpus is scored")


# ---- running it ----


def run_checks(root: Path, final: bool = False) -> list[Item]:
    r = Report()
    try:
        cfg = config.load(root)
        r.ok("p2.toml settings", f"section {cfg.section}, k {cfg.k}, chunks of {cfg.chunk_words} words")
    except config.ConfigError as error:
        r.fail("p2.toml settings", str(error).rstrip("."), "The other checks use the default settings until then.")
        cfg = config.Config(root=root)
    ctx = Context(root, cfg)
    results: dict = {}

    def keep_results():
        results.update(check_results(ctx, r))

    steps = [
        lambda: check_pinned(ctx, r),
        lambda: check_line_endings(ctx, r),
        lambda: check_shared_eval(ctx, r, check_corpus(ctx, r, "shared")),
        lambda: check_own_corpus_steps(ctx, r),
        lambda: check_own_eval(ctx, r),
        lambda: check_regenerate(ctx, r, check_runs(ctx, r)),
        keep_results,
        lambda: check_answers(ctx, r),
        lambda: check_judged(ctx, r),
        lambda: check_complete_runs(ctx, r),
        lambda: check_complete_own(ctx, r),
        lambda: check_complete_answers(ctx, r),
        lambda: check_complete_writing(ctx, r),
    ]
    if cfg.section == "598E":
        steps.append(lambda: check_rider(ctx, r))
    steps.append(lambda: check_notes(ctx, r, results))
    for step in steps:
        try:
            step()
        except Exception as error:  # report, do not hide: the student needs the line and the instructor the error
            r.fail("p2 check itself", f"something unexpected happened ({type(error).__name__}: {error})",
                   "Show this line to the instructor; the checks after it may be incomplete.")  # fmt: skip
    return r.items


def check_own_corpus_steps(ctx: Context, r: Report) -> None:
    if check_corpus(ctx, r, "own"):
        check_own_size(ctx, r)
        check_license(ctx, r)
        check_ingest_report(ctx, r)


def run(args, cfg_root: Path) -> int:
    items = run_checks(cfg_root, final=args.final)
    for item in items:
        print(item.line())
    fails = sum(1 for i in items if i.status == "FAIL")
    todos = sum(1 for i in items if i.status == "TODO")
    notes = sum(1 for i in items if i.status == "NOTE")
    passes = len(items) - fails - todos - notes
    print(f"{passes} passed, {todos} to do, {fails} failed" + (f", {notes} note(s)." if notes else "."))
    if fails:
        print("Fix the FAIL lines first; each one says what to do.")
    elif todos and args.final:
        print("--final is the submission bar, so every TODO line has to be done.")
    elif todos:
        print("Lines marked TODO are work still to do; they do not fail this check.")
    else:
        print("Everything checks out.")
    if notes:
        print("Lines marked NOTE never fail the check; they point at something worth a second look in EVAL.md.")
    return 1 if fails or (args.final and todos) else 0


if __name__ == "__main__" and sys.argv[1:] == ["--write-pins"]:
    PINS_FILE.write_text(render_pins(compute_pins(config.find_root())), encoding="utf-8", newline="\n")
    print(f"Wrote {PINS_FILE}")
