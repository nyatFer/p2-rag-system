"""`p2 score`: score every run file that has qrels, compare every pair of systems, check the answers files,
write results/results.json, and rewrite the tables between the p2 markers in EVAL.md.

A table in EVAL.md sits between two markers with the same NAME:

    <!-- p2:begin NAME -->
    (p2 score writes this part; anything you type here is replaced)
    <!-- p2:end NAME -->

The names:
    shared-practice            the systems on the shared practice queries: recall@10, MRR@10, nDCG@10
    shared-practice-classes    the same per query class (add -recall, -mrr or -ndcg for one metric)
    shared-practice-pairs      every pair of systems: mean difference, paired 95% interval, minimum
                               detectable difference, reading (add -recall, -mrr or -ndcg for one metric)
    own, own-classes, own-pairs   the same on your own corpus and gold set
    own-origins                the own-corpus scores for the queries you wrote (hand) and the ones a model
                               drafted (claude), hand minus claude with an unpaired 95% interval, whether
                               bm25's lead over dense differs between the two (stretch option 6), and the
                               content-word overlap of each group's queries with their relevant documents
    own-ablation               each ablation run against each of the other systems on your corpus
    own-ablation-queries       each ablation against the system it varies: the queries it helped and hurt
    answers                    each answers file: verified share, not_found share, declined share
    repeats                    598E: the repeated reranker runs and answers files, with their spread
    stretch-judge              stretch option 2: Claude's verdicts on claims against yours (p2 judge)
    stretch-cost               stretch option 3: the Claude calls, tokens and seconds in every trace, and
                               what a cheaper system or answers file saves against what it loses
    stretch-agent              stretch option 5: a stretch run (the grep agent) against the four systems
                               on the same queries (add -recall, -mrr or -ndcg to stretch-cost or
                               stretch-agent for another metric than MRR@10)
p2 check recomputes every one of these tables, so edit your prose around them, never inside.

How own-ablation-queries finds the system an ablation varies (its base): the file of the ablation's
system may say so in one line, BASE = "dense"; without that line, the base is the longest part of the
system's name before an underscore that names a system with a run on your own corpus (bm25_lab varies
bm25, dense_potion varies dense). p2 reads that line without running the file.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from p2 import metrics, paths, stats, stretch, textproc, verify
from p2 import corpus as corpus_mod
from p2 import retrievers as retriever_pkg
from p2.retrievers import CANONICAL
from p2.runfile import read_qrels, read_queries, read_run, run_tag

METRIC_NAMES = {"recall@10": "Recall@10", "mrr@10": "MRR@10", "ndcg@10": "nDCG@10"}
METRIC_SLUGS = {"recall": "recall@10", "mrr": "mrr@10", "ndcg": "ndcg@10"}
SETS = {"shared-practice": ("shared", "practice"), "shared-test": ("shared", "test"), "own": ("own", "own")}
VIEWS = ("classes", "origins", "pairs", "ablation")
OWN_ONLY_VIEWS = ("origins", "ablation")
# The tables EVAL.md must keep (a name with a metric added, such as shared-practice-pairs-mrr, counts).
REQUIRED_BLOCKS = (
    "shared-practice", "shared-practice-classes", "shared-practice-pairs", "answers", "own", "own-classes", "own-origins", "own-pairs",
    "own-ablation", "own-ablation-queries", "stretch-judge", "stretch-cost", "stretch-agent",
)  # fmt: skip
# Blocks that take a metric after their name (stretch-cost-ndcg), besides the set views.
METRIC_BLOCKS = ("stretch-cost", "stretch-agent")
# The part of the stretch-cost table that comes from traces/retrieval/, which git ignores: it starts
# with this sentence, and p2 check does not compare it (it would differ on every machine).
LOCAL_HEADING = "Systems that do not call Claude, from traces/retrieval/"
RIDER_BLOCKS = ("repeats",)
BLOCK_RE = re.compile(r"<!-- p2:begin (?P<name>[A-Za-z0-9_.-]+) -->(?P<body>.*?)<!-- p2:end (?P=name) -->", re.S)
BEGIN_RE = re.compile(r"<!-- p2:begin ([A-Za-z0-9_.-]+) -->")
END_RE = re.compile(r"<!-- p2:end ([A-Za-z0-9_.-]+) -->")
EVAL_FILE = "EVAL.md"
TOO_EASY = 0.95


@dataclass(frozen=True)
class RunRef:
    corpus: str
    query_set: str
    name: str
    path: Path
    repeat: int | None = None
    ablation: bool = False
    stretch: bool = False

    @property
    def system(self) -> str:
        """The name the tables show: the file's name, with ablation/ or stretch/ in front for those runs."""
        return f"ablation/{self.name}" if self.ablation else f"stretch/{self.name}" if self.stretch else self.name

    @property
    def set_key(self) -> str:
        return f"{self.corpus}/{self.query_set}"


def discover(root: Path) -> tuple[list[RunRef], list[Path]]:
    """Every run file under runs/ that follows the naming rules, and the files that do not."""
    found: list[RunRef] = []
    odd: list[Path] = []
    base = root / "runs"
    if not base.is_dir():
        return found, odd
    for path in sorted(p for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")):
        where = path.relative_to(base).parts
        parts = path.stem.split(".") if path.suffix == ".trec" else []
        ref = None
        if parts and all(paths.LABEL_RE.match(p) for p in parts):
            repeat = None
            if len(parts) > 1 and paths.REPEAT_RE.match(parts[-1]):
                repeat = int(parts[-1][1:])
                parts = parts[:-1]
            if where[:-1] == ("shared",) and len(parts) == 2 and parts[1] in paths.QUERY_SETS["shared"]:
                ref = RunRef("shared", parts[1], parts[0], path, repeat)
            elif where[:-1] == ("own",) and len(parts) == 1:
                ref = RunRef("own", "own", parts[0], path, repeat)
            elif where[:-1] == ("own", "ablation") and len(parts) == 1 and repeat is None:
                ref = RunRef("own", "own", parts[0], path, None, ablation=True)
            elif where[:-1] == ("own", "stretch") and len(parts) == 1 and repeat is None:
                ref = RunRef("own", "own", parts[0], path, None, stretch=True)
        if ref is None:
            odd.append(path)
        else:
            found.append(ref)
    return found, odd


def system_order(name: str) -> tuple:
    if name.startswith("ablation/"):
        return (2, name)
    return (0, CANONICAL.index(name), name) if name in CANONICAL else (1, name)


def r6(x: float) -> float:
    return round(float(x), 6)


def _pairs(names: list[str], per_query: dict[str, dict[str, dict[str, float]]], judged: list[str]) -> list[dict]:
    out = []
    for j, a in enumerate(names):
        for b in names[:j]:
            for m in metrics.METRICS:
                va = [per_query[a][q][m] for q in judged]
                vb = [per_query[b][q][m] for q in judged]
                boot = stats.paired_bootstrap(va, vb)
                same = all(x == y for x, y in zip(va, vb))
                out.append({
                    "a": a, "b": b, "metric": m, "n": boot["n"],
                    "mean_diff": r6(boot["mean_diff"]), "ci95": [r6(v) for v in boot["ci95"]],
                    "sd_diff": r6(boot["sd_diff"]), "mdd80": None if same else r6(stats.mdd(boot["sd_diff"], boot["n"])),
                    "reading": stats.IDENTICAL if same else stats.reading(boot["ci95"], a, b),
                })  # fmt: skip
    return out


def score_set(root: Path, corpus: str, query_set: str, refs: list[RunRef], qrels_path: Path) -> tuple[dict | None, list[str]]:
    """The results for one corpus and query set, or None when there is nothing to score."""
    notes: list[str] = []
    qpath = paths.queries(root, corpus, query_set)
    if not qrels_path.is_file() or not qpath.is_file():
        return None, notes
    queries, _ = read_queries(qpath)
    qrels, _ = read_qrels(qrels_path)
    judged = metrics.judged_queries(qrels, [q.qid for q in queries])
    mine = [r for r in refs if r.corpus == corpus and r.query_set == query_set and not r.stretch]
    if not judged or not mine:
        return None, notes
    classes = {q.qid: q.cls for q in queries}
    class_names = list(dict.fromkeys(classes[q] for q in judged))
    origins = {q.qid: q.origin for q in queries}
    origin_names = [o for o in ("hand", "claude") if any(origins[q] == o for q in judged)]
    n_docs = len(corpus_mod.doc_files(paths.docs_dir(root, corpus)))
    graded = any(r > 1 for rels in qrels.values() for r in rels.values())
    result = {
        "corpus": corpus, "queries": query_set,
        "queries_file": paths.rel(root, qpath), "qrels_file": paths.rel(root, qrels_path),
        "n_queries": len(judged), "n_documents": n_docs, "graded": graded,
        "chance_recall@10": r6(min(1.0, 10 / n_docs)) if n_docs else None,
        "systems": {}, "pairs": [],
    }  # fmt: skip
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for ref in sorted((r for r in mine if r.repeat is None), key=lambda r: system_order(r.system)):
        try:
            run = read_run(ref.path)
            pq = metrics.evaluate(run, qrels, judged)
        except ValueError as error:
            notes.append(f"{paths.rel(root, ref.path)} was not scored: {error}")
            continue
        per_query[ref.system] = pq
        result["systems"][ref.system] = {
            "run": paths.rel(root, ref.path),
            "mean": {m: r6(metrics.mean(pq, m)) for m in metrics.METRICS},
            "by_class": {
                c: {"n": sum(1 for q in judged if classes[q] == c), **{m: r6(metrics.mean(pq, m, [q for q in judged if classes[q] == c])) for m in metrics.METRICS}}
                for c in class_names
            },
            "per_query": {q: {m: r6(v) for m, v in pq[q].items()} for q in judged},
        }
        if origin_names:
            result["systems"][ref.system]["by_origin"] = {
                o: {"n": sum(1 for q in judged if origins[q] == o), **{m: r6(metrics.mean(pq, m, [q for q in judged if origins[q] == o])) for m in metrics.METRICS}}
                for o in origin_names
            }
    names = list(result["systems"])
    result["pairs"] = _pairs(names, per_query, judged)
    if origin_names:
        result["origins"] = origin_analysis(root, corpus, result, per_query, judged, origins, qrels, queries)
    ablations = {r.system: r for r in mine if r.ablation and r.repeat is None and r.system in per_query}
    if ablations:
        result["ablation_queries"] = {name: ablation_queries(ref, result, per_query, judged, classes, origins) for name, ref in ablations.items()}
    repeats: dict[str, dict] = {}
    for ref in sorted((r for r in mine if r.repeat is not None), key=lambda r: (system_order(r.system), r.repeat)):
        try:
            pq = metrics.evaluate(read_run(ref.path), qrels, judged)
        except ValueError as error:
            notes.append(f"{paths.rel(root, ref.path)} was not scored: {error}")
            continue
        entry = repeats.setdefault(ref.system, {"runs": [], "repeat": [], "values": {m: [] for m in metrics.METRICS}, "vs": {}})
        entry["runs"].append(paths.rel(root, ref.path))
        entry["repeat"].append(ref.repeat)
        for m in metrics.METRICS:
            entry["values"][m].append(r6(metrics.mean(pq, m)))
        for other in names:
            if other == ref.system:
                continue
            for m in metrics.METRICS:
                boot = stats.paired_bootstrap([pq[q][m] for q in judged], [per_query[other][q][m] for q in judged])
                entry["vs"].setdefault(other, {}).setdefault(m, []).append(
                    {"repeat": ref.repeat, "mean_diff": r6(boot["mean_diff"]), "ci95": [r6(v) for v in boot["ci95"]], "reading": stats.reading(boot["ci95"], f"{ref.system}.r{ref.repeat}", other)}
                )
    for entry in repeats.values():
        entry["spread"] = {m: {"min": min(v), "max": max(v), "range": r6(max(v) - min(v)), "sd": r6(stats.sd(v))} for m, v in entry["values"].items()}
    if repeats:
        result["repeats"] = repeats
    return result, notes


def content_words(text: str) -> set[str]:
    """The distinct content words of a text: the 12 lab's tokens ([a-z0-9]+, lower-cased, stopwords dropped)."""
    return set(textproc.tokens_lab(text))


def origin_analysis(root: Path, corpus: str, result: dict, per_query: dict, judged: list[str], origins: dict, qrels: dict, queries) -> dict:
    """Hand against claude queries (stretch option 6): each system's hand minus claude with an unpaired
    interval, the interaction (bm25 minus dense on claude queries) minus (the same on hand queries), and
    the mean content-word overlap of each group's queries with their best relevant document."""
    groups = {o: [q for q in judged if origins[q] == o] for o in ("hand", "claude")}
    both = all(groups.values())
    out: dict = {"n": {o: len(v) for o, v in groups.items()}, "interaction": None, "overlap": {}}
    if both:
        for name, pq in per_query.items():
            diff = {}
            for m in metrics.METRICS:
                boot = stats.unpaired_bootstrap([pq[q][m] for q in groups["hand"]], [pq[q][m] for q in groups["claude"]])
                diff[m] = {"mean_diff": r6(boot["mean_diff"]), "ci95": [r6(v) for v in boot["ci95"]], "reading": stats.reading(boot["ci95"], "hand", "claude")}
            result["systems"][name]["origin_diff"] = diff
        if "bm25" in per_query and "dense" in per_query:
            inter = {"a": "bm25", "b": "dense"}
            for m in metrics.METRICS:
                lead = {o: [per_query["bm25"][q][m] - per_query["dense"][q][m] for q in groups[o]] for o in groups}
                boot = stats.unpaired_bootstrap(lead["claude"], lead["hand"])
                inter[m] = {
                    "mean_diff": r6(boot["mean_diff"]), "ci95": [r6(v) for v in boot["ci95"]],
                    "lead_claude": r6(sum(lead["claude"]) / len(lead["claude"])), "lead_hand": r6(sum(lead["hand"]) / len(lead["hand"])),
                    "reading": interaction_reading(boot["ci95"]),
                }  # fmt: skip
            out["interaction"] = inter
    docs = corpus_mod.load(root, corpus).docs
    words: dict[str, set[str]] = {}
    texts = {q.qid: q.text for q in queries}
    for o, qids in groups.items():
        shares = []
        for q in qids:
            mine = content_words(texts.get(q, ""))
            relevant = [d for d, r in qrels.get(q, {}).items() if r > 0 and d in docs]
            if not mine or not relevant:
                continue
            for d in relevant:
                if d not in words:
                    words[d] = content_words(docs[d].text)
            shares.append(max(len(mine & words[d]) / len(mine) for d in relevant))
        if shares:
            out["overlap"][o] = {"n": len(shares), "mean": r6(sum(shares) / len(shares))}
    return out


def interaction_reading(ci95) -> str:
    lo, hi = ci95
    if lo > 0:
        return "bm25 gains more on claude queries"
    if hi < 0:
        return "bm25 gains more on hand queries"
    return "not distinguishable"


def base_constant(path: Path) -> str | None:
    """The string in a top-level `BASE = "..."` line of a Python file, read without running the file."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))  # a byte order mark from a Windows editor is not code
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):  # BASE: str = "dense"
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "BASE" for t in targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def ablation_base(system: str, others: list[str]) -> tuple[str | None, str]:
    """(the system an ablation's system varies, how that was decided); see the module docstring."""
    module = Path(retriever_pkg.__file__).parent / f"{system}.py"
    declared = base_constant(module)
    if declared is not None:
        if declared in others:
            return declared, f'BASE = "{declared}" in p2/retrievers/{system}.py'
        return None, f'p2/retrievers/{system}.py says BASE = "{declared}", which has no run on your own corpus'
    parts = system.split("_")
    for i in range(len(parts) - 1, 0, -1):
        guess = "_".join(parts[:i])
        if guess in others:
            return guess, f"the name {system} starts with {guess}_"
    return None, f'no base found: add a line BASE = "name of the system it varies" to p2/retrievers/{system}.py'


def ablation_queries(ref: RunRef, result: dict, per_query: dict, judged: list[str], classes: dict, origins: dict) -> dict:
    """The queries an ablation helped and hurt against its base, by reciprocal rank (MRR@10 per query)."""
    system = run_tag(ref.path) or ref.name
    others = [n for n in result["systems"] if not n.startswith("ablation/")]
    base, rule = ablation_base(system, others)
    entry: dict = {"system": system, "base": base, "rule": rule, "helped": [], "hurt": [], "unchanged": 0}
    if base is None:
        return entry
    for q in judged:
        before, after = per_query[base][q]["mrr@10"], per_query[ref.system][q]["mrr@10"]
        row = {"qid": q, "class": classes[q], "origin": origins.get(q) or "", "base_rr": r6(before), "rr": r6(after), "change": r6(after - before)}
        if after > before:
            entry["helped"].append(row)
        elif after < before:
            entry["hurt"].append(row)
        else:
            entry["unchanged"] += 1
    entry["helped"].sort(key=lambda r: (-r["change"], r["qid"]))
    entry["hurt"].sort(key=lambda r: (r["change"], r["qid"]))
    return entry


def score_answers(root: Path, cfg) -> tuple[dict, dict, list[str]]:
    """(answers results, answer repeats, notes) for every answers file under answers/."""
    results: dict[str, dict] = {}
    notes: list[str] = []
    chunk_cache: dict[tuple, dict[str, str]] = {}
    for path in paths.answers_files(root):
        corpus = path.parent.name
        try:
            data = verify.load_answers(path)
        except verify.AnswersError as error:
            notes.append(f"{paths.rel(root, path)} was not checked: it {error}")
            continue
        chunking = data.get("chunking") or {}
        key = (data.get("corpus", corpus), int(chunking.get("words", cfg.chunk_words)), int(chunking.get("overlap", cfg.chunk_overlap)))
        if key not in chunk_cache:
            chunk_cache[key] = verify.chunk_texts(data, root, cfg)
        totals = verify.evaluate(data, chunk_cache[key], label=path.stem)
        t = totals.as_dict()
        entry = {
            "file": paths.rel(root, path), "corpus": corpus, "system": data.get("system"), "model": data.get("model"), **t,
            "verified_share": r6(t["verified"] / t["n_in"]) if t["n_in"] else None,
            "verified_wilson95": [r6(v) for v in stats.wilson(t["verified"], t["n_in"])],
            "not_found_in_share": r6(t["not_found_in"] / t["n_in"]) if t["n_in"] else None,
            "declined_out_share": r6(t["declined_out"] / t["n_out"]) if t["n_out"] else None,
            "declined_out_wilson95": [r6(v) for v in stats.wilson(t["declined_out"], t["n_out"])],
            "passed": dict(totals.passed),
        }  # fmt: skip
        results[f"{corpus}/{path.stem}"] = entry
    # The repeats of one label answer the same questions, so they are not independent trials: each
    # repeat keeps its own Wilson interval, and the spread is the range across repeats (no pooled interval).
    repeats: dict[str, dict] = {}
    for key, entry in results.items():
        parts = key.split("/", 1)[1].split(".")
        if len(parts) == 2 and paths.REPEAT_RE.match(parts[1]):
            group = repeats.setdefault(f"{entry['corpus']}/{parts[0]}", {"files": [], "verified_share": [], "declined_out_share": []})
            group["files"].append(entry["file"])
            group["verified_share"].append(entry["verified_share"])
            group["declined_out_share"].append(entry["declined_out_share"])
    for group in repeats.values():
        group["n_repeats"] = len(group["files"])
        for f in ("verified_share", "declined_out_share"):
            values = [v for v in group[f] if v is not None]
            group[f + "_range"] = r6(max(values) - min(values)) if values else None
    return results, repeats, notes


def compute(root: Path, cfg, test_qrels: Path | None = None) -> tuple[dict, list[str]]:
    """Everything p2 score writes to results.json, and notes about files it could not use."""
    refs, odd = discover(root)
    notes = [f"{paths.rel(root, p)} does not follow the run-file naming rules, so it was not scored" for p in odd]
    sets: dict[str, dict] = {}
    plan = [("shared", "practice", paths.qrels(root, "shared", "practice")), ("own", "own", paths.qrels(root, "own", "own"))]
    if test_qrels:
        plan.insert(1, ("shared", "test", Path(test_qrels)))
    for corpus, query_set, qrels_path in plan:
        result, set_notes = score_set(root, corpus, query_set, refs, qrels_path)
        notes += set_notes
        if result:
            sets[f"{corpus}/{query_set}"] = result
    answers, answer_repeats, answer_notes = score_answers(root, cfg)
    notes += answer_notes
    judged, judge_notes = stretch.judge(root)
    agent, agent_notes = stretch.agent(root, [(r.name, r.path) for r in refs if r.stretch], sets.get("own/own"))
    notes += judge_notes + agent_notes
    results = {
        "written_by": "uv run p2 score",
        "note": "p2 check recomputes this file from the committed runs, qrels, answers and traces; do not edit it by hand.",
        "metrics": list(metrics.METRICS),
        "sets": sets,
        "answers": answers,
        "stretch": {"judge": judged, "cost": stretch.cost(root, sets, answers), "agent": agent},
    }
    if answer_repeats:
        results["answer_repeats"] = answer_repeats
    return results, notes


def comparable(results: dict) -> dict:
    """A results structure without the part p2 check cannot recompute in CI: the seconds of the systems
    that do not call Claude, from traces/retrieval/, which git ignores."""
    out = copy.deepcopy(results) if isinstance(results, dict) else results
    cost = out.get("stretch", {}).get("cost") if isinstance(out, dict) and isinstance(out.get("stretch"), dict) else None
    if isinstance(cost, dict):
        cost.pop("local", None)
    return out


def checked_part(name: str, body: str) -> str:
    """The part of an EVAL.md block that p2 check compares (all of it, except the local part of stretch-cost)."""
    if name == "stretch-cost" or name.startswith("stretch-cost-"):
        return body.split(LOCAL_HEADING, 1)[0]
    return body


# ---- rendering the EVAL.md tables ----


def f3(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def signed(x: float) -> str:
    """A signed number at 3 decimals; a value that rounds to zero but is not zero keeps enough
    decimals to show its sign (-0.0002), so an interval that excludes zero never reads [.., -0.000]."""
    x = float(x)
    if x == 0:
        return "+0.000"
    for places in (3, 4, 5, 6):
        if round(x, places) != 0:
            return f"{x:+.{places}f}"
    return f"{x:+.6f}"


def interval(ci) -> str:
    return f"[{signed(ci[0])}, {signed(ci[1])}]"


def wilson_text(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def share(k: int, n: int) -> str:
    return f"{k} of {n} ({100 * k / n:.0f}%)" if n else "no questions"


def table(header: list[str], rows: list[list[str]], right: int = 1, text_last: bool = False) -> str:
    """A Markdown table: the first `right` columns align left and the numbers after them right;
    text_last keeps a last column of words (a reading) aligned left too."""
    align = ["---"] * right + ["---:"] * (len(header) - right)
    if text_last:
        align[-1] = "---"
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _metrics_for(slug: str | None) -> list[str]:
    return [METRIC_SLUGS[slug]] if slug else list(metrics.METRICS)


def render_systems(s: dict) -> str:
    rows = [[name] + [f3(v["mean"][m]) for m in metrics.METRICS] for name, v in s["systems"].items()]
    chance = s.get("chance_recall@10")
    note = f"{s['n_queries']} judged queries; nDCG uses {'graded' if s['graded'] else 'binary'} labels"
    note += f"; a random ranking of the {s['n_documents']:,} documents would get recall@10 near {chance:.3f}." if chance is not None else "."
    return table(["System"] + [METRIC_NAMES[m] for m in metrics.METRICS], rows) + "\n\n" + note


def render_classes(s: dict, slug: str | None, key: str = "by_class") -> str:
    wanted = _metrics_for(slug)
    first = next(iter(s["systems"].values()))
    if not first.get(key):
        return "_Your queries have no origin column yet, so there is nothing to split._"
    groups = list(first[key])
    header = ["System", "Metric"] + [f"{c} (n={first[key][c]['n']})" for c in groups]
    rows = [[name, METRIC_NAMES[m]] + [f3(v[key][c][m]) for c in groups] for name, v in s["systems"].items() for m in wanted]
    return table(header, rows, right=2)


def render_pairs(pairs: list[dict], slug: str | None) -> str:
    wanted = set(_metrics_for(slug))
    rows = [
        [f"{p['a']} minus {p['b']}", METRIC_NAMES[p["metric"]], signed(p["mean_diff"]), interval(p["ci95"]), f3(p["mdd80"]), p["reading"]]
        for p in pairs
        if p["metric"] in wanted
    ]
    if not rows:
        return "_Only one system is scored here, so there is nothing to compare yet._"
    note = "The interval is a paired bootstrap (10,000 resamples of the queries); MDD is the smallest difference this many queries detect 80% of the time."
    if any(p["reading"] == stats.IDENTICAL for p in pairs if p["metric"] in wanted):
        note += " Two systems that score the same on every query have no interval to speak of, so their MDD is shown as -."
    return table(["Comparison", "Metric", "Mean difference", "95% interval", "MDD", "Reading"], rows, right=2, text_last=True) + "\n\n" + note


ABLATION_PLACEHOLDER = "_No ablation runs yet: `uv run p2 run --corpus own --system NAME --ablation` writes one to runs/own/ablation/, then `uv run p2 score` fills this table._"
JUDGE_PLACEHOLDER = (
    "_No judged answers yet (stretch option 2): `uv run p2 judge answers/shared/LABEL.json` writes them, your own verdicts go in"
    " answers/shared/LABEL.calibration.tsv, and then `uv run p2 score` fills this table._"
)
COST_PLACEHOLDER = "_No Claude traces yet: a run of a system that calls Claude, `p2 answer` and `p2 judge` write them in traces/, and then `uv run p2 score` fills this table._"
AGENT_PLACEHOLDER = (
    "_No stretch runs yet (stretch option 5): `uv run p2 run --corpus own --queries eval/own/agent.queries.tsv --system agent --stretch`"
    " writes one to runs/own/stretch/, then `uv run p2 score` fills this table._"
)
SET_LABELS = {"shared/practice": "shared practice", "own/own": "own"}
MIN_GROUP = 20


def render_origins(s: dict, slug: str | None) -> str:
    first = next(iter(s["systems"].values()))
    if not first.get("by_origin"):
        return "_Your queries have no origin column yet, so there is nothing to split._"
    wanted = _metrics_for(slug)
    groups = list(first["by_origin"])
    o = s.get("origins") or {}
    both = len(groups) == 2 and all(v.get("origin_diff") for v in s["systems"].values())
    header = ["System", "Metric"] + [f"{g} (n={first['by_origin'][g]['n']})" for g in groups]
    if both:
        header += ["hand minus claude", "95% interval", "Reading"]
    rows = []
    for name, v in s["systems"].items():
        for m in wanted:
            row = [name, METRIC_NAMES[m]] + [f3(v["by_origin"][g][m]) for g in groups]
            if both:
                d = v["origin_diff"][m]
                row += [signed(d["mean_diff"]), interval(d["ci95"]), d["reading"]]
            rows.append(row)
    parts = [table(header, rows, right=2, text_last=both)]
    if both:
        parts.append("The hand and claude queries are different queries, so the interval of hand minus claude comes from resampling the queries within each group (10,000 times), not from pairing them.")
    else:
        parts.append(f"All your judged queries have the origin {groups[0]}, so there is nothing to compare.")
    inter = o.get("interaction")
    if inter:
        irows = [[METRIC_NAMES[m], signed(inter[m]["lead_claude"]), signed(inter[m]["lead_hand"]), signed(inter[m]["mean_diff"]), interval(inter[m]["ci95"]), inter[m]["reading"]] for m in wanted]
        parts.append("Do model-written queries favor keyword search? bm25's lead over dense (bm25 minus dense, per query) on the claude queries and on the hand queries, and claude minus hand:")
        parts.append(table(["Metric", "Lead on claude", "Lead on hand", "Claude minus hand", "95% interval", "Reading"], irows, right=1, text_last=True))
    elif both:
        parts.append("Whether bm25's lead over dense differs between the two groups needs runs of both bm25 and dense on your own corpus.")
    overlap = o.get("overlap") or {}
    if overlap:
        bits = [f"{g} {overlap[g]['mean']:.2f} ({overlap[g]['n']} queries)" for g in ("hand", "claude") if g in overlap]
        parts.append("Mean content-word overlap, the share of a query's content words found in its best relevant document: " + ", ".join(bits) + ".")
    n = o.get("n") or {}
    if both and min(n.values()) < MIN_GROUP:
        parts.append(f"Stretch option 6 asks for at least {MIN_GROUP} judged queries of each origin; here there are {n['hand']} hand and {n['claude']} claude.")
    return "\n\n".join(parts)


def render_ablation_queries(s: dict | None) -> str:
    entries = (s or {}).get("ablation_queries") or {}
    if not entries:
        return ABLATION_PLACEHOLDER
    parts = []
    for name, e in entries.items():
        if e["base"] is None:
            parts.append(f"{name}: {e['rule']}.")
            continue
        head = f"{name} against {e['base']} ({e['rule']}): it helped {len(e['helped'])} queries, hurt {len(e['hurt'])} and left {e['unchanged']} unchanged."
        rows = [[r["qid"], r["class"], r["origin"] or "-", f3(r["base_rr"]), f3(r["rr"]), signed(r["change"]), effect] for effect in ("helped", "hurt") for r in e[effect]]
        if rows:
            head += "\n\n" + table(["Query", "Class", "Origin", f"{e['base']} RR", f"{name} RR", "Change", "Effect"], rows, right=3, text_last=True)
        parts.append(head)
    parts.append("RR is the reciprocal rank of the first relevant document in the top 10 (0 when none is there), the per-query value behind MRR@10; the queries at the top of each list are the ones to read.")
    return "\n\n".join(parts)


def render_stretch_judge(results: dict) -> str:
    j = (results.get("stretch") or {}).get("judge") or {}
    files = j.get("files") or {}
    if not files:
        return JUDGE_PLACEHOLDER
    rows = []
    for key, e in files.items():
        compared = e["compared"]
        rows.append([
            key.split("/", 1)[1], str(e["claims_judged"]), str(e["judge_counts"]["supported"]), str(e["judge_counts"]["partly"]), str(e["judge_counts"]["not"]),
            str(e["labeled"]), str(compared), share(e["agree"], compared) if compared else "-", wilson_text(e["agreement_wilson95"]) if compared else "-",
        ])  # fmt: skip
    pooled = j.get("pooled")
    if pooled and len(files) > 1:
        rows.append(["all files", "", "", "", "", "", str(pooled["compared"]), share(pooled["agree"], pooled["compared"]), wilson_text(pooled["agreement_wilson95"])])
    header = ["Answers file", "Judged by Claude", "supported", "partly", "not", "Labeled by you", "Labeled by both", "Agree", "95% interval"]
    parts = [table(header, rows)]
    if not pooled:
        parts.append(f"Nothing to compare yet: label at least {MIN_GROUP} claims in a calibration file beside a judged file.")
        return "\n\n".join(parts)
    verdicts = list(pooled["matrix"])
    mrows = [[v] + [str(pooled["matrix"][v][w]) for w in verdicts] + [str(sum(pooled["matrix"][v].values()))] for v in verdicts]
    mrows.append(["total"] + [str(sum(pooled["matrix"][v][w] for v in verdicts)) for w in verdicts] + [str(pooled["compared"])])
    parts.append(f"Claude's verdict (rows) against yours (columns), over the {pooled['compared']} claims you both labeled:")
    parts.append(table(["Claude", *(f"you: {w}" for w in verdicts), "Total"], mrows))
    note = "Agree is the share of the claims you both labeled where Claude's verdict is yours, with a Wilson 95% interval."
    if pooled["compared"] < MIN_GROUP:
        note += f" That is {pooled['compared']} claims, fewer than the {MIN_GROUP} the stretch asks for."
    parts.append(note)
    return "\n\n".join(parts)


def render_stretch_cost(results: dict, slug: str | None) -> str:
    c = (results.get("stretch") or {}).get("cost") or {}
    metric = METRIC_SLUGS[slug] if slug else "mrr@10"
    parts = []
    traces = c.get("traces") or []
    if traces:
        rows = []
        for t in traces:
            n = t["spans"] or 1
            rows.append([
                t["file"], t["system"] or "-", str(t["spans"]), str(t["calls"]), str(t["saved"]), f"{t['input_tokens']:,}", f"{t['output_tokens']:,}", f"{t['seconds']:.1f}",
                f"{t['input_tokens'] / n:,.0f}", f"{t['output_tokens'] / n:,.0f}", f"{t['seconds'] / n:.1f}",
            ])  # fmt: skip
        header = ["Trace", "System", "Spans", "Claude calls", "Saved", "Input tokens", "Output tokens", "Seconds", "Input per span", "Output per span", "Seconds per span"]
        parts.append("Every trace in traces/ that records a Claude call:\n\n" + table(header, rows, right=2))
        parts.append(
            "A span is one query of a run, one question of an answers file, or one claim of a judged file."
            " A saved reply costs nothing now, so it counts the tokens and the seconds of the call that made it."
        )
    else:
        parts.append(COST_PLACEHOLDER)
    pairs = [p for p in c.get("run_pairs") or [] if p["metric"] == metric]
    if pairs:
        rows = [
            [f"{SET_LABELS.get(p['set'], p['set'])}: {p['cheaper']} minus {p['dearer']}", str(p["n"]), signed(p["mean_diff"]), interval(p["ci95"]),
             f"{p['input_saved_per_query']:,.0f}", f"{p['seconds_saved_per_query']:.1f}", p["reading"]]
            for p in pairs
        ]  # fmt: skip
        header = ["Cheaper minus dearer", "Queries", f"{METRIC_NAMES[metric]} difference", "95% interval", "Input tokens saved per query", "Seconds saved per query", "Reading"]
        parts.append("Every pair of systems on the same queries where at least one calls Claude, the one with fewer Claude input tokens per query first:\n\n" + table(header, rows, text_last=True))
        parts.append("The difference is the cheaper system's score minus the dearer one's, with its paired interval; the savings count Claude calls only, so a system that makes none saves all of the other's.")
    answer_pairs = c.get("answers_pairs") or []
    if answer_pairs:
        rows = [
            [f"{p['corpus']}: {p['cheaper']} minus {p['dearer']}", str(p["n"]), signed(p["mean_diff"]), interval(p["ci95"]),
             f"{p['input_saved_per_question']:,.0f}", f"{p['seconds_saved_per_question']:.1f}", p["reading"]]
            for p in answer_pairs
        ]  # fmt: skip
        header = ["Cheaper minus dearer", "Questions", "Passing share difference", "95% interval", "Input tokens saved per question", "Seconds saved per question", "Reading"]
        parts.append("Every two answers files on the same questions, the one with fewer input tokens per question first:\n\n" + table(header, rows, text_last=True))
        parts.append("A question passes when its `p2 verify` line is a PASS: every quote is in a chunk that was retrieved, and the question is declined exactly when it should be.")
    local = c.get("local") or []
    if local:
        rows = [[t["file"], t["system"] or "-", str(t["spans"]), f"{t['seconds']:.2f}", f"{1000 * t['seconds'] / (t['spans'] or 1):.1f}"] for t in local]
        parts.append(
            f"{LOCAL_HEADING} on the machine that last ran `p2 score` (git ignores that folder, so `p2 check` does not compare this part):\n\n"
            + table(["Trace", "System", "Queries", "Seconds", "Milliseconds per query"], rows, right=2)
        )
    return "\n\n".join(parts)


def render_stretch_agent(results: dict, slug: str | None) -> str:
    a = (results.get("stretch") or {}).get("agent") or {}
    if not a:
        return AGENT_PLACEHOLDER
    parts = []
    for name, e in a.items():
        head = f"{name} on {e['n_queries']} of your own queries, {len(e['judged'])} of them with a relevant document"
        head += f" ({', '.join(e['judged'])})." if e["judged"] else "."
        rows = [
            [sys_name] + [f3(v["mean"][m]) for m in metrics.METRICS] + ([f"{v['input_tokens']:,.0f}", f"{v['seconds']:.1f}"] if v["claude"] else ["-", "-"])
            for sys_name, v in e["systems"].items()
        ]
        parts.append(head + "\n\n" + table(["System"] + [METRIC_NAMES[m] for m in metrics.METRICS] + ["Claude input tokens per query", "Claude seconds per query"], rows))
        parts.append(render_pairs(e["pairs"], slug or "mrr"))
    parts.append("Systems that make no Claude calls show -; their own seconds per query are in the stretch-cost table on your machine.")
    return "\n\n".join(parts)


def render_answers(results: dict) -> str:
    rows = []
    for key, a in results["answers"].items():
        if paths.REPEAT_RE.match(key.rsplit(".", 1)[-1]) and "." in key:
            continue
        rows.append([
            key.split("/", 1)[1], str(a["system"]),
            share(a["verified"], a["n_in"]), wilson_text(a["verified_wilson95"]),
            share(a["not_found_in"], a["n_in"]),
            share(a["declined_out"], a["n_out"]), wilson_text(a["declined_out_wilson95"]),
            f"{a['claims_verified']} of {a['claims']}",
        ])  # fmt: skip
    if not rows:
        return "_No answers files yet: run `uv run p2 answer --corpus shared --system NAME`, then `uv run p2 score`._"
    header = ["Label", "System", "Verified (in)", "95% interval", "not_found (in)", "Declined (out)", "95% interval", "Quotes found"]
    return table(header, rows, right=2)


def render_repeats(results: dict) -> str:
    parts = []
    for key, s in results["sets"].items():
        for name, entry in (s.get("repeats") or {}).items():
            rows = [[f"{name}.r{r}"] + [f3(entry["values"][m][i]) for m in metrics.METRICS] for i, r in enumerate(entry["repeat"])]
            rows.append(["range (max minus min)"] + [f3(entry["spread"][m]["range"]) for m in metrics.METRICS])
            parts.append(f"Repeated runs of {name} on {key}:\n\n" + table(["Run"] + [METRIC_NAMES[m] for m in metrics.METRICS], rows))
            against = ["hybrid"] if "hybrid" in entry["vs"] else list(entry["vs"])
            vs_rows = [
                [f"{name}.r{c['repeat']} minus {other}", METRIC_NAMES[m], signed(c["mean_diff"]), interval(c["ci95"]), c["reading"]]
                for other in against
                for m in metrics.METRICS
                for c in entry["vs"][other][m]
            ]
            if vs_rows:
                parts.append(table(["Comparison", "Metric", "Mean difference", "95% interval", "Reading"], vs_rows, right=2, text_last=True))
    for key, g in (results.get("answer_repeats") or {}).items():
        rows = []
        for f in g["files"]:
            a = results["answers"][f"{key.split('/', 1)[0]}/{Path(f).stem}"]
            rows.append([Path(f).stem, share(a["verified"], a["n_in"]), wilson_text(a["verified_wilson95"]), share(a["declined_out"], a["n_out"]), wilson_text(a["declined_out_wilson95"])])
        rows.append(["range (max minus min)", f3(g["verified_share_range"]), "", f3(g["declined_out_share_range"]), ""])
        note = "Each repeat answers the same questions, so the repeats are not independent trials and their counts are not pooled into one interval."
        parts.append(f"Repeated answers {key}:\n\n" + table(["File", "Verified (in)", "95% interval", "Declined (out)", "95% interval"], rows) + "\n\n" + note)
    if not parts:
        return "_No repeated runs or answers files yet (598E: `p2 run ... --repeat 3` and `p2 answer ... --repeat 3`)._"
    return "\n\n".join(parts)


def render(name: str, results: dict) -> str | None:
    """The text of one EVAL.md block, or None when the name is not one p2 knows."""
    if name == "answers":
        return render_answers(results)
    if name == "repeats":
        return render_repeats(results)
    if name == "stretch-judge":
        return render_stretch_judge(results)
    if name == "own-ablation-queries":
        return render_ablation_queries(results["sets"].get("own/own"))
    for base in METRIC_BLOCKS:
        if name == base or (name.startswith(base + "-") and name[len(base) + 1 :] in METRIC_SLUGS):
            slug = None if name == base else name[len(base) + 1 :]
            return render_stretch_cost(results, slug) if base == "stretch-cost" else render_stretch_agent(results, slug)
    for set_name, (corpus, query_set) in sorted(SETS.items(), key=lambda kv: -len(kv[0])):
        if name != set_name and not name.startswith(set_name + "-"):
            continue
        rest = name[len(set_name) + 1 :].split("-") if name != set_name else []
        view = rest[0] if rest else None
        slug = rest[1] if len(rest) > 1 else None
        if len(rest) > 2 or (view and view not in VIEWS) or (slug and slug not in METRIC_SLUGS) or (view in OWN_ONLY_VIEWS and corpus != "own"):
            return None
        s = results["sets"].get(f"{corpus}/{query_set}")
        if view == "ablation" and not (s and any(n.startswith("ablation/") for n in s["systems"])):
            return ABLATION_PLACEHOLDER
        if not s or not s["systems"]:
            where = "--corpus own" if corpus == "own" else f"--corpus shared --queries {query_set}"
            return f"_No scored runs here yet: run `uv run p2 run {where} --system bm25` (or `--all`), then `uv run p2 score`._"
        if view is None:
            return render_systems(s)
        if view == "classes":
            return render_classes(s, slug)
        if view == "origins":
            return render_origins(s, slug)
        if view == "pairs":
            return render_pairs(s["pairs"], slug)
        ablations = [p for p in s["pairs"] if p["a"].startswith("ablation/") != p["b"].startswith("ablation/")]
        return render_pairs(ablations, slug)
    return None


def blocks(text: str) -> dict[str, str]:
    """{name: body} of every complete block in an EVAL.md text."""
    return {m.group("name"): m.group("body") for m in BLOCK_RE.finditer(text)}


def marker_problems(text: str) -> list[str]:
    """Markers without a partner, or a name used twice."""
    begins, ends = BEGIN_RE.findall(text), END_RE.findall(text)
    problems = []
    for name in sorted(set(begins) | set(ends)):
        if begins.count(name) != ends.count(name):
            problems.append(f"the {name} block is missing its {'end' if begins.count(name) > ends.count(name) else 'begin'} marker")
        elif begins.count(name) > 1:
            problems.append(f"the {name} block appears more than once")
    return problems


def rewrite(text: str, results: dict) -> tuple[str, list[str], list[str]]:
    """(new text, names rewritten, names p2 does not know). Unknown blocks are left as they are."""
    done: list[str] = []
    unknown: list[str] = []

    def replace(m: re.Match) -> str:
        name = m.group("name")
        body = render(name, results)
        if body is None:
            unknown.append(name)
            return m.group(0)
        done.append(name)
        return f"<!-- p2:begin {name} -->\n{body}\n<!-- p2:end {name} -->"

    return BLOCK_RE.sub(replace, text), done, unknown


NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def same_text(a: str, b: str, tolerance: float = 0.0011) -> bool:
    """True when two block texts differ at most in their numbers, by no more than the tolerance."""
    a, b = a.strip(), b.strip()
    if NUMBER.sub("#", a) != NUMBER.sub("#", b):
        return False
    return all(abs(float(x) - float(y)) <= tolerance for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)))


def same_results(a, b, tolerance: float = 0.0005, where: str = "results") -> str | None:
    """None when two results structures match (numbers to 3 decimals), else where they first differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            extra = sorted(set(a) ^ set(b))
            return f"{where}.{extra[0]}"
        for k in a:
            found = same_results(a[k], b[k], tolerance, f"{where}.{k}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return where
        for i, (x, y) in enumerate(zip(a, b)):
            found = same_results(x, y, tolerance, f"{where}[{i}]")
            if found:
                return found
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a == b else where
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return None if abs(a - b) <= tolerance else where
    return None if a == b else where


def dumps(results: dict) -> str:
    return json.dumps(results, indent=2, ensure_ascii=False) + "\n"


SET_NAMES = {"shared/practice": "shared practice queries", "shared/test": "shared test queries", "own/own": "own corpus"}


def missing_blocks(text: str, section: str) -> list[str]:
    """The tables EVAL.md must keep that it does not have (a block with a metric added counts)."""
    names = set(blocks(text))
    wanted = REQUIRED_BLOCKS + (RIDER_BLOCKS if section == "598E" else ())
    return [w for w in wanted if not any(n == w or (n.startswith(w + "-") and n[len(w) + 1 :] in METRIC_SLUGS) for n in names)]


def print_summary(results: dict) -> None:
    for key, s in results["sets"].items():
        print(f"\n{SET_NAMES.get(key, key)}: {s['n_queries']} judged queries")
        width = max(len(n) for n in s["systems"]) if s["systems"] else 6
        print(f"  {'system':<{width}}  recall@10  MRR@10  nDCG@10")
        for name, v in s["systems"].items():
            print(f"  {name:<{width}}  {v['mean']['recall@10']:9.3f}  {v['mean']['mrr@10']:6.3f}  {v['mean']['ndcg@10']:7.3f}")
        for p in s["pairs"]:
            if p["metric"] == "mrr@10":
                print(f"  MRR@10 {p['a']} minus {p['b']}: {signed(p['mean_diff'])} {interval(p['ci95'])}, {p['reading']}")
        if key == "own/own" and "bm25" in s["systems"]:
            best = max(v["mean"]["mrr@10"] for v in s["systems"].values())
            if s["systems"]["bm25"]["mean"]["recall@10"] >= TOO_EASY or best >= TOO_EASY:
                print("  Note: a score of 0.95 or more suggests your queries are too easy to tell the systems apart; harder queries make the comparison mean more.")
    for key, a in results["answers"].items():
        print(f"\nanswers {key.split('/', 1)[1]}: verified {share(a['verified'], a['n_in'])}, not_found in {share(a['not_found_in'], a['n_in'])}, declined out {share(a['declined_out'], a['n_out'])}")


def run(args, cfg) -> int:
    root = cfg.root
    results, notes = compute(root, cfg, getattr(args, "test_qrels", None))
    for note in notes:
        print(f"Note: {note}.", file=sys.stderr)
    if getattr(args, "test_qrels", None):
        text = dumps(results)
        if args.out:
            paths.write_text(Path(args.out), text)
            print(f"Wrote {args.out} (results.json and EVAL.md are unchanged).")
        else:
            sys.stdout.write(text)
        return 0
    out = paths.results_file(root)
    paths.write_text(out, dumps(results))
    print(f"Wrote {paths.rel(root, out)}.")
    eval_path = root / EVAL_FILE
    if eval_path.is_file():
        text = eval_path.read_text(encoding="utf-8")
        problems = marker_problems(text)
        for problem in problems:
            print(f"EVAL.md: {problem}; put the marker back as in the template.")
        new, done, unknown = rewrite(text, results)
        if new != text:
            paths.write_text(eval_path, new)
        print(f"Rewrote {len(done)} table(s) in EVAL.md" + (f": {', '.join(done)}." if done else "."))
        if unknown:
            print(f"EVAL.md has block names p2 does not know, left as they were: {', '.join(unknown)}.")
    else:
        print("There is no EVAL.md, so no tables were rewritten.")
    print_summary(results)
    return 0
