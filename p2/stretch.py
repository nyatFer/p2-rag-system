"""The numbers behind the stretch tables of EVAL.md (`p2 score` writes them; p2/score.py renders them).

- judge(root): stretch option 2, the claim-level faithfulness check. For every answers file with a
  judged file (`p2 judge`) or a calibration file (your own verdicts), Claude's verdict counts, and for
  the claims you both labeled the agreement with a Wilson 95% interval and the 3 by 3 table of
  Claude's verdicts against yours, per file and pooled over all files.
- cost(root, sets, answers): stretch option 3, cost and latency. Every trace in traces/ with a Claude
  call added up (calls, saved replies, tokens, seconds), then every pair of systems on the same queries where
  at least one calls Claude, cheaper minus dearer, with the metric difference and its paired interval
  next to the tokens and seconds the cheaper one saves; the same for answers files on the same
  questions; and, from traces/retrieval/ when it exists on this machine, the seconds of the systems
  that do not call Claude. Git ignores traces/retrieval/, so that last part is kept under "local",
  and p2 check leaves it out when it compares (see comparable() in p2/score.py).
- agent(root, stretch_runs, own): stretch option 5, a stretch run (the grep agent) on some of your
  own queries, against bm25, dense, hybrid and rerank on the same queries, with the Claude tokens and
  seconds per query.
The seconds of a span are its duration plus, for replies saved from an earlier run, the seconds those
calls took when they were made (p2.saved_seconds), so a run made from saved replies still shows what
its calls cost.
"""

from __future__ import annotations

from pathlib import Path

from p2 import judge as judge_mod
from p2 import metrics, paths, stats, trace, verify
from p2.retrievers import CANONICAL
from p2.runfile import read_qrels, read_run


def r6(x: float) -> float:
    return round(float(x), 6)


# ---- option 2: the judge against your own verdicts ----


def _matrix() -> dict[str, dict[str, int]]:
    return {j: {s: 0 for s in judge_mod.VERDICTS} for j in judge_mod.VERDICTS}


def judge(root: Path) -> tuple[dict, list[str]]:
    """({"files": {...}, "pooled": {...} or None}, notes)."""
    files: dict[str, dict] = {}
    notes: list[str] = []
    pooled = _matrix()
    for answers in paths.answers_files(root):
        judged_path, calibration_path = paths.judged_file(answers), paths.calibration_file(answers)
        if not judged_path.is_file() and not calibration_path.is_file():
            continue
        key = f"{answers.parent.name}/{paths.answers_stem(answers)}"
        entry: dict = {
            "answers_file": paths.rel(root, answers), "judged_file": None, "calibration_file": None,
            "claims_judged": 0, "judge_counts": {v: 0 for v in judge_mod.VERDICTS}, "judge_errors": 0,
            "labeled": 0, "compared": 0, "agree": 0,
        }  # fmt: skip
        verdicts: dict[str, str] = {}
        if judged_path.is_file():
            try:
                data = judge_mod.load_judged(judged_path)
            except judge_mod.JudgedError as error:
                notes.append(f"{paths.rel(root, judged_path)} was not used: it {error}")
                data = None
            if data is not None:
                entry["judged_file"] = paths.rel(root, judged_path)
                entry["claims_judged"] = len(data["claims"])
                for record in data["claims"]:
                    if record["verdict"] in judge_mod.VERDICTS:
                        verdicts[record["id"]] = record["verdict"]
                        entry["judge_counts"][record["verdict"]] += 1
                    else:
                        entry["judge_errors"] += 1
        labels: dict[str, str] = {}
        if calibration_path.is_file():
            try:
                known = {c["id"] for c in judge_mod.claims_of(verify.load_answers(answers))}
            except verify.AnswersError:
                known = None
            labels, problems = judge_mod.read_calibration(calibration_path, known)
            if problems:
                notes.append(f"{paths.rel(root, calibration_path)}: {problems[0]}")
            entry["calibration_file"] = paths.rel(root, calibration_path)
            entry["labeled"] = len(labels)
        matrix = _matrix()
        for cid, mine in labels.items():
            theirs = verdicts.get(cid)
            if theirs is None:
                continue
            matrix[theirs][mine] += 1
            pooled[theirs][mine] += 1
        entry["compared"] = sum(sum(row.values()) for row in matrix.values())
        entry["agree"] = sum(matrix[v][v] for v in judge_mod.VERDICTS)
        entry["agreement"] = r6(entry["agree"] / entry["compared"]) if entry["compared"] else None
        entry["agreement_wilson95"] = [r6(v) for v in stats.wilson(entry["agree"], entry["compared"])]
        entry["matrix"] = matrix
        files[key] = entry
    compared = sum(sum(row.values()) for row in pooled.values())
    agree = sum(pooled[v][v] for v in judge_mod.VERDICTS)
    pooled_entry = None
    if compared:
        pooled_entry = {
            "compared": compared, "agree": agree, "agreement": r6(agree / compared),
            "agreement_wilson95": [r6(v) for v in stats.wilson(agree, compared)], "matrix": pooled,
        }  # fmt: skip
    return {"files": files, "pooled": pooled_entry}, notes


# ---- option 3: cost and latency ----


def trace_totals(path: Path) -> dict | None:
    """Calls, saved replies, tokens and seconds of one trace file, or None when it cannot be read."""
    try:
        spans = trace.read(path)
    except (OSError, ValueError):
        return None
    systems = [str(s.get("p2.system")) for s in spans if s.get("p2.system")]
    return {
        "system": max(sorted(set(systems)), key=systems.count) if systems else None,
        "spans": len(spans),
        "calls": sum(int(s.get("p2.claude_calls") or 0) for s in spans),
        "saved": sum(int(s.get("p2.cached_calls") or 0) for s in spans),
        "input_tokens": sum(int(s.get("gen_ai.usage.input_tokens") or 0) for s in spans),
        "output_tokens": sum(int(s.get("gen_ai.usage.output_tokens") or 0) for s in spans),
        "seconds": r6(sum(float(s.get("duration_ms") or 0) / 1000 + float(s.get("p2.saved_seconds") or 0) for s in spans)),
    }


def span_costs(path: Path) -> dict[str, dict] | None:
    """{qid: {input_tokens, output_tokens, seconds, calls}} of a trace, or None when there is none."""
    if not path.is_file():
        return None
    try:
        spans = trace.read(path)
    except (OSError, ValueError):
        return None
    out = {}
    for s in spans:
        out.setdefault(str(s.get("p2.qid")), {
            "input_tokens": int(s.get("gen_ai.usage.input_tokens") or 0), "output_tokens": int(s.get("gen_ai.usage.output_tokens") or 0),
            "seconds": float(s.get("duration_ms") or 0) / 1000 + float(s.get("p2.saved_seconds") or 0), "calls": int(s.get("p2.claude_calls") or 0),
        })  # fmt: skip
    return out


def per_item(costs: dict[str, dict] | None, qids: list[str] | None = None) -> dict:
    """Mean Claude cost per query (or question, or claim) over `qids` (default: every span); zero
    when the system makes no Claude calls."""
    if not costs or not any(c["calls"] for c in costs.values()):
        return {"claude": False, "input_tokens": 0.0, "output_tokens": 0.0, "seconds": 0.0}
    keys = [q for q in (qids if qids is not None else list(costs)) if q in costs]
    n = len(keys) or 1
    return {
        "claude": True,
        "input_tokens": r6(sum(costs[q]["input_tokens"] for q in keys) / n),
        "output_tokens": r6(sum(costs[q]["output_tokens"] for q in keys) / n),
        "seconds": r6(sum(costs[q]["seconds"] for q in keys) / n),
    }


def _run_trace(root: Path, run: str) -> Path:
    """The committed Claude trace of a run file (runs/shared/rerank.practice.trec, runs/own/rerank.trec)."""
    parts = Path(run).parts
    stem = Path(run).name[: -len(".trec")]
    if parts[1] == "shared":
        name, query_set = stem.split(".")[:2]
        return paths.trace_file(root, "shared", query_set, name, calls_claude=True)
    ablation = len(parts) > 3 and parts[2] == "ablation"
    stretch = len(parts) > 3 and parts[2] == "stretch"
    return paths.trace_file(root, "own", "own", stem, ablation=ablation, calls_claude=True, stretch=stretch)


def run_pairs(root: Path, sets: dict) -> list[dict]:
    """Cheaper minus dearer for every pair of systems on the same queries where at least one calls Claude."""
    out = []
    for key in ("shared/practice", "own/own"):
        s = sets.get(key)
        if not s:
            continue
        names = [n for n in s["systems"] if not n.startswith("ablation/")]
        cost = {n: per_item(span_costs(_run_trace(root, s["systems"][n]["run"]))) for n in names}
        for p in s["pairs"]:
            a, b = p["a"], p["b"]
            if a not in cost or b not in cost or not (cost[a]["claude"] or cost[b]["claude"]):
                continue
            if (cost[a]["input_tokens"], a) > (cost[b]["input_tokens"], b):
                cheap, dear, sign = b, a, -1
            else:
                cheap, dear, sign = a, b, 1
            out.append({
                "set": key, "cheaper": cheap, "dearer": dear, "metric": p["metric"], "n": p["n"],
                "mean_diff": r6(sign * p["mean_diff"]), "ci95": sorted(r6(sign * v) for v in p["ci95"]),
                "reading": p["reading"] if p["reading"] == stats.IDENTICAL else stats.reading(sorted(sign * v for v in p["ci95"]), cheap, dear),
                "input_saved_per_query": r6(cost[dear]["input_tokens"] - cost[cheap]["input_tokens"]),
                "seconds_saved_per_query": r6(cost[dear]["seconds"] - cost[cheap]["seconds"]),
            })  # fmt: skip
    return out


def answers_pairs(root: Path, answers: dict) -> list[dict]:
    """Cheaper minus dearer for every two answers files on the same questions (repeats left out)."""
    entries = []
    for key, a in answers.items():
        stem = key.split("/", 1)[1]
        if "." in stem or not a.get("passed"):
            continue
        costs = span_costs(paths.answers_trace(root, stem))
        if costs is None:  # no trace: the answers file's own records (the answer calls only)
            try:
                records = verify.load_answers(root / a["file"])["answers"]
            except verify.AnswersError:
                records = []
            costs = {
                str(r["qid"]): {"input_tokens": int(r.get("input_tokens") or 0), "output_tokens": int(r.get("output_tokens") or 0), "seconds": float(r.get("seconds") or 0), "calls": 1}
                for r in records
            }
        entries.append((key, a, per_item(costs)))
    out = []
    for i, (ka, a, ca) in enumerate(entries):
        for kb, b, cb in entries[i + 1 :]:
            if a["corpus"] != b["corpus"] or set(a["passed"]) != set(b["passed"]):
                continue
            (kc, c, cc), (kd, d, cd) = sorted([(ka, a, ca), (kb, b, cb)], key=lambda e: (e[2]["input_tokens"], e[0]))
            qids = sorted(c["passed"])
            boot = stats.paired_bootstrap([float(c["passed"][q]) for q in qids], [float(d["passed"][q]) for q in qids])
            cheap, dear = kc.split("/", 1)[1], kd.split("/", 1)[1]
            out.append({
                "corpus": c["corpus"], "cheaper": cheap, "dearer": dear, "n": boot["n"],
                "mean_diff": r6(boot["mean_diff"]), "ci95": [r6(v) for v in boot["ci95"]], "reading": stats.reading(boot["ci95"], cheap, dear),
                "input_saved_per_question": r6(cd["input_tokens"] - cc["input_tokens"]),
                "seconds_saved_per_question": r6(cd["seconds"] - cc["seconds"]),
            })  # fmt: skip
    return out


def cost(root: Path, sets: dict, answers: dict) -> dict:
    traces = []
    for path in sorted((root / "traces").glob("*.jsonl")) if (root / "traces").is_dir() else []:
        totals = trace_totals(path)
        if totals is not None and totals["calls"]:  # an old trace of a system without Claude calls says nothing about cost
            traces.append({"file": paths.rel(root, path), **totals})
    local = []
    for path in sorted((root / "traces" / "retrieval").glob("*.jsonl")) if (root / "traces" / "retrieval").is_dir() else []:
        totals = trace_totals(path)
        if totals is not None:
            local.append({"file": paths.rel(root, path), "system": totals["system"], "spans": totals["spans"], "seconds": totals["seconds"]})
    return {"traces": traces, "run_pairs": run_pairs(root, sets), "answers_pairs": answers_pairs(root, answers), "local": local}


# ---- option 5: a stretch run (the grep agent) against the four systems ----


def agent(root: Path, stretch_runs: list[tuple[str, Path]], own: dict | None) -> tuple[dict, list[str]]:
    """{"stretch/<name>": {...}} for every stretch run, and notes about runs that could not be used."""
    out: dict[str, dict] = {}
    notes: list[str] = []
    qrels_path = paths.qrels(root, "own", "own")
    qrels = read_qrels(qrels_path)[0] if qrels_path.is_file() else {}
    for name, path in stretch_runs:
        try:
            run = read_run(path)
        except ValueError as error:
            notes.append(f"{paths.rel(root, path)} was not scored: {error}")
            continue
        # A query the agent found nothing for has no line in the run file, but its span in the committed
        # trace still says it was asked, so it scores 0 instead of dropping out of the comparison.
        spans = span_costs(_run_trace(root, paths.rel(root, path)))
        asked = list(run) + [q for q in (spans or {}) if q not in run]
        judged = metrics.judged_queries(qrels, asked)
        entry: dict = {"run": paths.rel(root, path), "n_queries": len(asked), "judged": judged, "systems": {}, "pairs": []}
        system = f"stretch/{name}"
        per_query = {system: metrics.evaluate(run, qrels, judged)}
        costs = {system: per_item(spans, judged)}
        for other in CANONICAL:
            v = (own or {}).get("systems", {}).get(other)
            if v is None:
                continue
            per_query[other] = {q: v["per_query"][q] for q in judged if q in v["per_query"]}
            costs[other] = per_item(span_costs(_run_trace(root, v["run"])), judged)
        for sys_name, pq in per_query.items():
            entry["systems"][sys_name] = {"mean": {m: r6(metrics.mean(pq, m, judged)) for m in metrics.METRICS}, **costs[sys_name]}
        if judged:
            for other in [n for n in per_query if n != system]:
                if set(per_query[other]) != set(judged):
                    continue
                for m in metrics.METRICS:
                    va = [per_query[system][q][m] for q in judged]
                    vb = [per_query[other][q][m] for q in judged]
                    boot = stats.paired_bootstrap(va, vb)
                    same = all(x == y for x, y in zip(va, vb))
                    entry["pairs"].append({
                        "a": system, "b": other, "metric": m, "n": boot["n"],
                        "mean_diff": r6(boot["mean_diff"]), "ci95": [r6(v) for v in boot["ci95"]], "sd_diff": r6(boot["sd_diff"]),
                        "mdd80": None if same else r6(stats.mdd(boot["sd_diff"], boot["n"])),
                        "reading": stats.IDENTICAL if same else stats.reading(boot["ci95"], system, other),
                    })  # fmt: skip
        out[system] = entry
    return out, notes
