"""`p2 judge FILE`: ask Claude, one claim at a time, whether a claim's quote supports the claim (stretch option 2).

`p2 verify` proves that a quote is a real piece of the chunk it cites; it cannot tell whether the
quote supports the claim next to it. This command asks Claude exactly that, once per claim:
- Claude gets the question, the claim and its quote, with prompts/judge.txt (your prompt, its lines
  that start with # removed) as the system prompt, and its reply is forced into
  {"verdict": "supported" | "partly" | "not", "reason": one sentence}.
- Every call goes through p2.claude.call, so replies are saved in .cache/claude/ (--fresh asks
  again), a run of more than 5 calls asks first (--yes), and the calls land in a trace.
- It writes answers/<corpus>/<label>.judged.json beside the answers file and the trace
  traces/judge-<label>.jsonl; commit both, because p2 check compares the verdicts with the trace.
A claim's id is its question id and its number in that answer: a01-c1 is the first claim of a01.

How far to trust the judge is the point of the stretch, so you check it: label at least 20 claims
yourself in answers/<corpus>/<label>.calibration.tsv, one line per claim, tab-separated:

    a01-c1	supported	the quote gives the 80 percent figure word for word
    a02-c1	partly	the quote says seat belts are provided, not that they are worn

No command writes that file; it holds your own verdicts. `p2 score` compares the two in the
stretch-judge table of EVAL.md: the share of claims where you agree, with a Wilson 95% interval, and
a 3 by 3 table of Claude's verdicts against yours.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from p2 import claude, paths, trace, verify
from p2.runfile import data_lines
from p2.trace import Tracer

VERDICTS = ("supported", "partly", "not")
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS), "description": "supported, partly or not"},
        "reason": {"type": "string", "description": "one sentence that says why"},
    },
    "required": ["verdict", "reason"],
}
USER_PROMPT = """Question: {question}

Claim: {claim}

Quote: {quote}"""
PROMPT_FILE = Path("prompts") / "judge.txt"
CALIBRATION_MIN = 20
CLAIM_FIELDS = ("id", "qid", "question", "claim", "chunk_id", "quote", "verdict", "reason")


class JudgedError(ValueError):
    pass


def claim_id(qid: str, n: int) -> str:
    return f"{qid}-c{n}"


def claims_of(data: dict) -> list[dict]:
    """Every claim of an answers file, in order, with its id, question, text, chunk id and quote."""
    out = []
    for record in data["answers"]:
        for n, claim in enumerate(record.get("claims") or [], 1):
            claim = claim if isinstance(claim, dict) else {}
            out.append({
                "id": claim_id(str(record["qid"]), n), "qid": str(record["qid"]), "question": str(record.get("question", "")),
                "claim": str(claim.get("text", "")), "chunk_id": str(claim.get("chunk_id", "")), "quote": str(claim.get("quote", "")),
            })  # fmt: skip
    return out


def load_system_prompt(root: Path) -> str:
    """prompts/judge.txt without the lines that start with #."""
    path = root / PROMPT_FILE
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#")).strip()


def build_prompt(claim: dict) -> str:
    return USER_PROMPT.format(question=claim["question"], claim=claim["claim"], quote=claim["quote"])


def judge_one(claim: dict, quote_found: bool, system_prompt: str, cfg, tracer: Tracer) -> tuple[dict, bool]:
    """Ask Claude about one claim; the record for the judged file, and whether the reply was a saved one."""
    with tracer.span("judge", "chat", "judge", claim["id"], model=cfg.model) as span:
        reply = claude.call(build_prompt(claim), SCHEMA, system=system_prompt, model=cfg.model, cache_dir=claude.cache_folder(cfg))
        record = {
            **claim, "quote_found": quote_found, "verdict": None, "reason": "",
            "seconds": round(reply.seconds, 1), "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens, "model": reply.model,
        }  # fmt: skip
        verdict = (reply.output or {}).get("verdict") if reply.error is None else None
        if reply.error:
            record["error"] = reply.error
        elif verdict not in VERDICTS:
            record["error"] = f"the reply's verdict was {verdict!r}, not supported, partly or not"
        else:
            record["verdict"] = verdict
            record["reason"] = str((reply.output or {}).get("reason", ""))
        span["p2.verdict"] = record["verdict"]
        if record.get("error"):
            span["p2.error"] = record["error"]
        return record, reply.cached


def resolve(root: Path, given: str) -> Path:
    path = Path(given)
    if not path.is_absolute() and not path.exists():
        path = root / path
    return path


def run(args, cfg) -> int:
    root = cfg.root
    path = resolve(root, args.file)
    if path.name.endswith(paths.JUDGED_SUFFIX):
        print(f"{paths.rel(root, path)} is a judged file; give p2 judge the answers file it came from, {paths.rel(root, path.with_name(paths.answers_stem(path) + '.json'))}.")
        return 2
    if not path.is_file():
        print(f"{paths.rel(root, path)} is not a file; give p2 judge an answers file, such as answers/shared/rerank.json.")
        return 2
    try:
        data = verify.load_answers(path)
    except verify.AnswersError as error:
        print(f"{paths.rel(root, path)} {error}.")
        return 1
    claims = claims_of(data)
    if args.only:
        wanted = [q.strip() for q in args.only.split(",") if q.strip()]
        unknown = [q for q in wanted if q not in {r["qid"] for r in data["answers"]}]
        if unknown:
            print(f"{', '.join(unknown)} is not a question id in {paths.rel(root, path)}.")
            return 2
        claims = [c for c in claims if c["qid"] in wanted]
    if not claims:
        print(f"{paths.rel(root, path)} has no claims to judge" + (" among the questions you named." if args.only else "; every question was declined or its call failed."))
        return 1
    system_prompt = load_system_prompt(root)
    if not system_prompt:
        print(f"{PROMPT_FILE.as_posix()} is missing or empty once the # lines are removed; write your instructions there (the template's file is a starting point).")
        return 1
    if args.fresh:
        cfg = cfg.replace(cache_claude=False)
    cache = claude.cache_folder(cfg)
    saved = 0
    if cache is not None:
        saved = sum((cache / f"{claude.cache_key(build_prompt(c), SCHEMA, cfg.model, system_prompt)}.json").is_file() for c in claims)
    calls = len(claims) - saved
    if not claude.confirm_calls(calls, "This p2 judge", args.yes):
        return 2
    note = f", and {saved} saved repl{'y' if saved == 1 else 'ies'} that cost nothing" if saved else ""
    print(f"{len(claims)} claim(s) from {paths.rel(root, path)}: {calls} new claude -p call(s) on {cfg.model}{note}.")
    chunks = verify.chunk_texts(data, root, cfg)
    stem = paths.answers_stem(path)
    records = []
    started = time.perf_counter()
    with Tracer(paths.judge_trace(root, stem)) as tracer:
        for n, claim in enumerate(claims, 1):
            found = claim["chunk_id"] in chunks and bool(claim["quote"]) and verify.norm(claim["quote"]) in verify.norm(chunks[claim["chunk_id"]])
            record, was_saved = judge_one(claim, found, system_prompt, cfg, tracer)
            records.append(record)
            cost = "saved reply, no call" if was_saved else f"{record['seconds']:.1f} s"
            state = f", ERROR {record['error']}" if record.get("error") else ""
            # The verdicts are not printed, so you can label the claims yourself before you see them.
            print(f"[{n}/{len(claims)}] {claim['id']} {cost}, {record['input_tokens']:,} input tokens{state}", flush=True)
    out = paths.judged_file(path)
    judged = {
        "label": stem, "answers_file": paths.rel(root, path), "corpus": data.get("corpus", path.parent.name),
        "model": cfg.model, "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16],
        "only": [c.strip() for c in args.only.split(",") if c.strip()] if args.only else None, "claims": records,
    }  # fmt: skip
    paths.write_text(out, json.dumps(judged, indent=2, ensure_ascii=False) + "\n")
    counts = {v: sum(1 for r in records if r["verdict"] == v) for v in VERDICTS}
    failed = sum(1 for r in records if r["verdict"] is None)
    print(f"Wrote {paths.rel(root, out)} and {paths.rel(root, paths.judge_trace(root, stem))} in {time.perf_counter() - started:.0f} s.")
    print(f"Claude judged {len(records) - failed} of {len(records)} claim(s): {counts['supported']} supported, {counts['partly']} partly, {counts['not']} not.")
    calibration = paths.calibration_file(path)
    print(
        f"Before you open the judged file, label at least {CALIBRATION_MIN} claims yourself in {paths.rel(root, calibration)}, "
        "one line each: claim id, supported, partly or not, and a note, separated by tabs. No command writes that file, so Claude's verdicts cannot steer yours."
    )
    total = len(claims_of(data))
    if total < CALIBRATION_MIN:
        print(f"This answers file has {total} claims in all, fewer than {CALIBRATION_MIN}, so judge a second answers file as well and label claims from both.")
    if args.only:
        print("This judged file covers only the questions you named; run p2 judge again without --only to judge them all.")
    return 0


# ---- reading the files back (p2 check and p2 score) ----


def load_judged(path: Path) -> dict:
    """A judged file with its shape checked; raises JudgedError with a one-sentence reason."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JudgedError(f"cannot be read as JSON ({type(error).__name__})") from None
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise JudgedError('is not a judged file (it needs a "claims" list)')
    for i, record in enumerate(data["claims"], 1):
        if not isinstance(record, dict) or any(f not in record for f in CLAIM_FIELDS):
            raise JudgedError(f"claim {i} needs the fields {', '.join(CLAIM_FIELDS)}")
        if record["verdict"] is not None and record["verdict"] not in VERDICTS:
            raise JudgedError(f"claim {record['id']} has the verdict {record['verdict']!r}, not supported, partly or not")
        if record["verdict"] is None and not record.get("error"):
            raise JudgedError(f"claim {record['id']} has no verdict and no error")
    return data


def read_calibration(path: Path, known: set[str] | None = None) -> tuple[dict[str, str], list[str]]:
    """({claim id: your verdict}, problems) from a calibration file. A first line that reads
    claim, verdict, note (a header) and lines that start with # are skipped."""
    labels: dict[str, str] = {}
    problems: list[str] = []
    for n, line in data_lines(Path(path)):
        parts = [p.strip() for p in line.split("\t")]
        if not labels and not problems and [p.lower() for p in parts[:2]] == ["claim", "verdict"]:
            continue
        if len(parts) < 2 or not parts[0]:
            problems.append(f"line {n} needs at least two tab-separated fields: the claim id and your verdict (then a note)")
            continue
        cid, verdict = parts[0], parts[1].lower()
        if verdict not in VERDICTS:
            problems.append(f"line {n} ({cid}) has the verdict {parts[1]!r}, not supported, partly or not")
            continue
        if cid in labels:
            problems.append(f"line {n} labels {cid} a second time")
            continue
        if known is not None and cid not in known:
            problems.append(f"line {n} names {cid}, which is not a claim of its answers file (ids look like a01-c1)")
            continue
        labels[cid] = verdict
    return labels, problems


def integrity(root: Path, judged_path: Path) -> list[str]:
    """Why a committed judged file cannot be trusted, as short sentences (empty when it is fine):
    its answers file is gone or changed since it was judged, or its trace does not back its verdicts."""
    try:
        data = load_judged(judged_path)
    except JudgedError as error:
        return [f"it {error}"]
    answers = judged_path.with_name(paths.answers_stem(judged_path) + ".json")
    if not answers.is_file():
        return [f"its answers file {paths.rel(root, answers)} is gone"]
    try:
        current = {c["id"]: c for c in claims_of(verify.load_answers(answers))}
    except verify.AnswersError as error:
        return [f"its answers file {paths.rel(root, answers)} {error}"]
    problems = []
    for record in data["claims"]:
        now = current.get(record["id"])
        if now is None or any(now[f] != record[f] for f in ("qid", "claim", "chunk_id", "quote")):
            problems.append(f"{record['id']} is not the same claim in {paths.rel(root, answers)} any more, so the answers changed after they were judged")
            break
    if not data.get("only") and len(data["claims"]) != len(current):
        problems.append(f"it judges {len(data['claims'])} claims and {paths.rel(root, answers)} has {len(current)}")
    trace_path = paths.judge_trace(root, paths.answers_stem(judged_path))
    if not trace_path.is_file():
        return problems + [f"{paths.rel(root, trace_path)} is missing; `p2 judge` writes it with the judged file as the record of the Claude calls"]
    try:
        spans = {str(s.get("p2.qid")): s for s in trace.read(trace_path)}
    except (OSError, ValueError):
        return problems + [f"{paths.rel(root, trace_path)} cannot be read"]
    for record in data["claims"]:
        span = spans.get(record["id"])
        if span is None:
            problems.append(f"{paths.rel(root, trace_path)} has no span for {record['id']}")
            break
        if int(span.get("p2.claude_calls") or 0) < 1:
            problems.append(f"{paths.rel(root, trace_path)} records no Claude call for {record['id']}")
            break
        if span.get("p2.verdict") != record["verdict"]:
            problems.append(f"{record['id']} has the verdict {record['verdict']!r} where its trace has {span.get('p2.verdict')!r}")
            break
    if data["claims"] and not problems and sum(int(spans[r["id"]].get("gen_ai.usage.input_tokens") or 0) for r in data["claims"]) <= 0:
        problems.append(f"{paths.rel(root, trace_path)} records no input tokens at all")
    return problems
