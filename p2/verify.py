"""The quote check on an answers file: do the citations point at chunks the model saw, and are the quotes really in them?

For every answer it checks, with no model involved (the 13 lab's verify.py, on chunks):
  1. every claim's chunk_id is one of the chunk ids retrieved for that question;
  2. every claim's quote appears in the chunk it cites, after lower-casing and collapsing runs of whitespace;
  3. an out-of-corpus question is answered not_found with no claims, and an in-corpus question is not answered not_found.
Totals for a file:
  verified        in-corpus questions with at least one claim whose claims all pass checks 1 and 2
  not_found in    in-corpus questions answered not_found (a failure: the answer was in the corpus)
  declined out    out-of-corpus questions answered not_found with no claims (what you want)

What a pass proves: the quote is a real piece of the chunk it cites.
What it cannot prove: that the quote supports the claim. Only reading it tells you.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from p2 import corpus as corpus_mod
from p2 import paths

RECORD_FIELDS = ("qid", "question", "kind", "system", "retrieved", "answer", "not_found", "claims", "seconds", "input_tokens", "output_tokens", "model")


class AnswersError(ValueError):
    pass


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


@dataclass
class Totals:
    label: str
    n_in: int = 0
    n_out: int = 0
    verified: int = 0
    not_found_in: int = 0
    declined_out: int = 0
    claims: int = 0
    claims_verified: int = 0
    lines: list[str] = field(default_factory=list)
    passed: dict[str, bool] = field(default_factory=dict)  # qid: True when its line is a PASS

    def as_dict(self) -> dict:
        return {
            "n_in": self.n_in, "n_out": self.n_out,
            "verified": self.verified, "not_found_in": self.not_found_in, "declined_out": self.declined_out,
            "claims": self.claims, "claims_verified": self.claims_verified,
        }  # fmt: skip

    def summary(self) -> str:
        def pct(a, b):
            return f"{a} of {b} ({100 * a / b:.0f}%)" if b else "no questions"

        return f"Answers {self.label} | verified {pct(self.verified, self.n_in)} | not_found in {pct(self.not_found_in, self.n_in)} | declined out {pct(self.declined_out, self.n_out)}"


def load_answers(path: Path) -> dict:
    """An answers file, with its shape checked; raises AnswersError with a one-sentence reason."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnswersError(f"cannot be read as JSON ({type(error).__name__})") from None
    if not isinstance(data, dict) or not isinstance(data.get("answers"), list):
        raise AnswersError('is not an answers file (it needs an "answers" list)')
    for i, record in enumerate(data["answers"], 1):
        if not isinstance(record, dict):
            raise AnswersError(f"record {i} is not an object")
        missing = [f for f in RECORD_FIELDS if f not in record]
        if missing:
            raise AnswersError(f"record {i} ({record.get('qid', '?')}) has no {', '.join(missing)}")
        if record["kind"] not in ("in", "out"):
            raise AnswersError(f"record {i} ({record['qid']}) has kind {record['kind']!r}, not in or out")
        if not isinstance(record["retrieved"], list) or not isinstance(record["claims"], list):
            raise AnswersError(f"record {i} ({record['qid']}) needs retrieved and claims to be lists")
    return data


def chunk_texts(data: dict, root: Path, cfg) -> dict[str, str]:
    """The chunks the answers were made from: the file's corpus, cut with the file's chunking settings."""
    chunking = data.get("chunking") or {}
    words = int(chunking.get("words", cfg.chunk_words))
    overlap = int(chunking.get("overlap", cfg.chunk_overlap))
    corpus = corpus_mod.load(root, data.get("corpus", "shared"))
    return corpus.chunk_texts(words, overlap)


def judge(record: dict, chunks: dict[str, str]) -> list[str]:
    """Problems with one record's claims (checks 1 and 2)."""
    problems = []
    retrieved = set(record.get("retrieved") or [])
    for claim in record.get("claims") or []:
        cid = claim.get("chunk_id") if isinstance(claim, dict) else None
        quote = claim.get("quote") if isinstance(claim, dict) else None
        if cid not in retrieved:
            problems.append(f"cites {cid} which was not retrieved")
        elif cid not in chunks:
            problems.append(f"cites {cid} which is not a chunk of the corpus")
        elif not quote or norm(quote) not in norm(chunks[cid]):
            problems.append(f'quote not found in {cid}: "{str(quote)[:60]}"')
    return problems


def evaluate(data: dict, chunks: dict[str, str], label: str = "?") -> Totals:
    """Per-question lines and the totals for one answers file."""
    t = Totals(label=label)
    for r in data["answers"]:
        claims = r.get("claims") or []
        problems = judge(r, chunks)
        t.claims += len(claims)
        t.claims_verified += len(claims) - len(problems)
        kind = r.get("kind")
        if kind == "out":
            t.n_out += 1
            if r.get("not_found") and not claims:
                t.declined_out += 1
            else:
                problems.append("an out-of-corpus question should be not_found with no claims")
        else:
            t.n_in += 1
            if r.get("not_found"):
                t.not_found_in += 1
                problems.append("answered not_found, but the corpus has the answer")
            elif not claims:
                problems.append("no claims, so nothing to verify")
            elif not judge(r, chunks):
                t.verified += 1
        if r.get("error"):
            problems.append(f"the call failed: {r['error']}")
        status = "PASS" if not problems else "FAIL"
        t.passed[str(r.get("qid", "?"))] = not problems
        what = "not_found" if r.get("not_found") else f"{len(claims)} claim" + ("" if len(claims) == 1 else "s")
        t.lines.append(f"{r.get('qid', '?'):<5} {kind:<4} {status} {what}" + (": " + "; ".join(problems) if problems else ""))
    return t


def mechanics(data: dict, chunks: dict[str, str]) -> list[str]:
    """Integrity problems: every retrieved id is a chunk of the corpus, and every citation is a string."""
    problems = []
    for r in data["answers"]:
        unknown = [c for c in r["retrieved"] if c not in chunks]
        if unknown:
            problems.append(f"{r['qid']} retrieved {unknown[0]}, which is not a chunk of the corpus with the file's chunking")
        for claim in r["claims"]:
            if not isinstance(claim, dict) or not all(isinstance(claim.get(f), str) for f in ("text", "chunk_id", "quote")):
                problems.append(f"{r['qid']} has a claim without text, chunk_id and quote")
                break
    return problems


def latest_answers(root: Path) -> Path | None:
    found = sorted(paths.answers_files(root), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def run(args, cfg) -> int:
    """`p2 verify [FILE]`: print the quote check for one answers file (default: the newest)."""
    root = cfg.root
    path = Path(args.file) if args.file else latest_answers(root)
    if path is None:
        print("There are no answers files yet; run `uv run p2 answer --corpus shared --system NAME` first.")
        return 1
    if not path.is_absolute() and not path.exists():
        path = root / path
    try:
        data = load_answers(path)
    except AnswersError as error:
        print(f"{paths.rel(root, path)} {error}.")
        return 1
    totals = evaluate(data, chunk_texts(data, root, cfg), label=data.get("label", path.stem))
    print(f"Checking {paths.rel(root, path)}")
    print("\n".join(totals.lines))
    print(totals.summary())
    print(f"Claims: {totals.claims_verified} of {totals.claims} quotes found in the chunk they cite.")
    print("A PASS means each quote is really in the chunk it cites; it does not mean the quote supports the claim.")
    return 0
