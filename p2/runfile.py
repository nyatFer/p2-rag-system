"""Read, write and validate the TREC files: run files, qrels, queries and questions.

Run file, one line per query and document:   qid Q0 docid rank score tag
Qrels, one line per judgment:                  qid 0 docid rel
Queries (tab-separated):                       qid  class  text  [origin]
Questions (tab-separated):                     qid  in|out  question  gold-docids|-

Ordering rule, used by every writer and by the metrics: higher score first, and equal scores in
document id order (ascending). Scores are compared after rounding to 6 decimals, the precision a
run file keeps, so the order in memory and the order in the file always agree.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from p2.paths import write_text

DOCID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SCORE_RE = re.compile(r"^-?[0-9]+\.[0-9]{6}$")
ORIGINS = ("hand", "claude")
KINDS = ("in", "out")
MAX_SHOWN = 3


def order(pairs: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """(id, score) pairs best first; equal scores (at 6 decimals) in id order."""
    return sorted(((i, float(s)) for i, s in pairs), key=lambda p: (-round(p[1], 6), p[0]))


def data_lines(path: Path) -> list[tuple[int, str]]:
    """(line number, line) for every line that is not blank and does not start with #.

    A byte order mark at the start (Excel and Notepad add one when they save) is ignored."""
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip() and not line.startswith("#"):
            out.append((n, line))
    return out


def summarize(problems: list[str]) -> str:
    """The first few problems, and how many more there are."""
    shown = "; ".join(problems[:MAX_SHOWN])
    more = len(problems) - MAX_SHOWN
    return shown + (f"; and {more} more" if more > 0 else "")


# ---- run files ----


def format_run(results: dict[str, list[tuple[str, float]]], tag: str, k: int) -> str:
    """The text of a run file: queries in the given order, each query's documents ordered and cut at k."""
    lines = []
    for qid, pairs in results.items():
        for rank, (docid, score) in enumerate(order(pairs)[:k], 1):
            lines.append(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_run(path: Path, results: dict[str, list[tuple[str, float]]], tag: str, k: int) -> None:
    write_text(path, format_run(results, tag, k))


def read_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    """{qid: [(docid, score), ...]} in file order. Raises ValueError on a line that cannot be parsed;
    use validate_run for a full report."""
    run: dict[str, list[tuple[str, float]]] = {}
    for n, line in data_lines(path):
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{path.name} line {n} has {len(parts)} fields, not 6")
        try:
            score = float(parts[4])
        except ValueError:
            raise ValueError(f"{path.name} line {n} has a score that is not a number: {parts[4]}") from None
        run.setdefault(parts[0], []).append((parts[2], score))
    return run


def run_tag(path: Path) -> str | None:
    """The tag of the first line of a run file, or None if it has no lines."""
    for _n, line in data_lines(path):
        parts = line.split()
        return parts[5] if len(parts) == 6 else None
    return None


def validate_run(path: Path, k: int | None = None, docids: set[str] | None = None, qids: Iterable[str] | None = None) -> list[str]:
    """Every way the file breaks the run-file contract, as short sentences (empty if it is valid).

    k: the most lines a query may have. docids: the corpus, to catch unknown documents.
    qids: the queries the run should cover, in order.
    """
    problems: list[str] = []
    try:
        lines = data_lines(path)
    except (OSError, UnicodeDecodeError) as error:
        return [f"cannot be read as UTF-8 text ({type(error).__name__})"]
    if not lines:
        return ["has no lines"]
    seen_q: list[str] = []
    per_query: dict[str, list[tuple[str, float, str]]] = {}
    tags = set()
    for n, line in lines:
        parts = line.split()
        if len(parts) != 6:
            problems.append(f"line {n} has {len(parts)} fields, not 6")
            continue
        qid, q0, docid, rank, score, tag = parts
        tags.add(tag)
        if q0 != "Q0":
            problems.append(f"line {n} has {q0} in the second column, not Q0")
        if not DOCID_RE.match(docid):
            problems.append(f"line {n} has a document id that is not allowed: {docid}")
        if not SCORE_RE.match(score):
            problems.append(f"line {n} has score {score}, not a number with 6 decimals")
            continue
        if seen_q and seen_q[-1] != qid and qid in per_query:
            problems.append(f"the lines of {qid} are not all together (line {n})")
        if not seen_q or seen_q[-1] != qid:
            seen_q.append(qid)
        rows = per_query.setdefault(qid, [])
        if not rank.isdigit() or int(rank) != len(rows) + 1:
            problems.append(f"line {n} has rank {rank}, expected {len(rows) + 1}")
        rows.append((docid, float(score), score))
    if len(tags) > 1:
        problems.append(f"has more than one tag ({', '.join(sorted(tags))})")
    for qid, rows in per_query.items():
        ids = [d for d, _, _ in rows]
        dupes = sorted({d for d in ids if ids.count(d) > 1})
        if dupes:
            problems.append(f"{qid} lists {', '.join(dupes)} more than once")
        if k is not None and len(rows) > k:
            problems.append(f"{qid} has {len(rows)} lines, more than k = {k}")
        for (d1, s1, _), (d2, s2, _) in zip(rows, rows[1:]):
            if s2 > s1:
                problems.append(f"{qid} has a score that goes up ({d1} {s1:.6f}, then {d2} {s2:.6f})")
                break
            if s2 == s1 and d2 < d1:
                problems.append(f"{qid} has equal scores not in document id order ({d1}, then {d2})")
                break
        if docids is not None:
            unknown = sorted({d for d in ids if d not in docids})
            if unknown:
                problems.append(f"{qid} names {len(unknown)} document(s) not in the corpus, such as {unknown[0]}")
    if qids is not None:
        wanted = list(qids)
        extra = sorted(set(per_query) - set(wanted))
        missing = [q for q in wanted if q not in per_query]
        if extra:
            problems.append(f"has queries that are not in the queries file, such as {extra[0]}")
        if missing:
            problems.append(f"covers {len(wanted) - len(missing)} of {len(wanted)} queries (missing {missing[0]})")
    return problems


# ---- qrels ----


def read_qrels(path: Path) -> tuple[dict[str, dict[str, int]], list[str]]:
    """({qid: {docid: rel}}, problems). Lines with problems are skipped."""
    qrels: dict[str, dict[str, int]] = {}
    problems: list[str] = []
    for n, line in data_lines(path):
        parts = line.split()
        if len(parts) != 4:
            problems.append(f"line {n} has {len(parts)} fields, not 4 (qid 0 docid rel)")
            continue
        qid, _iteration, docid, rel = parts
        if not re.fullmatch(r"-?[0-9]+", rel):
            problems.append(f"line {n} has relevance {rel}, not a whole number")
            continue
        if not DOCID_RE.match(docid):
            problems.append(f"line {n} has a document id that is not allowed: {docid}")
            continue
        if docid in qrels.get(qid, {}):
            problems.append(f"line {n} judges {qid} {docid} a second time")
            continue
        qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels, problems


def relevant(qrels: dict[str, dict[str, int]]) -> dict[str, set[str]]:
    """{qid: set of docids with rel > 0}, only for queries that have at least one."""
    out = {}
    for qid, judged in qrels.items():
        docs = {d for d, r in judged.items() if r > 0}
        if docs:
            out[qid] = docs
    return out


# ---- queries and questions ----


@dataclass(frozen=True)
class Query:
    qid: str
    cls: str
    text: str
    origin: str = ""


@dataclass(frozen=True)
class Question:
    qid: str
    kind: str
    text: str
    gold: tuple[str, ...] = ()


def read_queries(path: Path, require_origin: bool = False) -> tuple[list[Query], list[str]]:
    """(queries, problems) from a tab-separated queries file. Lines with problems are skipped."""
    queries: list[Query] = []
    problems: list[str] = []
    seen = set()
    for n, line in data_lines(path):
        parts = line.split("\t")
        if len(parts) < 3:
            problems.append(f"line {n} has {len(parts)} tab-separated fields, not at least 3 (qid, class, text)")
            continue
        qid, cls, text = parts[0].strip(), parts[1].strip(), parts[2].strip()
        origin = parts[3].strip() if len(parts) > 3 else ""
        if not qid or any(c.isspace() for c in qid):
            problems.append(f"line {n} has a query id with spaces or nothing in it")
            continue
        if qid in seen:
            problems.append(f"line {n} uses query id {qid} a second time")
            continue
        if not cls or " " in cls:
            problems.append(f"line {n} has a class that is empty or has spaces: {cls!r}")
        if not text:
            problems.append(f"line {n} ({qid}) has no query text")
        if require_origin and origin not in ORIGINS:
            problems.append(f"line {n} ({qid}) needs a fourth column origin, hand or claude")
        elif origin and origin not in ORIGINS:
            problems.append(f"line {n} ({qid}) has origin {origin!r}, not hand or claude")
        seen.add(qid)
        queries.append(Query(qid, cls, text, origin))
    return queries, problems


def read_questions(path: Path) -> tuple[list[Question], list[str]]:
    """(questions, problems) from questions.tsv: qid, in or out, the question, gold docids (comma-separated, or -)."""
    questions: list[Question] = []
    problems: list[str] = []
    seen = set()
    for n, line in data_lines(path):
        parts = line.split("\t")
        if len(parts) < 3:
            problems.append(f"line {n} has {len(parts)} tab-separated fields, not at least 3 (qid, kind, question)")
            continue
        qid, kind, text = parts[0].strip(), parts[1].strip(), parts[2].strip()
        gold_field = parts[3].strip() if len(parts) > 3 else "-"
        gold = tuple(g.strip() for g in gold_field.split(",") if g.strip() and g.strip() != "-")
        if qid in seen:
            problems.append(f"line {n} uses question id {qid} a second time")
            continue
        if kind not in KINDS:
            problems.append(f"line {n} ({qid}) has kind {kind!r}, not in or out")
        if kind == "in" and not gold:
            problems.append(f"line {n} ({qid}) is an in question with no gold document")
        seen.add(qid)
        questions.append(Question(qid, kind, text, gold))
    return questions, problems
