"""Where every file of the repository lives, in one place (section 1 of the layout)."""

from __future__ import annotations

import re
from pathlib import Path

CORPORA = ("shared", "own")
QUERY_SETS = {"shared": ("practice", "test"), "own": ("own",)}
# A label names a run or answers file; dots are not allowed because they separate the query set
# and the repeat number in a file name (rerank.practice.r2.trec).
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REPEAT_RE = re.compile(r"^r([1-9][0-9]*)$")


def corpus_dir(root: Path, corpus: str) -> Path:
    return root / "corpora" / corpus


def docs_dir(root: Path, corpus: str) -> Path:
    return corpus_dir(root, corpus) / "docs"


def manifest(root: Path, corpus: str) -> Path:
    return corpus_dir(root, corpus) / "manifest.tsv"


def queries(root: Path, corpus: str, query_set: str) -> Path:
    if corpus == "own":
        return root / "eval" / "own" / "queries.tsv"
    return root / "eval" / "shared" / f"{query_set}.queries.tsv"


def qrels(root: Path, corpus: str, query_set: str) -> Path:
    if corpus == "own":
        return root / "eval" / "own" / "qrels.txt"
    return root / "eval" / "shared" / f"{query_set}.qrels.txt"


def questions(root: Path, corpus: str = "shared") -> Path:
    return root / "eval" / corpus / "questions.tsv"


def run_file(root: Path, corpus: str, query_set: str, name: str, repeat: int | None = None, ablation: bool = False, stretch: bool = False) -> Path:
    """runs/shared/<name>.<set>[.rN].trec, runs/own/<name>[.rN].trec, runs/own/ablation/<name>.trec, or
    runs/own/stretch/<name>.trec (a stretch run on some of your own queries, such as the grep agent's)."""
    suffix = f".r{repeat}" if repeat else ""
    if corpus == "own":
        folder = root / "runs" / "own" / ("ablation" if ablation else "stretch" if stretch else "")
        return folder / f"{name}{suffix}.trec"
    return root / "runs" / "shared" / f"{name}.{query_set}{suffix}.trec"


def trace_file(
    root: Path, corpus: str, query_set: str, name: str, repeat: int | None = None, ablation: bool = False, *, calls_claude: bool, stretch: bool = False
) -> Path:
    """The trace of a run. A system that calls Claude writes it in traces/, where it is committed as the
    record of what the calls cost and what Claude returned; the others write it in traces/retrieval/,
    which git ignores because it changes on every run."""
    suffix = f".r{repeat}" if repeat else ""
    middle = f"ablation-{name}" if ablation else f"stretch-{name}" if stretch else name
    folder = root / "traces" if calls_claude else root / "traces" / "retrieval"
    if stretch:
        return folder / f"{corpus}-{middle}{suffix}.jsonl"
    return folder / f"{corpus}-{query_set}-{middle}{suffix}.jsonl"


def answers_file(root: Path, corpus: str, label: str, repeat: int | None = None) -> Path:
    suffix = f".r{repeat}" if repeat else ""
    return root / "answers" / corpus / f"{label}{suffix}.json"


def answers_trace(root: Path, label: str, repeat: int | None = None) -> Path:
    suffix = f".r{repeat}" if repeat else ""
    return root / "traces" / f"answers-{label}{suffix}.jsonl"


# Beside an answers file answers/<corpus>/<stem>.json (stem: a label, or label.rN for a repeat), stretch
# option 2 keeps <stem>.judged.json (written by `p2 judge`) and <stem>.calibration.tsv (written by you).
JUDGED_SUFFIX = ".judged.json"
CALIBRATION_SUFFIX = ".calibration.tsv"


def answers_files(root: Path) -> list[Path]:
    """Every answers file under answers/<corpus>/, without the judged files that sit beside them."""
    return sorted(p for p in (root / "answers").glob("*/*.json") if not p.name.endswith(JUDGED_SUFFIX))


def answers_stem(path: Path) -> str:
    """The label of an answers, judged or calibration file: rerank for rerank.json, rerank.judged.json
    and rerank.calibration.tsv; rerank.r1 for rerank.r1.json."""
    name = Path(path).name
    for suffix in (JUDGED_SUFFIX, CALIBRATION_SUFFIX, ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def judged_file(answers: Path) -> Path:
    return Path(answers).with_name(answers_stem(answers) + JUDGED_SUFFIX)


def calibration_file(answers: Path) -> Path:
    return Path(answers).with_name(answers_stem(answers) + CALIBRATION_SUFFIX)


def judge_trace(root: Path, stem: str) -> Path:
    return root / "traces" / f"judge-{stem}.jsonl"


def results_file(root: Path) -> Path:
    return root / "results" / "results.json"


def rel(root: Path, path: Path) -> str:
    """A path relative to the repository, with forward slashes on every system."""
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with \\n line endings, creating the folder first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
