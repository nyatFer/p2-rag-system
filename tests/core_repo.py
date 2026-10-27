"""A tiny P2 repository for the core tests: a shared corpus, practice and test queries, qrels and questions."""

from __future__ import annotations

from pathlib import Path

DOCS = {
    "cfr30-75.403": ("Maintenance of incombustible content of rock dust", "Where rock dust is required to be applied, it shall be distributed upon the top, floor, and sides of all underground areas of a coal mine and maintained in such quantities that the incombustible content of the combined coal dust, rock dust, and other dust shall be not less than 80 percent."),
    "cfr30-75.400": ("Accumulation of combustible materials", "Coal dust, including float coal dust deposited on rock-dusted surfaces, loose coal, and other combustible materials, shall be cleaned up and not be permitted to accumulate in active workings, or on diesel-powered and electric equipment therein."),
    "cfr30-56.14131": ("Seat belts for haulage trucks", "Seat belts shall be provided and worn in haulage trucks. Seat belts shall meet the requirements of SAE J386, Operator Restraint System for Off-Road Work Machines."),
    "cfr30-57.14131": ("Seat belts for haulage trucks", "Seat belts shall be provided and worn in haulage trucks in underground and surface areas of underground mines."),
    "cfr30-77.1710": ("Protective clothing; requirements", "Each employee working in a surface coal mine shall be required to wear protective clothing and devices including hard hats and safety-toe shoes."),
    "cfr30-56.14107": ("Moving machine parts", "Moving machine parts shall be guarded to protect persons from contacting gears, sprockets, chains, drive, head, tail, and takeup pulleys, flywheels, couplings, shafts, fan blades, and similar moving parts that can cause injury."),
}
PRACTICE_QUERIES = [
    ("p01", "identifier", "75.403"),
    ("p02", "paraphrase", "Do haul truck drivers have to buckle up?"),
    ("p03", "mixed", "rock dust incombustible content 80 percent"),
]
PRACTICE_QRELS = [("p01", "cfr30-75.403", 1), ("p02", "cfr30-56.14131", 1), ("p02", "cfr30-57.14131", 1), ("p03", "cfr30-75.403", 1)]
TEST_QUERIES = [("t01", "identifier", "SAE J386"), ("t02", "paraphrase", "guards on gears and pulleys")]
QUESTIONS = [
    ("a01", "in", "What share of the dust must be incombustible where rock dust is required?", "cfr30-75.403"),
    ("a02", "in", "Must seat belts be worn in haulage trucks?", "cfr30-56.14131,cfr30-57.14131"),
    ("a03", "out", "How often must cranes be inspected on a construction site?", "-"),
]
EVAL_MD = """# EVAL

Prose that the student writes. TODO: write it.

<!-- p2:begin shared-practice -->
<!-- p2:end shared-practice -->

<!-- p2:begin shared-practice-pairs-mrr -->
<!-- p2:end shared-practice-pairs-mrr -->

<!-- p2:begin answers -->
<!-- p2:end answers -->

<!-- p2:begin shared-practice-classes -->
<!-- p2:end shared-practice-classes -->

<!-- p2:begin own -->
<!-- p2:end own -->

<!-- p2:begin own-classes -->
<!-- p2:end own-classes -->

<!-- p2:begin own-origins -->
<!-- p2:end own-origins -->

<!-- p2:begin own-pairs -->
<!-- p2:end own-pairs -->

<!-- p2:begin own-ablation -->
<!-- p2:end own-ablation -->

<!-- p2:begin own-ablation-queries -->
<!-- p2:end own-ablation-queries -->

<!-- p2:begin stretch-judge -->
<!-- p2:end stretch-judge -->

<!-- p2:begin stretch-cost -->
<!-- p2:end stretch-cost -->

<!-- p2:begin stretch-agent -->
<!-- p2:end stretch-agent -->

<!-- p2:begin someone-elses-block -->
left alone
<!-- p2:end someone-elses-block -->
"""
P2_TOML = """section = "498E"

[run]
k = 10

[chunking]
words = 20
overlap = 5

[bm25]
tokenizer = "codes"

[answer]
top = 3

[claude]
model = "sonnet"
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def make_repo(root: Path, with_eval_md: bool = True) -> Path:
    write(root / "p2.toml", P2_TOML)
    manifest = ["docid\ttitle\tsource\tlicense\tnotes"]
    for docid, (title, body) in DOCS.items():
        write(root / "corpora" / "shared" / "docs" / f"{docid}.md", f"# \u00a7 {docid.split('-', 1)[1]} {title}\n\n{body}\n")
        manifest.append(f"{docid}\t{title}\thttps://www.ecfr.gov/current/title-30\tus-gov-public-domain\t")
    write(root / "corpora" / "shared" / "manifest.tsv", "\n".join(manifest) + "\n")
    write(root / "corpora" / "own" / "manifest.tsv", "docid\ttitle\tsource\tlicense\tnotes\n")
    (root / "corpora" / "own" / "docs").mkdir(parents=True, exist_ok=True)
    write(root / "eval" / "shared" / "practice.queries.tsv", "".join(f"{q}\t{c}\t{t}\n" for q, c, t in PRACTICE_QUERIES))
    write(root / "eval" / "shared" / "practice.qrels.txt", "".join(f"{q} 0 {d} {r}\n" for q, d, r in PRACTICE_QRELS))
    write(root / "eval" / "shared" / "test.queries.tsv", "".join(f"{q}\t{c}\t{t}\n" for q, c, t in TEST_QUERIES))
    write(root / "eval" / "shared" / "questions.tsv", "".join(f"{q}\t{k}\t{t}\t{g}\n" for q, k, t, g in QUESTIONS))
    write(root / "eval" / "own" / "queries.tsv", "# qid<TAB>class<TAB>text<TAB>origin\n")
    write(root / "eval" / "own" / "qrels.txt", "# qid 0 docid rel\n")
    write(root / "prompts" / "answer.txt", "Answer only from the chunks.\n# a note\n")
    if with_eval_md:
        write(root / "EVAL.md", EVAL_MD)
        write(root / "DECISIONS.md", "# DECISIONS\n\nTODO: your decisions.\n")
    return root
