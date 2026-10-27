"""The `p2` command. `uv run p2 --help` lists the commands; `uv run p2 COMMAND --help` lists a command's options.

    p2 run      write a run file (and a trace) for one system, or for every system with --all
    p2 score    score the runs and answers, write results/results.json, rewrite the EVAL.md tables
    p2 answer   answer the shared questions with Claude, citing chunks
    p2 verify   print the quote check for one answers file
    p2 judge    ask Claude whether each claim's quote supports the claim (stretch option 2)
    p2 ingest   turn a folder of PDF, HTML, Markdown or text files into your own corpus
    p2 license  check the licenses of your own corpus and write LICENSES.md
    p2 check    is everything well-formed and reproducible, and what is left to do
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

from p2 import config, metrics, paths, retrievers
from p2 import corpus as corpus_mod
from p2.claude import CONFIRM_ABOVE, confirm_calls
from p2.limits import OWN_MIN_DOCS, SUGGESTED_PART_PAGES
from p2.runfile import read_qrels, read_queries, run_tag, write_run
from p2.trace import Tracer


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p2", description="P2: retrieval systems over two corpora, measured, plus cited answers.")
    p.add_argument("--root", type=Path, help="the P2 repository (default: the nearest folder at or above this one with p2.toml)")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    run = sub.add_parser("run", help="write a run file for one system (or every system with --all)")
    run.add_argument("--corpus", choices=paths.CORPORA, help="shared or own")
    run.add_argument("--queries", help="practice or test (shared corpus), own (own corpus), or the path of a queries file")
    run.add_argument("--system", help="a system in p2/retrievers/ (bm25, dense, hybrid, rerank, or one you added)")
    run.add_argument("--k", type=int, help="documents per query (default: k in p2.toml; p2 check expects that value)")
    run.add_argument("--label", help="name the run file after this label instead of the system")
    run.add_argument("--ablation", action="store_true", help="own corpus only: write the run to runs/own/ablation/")
    run.add_argument("--stretch", action="store_true", help="own corpus only: a stretch run on a queries file of lines copied from eval/own/queries.tsv, written to runs/own/stretch/ (the grep agent of stretch option 5)")
    run.add_argument("--repeat", type=int, default=0, metavar="N", help="598E: run N times with fresh Claude replies, into .r1 to .rN files")
    run.add_argument("--fresh", action="store_true", help="ask Claude again instead of using saved replies")
    run.add_argument("--extra-corpus", type=Path, metavar="DIR", help="add the documents in DIR for this run only (never written to the repo)")
    run.add_argument("--out", type=Path, help="write the run file here instead of runs/ (default for --extra-corpus or a queries file of your own: .cache/extra/)")
    run.add_argument("--all", action="store_true", help="bm25, dense, hybrid and rerank on every query set you have, and every other run file you have again")
    run.add_argument("--with-claude", action="store_true", help="with --all: also run the systems that call Claude")
    run.add_argument("--yes", action="store_true", help=f"go ahead without asking when a command makes more than {CONFIRM_ABOVE} claude -p calls")

    sc = sub.add_parser("score", help="score runs and answers, write results/results.json, rewrite the EVAL.md tables")
    sc.add_argument("--test-qrels", type=Path, help=argparse.SUPPRESS)  # the instructor's hidden test judgments
    sc.add_argument("--out", type=Path, help=argparse.SUPPRESS)

    an = sub.add_parser("answer", help="answer the shared questions with Claude, citing chunks")
    an.add_argument("--corpus", choices=paths.CORPORA, default="shared", help="the corpus whose eval/<corpus>/questions.tsv to answer (default shared)")
    an.add_argument("--system", required=True, help="the system that retrieves the chunks (it needs search_chunks)")
    an.add_argument("--label", help="answers go to answers/<corpus>/<label>.json (default: the system name)")
    an.add_argument("--only", help="only these question ids, for example a01,a09")
    an.add_argument("--repeat", type=int, default=0, metavar="N", help="598E: answer N times with fresh Claude replies, into .r1 to .rN files")
    an.add_argument("--fresh", action="store_true", help="ask Claude again instead of using saved replies")
    an.add_argument("--yes", action="store_true", help=f"go ahead without asking when this makes more than {CONFIRM_ABOVE} claude -p calls")

    ve = sub.add_parser("verify", help="print the quote check for one answers file")
    ve.add_argument("file", nargs="?", help="an answers file (default: the newest one)")

    ju = sub.add_parser("judge", help="ask Claude whether each claim's quote supports the claim, one call per claim (stretch option 2)")
    ju.add_argument("file", metavar="FILE", help="an answers file, such as answers/shared/rerank.json")
    ju.add_argument("--only", help="only the claims of these question ids, for example a01,a02")
    ju.add_argument("--fresh", action="store_true", help="ask Claude again instead of using saved replies")
    ju.add_argument("--yes", action="store_true", help=f"go ahead without asking when this makes more than {CONFIRM_ABOVE} claude -p calls")

    ing = sub.add_parser("ingest", help="turn a folder of documents into your own corpus (needs `uv sync --group ingest`)")
    ing.add_argument("src_dir", metavar="SRC_DIR", nargs="?", help="the folder of PDF, HTML, Markdown or text files")
    ing.add_argument("--into", default="own", choices=["own"], help="the corpus to add to (own)")
    ing.add_argument("--license", help="the license of these documents, from the vocabulary in the README")
    ing.add_argument("--source", help="where they came from (a URL or a citation); {name} and {stem} stand for each file's name, as in https://example.org/reports/{name}")
    ing.add_argument("--part-pages", type=int, metavar="N", help=f"split every PDF longer than N pages into parts of N pages, each its own document (try {SUGGESTED_PART_PAGES} when long documents leave you under {OWN_MIN_DOCS}); web pages and text files are not affected")
    ing.add_argument("--report", action="store_true", help="rewrite corpora/own/INGEST.md for the documents there now (after you removed some), without reading a folder")

    lic = sub.add_parser("license", help="check the licenses of your own corpus and write LICENSES.md")
    lic.add_argument("--online", action="store_true", help="also ask Crossref, arXiv and PubMed Central (needs the internet)")

    ch = sub.add_parser("check", help="is everything well-formed and reproducible, and what is left to do")
    ch.add_argument("--final", action="store_true", help="the submission bar: also fail on anything still to do")
    return p


# ---- p2 run ----


def resolve_queries(root: Path, corpus: str, queries: str | None) -> tuple[str, Path]:
    """(query set name, queries file) for --corpus and --queries."""
    if queries is None:
        queries = "own" if corpus == "own" else "practice"
    if queries in paths.QUERY_SETS[corpus]:
        return queries, paths.queries(root, corpus, queries)
    if queries in ("practice", "test", "own"):
        raise SystemExit(f"The {corpus} corpus goes with --queries {' or '.join(paths.QUERY_SETS[corpus])}, not {queries}.")
    path = Path(queries)
    if not path.is_file():
        raise SystemExit(f"--queries {queries} is neither a query set name nor a file.")
    return path.stem.split(".")[0], path


def checked(name: str, hits, corpus: corpus_mod.Corpus, k: int) -> list[tuple[str, float]]:
    """The hits of one search, after checking they keep the interface's promises."""
    hits = [(str(d), float(s)) for d, s in hits]
    ids = [d for d, _ in hits]
    if len(hits) > k:
        raise SystemExit(f"{name}.search returned {len(hits)} documents for k = {k}; return at most k.")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{name}.search returned a document twice; return each document once (corpus.doc_level keeps a document's best chunk).")
    unknown = [d for d in ids if d not in corpus.docs]
    if unknown:
        raise SystemExit(f"{name}.search returned {unknown[0]}, which is not a document id of the corpus (a chunk id needs corpus.docid_of).")
    return hits


def where(corpus: str, query_set: str) -> str:
    """How a corpus and query set read in a sentence: "shared practice", "the own corpus"."""
    return "the own corpus" if corpus == "own" else f"shared {query_set}"


def run_one(cfg: config.Config, corpus: corpus_mod.Corpus, name: str, query_set: str, queries, out: Path, trace_path: Path, k: int, label: str) -> float | None:
    """Run one system over a query set, write the run file and the trace; returns MRR@10 when qrels exist."""
    started = time.perf_counter()  # building the index (encoding the chunks, say) is part of the time
    system = retrievers.build(name, corpus, cfg)
    model = getattr(system, "model", None)
    calls_claude = retrievers.needs_claude(name)
    results = {}
    input_tokens = 0
    with Tracer(trace_path) as tracer:
        for n, q in enumerate(queries, 1):
            if calls_claude:
                # before the call, so a warning your reranker prints sits under the query it belongs to
                print(f"  [{n}/{len(queries)}] {q.qid}", flush=True)
            with tracer.span(f"retrieval {name}", "retrieval", name, q.qid, model=model) as span:
                hits = checked(name, system.search(q.text, k), corpus, k)
                span["p2.top_ids"] = [d for d, _ in hits]
            input_tokens += int(span["gen_ai.usage.input_tokens"] or 0)
            results[q.qid] = hits
    write_run(out, results, tag=name, k=k)
    seconds = time.perf_counter() - started
    root = cfg.root
    line = f"{label}: wrote {paths.rel(root, out)} ({len(queries)} queries, {seconds:.1f} s"
    if calls_claude and queries:
        line += f", {seconds / len(queries):.1f} s and {input_tokens / len(queries):,.0f} input tokens per query)"
    else:
        line += ")"
    qrels_path = paths.qrels(root, corpus.name, query_set) if query_set in ("practice", "test", "own") else None
    mrr = None
    if qrels_path and qrels_path.is_file():
        qrels, _ = read_qrels(qrels_path)
        judged = metrics.judged_queries(qrels, [q.qid for q in queries])
        if judged:
            mrr = metrics.mean(metrics.evaluate(results, qrels, judged), "mrr@10")
            line += f", MRR@10 {mrr:.3f} on {len(judged)} judged queries"
    print(line, flush=True)
    return mrr


def cmd_run(args, cfg: config.Config) -> int:
    root = cfg.root
    if args.all:
        return run_all(args, cfg)
    if not args.corpus or not args.system:
        print("p2 run needs --corpus and --system (or --all); for example: uv run p2 run --corpus shared --queries practice --system bm25")
        return 2
    if args.system not in retrievers.names():
        print(f"There is no system called {args.system!r}; the systems are {', '.join(retrievers.names())}.")
        return 2
    label = args.label or args.system
    if not paths.LABEL_RE.match(label):
        print(f"The label {label!r} can use lower-case letters, digits, _ and - only; pick another.")
        return 2
    if label != args.system and label in retrievers.names():
        print(f"The label {label!r} is the name of another system, and p2 check reads a run file named after a system as that system's; pick another label.")
        return 2
    if args.ablation and args.corpus != "own":
        print("--ablation goes with --corpus own: ablations are measured on your own corpus.")
        return 2
    if args.stretch and (args.corpus != "own" or args.ablation or args.repeat or args.out or args.extra_corpus or not args.queries):
        print("--stretch goes with --corpus own and --queries FILE, a file of lines copied from eval/own/queries.tsv (for example 10 of them for the grep agent), and without --ablation, --repeat, --out or --extra-corpus.")
        return 2
    query_set, qpath = resolve_queries(root, args.corpus, args.queries)
    if not qpath.is_file():
        print(f"{paths.rel(root, qpath)} is missing; write the queries first.")
        return 1
    queries, problems = read_queries(qpath, require_origin=args.corpus == "own" and qpath == paths.queries(root, "own", "own"))
    if problems:
        print(f"{paths.rel(root, qpath)}: {problems[0]}; fix the file first.")
        return 1
    if not queries:
        print(f"{paths.rel(root, qpath)} has no queries yet.")
        return 1
    if args.stretch:
        problem = stretch_queries_problem(root, queries)
        if problem:
            print(f"{paths.rel(root, qpath)}: {problem}")
            return 1
        query_set = "own"
    k = args.k or cfg.k
    if args.repeat or args.fresh:
        cfg = cfg.replace(cache_claude=False)
    try:
        corpus = corpus_mod.load(root, args.corpus, extra=args.extra_corpus)
    except ValueError as error:
        print(error)
        return 1
    if not corpus.docs:
        print(f"corpora/{args.corpus}/docs/ has no documents yet.")
        return 1
    calls_claude = retrievers.needs_claude(args.system)
    if calls_claude:
        calls = len(queries) * max(1, args.repeat)
        saved = "every repeat asks Claude again" if args.repeat else ("--fresh asks Claude again" if args.fresh else "fewer if replies are saved from before")
        if not confirm_calls(calls, f"This run of {args.system}", args.yes):
            return 1 if sys.stdin is not None and sys.stdin.isatty() else 2
        print(f"{args.system} calls claude -p once per query: {calls} call(s) on {cfg.model}, {saved}.")
    for repeat in range(1, args.repeat + 1) if args.repeat else [None]:
        if args.out:
            out = args.out if repeat is None else args.out.with_name(f"{args.out.stem}.r{repeat}{args.out.suffix}")
            trace_path = out.with_name(out.stem + ".trace.jsonl")
        elif args.stretch:
            # A stretch run uses some of your own queries; it is committed with its trace, like a run in runs/.
            out = paths.run_file(root, "own", "own", label, stretch=True)
            trace_path = paths.trace_file(root, "own", "own", label, calls_claude=calls_claude, stretch=True)
        elif args.extra_corpus or qpath != paths.queries(root, args.corpus, query_set):
            # A run on extra documents or on a queries file of your own is not part of the contract,
            # so it goes to .cache/extra/ (ignored by git) instead of runs/.
            out = root / ".cache" / "extra" / f"{label}.{query_set}{f'.r{repeat}' if repeat else ''}.trec"
            trace_path = out.with_name(out.stem + ".trace.jsonl")
        else:
            out = paths.run_file(root, args.corpus, query_set, label, repeat, args.ablation)
            trace_path = paths.trace_file(root, args.corpus, query_set, label, repeat, args.ablation, calls_claude=calls_claude)
        try:
            run_one(cfg, corpus, args.system, query_set, queries, out, trace_path, k, label + (f".r{repeat}" if repeat else "") + f" on {where(args.corpus, query_set)}")
        except NotImplementedError as error:
            print(error)
            return 1
    return 0


def stretch_queries_problem(root: Path, queries) -> str | None:
    """Why a stretch run's queries are not lines of your own gold set, or None when they are."""
    own_path = paths.queries(root, "own", "own")
    own = {q.qid: q for q in read_queries(own_path)[0]} if own_path.is_file() else {}
    for q in queries:
        if q.qid not in own:
            return f"{q.qid} is not a query in eval/own/queries.tsv; a stretch run uses queries of your own gold set, so its scores can be compared with your other runs."
        if own[q.qid].text != q.text:
            return f"the text of {q.qid} differs from its line in eval/own/queries.tsv; copy the lines as they are."
    return None


def run_all(args, cfg: config.Config) -> int:
    """bm25, dense, hybrid and rerank on every query set, then every other run file again.

    A system you added (an ablation variant, say) runs only where it already has a run file, so --all
    never spreads it over query sets it was not meant for."""
    from p2.score import discover

    root = cfg.root
    names = retrievers.names()
    canonical = [n for n in retrievers.CANONICAL if n in names]
    sets = []  # (corpus, query set) pairs that have documents and queries
    for corpus in paths.CORPORA:
        if not corpus_mod.doc_files(paths.docs_dir(root, corpus)):
            print(f"Skipping the {corpus} corpus: it has no documents yet.")
            continue
        for query_set in paths.QUERY_SETS[corpus]:
            qpath = paths.queries(root, corpus, query_set)
            if not qpath.is_file() or not read_queries(qpath)[0]:
                print(f"Skipping {paths.rel(root, qpath)}: it has no queries yet.")
                continue
            sets.append((corpus, query_set))
    jobs = [(corpus, query_set, name, name, False) for corpus, query_set in sets for name in canonical]  # (corpus, query set, system, label, ablation)
    refs, _odd = discover(root)
    for ref in refs:
        # repeats and stretch runs are made by a command of their own, never again by --all
        if ref.repeat is not None or ref.stretch or (ref.corpus, ref.query_set) not in sets or (not ref.ablation and ref.name in canonical):
            continue
        tag = run_tag(ref.path)
        if tag in names:
            jobs.append((ref.corpus, ref.query_set, tag, ref.name, ref.ablation))
    others = sorted(set(names) - set(canonical) - {job[2] for job in jobs})
    if others:
        print(f"Not run: {', '.join(others)}. --all runs the four systems of the brief everywhere, and a system you added only where it already has a run file.")
    claude_jobs = [job for job in jobs if retrievers.needs_claude(job[2])]
    if claude_jobs and args.with_claude:
        sizes = {(c, q): len(read_queries(paths.queries(root, c, q))[0]) for c, q in sets}
        calls = sum(sizes[(c, q)] for c, q, *_ in claude_jobs)
        systems = ", ".join(sorted({j[3] for j in claude_jobs}))
        if not confirm_calls(calls, f"Running {systems} on {len(claude_jobs)} query set(s)", args.yes):
            return 1 if sys.stdin is not None and sys.stdin.isatty() else 2
        print(f"With --with-claude this runs {systems} on {len(claude_jobs)} query set(s): up to {calls} claude -p call(s) on {cfg.model}, fewer if replies are saved from before.")
    status = 0
    loaded: dict[str, corpus_mod.Corpus] = {}
    for corpus_name, query_set, name, label, ablation in jobs:
        shown = f"{'ablation/' if ablation else ''}{label} on {where(corpus_name, query_set)}"
        calls_claude = retrievers.needs_claude(name)
        if calls_claude and not args.with_claude:
            print(f"{shown}: skipped, it calls Claude (add --with-claude to run it too).")
            continue
        corpus = loaded.setdefault(corpus_name, corpus_mod.load(root, corpus_name))
        queries, problems = read_queries(paths.queries(root, corpus_name, query_set), require_origin=corpus_name == "own")
        if problems:
            print(f"{paths.rel(root, paths.queries(root, corpus_name, query_set))}: {problems[0]}; fix the file first.")
            status = 1
            continue
        out = paths.run_file(root, corpus_name, query_set, label, None, ablation)
        trace_path = paths.trace_file(root, corpus_name, query_set, label, None, ablation, calls_claude=calls_claude)
        try:
            run_one(cfg, corpus, name, query_set, queries, out, trace_path, args.k or cfg.k, shown)
        except NotImplementedError:
            print(f"{shown}: not built yet (p2/retrievers/{name}.py still raises NotImplementedError).")
    return status


# ---- the other commands ----


def lazy(module_name: str, command: str, hint: str):
    """Import a module that another part of the template provides, or say it is not there yet."""
    try:
        return importlib.import_module(f"p2.{module_name}")
    except ModuleNotFoundError as error:
        if error.name == f"p2.{module_name}":
            print(f"p2 {command} is not built yet in this copy of the template.")
            return None
        print(f"p2 {command} needs the {error.name} package; {hint}")
        return None


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # a Windows console cannot print every character
        except (AttributeError, ValueError):
            pass
    args = parser().parse_args(argv)
    if not args.command:
        parser().print_help()
        return 0
    root = args.root.resolve() if args.root else None
    if args.command == "check":
        from p2 import check

        try:
            root = root or config.find_root()
        except config.ConfigError as error:
            print(error)
            return 1
        return check.run(args, root)
    try:
        cfg = config.load(root)
    except config.ConfigError as error:
        print(error)
        return 1
    args.root = cfg.root
    if args.command == "run":
        return cmd_run(args, cfg)
    if args.command == "score":
        from p2 import score

        return score.run(args, cfg)
    if args.command == "answer":
        from p2 import answer

        return answer.run(args, cfg)
    if args.command == "verify":
        from p2 import verify

        return verify.run(args, cfg)
    if args.command == "judge":
        from p2 import judge

        return judge.run(args, cfg)
    if args.command == "ingest":
        module = lazy("ingest", "ingest", "install the converters with `uv sync --group ingest` and run the command again.")
        return 1 if module is None else int(module.run(args) or 0)
    if args.command == "license":
        module = lazy("license", "license", "run `uv sync` and try again.")
        return 1 if module is None else int(module.run(args) or 0)
    return 2


if __name__ == "__main__":
    sys.exit(main())
