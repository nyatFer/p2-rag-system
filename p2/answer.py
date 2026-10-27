"""`p2 answer`: answer the shared questions from retrieved chunks with Claude, citing chunk ids and quotes.

How one question is answered (the 13 lab's answer.py, on chunks):
- The named system's search_chunks(question, top) gives the top chunks ([answer] top in p2.toml).
- They are sent to Claude, each labeled with its chunk id, followed by the question.
- prompts/answer.txt is the system prompt (--system-prompt), with its lines that start with # removed,
  so the only instructions the model gets are yours.
- The reply is forced into {"answer", "not_found", "claims": [{"text", "chunk_id", "quote"}]}.
- Replies are saved in .cache/claude/, so answering the same chunks with the same prompt again costs
  nothing; --fresh asks Claude again, and every --repeat is fresh.
- answers/<corpus>/<label>.json keeps, for every question, the chunk ids that were retrieved, so the
  quote check can test that each citation points at a chunk the model really saw.
Your work here is the prompt and the choices (system, top, chunking); the plumbing is done.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from p2 import claude, paths, retrievers, verify
from p2 import corpus as corpus_mod
from p2.runfile import read_questions
from p2.trace import Tracer

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "the answer to the question"},
        "not_found": {"type": "boolean", "description": "true if the question was not answered"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "one statement in the answer"},
                    "chunk_id": {"type": "string", "description": "the id of the chunk the statement comes from"},
                    "quote": {"type": "string", "description": "a quote from that chunk"},
                },
                "required": ["text", "chunk_id", "quote"],
            },
        },
    },
    "required": ["answer", "not_found", "claims"],
}
USER_PROMPT = """{chunks}

Question: {question}"""
PROMPT_FILE = Path("prompts") / "answer.txt"


def load_system_prompt(root: Path) -> str:
    """prompts/answer.txt without the lines that start with #."""
    lines = (root / PROMPT_FILE).read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#")).strip()


def format_chunks(hits: list[tuple[str, float]], texts: dict[str, str]) -> str:
    return "\n\n".join(f'<chunk id="{cid}">\n{texts[cid].strip()}\n</chunk>' for cid, _ in hits)


def build_prompt(hits: list[tuple[str, float]], texts: dict[str, str], question) -> str:
    return USER_PROMPT.format(chunks=format_chunks(hits, texts), question=question.text)


def answer_one(system, name: str, question, texts: dict[str, str], system_prompt: str, cfg, tracer: Tracer) -> tuple[dict, bool]:
    """Retrieve, ask Claude, and return the record for one question, and whether the reply was a saved one."""
    with tracer.span(f"answer {name}", "chat", name, question.qid, model=cfg.model) as span:
        hits = system.search_chunks(question.text, cfg.answer_top)
        ids = [cid for cid, _ in hits]
        span["p2.top_ids"] = ids
        prompt = build_prompt(hits, texts, question)
        reply = claude.call(prompt, SCHEMA, system=system_prompt, model=cfg.model, cache_dir=claude.cache_folder(cfg))
        record = {
            "qid": question.qid, "question": question.text, "kind": question.kind, "system": name,
            "retrieved": ids, "answer": "", "not_found": False, "claims": [],
            "seconds": round(reply.seconds, 1), "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens, "model": reply.model,
        }  # fmt: skip
        if reply.error:
            record["error"] = reply.error
            span["p2.error"] = reply.error
        else:
            out = reply.output
            claims = [c for c in (out.get("claims") or []) if isinstance(c, dict)]
            record.update(answer=str(out.get("answer", "")), not_found=bool(out.get("not_found")), claims=claims)
        return record, reply.cached


def run(args, cfg) -> int:
    root = cfg.root
    name = args.system
    label = args.label or name
    if not paths.LABEL_RE.match(label):
        print(f"The label {label!r} can use lower-case letters, digits, _ and - only; pick another.")
        return 2
    qpath = paths.questions(root, args.corpus)
    if not qpath.is_file():
        print(f"{paths.rel(root, qpath)} is missing; the questions for {args.corpus} live there.")
        return 1
    questions, problems = read_questions(qpath)
    if problems:
        print(f"{paths.rel(root, qpath)}: {problems[0]}; fix the file first.")
        return 1
    if args.only:
        wanted = [q.strip() for q in args.only.split(",") if q.strip()]
        unknown = [q for q in wanted if q not in {x.qid for x in questions}]
        if unknown:
            print(f"{', '.join(unknown)} is not a question id in {paths.rel(root, qpath)}.")
            return 2
        questions = [q for q in questions if q.qid in wanted]
    system_prompt = load_system_prompt(root)
    if not system_prompt:
        print(f"{PROMPT_FILE.as_posix()} is empty once the # lines are removed; write your instructions there.")
        return 1
    repeats = list(range(1, args.repeat + 1)) if args.repeat else [None]
    if args.repeat or args.fresh:
        cfg = cfg.replace(cache_claude=False)
    corpus = corpus_mod.load(root, args.corpus)
    try:
        system = retrievers.build(name, corpus, cfg)
    except NotImplementedError as error:
        print(str(error))
        return 1
    if not callable(getattr(system, "search_chunks", None)):
        print(f"{name} has no search_chunks(text, k), which p2 answer needs; add it (ChunkScorer gives it for free) or answer with another system.")
        return 1
    texts = corpus.chunk_texts(cfg.chunk_words, cfg.chunk_overlap)
    asks = len(questions) * len(repeats)
    if retrievers.needs_claude(name):
        # the system itself calls Claude to retrieve, so each question costs one call more
        calls = 2 * asks
        plan = f"up to {calls} claude -p call(s) on {cfg.model}, two per question because {name} calls Claude to retrieve"
    else:
        saved = 0
        cache = claude.cache_folder(cfg)
        if cache is not None:
            for q in questions:
                prompt = build_prompt(system.search_chunks(q.text, cfg.answer_top), texts, q)
                saved += (cache / f"{claude.cache_key(prompt, SCHEMA, cfg.model, system_prompt)}.json").is_file()
        calls = asks - saved
        note = f", and {saved} saved repl{'y' if saved == 1 else 'ies'} that cost nothing" if saved else ""
        plan = f"{calls} new claude -p call(s) on {cfg.model}{note}"
    if not claude.confirm_calls(calls, f"This p2 answer with {name}", getattr(args, "yes", False)):
        return 2
    print(f"{len(questions)} question(s), top {cfg.answer_top} chunks from {name}, {len(repeats)} time(s): {plan}.")
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    for repeat in repeats:
        records = []
        started = time.perf_counter()
        with Tracer(paths.answers_trace(root, label, repeat)) as tracer:
            for n, q in enumerate(questions, 1):
                record, was_saved = answer_one(system, name, q, texts, system_prompt, cfg, tracer)
                records.append(record)
                state = "ERROR " + record["error"] if record.get("error") else ("not_found" if record["not_found"] else f"{len(record['claims'])} claim(s)")
                cost = "saved reply, no call" if was_saved else f"{record['seconds']:.1f} s"
                print(f"[{n}/{len(questions)}] {q.qid} {cost}, {record['input_tokens']:,} input tokens, {state}", flush=True)
        data = {
            "label": label, "system": name, "corpus": args.corpus, "model": cfg.model,
            "chunking": {"words": cfg.chunk_words, "overlap": cfg.chunk_overlap}, "top": cfg.answer_top,
            "prompt_sha256": prompt_hash, "repeat": repeat, "answers": records,
        }  # fmt: skip
        out = paths.answers_file(root, args.corpus, label, repeat)
        paths.write_text(out, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        totals = verify.evaluate(data, texts, label=label + (f".r{repeat}" if repeat else ""))
        print(f"Wrote {paths.rel(root, out)} in {time.perf_counter() - started:.0f} s.")
        print(totals.summary())
    if args.only:
        print("This file covers only the questions you named; answer all of them under one label before you submit.")
    return 0
