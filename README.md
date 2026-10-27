# Project 2 - Build a RAG System and Measure It

**CSCI 498E / 598E - Coding with AI Agents**

| | |
|---|---|
| **Weight** | 20% of your final grade |
| **Points** | 200 on Canvas: 150 for this project and 50 for a video, handed in together in one discussion post |
| **Released** | Tuesday, October 6 (lecture 12) |
| **Due** | Tuesday, October 27, 11:59 pm, one deadline for everything |
| **Work** | Individual; 598E adds a rider (below) |

You hand in three things:

1. A **public GitHub repo** made from this template, whose URL you post in the Project 2 discussion on Canvas.
2. The **work inside it**: three retrievers you built, run files and answers you generated, your own corpus and gold set, `EVAL.md` and `DECISIONS.md` written in your own words, and for 598E a `PREREG.md`.
3. A **5 to 10 minute video**, in the same discussion post.

---

## What this project is

You build a system that searches a collection of documents and answers questions from what it finds.
Then you measure which search method works best on questions you wrote yourself, and you report what your numbers do and do not let you claim.

The system is not the point; the measurement and the judgment are.
Lecture 12 told you that no retriever wins everywhere, and here you test that claim yourself, on two corpora, with intervals that say how much to trust each difference.

The project has two stages, and one deadline for both.

- **Stage 1: the shared corpus.**
  Everyone works on the same collection, **30 CFR Chapter I**: the federal mine safety and health regulations, one section per file.
  It is public domain, it is a Mines subject, and I hold the answer key.
  You build `dense`, `hybrid` and `rerank` from the code you already wrote in labs 12 and 13, next to the `bm25` baseline that ships in this repo.
  You run all four on 20 practice queries (answers public, so you can score yourself) and 40 test queries (answers private, scored by me), and you answer 12 questions with citations.
- **Stage 2: your own corpus.**
  You choose a collection you are allowed to publish, ingest it, write your own gold set, run the same four systems, run one ablation, and analyze where the best system fails.
- **The cross-corpus question.**
  Did the best system on the shared corpus stay the best on yours, and why or why not?
  Lecture 12's claim becomes something you measured.

I know two stages is more work than one corpus would be.
I chose it for five reasons.
Building a retrieval pipeline and preparing a corpus are different skills, and stage 1 means you are not stuck converting PDFs during the first week.
I have no teaching assistants, so I would rather debug one corpus for everyone than thirty-seven different ones.
The private test queries give the whole class one comparable number, which corpora of your own cannot.
Moving your pipeline to a second corpus shows that it is general and not tuned to the first.
And the comparison between the two is the thing I most want you to take away.

### The four systems

| System | What it is | Who writes it |
|---|---|---|
| `bm25` | BM25 keyword search, the lab-calibrated version from 12 | provided |
| `dense` | embeddings of your choice (the potion or bge model from 12, or another) and a nearest-neighbor search | you |
| `hybrid` | fusion of two systems, for example reciprocal rank fusion of `bm25` and `dense` from 12 | you |
| `rerank` | a `claude -p` listwise reranker over a first-stage top 20, as in lab 13 | you |

A retriever is a module in `p2/retrievers/` with a `build(corpus, cfg)` function that returns an object whose `.search(text, k)` gives a list of `(docid, score)` pairs, and a constant `NEEDS_CLAUDE`.
`p2/retrievers/bm25.py` is a working example, and the three you write ship as stubs that raise `NotImplementedError` and point at the lab file that shows the idea.
You may add more systems, for an ablation or because you are curious.

### What a program checks and what I read

`uv run p2 check` is the check that runs on your laptop and in GitHub's CI on every push.
It runs every system that does not call Claude again and compares the result with the run files you committed, checks the runs of the systems that do call Claude against the traces you committed with them, recomputes the scores and the tables in `EVAL.md`, and checks the format of everything else.
At the end I run the instructor's copy of `p2 check --final` on your repo, so editing the checker in your copy cannot help you.
I score your test run files against the answer key I keep private, and I re-run your retrievers on a few documents I add that you have never seen, to confirm that the run files you committed really come from your code.
The test score goes back to you as feedback.
It counts only as complete and reproducible, so it is there to catch a pipeline that works on the practice queries and nothing else, and there is no leaderboard to tune for.
I read `EVAL.md` and `DECISIONS.md` and watch the video myself, because the judgment in them is what I am grading.

---

## Start here

Every command below runs from the root of your own P2 repo, `work/p2-rag` once you have cloned it, unless it says otherwise, and every `p2` command starts with `uv run`.
Say "In work/p2-rag, run ..." to Claude Code, or run it yourself in a terminal in that folder.

1. Click **Use this template** at the top of this repo, then **Create a new repository**.
2. **Name it `p2-rag-system`** and make it **Public**.
   Public, because I need to read it and because the whole project rests on committing only text you are allowed to publish.
3. Clone it into `work/`, your own space in the course repo: say **"clone my P2 repo into work/p2-rag"** to Claude Code, or in a terminal at the root of the course repo run `gh repo clone <your-github-username>/p2-rag-system work/p2-rag`.
   Git ignores `work/`, so this repo never collides with a course update.
   Keep working in the same Claude Code session and say where the work goes (`work/p2-rag`).
   At the start of every Claude Code session you use for P2, the first one and every one after, ask it to read `work/p2-rag/CLAUDE.md`.
   It is a short orientation for the agent, including the rule to tell you how many Claude calls a command makes before it runs it, and Claude Code loads it only after it reads a file in that folder, so running commands there is not enough.
4. **Install `uv`** if the prep for lecture 12 did not already.
   Check with `uv --version`; if it prints a version number, you are done.
   Otherwise, on macOS or Linux run `curl -LsSf https://astral.sh/uv/install.sh | sh`, and on Windows, in PowerShell, run `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, then open a new terminal (on Windows, Git Bash) so it is on your PATH.
5. Run `uv sync`.
   It installs the right Python and every package this repo needs, which takes a minute or two the first time.
6. Download the two embedding models by running the lab 12 warm-up once more.
   This one command runs from the root of the course repo, the folder you start Claude Code in, and not from `work/p2-rag`: `uv run class/12-embeddings-and-retrieval/lab/starter/warmup.py`.
   It downloads potion-retrieval-32M and bge-small (about 200 MB) into the Hugging Face cache, which this repo reads too, checks that each one loads, and ends with the word `ready`.
   If you ran it for lecture 12, it finishes in seconds.
   If you skip it, your first `dense` run downloads the same models, so nothing breaks, but you will do it in the middle of your work and not before.
7. Open `p2.toml` and check that `section` matches your course, `"498E"` or `"598E"`.
8. Run `uv run p2 check`.
   On a fresh copy it prints `PASS` lines and `TODO` lines and exits cleanly: a `TODO` line is work you have not done yet and is not an error, and only a `FAIL` line means something is broken.
   Later you may also see `NOTE` lines, which point at something in your gold set worth a second look and never fail the check.
9. Commit and push, open the **Actions** tab of your repo, and wait for the `p2-check` run to go green.
   This first run takes a minute or two, because there is nothing to regenerate yet.
   Once you commit runs of `dense` or `hybrid`, the next run downloads the embedding model and encodes the corpus, which can take several minutes, and later runs reuse a cache.

Please do all of this on the first day, before you have started anything real.
The setup is the step with the most ways to go wrong, and you want it behind you while there is plenty of time.

### A pace that works

There are no checkpoints, and nothing is checked on these dates.
This is the pace I would keep, and I am saying it because the two slow parts of this project, reviewing licenses and judging documents by reading them, are the ones people leave to the last weekend.

- **By lecture 13 (Thu Oct 8):** setup done and CI green, `dense` and `hybrid` working on the practice queries.
- **By lecture 14 (Thu Oct 15):** stage 1 finished, with the reranker, the scores, and the cited answers; your own corpus chosen and its first `p2 license` run done, so you know the corpus is usable.
- **By Tue Oct 20:** the corpus ingested and the gold set written, using what lecture 14 teaches about gold sets.
  598E: `PREREG.md` committed before any judgment goes into your `qrels.txt`.
- **By Fri Oct 23:** the four systems on your corpus, scored, and the ablation done.
- **The last weekend:** failure analysis, cross-corpus comparison, `DECISIONS.md`, the video, and `uv run p2 check --final` on Monday.

Tue Oct 13 is Fall Break, so there is no class, and nothing is due.

---

## Stage 1: the shared corpus

### What is in the repo

- `corpora/shared/docs/` holds 30 CFR Chapter I from the eCFR, one section per file: `cfr30-75.403.md` is section 75.403, with its section number and heading as the first line.
  `NOTICE.md` pins the date of the eCFR copy, explains why it is public domain, and says what it is not: it is not an official legal edition, and nothing here is legal or compliance advice.
  `manifest.tsv` records each document's source and license.
- `eval/shared/practice.queries.tsv` and `practice.qrels.txt` hold the 20 practice queries (7 `identifier`, 7 `paraphrase`, 6 `mixed`, the same three classes as lab 12) and their judgments.
- `eval/shared/test.queries.tsv` holds the 40 test queries (13, 14 and 13).
  Their judgments are not in this repo.
- `eval/shared/questions.tsv` holds the 12 questions for the cited answers: 8 the corpus answers and 4 it does not.

A section is relevant to a query if it states the requirement the query asks about.
When the query does not name a kind of mine, the parallel sections in parts 56 (surface metal and nonmetal), 57 (underground metal and nonmetal), 75 (underground coal) and 77 (surface coal) are all relevant, because the same rule is written once for each.
A section that only cross-references the requirement is not relevant.
That one rule explains most of the surprises you will find when you read a failed query.

### Steps

1. **Read before you build.**
   Open three sections, `NOTICE.md`, five practice queries and their judgments.
   Notice how short the sections are, how often the same rule repeats across parts 56, 57, 75 and 77, and how little of a paraphrase query's wording appears in the section it wants.
   That last point is deliberate: `bm25` finds none of the seven practice paraphrase queries' answers in its top 10, which is what keyword search does when the words differ, not a bug in your setup.
2. **Run the baseline.**
   Say "In work/p2-rag, run `uv run p2 run --corpus shared --queries practice --system bm25`".
   It writes `runs/shared/bm25.practice.trec`; open it and look at the first lines.
   A query that shares words with fewer than ten sections, such as p01, ends with sections that score 0.000000 in document id order: that is padding up to ten lines, not a match.
3. **Build `dense`.**
   Ask Claude Code to read `class/12-embeddings-and-retrieval/lab/starter/retrieve.py` in the course repo and port the embedding arm into `p2/retrievers/dense.py`, using the helpers in `p2/embed.py`.
   The docstring of `p2/retrievers/__init__.py` explains the interface, including the `ChunkScorer` helper that gives you both document search and the chunk search that `p2 answer` needs, and `bm25.py` is the worked example.
   You choose the embedder, and the choice goes in `EVAL.md`.
   If you want a setting of your own, such as the model name, add a table to `p2.toml` (for example `[dense]` with `model = "..."`) and read it in your retriever with `cfg.table("dense")`.
   Run it the same way as the baseline with `--system dense`.
   Scale is the new problem: the lab had 260 articles and this corpus has thousands of sections, so the first run takes a few minutes while the vectors are computed, with a progress line every 15 seconds or so, and then they are cached in `.cache/vectors/`.
   Chunk settings live in `p2.toml`; a section is usually shorter than one chunk, but a few are long.
   A run is always at the document level: the best chunk's score becomes the document's score.
4. **Build `hybrid`.**
   Port the fusion from lab 12 or 13 (`class/13-rag-pipeline/lab/starter/pipeline.py`) into `p2/retrievers/hybrid.py`.
   You decide which two systems to fuse; `bm25` and `dense` is the usual choice, and `dense` with a second embedder is also fair.
5. **Build `rerank`.**
   Port the listwise reranker from lab 13 into `p2/retrievers/rerank.py`: a first-stage top 20, sent to Claude through `p2.claude.call`, which handles the full path, the JSON schema, the model, the saved replies, and the environment for you.
   Keep `NEEDS_CLAUDE = True` in that file, because `p2 check` runs every system again with Claude switched off, and it must not spend your plan doing so.
   It checks a reranker's runs against the trace that `p2 run` writes next to them in `traces/` instead, so commit the trace with the run file.
   Try it on two queries before the full run, because every call draws on your own plan (see the cost note below).
   To do that, copy two lines of `eval/shared/practice.queries.tsv` into `.cache/two.tsv` and run `uv run p2 run --corpus shared --queries .cache/two.tsv --system rerank`, which writes its run file to `.cache/extra/` and leaves `runs/` alone.
   Before any command that makes more than 5 Claude calls, `p2` stops and says how many: in a terminal it asks you, and when Claude Code runs it, it stops until the command is run again with `--yes`, so the agent has to bring the number to you first.
   Until a retriever is written it raises `NotImplementedError`, and `p2 run --all` skips it and lists it as work to do.
6. **Run everything.**
   Say "In work/p2-rag, run `uv run p2 run --all --with-claude`".
   It runs the four systems on the practice and test queries, 60 reranker calls in all, so it asks first; without `--with-claude` it skips the reranker.
   Commit the run files in `runs/shared/`.
7. **Score.**
   Say "In work/p2-rag, run `uv run p2 score`".
   It computes recall@10, MRR@10 and nDCG@10 for every run that has an answer key, per query class, plus the paired intervals and the minimum detectable difference for every pair of systems.
   Recall@10 is the share of a query's relevant documents that are in its top 10, and MRR@10 is 1 divided by the rank of the first relevant one (0 if none is in the top 10), both from lab 12.
   nDCG@10 adds up the relevant documents in the top 10, each divided by log2 of its rank plus 1 so a lower rank counts less, and divides that by the same sum for the best possible order, so 1.0 is a perfect ranking.
   A paired interval is the 95% range of the mean difference between two systems on the same queries, found by resampling the queries 10,000 times; lecture 14 covers both in depth.
   It writes `results/results.json` and fills the tables in `EVAL.md`.
   You then write what the intervals let you claim, in `EVAL.md` section 1.
8. **Answer the 12 questions.**
   Choose a system, and say why in `EVAL.md`.
   `prompts/answer.txt` is the starting prompt from lab 13, and you may edit it.
   Try two questions first, one the corpus answers and one it does not (the ids are in `eval/shared/questions.tsv`): `uv run p2 answer --corpus shared --system rerank --label rerank --only a01,a09`.
   Then run all twelve with the same command without `--only`, which writes `answers/shared/rerank.json`.
   Answering with `rerank` reranks each question before it answers, so it makes two Claude calls per question, 24 for all twelve; answering with `hybrid` makes one, and `p2` counts the replies it already has from your trial.
9. **Verify the answers.**
   Say "In work/p2-rag, run `uv run p2 verify answers/shared/rerank.json`".
   It prints a PASS or FAIL line per question and always finishes cleanly, because a FAIL here is a result to report, not a broken file.
   It checks that every citation is one of the chunks that was retrieved, that every quote appears in the chunk it cites (after lower-casing and collapsing whitespace), that the 4 unanswerable questions were declined with no claims, and that the 8 answerable ones were not.
   The check proves a quote is real.
   It cannot prove that the quote supports the claim next to it, so read three answers yourself and say so in `EVAL.md`.
   Claude may already know these regulations from training, which is why the answer must quote what you gave it, and why the four unanswerable questions are the real test.
10. **Run `uv run p2 check`**, commit, push, and check that CI is green.

### What the Claude calls cost

Anything that runs `claude -p` draws on your own Claude plan.
On this corpus, the course's own versions read about 12,000 input tokens per rerank call and about 7,000 per answer call, a little more than in the 13 lab because the sections are longer than the lab's articles; plan on that much.
A full pass of the reranker over the 20 practice queries, the 40 test queries and your own 30 or more is about 90 calls, roughly a million input tokens.
The 12 answers add about 80,000 input tokens when you answer with `hybrid`, and about 230,000 with `rerank`, which reranks each question before it answers.
That is about three times the 13 lab.
Spread it over several days and try a few queries before every full run.
`p2` saves every Claude reply in `.cache/`, so running the same thing again costs nothing, but a change to your prompt, your candidates or the model asks Claude again, and so do `--fresh` and `--repeat`.
If you reach your usage limit, `/usage` shows where you stand and it resets on its own schedule; the run files you already committed are what count.

---

## Stage 2: your own corpus

### Choose a corpus you may publish

Your corpus goes in a public repo, and CI re-runs retrieval on it.
So it has to be text you may republish, and I do not allow private corpora in this project, not even the pattern of keeping PDFs on your laptop and committing only a manifest.
I know that costs some of you the corpus you would most like to use, because many fields keep their documents behind publisher licenses.
The reason is that CI and I cannot check retrieval on documents we cannot see.
If your field's documents are copyrighted, pick an openly licensed slice of the field.

**Allowed**, each with its license in the manifest:

- Your own writing, entirely yours: reports, notes, essays, code documentation (`own-work`).
- US federal government works (`us-gov-public-domain`), except contractor-written reports and the third-party material that appears inside government reports.
- CC0, CC BY and CC BY-SA text, including Wikipedia (CC BY-SA 4.0, with attribution) and open-access articles whose own license is one of those.
- US public-domain books.
- Openly licensed software documentation (MIT, Apache 2.0, BSD, CC BY), such as Kubernetes.
- PubMed Central open-access articles whose per-article license is CC0, CC BY or CC BY-SA, and arXiv papers under CC0, CC BY or CC BY-SA (arXiv abstracts are CC0).

**Not allowed:**

- Publisher PDFs, textbooks, and engineering standards (ASTM, ISO, IEEE).
- Other courses' materials, and employer or internship documents.
- An advisor's unpublished data, unless you have their written permission.
- Private information about people: student records, patient data, survey or interview answers, home addresses or personal phone numbers, private messages, and data about the members of your lab.
  Those people never agreed to be in a public repo, and git history keeps whatever you commit.
  Names and work contact details that the publisher printed in a public document are fine, such as an author's email on a paper or the "Prepared by" line on a USGS chapter.
- Anything export-controlled.
- Anything licensed NC (non-commercial) or ND (no derivatives), because whether chunks and embeddings count as adaptations has no settled answer.
- Anything with no license at all: a missing copyright notice does not make a work free.

I am not telling you that using copyrighted text in a retrieval system is safe.
Whether it is fair use depends on the facts, and recent court decisions have gone against at least one company that did it.
I am not a lawyer, and this is not legal advice.
The rule for this project is simpler and stricter: commit only text you can show you may republish.

If you have no corpus of your own, choose from this short list:

- **OSHA's rules, 29 CFR parts 1910 and 1926**, public domain, from the eCFR.
  Take a slice, such as one part or a few subparts, because the two parts together are about 1.2 million words, over the size guidance below.
  This text is the same kind as the shared corpus, so your cross-corpus comparison will be less interesting than for a different kind of text.
- **The USGS Mineral Commodity Summaries**, public domain as federal works, with one catch: the front matter of each volume says permission must be secured from the individual copyright owners for any copyrighted material inside it, such as a photo credited to a company.
  The license tool flags that sentence only where it appears, in the volume's first pages, and the per-commodity chapter PDFs, the easy way to reach 200 documents, do not repeat it.
  So a corpus of chapters can come back clean, and you still say in `EVAL.md` that the volume makes this reservation and why your chapters are text you may publish.
  Ingest each year's chapters with its own `--source`, for example `--source "https://pubs.usgs.gov/periodicals/mcs2024/{name}"`, so every document records the address of its own file.

The eCFR's website blocks scripts but its API does not, so ask Claude Code to write a small fetch script (I fetched 30 CFR that way).
Keep your raw downloads outside the repo, for example in `work/p2-raw/`, which from inside `work/p2-rag`, where every command runs, is `../p2-raw`.

Size rules, checked by `p2 check`:

- At least **200 documents** from one domain.
  Below that, every system finds nearly everything and the comparison measures nothing.
  A long document can be split into parts, and each part counts as a document (step 1 below).
- At most **1,000,000 tokens**, estimated as words times 1.4: `p2 check` warns above 800,000 and fails above 1,000,000.
- At most **25 MB** under `corpora/own/`, and 10 MB per file.
- Between 500 and 1,500 documents is a good size when your documents are short, as long as the corpus stays under the token limit; past about 1,500 you add CI time and not information.

The token limit is there for CI.
The first CI run that checks a `dense` run on your corpus encodes the whole corpus, and the check may run for 45 minutes.
The shared corpus, about 550,000 tokens, took 3 to 7 minutes to encode on GitHub's Linux machine in my tests, so a corpus near the limit costs about twice that, 6 to 14 minutes, once.
Later runs reuse a cache, and the same check then took 6 seconds.
I set the limit this high because most of you bring a corpus from your own field, where long documents are common, and the extra CI time is paid once, and only by those of you with a large corpus.

The two rules meet when your documents are long: 200 documents of about 3,500 words is about 700,000 words, or about 1,000,000 tokens.
So 200 whole documents much longer than that cannot fit under the limit.
If long documents leave you under 200, split them into parts of about 5 pages with `--part-pages 5` (step 1 below), and each part counts as a document.
For example, 70 papers of about 9,000 words and 15 pages each are only 70 documents whole, and 210 documents as 5-page parts, about 880,000 tokens in all.
In my tests a 5-page part was about 1,700 words of a NIST report and about 3,300 words of a two-column research paper, so 200 such parts stay under the limit.
`corpora/own/INGEST.md` lists the words in each part, so check yours, and if your parts run well over 3,500 words, use fewer pages, such as `--part-pages 3`.

### Steps

1. **Ingest it.**
   Install the converters with `uv sync --group ingest`, which CI never does, and run `uv run p2 ingest ../p2-raw --into own --license <id> --source "<where it came from>"`, where `<id>` is one of the license values listed in step 3.
   In `--source`, `{name}` stands for each file's name, so `--source "https://example.org/reports/{name}"` gives every document the address of its own file.
   PDFs go through pypdf, web pages through trafilatura, and `.md` and `.txt` files are read as they are.
   Word and PowerPoint files are refused, and the message tells you to export them to PDF first.
   Ingest runs a cleaning pass that strips markup, repairs ligatures and soft hyphens, re-joins words hyphenated across lines, drops page headers and footers, and drops a trailing reference list.
   This matters because raw converter output breaks the quote check on 8% to 41% of sentences in my tests, and cleaned text breaks it on 1% or less.
   Documents over about 15,000 words are split into parts of about 10 pages, with the page range in the id (`handbook__p011-020`), and exact duplicates are skipped.
   If long documents leave you under 200 (see the size rules above), add `--part-pages 5`: every PDF longer than 5 pages becomes parts of 5 pages with the same kind of id, and each part counts as a document.
   It splits PDFs only, because web pages and text files have no pages; those are split only above 15,000 words.
   Decide before you write any judgments, and use the same setting for every batch of PDFs, so your documents are of a similar size.
   If you already ingested a file cut another way, whole for example, ingest skips it and names the documents its text is already in; remove those documents and their rows in `corpora/own/manifest.tsv` first (Claude Code can do it), then run ingest again.
   If your documents have different licenses, run ingest once per license so each batch gets its own `--license` and `--source`, or edit the manifest rows by hand.
   Document ids come from file names and never change once your judgments mention them.
   Titles come from the PDF's metadata, the first heading or the first line; when several PDFs share one metadata title, as every chapter of a volume often does, each title starts with its file name instead.
2. **Read `corpora/own/INGEST.md`.**
   It has one line per document with its word count, characters per page and any flags.
   A document with fewer than about 400 characters per page is probably a scanned image with no text; drop it, or ask Claude Code about the OCR option in the ingest docs.
   Open three documents and read them, because a corpus you have not looked at will surprise you later.
   If you remove documents by hand later, `uv run p2 ingest --report` rewrites `INGEST.md` for the documents that are left.
3. **Check the licenses.**
   Run `uv run p2 license`.
   It writes `LICENSES.md` and fails when a document's `license` in the manifest is not on the allowed list: `us-gov-public-domain`, `public-domain`, `cc0-1.0`, `cc-by-2.0` to `cc-by-4.0`, `cc-by-sa-2.0` to `cc-by-sa-4.0`, `own-work`, `mit`, `apache-2.0`, `bsd-2-clause` and `bsd-3-clause`.
   Anything else fails, including `unknown`, an empty value, and anything NC or ND.
   It also scans each document's text for a copyright line, "All rights reserved", a publisher's name next to words about rights (such as "Published by Elsevier"), a CC BY-NC notice, "reprinted with permission", the USGS permission sentence, a contractor's notice and a "courtesy of" credit line.
   Each hit is a flag, and a flag passes only when the manifest's `notes` column says `reviewed: <reason>` for that document.
   A publisher's name on its own, such as "see ASTM E2500", is usually a citation, so `LICENSES.md` lists it as a mention and it needs no review.
   Ask Claude Code to "read `work/p2-rag/.claude/skills/license-check/SKILL.md` and walk me through my license failures", and it takes you through each failing or flagged document: it opens the source page, finds and quotes the license text, and you decide whether to record `reviewed: <reason with the quote>` or remove the document.
   Run `uv run p2 license --online` on your own machine as well; for a source with a DOI, an arXiv id or a PubMed Central id, it asks Crossref, arXiv or PubMed Central for the published license and reports any mismatch with yours.
   The tool gathers evidence and flags problems.
   It cannot prove a license, and you remain responsible for what you commit, so when you cannot quote the license for a document, take the document out.
4. **Write your gold set.**
   `eval/own/queries.tsv` has one query per line: `qid`, `class`, `text`, `origin`, separated by tabs.
   `eval/own/qrels.txt` has one judgment per line, `qid 0 docid 1`.
   - At least **30 queries**, at least **10 with `origin` `hand`**, and every query with at least one relevant document.
   - `hand` means you wrote the query yourself, from your own question or from reading the corpus.
     `claude` means a model drafted it, which is allowed for the rest, and you still judge relevance.
     The column exists so you can check whether the queries a model wrote favor the keyword systems, which they often do because they copy the document's own words; stretch option 6 measures it.
   - The `class` is a free word.
     The shared sets use `identifier`, `paraphrase` and `mixed`; use those or classes that fit your corpus, because per-class scores are reported.
   - Judge relevance by reading each document, and write your rule in `EVAL.md`.
     Find candidates from several places: the top 10 of more than one system, a keyword search, and your own reading.
     A gold set built only from what one system returned is biased toward that system.
   - Claude Code can search the corpus for candidates, and it can draft queries, but you decide what is relevant.
5. **Run the four systems.**
   Say "In work/p2-rag, run `uv run p2 run --all --with-claude`", which writes `runs/own/<system>.trec` for each system and asks before the reranker's calls.
6. **Score.**
   Run `uv run p2 score` again, and the own-corpus tables in `EVAL.md` fill in.
7. **Run one ablation.**
   An ablation changes one thing and keeps everything else the same, so you can say what that one thing did.
   Good candidates: the chunk size, the BM25 tokenizer (`tokenizer = "lab"` gives lab 12's simple one), a second embedding model, how many candidates the reranker sees, the fusion method.
   Write your hypothesis in `EVAL.md` before you run it.
   Make the variant its own system by adding a file to `p2/retrievers/`, for example `bm25_lab.py` or `dense_potion.py`; a file there is a system, and nothing else needs registering.
   A variant that only changes a setting can build with a changed copy of the settings, `cfg.replace(chunk_words=100)`, so you do not have to edit `p2.toml`, which every system shares.
   Run it with `--ablation`, as in `uv run p2 run --corpus own --system bm25_lab --ablation`, which writes the run file to `runs/own/ablation/`.
   From then on `p2 run --all` runs it again only as that ablation, not on every query set.
   Then run `uv run p2 score` and write what the interval lets you claim; the `own-ablation` table in `EVAL.md` compares it with your other systems.
   The `own-ablation-queries` table below it lists the queries your change helped and hurt against the system it varies, which `p2` finds from a line such as `BASE = "bm25"` in your variant's file, or else from its name: `bm25_lab` varies `bm25`.
8. **Write the analysis** (next section), then `uv run p2 check --final`.

---

## The analysis

### The failure analysis

Take the queries where your best system on your own corpus fell short, at least five, and read what it returned next to what was relevant.
Sort them by cause: the vocabulary of the query and the document differ, near-duplicate documents compete, a chunk cut the answer in half, the query was ambiguous, or the label was wrong.
The last one happens more than you would think, and finding it is a good result.
Then say what you would try, and what evidence would tell you whether it worked.
`EVAL.md` section 6 has a table for it.

### The cross-corpus comparison

Put the systems side by side on the two corpora, using the tables `p2 score` wrote.
Say whether the winner changed, and then explain why, using what you know about the two collections: how long the documents are, how much of their vocabulary is shared with the queries, whether identifiers matter, whether near-duplicates are common.
Say which explanations you tested and which are guesses.

### What "not distinguishable" means

With about 30 queries, many differences between systems are too small for your gold set to see.
For every pair of systems, `p2 score` gives the mean difference and its paired interval, which is the 95% range of that difference when the queries are resampled 10,000 times.
If the interval includes zero, you cannot say which system is better, and the right words are "not distinguishable on this gold set".
That is a different statement from "the two are equal".
It says the gold set was too small to tell.

`p2 score` also gives the minimum detectable difference, the smallest gap this many queries could reliably show.
If the typical per-query spread is about 0.3, then 30 queries can only detect a difference of about 0.15 in MRR, and a gap of 0.05 is invisible however real it is.
On the shared practice queries the spread between two systems is 0.3 at the smallest and over 0.6 at the largest, so most pairs need more queries than that.
So write the interval, say "not distinguishable" when it includes zero, never rank two systems by their averages alone, and say how large a difference you could have seen.
A claim that matches its interval is a good result even when the answer is "I cannot tell".

---

## The 598E rider

The rider is required for 598E and is one of the stretch options for 498E.
For a 598E student the rider is the stretch: you do not do the other options, and doing one as well adds no points, because the project is capped at 150.
Set `section = "598E"` in `p2.toml`, which makes `p2 check --final` verify the files below.
It has two parts, and I estimate 4 to 6 extra hours, mostly the larger gold set.

**Repeats, on the shared corpus.**
The Claude-based steps are not deterministic, so run them three times, asking Claude again each time.
For the reranker that is `uv run p2 run --corpus shared --queries practice --system rerank --repeat 3`, which saves `runs/shared/rerank.practice.r1.trec`, `.r2.trec` and `.r3.trec`.
For the cited answers it is `uv run p2 answer --corpus shared --system rerank --label rerank --repeat 3`, which saves `answers/shared/rerank.r1.json`, `.r2.json` and `.r3.json`.
`p2 score` reports the run-to-run spread, and Wilson 95% intervals on the share of quotes verified and the share of unanswerable questions correctly declined.
A Wilson interval is a confidence interval for a share that behaves well on small counts: 4 declined out of 4 still leaves an interval from about 0.5 to 1, which is the honest size of what 4 questions can show.
Then say whether the reranker's gain over its first stage survives the run-to-run variation.
The three reranker repeats and the three answer sets add about 130 Claude calls (60 for the reranker, and 72 for the answers, since answering with `rerank` makes two calls per question).

**A pre-registered, powered claim, on your own corpus.**
`PREREG.md` holds your hypothesis, the smallest effect that matters, the gold-set size you need to detect it (computed from the per-query spread you measured in stage 1), and the test.
You commit it before the first judgment appears in `eval/own/qrels.txt`, and `p2 check --final` reads `git log` to confirm the order.
Afterwards you report the outcome against it.
A well-powered null scores as well as a confirmation, so you have no reason to hedge the claim.
Pre-register a smallest effect of at least 0.15 in MRR@10 or nDCG@10, unless you argue for a larger one: a smaller gap is rarely worth a reranker's cost, and detecting it takes more queries than one project can judge.
The gold-set size then follows from the per-query spread you measure in stage 1.
For `rerank` against `hybrid` that spread is about 0.39 on the practice queries, so a gap of 0.15 needs about 55 queries, against the 30 that 498E uses.
Spreads differ by pair, from about 0.31 (`hybrid` against `bm25`) to about 0.66 (`dense` against `bm25`), and at 0.66 even a gap of 0.15 needs about 150 queries, so compare a pair whose spread you can afford, or take a larger gap and say why.
Say in `PREREG.md` how you chose the pair, the gap and the gold-set size.

---

## One stretch

The stretch is 25 of the 150 points.
For 498E it is one of the six options below, your choice.
For 598E it is the rider, which is required, and you do not do the other options: doing one as well adds no points, because the project is capped at 150.

Every option is graded the same way.
Each one below says what you hand in and what earns the 25 points, and each ends in `EVAL.md` section 8 with five things:

1. the option's name;
2. what you did, and the files that hold it;
3. the table `p2 score` wrote for it, by name;
4. what its interval lets you claim, and what it does not;
5. one thing you would do next, and what result would tell you it worked.

Pick your option early and try a small piece of it first, because the expensive ones draw on your Claude plan.

### 1. The 598E rider

The rider as the section above describes it: three repeats of the Claude steps on the shared corpus, and a pre-registered, powered claim on your own corpus.
Set `section = "598E"` in `p2.toml`, so `p2 check --final` checks its files; in 498E, set it too, because the setting only switches on the rider's checks.

**What you hand in:** the three reranker runs and the three answers files with their traces, `PREREG.md`, the `repeats` table and your write-up in `EVAL.md` section 9, and the word "rider" in section 8.
**What earns the 25 points:** both parts done as that section says, the run-to-run spread read against the reranker's gain, `PREREG.md` committed before your first judgment, and its outcome reported.

### 2. A claim-level faithfulness check

`p2 verify` proves that a quote is really in the chunk it cites, and it cannot tell whether the quote supports the claim next to it.
Here Claude judges exactly that, one claim at a time, and you measure how far to trust the judge by checking at least 20 claims yourself.

1. Read `prompts/judge.txt`, a plain starting prompt that is yours to change, like `prompts/answer.txt`.
2. Try two questions first, for example `uv run p2 judge answers/shared/rerank.json --only a01,a02` with the name of your own answers file, and then judge the whole file without `--only`.
   It makes one Claude call per claim, asks first above 5 calls, and writes `answers/shared/rerank.judged.json` and `traces/judge-rerank.jsonl`.
   Commit both, because `p2 check` compares the verdicts with the trace.
3. Label at least 20 claims yourself, before you open the judged file, so Claude's verdicts cannot steer yours.
   Write them in `answers/shared/rerank.calibration.tsv`, one line per claim with three fields separated by tabs: the claim id, your verdict (`supported`, `partly` or `not`), and a short note on why.
   A claim's id is its question id and its number in that answer, so `a01-c1` is the first claim of a01; read the claim and its quote in the answers file, open the cited chunk, and decide.
   No command writes this file, and Claude Code must not fill it in for you, because your verdicts are the measurement.
   Twelve questions give about 10 to 20 claims, so if your answers file has fewer than 20, judge a second one (another system, or a repeat) and label claims from both.
   If you run `p2 answer` again under the same label, its claims change, so judge and label them again.
4. Run `uv run p2 score`.
   The `stretch-judge` table in `EVAL.md` section 8 shows the share of claims where Claude's verdict is yours, with its Wilson 95% interval, and a 3 by 3 table of Claude's verdicts against yours.

If you change `prompts/judge.txt` after you have labeled claims, you are tuning the judge on your own labels, so say so, and report the agreement from before the change too.
A judge call is short: 1,400 to 4,500 input tokens in my tests, so judging 20 claims costs about as much as seven answer calls.

**What you hand in:** each judged file and its trace from `p2 judge` (`answers/shared/<label>.judged.json` and `traces/judge-<label>.jsonl`), your own `answers/shared/<label>.calibration.tsv`, the `stretch-judge` table, and the five things in `EVAL.md` section 8.
**What earns the 25 points:** at least 20 claims labeled by you and judged by Claude, with the judged files and traces committed; a claim about the judge that matches its interval (17 agreements in 20 claims has an interval of about 0.64 to 0.95, so "the judge is right 85% of the time" says more than 20 claims can show); and where the disagreements sit in the 3 by 3 table, with one or two of them read closely.

### 3. Cost and latency

Every Claude call you made is in a trace in `traces/`, with its tokens and how long it took.
This option asks whether a cheaper setup is good enough.

1. Run one cheaper variant on the practice queries, and commit it with its trace.
   Two that work well: the reranker over 10 candidates instead of 20 (copy `p2/retrievers/rerank.py` to `rerank_c10.py`, change the number of candidates, and run `uv run p2 run --corpus shared --queries practice --system rerank_c10`, 20 calls), or, if you answered with `rerank`, answering the 12 questions with `hybrid` (`uv run p2 answer --corpus shared --system hybrid`, 12 calls).
2. Run `uv run p2 score`.
   The `stretch-cost` table adds up every trace (calls, saved replies, input and output tokens, seconds, and the same per query), then lists every pair of systems on the same queries where at least one calls Claude, the cheaper one first, with the difference in MRR@10 and its paired interval next to the input tokens and seconds per query the cheaper one saves.
   Answers files on the same questions get the same comparison, on the share of questions whose `p2 verify` line is a PASS.
   If you want nDCG@10 instead of MRR@10, rename the block to `stretch-cost-ndcg` in both of its marker lines in `EVAL.md`.
   The table also lists the seconds of the systems that do not call Claude, from `traces/retrieval/` on your machine; git ignores that folder, so `p2 check` leaves that part out.

**What you hand in:** the cheaper variant's run file or answers file with its trace, the `stretch-cost` table, and the five things in `EVAL.md` section 8.
**What earns the 25 points:** the cost table for your own runs, one cheaper variant run on the practice queries and committed with its trace, and a judgment with numbers: what the variant saves per query, what it costs in quality with its interval, and whether you would make that trade, and for what use.

### 4. A second ablation, of a different kind

Your core ablation changed one thing; this one changes a thing of a different kind, so you learn about a second part of the pipeline.
The kinds are chunking; text processing or the tokenizer; the embedding model; the fusion method or its weights; and the reranker's candidates or passage length.
If your first ablation changed the tokenizer, for example, the second cannot be another tokenizer, but it can be the chunk size.

1. Write the hypothesis first, in `EVAL.md` section 8, in one sentence with a direction, and commit it before you run anything.
2. Make the variant its own system, as for the first ablation, and name the system it varies: add a line such as `BASE = "dense"` to the variant's file, or start the file's name with it, since `dense_potion.py` varies `dense` by the part of its name before the underscore.
3. Run it with `--ablation`, then `uv run p2 score`.
   The `own-ablation` table gives its intervals against every system, and the `own-ablation-queries` table lists the queries it helped and hurt against the system it varies, with the reciprocal rank on each side.

**What you hand in:** the hypothesis in `EVAL.md` section 8, committed before the run, the variant's file in `p2/retrievers/` and its run in `runs/own/ablation/`, the `own-ablation` and `own-ablation-queries` tables in `EVAL.md` section 5, and the five things in section 8.
**What earns the 25 points:** a second ablation of a different kind, its hypothesis committed before its run, its interval read honestly, and a look at the queries: read the two it helped most and the two it hurt most, and say whether your hypothesis explains them.

### 5. The grep agent from lab 12

In lab 12 you ran Claude Code itself as a retriever, searching the corpus with only the Grep, Glob and Read tools.
Here you put that agent beside your four systems on your own corpus.

1. Ask Claude Code to port `class/12-embeddings-and-retrieval/lab/starter/agent_search.py` from the course repo into `p2/retrievers/agent.py`, with `NEEDS_CLAUDE = True`.
   Its `search(text, k)` makes one call per query with `p2.claude.call(prompt, schema, model=cfg.model, cache_dir=claude.cache_folder(cfg), tools=claude.AGENT_TOOLS, cwd=paths.docs_dir(cfg.root, corpus.name))`, which runs `claude -p` on a fresh copy of `corpora/own/docs/` outside your repo, with those three tools fenced to that copy and nothing else, saves the reply, and records the tokens in the trace; the docstring of `p2/claude.py` explains it.
   The lab's schema returns up to 5 document ids; ask for up to 10, and drop any id that is not a document of your corpus.
   The agent sometimes replies with a path or a file name instead of an id, especially on Windows, so keep only the last part of each path and remove `.md` before that check.
2. Copy 10 lines of `eval/own/queries.tsv` into `eval/own/agent.queries.tsv`, a mix of classes and of both origins.
3. Try 2 of them first: put two lines in `.cache/agent2.tsv` and run `uv run p2 run --corpus own --queries .cache/agent2.tsv --system agent`, which writes to `.cache/extra/` and prints the input tokens per query.
4. Then run all 10: `uv run p2 run --corpus own --queries eval/own/agent.queries.tsv --system agent --stretch`, which reuses the two saved replies and writes `runs/own/stretch/agent.trec` and `traces/own-stretch-agent.jsonl`.
   Commit both, because `p2 check` checks the run against its trace, as it does for `rerank`.
   `--stretch` refuses a line that is not in your gold set, so the agent's scores can be compared with your other runs.
5. Run `uv run p2 score`.
   The `stretch-agent` table shows the agent and your four systems on those 10 queries, with recall@10, MRR@10, nDCG@10 and the Claude input tokens and seconds per query, and the paired interval of the agent against each system.

**The cost, in plain numbers.**
This is the most expensive option.
In lab 12 the agent read about 54,000 input tokens per query inside the course repo, on 260 short articles, and the course's `CLAUDE.md` files, which Claude Code reads in the folder it runs in and in every folder above it, were 9,000 to 16,000 of those.
`p2` runs the agent on a fresh copy of your documents outside your repo, so it loads no `CLAUDE.md` and cannot open your gold set, which keeps the comparison honest and every call a little cheaper.
On a corpus of 271 USGS chapters, run the way `p2` runs it now (a fresh copy of the documents, each tool fenced to that copy), it read about 22,500 to 27,700 input tokens per query in my tests, and my research measured 120,000 to 245,000 per question on a corpus of 5,183 abstracts.
So 10 queries on your corpus can be half a million to two and a half million input tokens, more than a full reranker pass over all your own queries.
Try 2 queries first, look at the input tokens per query that `p2 run` prints, and spread the rest over more than one day if you need to.

**What you hand in:** `p2/retrievers/agent.py`, `eval/own/agent.queries.tsv`, `runs/own/stretch/agent.trec` with `traces/own-stretch-agent.jsonl`, the `stretch-agent` table, and the five things in `EVAL.md` section 8.
**What earns the 25 points:** the agent run on 10 of your own queries, committed with its trace; its comparison with the four systems on the same queries; and a judgment on quality per token: with 10 queries most intervals include zero, so say what 10 queries can and cannot show, and what the agent would cost on your whole gold set.

### 6. Do model-written queries favor keyword search?

A query a model drafts from a document tends to copy that document's words, which is exactly what keyword search rewards.
This option tests that on your own gold set.

1. Have at least 20 judged queries with `origin` `hand` and at least 20 with `origin` `claude` in `eval/own/`, judged by the same relevance rule.
   That is 10 more hand-written queries than the core asks for, or more if you have fewer than 20 drafted ones.
2. Write your prediction in `EVAL.md` section 8 before you score: will `bm25`'s lead over `dense` be larger on the claude queries, and by about how much?
3. Run your systems again (`uv run p2 run --all --with-claude` asks before the reranker's calls), then `uv run p2 score`.
   The `own-origins` table in `EVAL.md` section 4 shows each system's mean on each group and the difference, hand minus claude, with a 95% interval from resampling the queries within each group.
   Below it is the number that answers the question, `bm25` minus `dense` on the claude queries minus the same on the hand queries, with its interval, and the mean content-word overlap of each group: the share of a query's content words that its best relevant document contains, which is the likely mechanism.
   For comparison, that overlap is 0.95 for the shared practice set's identifier queries, 0.11 for its paraphrases and 0.51 for its mixed queries.

**What you hand in:** at least 20 judged `hand` and 20 judged `claude` queries in `eval/own/`, your prediction in `EVAL.md` section 8, committed before you scored, the `own-origins` table in `EVAL.md` section 4, and the five things in section 8.
**What earns the 25 points:** at least 20 judged queries of each origin, the prediction written first, the interval of the interaction read honestly, and the overlap used as evidence for or against the mechanism, along with what else differs between your two groups, such as their classes and their length.

---

## How the 150 points are planned

This is how I plan to grade.
Canvas is the record, and if I change this table before the deadline I will announce it there.

| Part | Points | What earns it |
|---|---|---|
| Stage 1: the systems | 30 | `dense`, `hybrid` and `rerank` built and run on the shared corpus, run files for the practice and test queries committed and reproducible, CI passing |
| Stage 1: the scores | 10 | the practice scores with paired intervals in `EVAL.md`, and what they let you claim |
| Stage 1: cited answers | 15 | answers to the 12 questions, quotes verified, the 4 unanswerable ones declined |
| Stage 2: your corpus | 10 | at least 200 documents, size rules met, a clean license report |
| Stage 2: your gold set | 15 | at least 30 queries, at least 10 `hand`, each with a relevant document, and your relevance rule written down |
| Stage 2: the four systems | 10 | `bm25`, `dense`, `hybrid` and `rerank` run on your corpus, scored, with intervals |
| Stage 2: one ablation | 10 | one change, a hypothesis written first, a committed run, and a claim that matches the interval |
| Failure analysis | 10 | at least five failures, read and sorted by cause |
| Cross-corpus comparison | 10 | whether the winner changed and a tested or honestly labeled explanation |
| `DECISIONS.md` | 5 | five prompts answered in your own words, with specifics |
| One stretch | 25 | 498E: one of the six options in "One stretch" above, each graded the same way; 598E: the rider, required |

That is 55 points for stage 1, 45 for stage 2, 25 for the analysis and 25 for the stretch.
The hidden test queries are graded as complete and reproducible only, and their score is feedback.

For 598E the rider is the stretch: you do not do the other options, and doing one as well adds no points, because the project is capped at 150.
Whichever stretch you do, write it up in `EVAL.md` section 8.

I would rather say this now than have you find out later: without a stretch, the most this project can reach is 125 of 150, and with the video it is 175 of 200.
That is 87.5% on this project, so here an A needs a stretch.
The stretch is worth doing properly and not worth doing in the last hour.

---

## The video

The video is worth 50 points, and it goes in the same Canvas discussion post as your repo URL.
Record 5 to 10 minutes, with any tool and any quality; a phone or a screen recording with your voice over it is fine.
Cover three things, in whatever order feels natural:

1. **The project.**
   What you built and what you found: show the repo, one table from `EVAL.md`, and say what the numbers do and do not let you claim.
2. **Your observations.**
   What surprised you, which system won where, and what you would not have guessed from lecture 12.
3. **The challenges and how you solved them.**
   A real one, in your words.
   Where you overruled the agent is a good story to tell here.

I am asking for this because the code and the runs can come from an agent and the video cannot.
It is your voice, your reasons and your story, and that matters more when a model wrote most of the code.
I care about the story and not about production quality.

---

## The deadline

Everything is due **Tuesday, October 27, at 11:59 pm**.
There are no checkpoints, because I would rather you finish the whole thing by the deadline than turn something in halfway.
On Canvas you make one post in the Project 2 discussion, with the URL of your repo and your video.
Push your last commit before you post.

Late work loses 20%, and nothing is accepted more than a week late.
Canvas is the system of record for the deadline.
If Canvas and this file ever disagree, Canvas wins, and please tell me so I can fix it.

---

## Ground rules

- **Using Claude Code is the point**, not something to disclose or apologize for.
  This is a course about driving agents well.
- **Your repo is public, and so is its history.**
  Do not commit API keys, passwords, private information about people (the corpus rules above say what that covers), or other people's private documents.
  A secret you delete in a later commit is still readable.
  You never need an API key in this project, because `claude -p` uses your Claude Code sign-in.
- **Commit only text you may publish**, with its license recorded.
- **The numbers must come from your code.**
  Run files, their traces, answers files, judged files, `results/results.json` and the tables in `EVAL.md` are written by commands.
  `p2 check` runs your systems again, checks the reranker's runs against their traces, and recomputes the scores and tables, and I also run your retrievers on documents you have not seen and score your test runs.
  Do not edit those files by hand, and do not hard-code answers.
- **Write the judgment parts yourself:** your `hand` queries, your relevance labels, `DECISIONS.md`, the prose in `EVAL.md`, `PREREG.md`, and for stretch option 2 your verdicts in the calibration file.
  The agent can find candidates and explain numbers.
- **Tune on the practice queries, never on the test queries.**
  You cannot see the test judgments, so the test score is the honest check on whether your pipeline generalizes.
- **What to edit and what not to:** the three retrievers and any system you add, `prompts/answer.txt`, `prompts/judge.txt`, `p2.toml`, your corpus and gold set, and your own writing are yours.
  The rest of `p2/`, `tests/`, the shared corpus and queries, the workflows, and the license skill belong to the course, and the instructor's copy of the checker is what grades you, so changing yours cannot help.
  `CLAUDE.md` has the exact list.
  If you think one of those files has a bug, tell me; that is a bug too, and I would rather fix it for everyone.

---

## When it breaks

- **CI is red.**
  Run `uv run p2 check` on your own machine, and read the `FAIL` lines; each one says what to do next.
  The usual causes are a committed run that no longer reproduces because you changed your code or `p2.toml` after generating it (run it again and commit the new file), a reranker run committed without its trace in `traces/`, a `results/results.json` or `EVAL.md` table that is out of date (run `uv run p2 score`), a carriage return in a file under `corpora/` (set your editor to LF line endings), and a failing license or a corpus over the size limits.
  A `TODO` line is not a failure.
- **A model download fails or hangs.**
  Check your internet connection and run the warm-up again; files that finished downloading are kept.
  A line about "unauthenticated requests to the HF Hub" is only a notice.
  On a Mac on the Mines network, if every download says "Could not resolve host" while your browser works, iCloud Private Relay is a known cause: in System Settings, open Wi-Fi, Details for the Mines network, and turn off Limit IP Address Tracking.
- **`claude -p` fails or asks you to sign in.**
  Open `claude` once in a terminal and sign in, then run the command again.
  `p2/claude.py` removes `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the child process, and prints a line when it does, so a key left over from another course is never billed for your runs.
  If `claude` is not found, check that `claude --version` works in the same terminal.
  On Windows, `p2` refuses a `claude.cmd` from an npm install, because Windows cannot pass a multi-line prompt through it safely; install Claude Code with the official installer from `setup.md` instead.
- **You hit your usage limit.**
  Wait for the reset, do the parts that do not need Claude in the meantime (the corpus, the gold set, the analysis), and spread the Claude steps over more days.
- **Windows or an Intel Mac.**
  The template is written to run on both, and its own tests run on Windows and macOS, but those two are the setups I have tested least.
  On Windows, use Git Bash for every command.
  `uv sync` picks a Python below 3.14 on purpose, because that range keeps Intel Macs working, so leave `.python-version` and the `requires-python` line in `pyproject.toml` alone.
  An Intel Mac needs macOS 13 or newer, because that is the oldest system the embedding library ships for.
  A Windows laptop with an ARM processor (a Snapdragon, say) has no packages of its own for the embedding library, so run `uv sync --python cpython-3.12-windows-x86_64-none` once, which uses the Intel version of Python under Windows' emulation; tell me if it fails.
  If something fails, send me the exact output.
- **The stubs raise `NotImplementedError`.**
  That is expected until you write the retriever; the message names the lab file that shows the idea.

Still stuck after 20 minutes?
Come to office hours: Tue and Thu, 9 to 10 am and 2 to 3 pm, CTLM 246L.
Bring the exact error message, because the words on the screen usually save us half the debugging time.

And if anything in this brief is unclear or seems wrong, tell me.
That is a bug too, and I would rather fix it than have you guess.
