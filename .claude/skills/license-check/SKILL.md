---
name: license-check
description: Walk the student through every failing or flagged document in their own P2 corpus - open the source page, find and quote the license text, decide with the student, then record `reviewed: <reason with the quote>` in the manifest notes or remove the document. Never marks a document reviewed without a quote from the source. Use when `uv run p2 license` reports FAIL or FLAG, when `p2 check` fails on "own corpus licenses", when the student says "run the license-check skill", "walk me through my license failures", or asks whether a document may go in the repo.
---

# License check - one document at a time

The repo is public, and everything in `corpora/own/` is republished the moment the student pushes.
`uv run p2 license` gathers evidence and flags problems, but it cannot prove a license, so each failing or flagged document needs a person to read the source.
Your job is to do the legwork with the student: open the source, find the license sentence, quote it exactly, and help them decide.
The decision and the responsibility stay with the student, and you are not a lawyer, so say what the page says and leave the legal conclusion out.

## The one rule

**Never write `reviewed:` for a document unless you have a quote from the source that you or the student actually read.**
A quote is the exact words from the page, in quotation marks, with the URL where they appear.
If you cannot reach the page, say so and ask the student to open it and paste the license sentence.
If the student says "just mark it reviewed", explain kindly why you will not: a GitHub takedown notice gives about one business day to remove the document it names, and the quote is what shows the decision was made on evidence.
Then offer the two honest routes, which are to find the quote together or to remove the document.
Removing a document is always fine, and it costs little, because the count of 200 includes parts of long documents.

## Start

1. Run `uv run p2 license` and read `LICENSES.md`.
   Say how many documents are failing, how many are flagged, and that you will take them one at a time, failing ones first.
2. If any `source` holds a DOI, an arXiv id or a PubMed Central id, offer `uv run p2 license --online`.
   It asks Crossref, arXiv and the PMC open-access dataset what license they publish, and a mismatch with the manifest counts as a failure.
   It needs the internet and runs on the student's machine only, never in CI.
   "Could not check" means the service did not answer, and it is not a failure.
3. Do not start editing yet.
   Show the first document and its evidence.

## For each document

**1. Show the evidence.**
Give the docid, title, source, the `license` in the manifest, and the reason from the report: a license problem, the matching lines from the text scan, or an online mismatch.
Read the matching lines in `corpora/own/docs/<docid>.md` so you know what they say in context.
A hit on "a line ingest removed" is in the manifest's `notes` (`cleaning removed: ...`): ingest saw it in the raw file, for example a publisher footer on every page, and its cleaning pass dropped it, so open the original file or the source page to read it.
A "mentions" entry in `LICENSES.md` (a publisher's name with no words about rights on its line, such as "see ASTM E2500") is not a flag and needs nothing from the student.

**2. Open the source.**
The `source` column is a URL or a citation.
Fetch the page, or ask the student to open it if you cannot.
These addresses usually lead to the license statement:

- A DOI: `https://doi.org/<doi>`, then look for "License", "Copyright", "Rights and permissions" or the footer of the article page.
- An arXiv id: `https://arxiv.org/abs/<id>`, where the license link sits under the download options on the right.
- A PubMed Central id: `https://pmc.ncbi.nlm.nih.gov/articles/<PMCID>/`, where the copyright and license statement is near the title and again at the end of the article.
- A government report: the agency's page for the report and its copyright or usage page, for example `https://www.nist.gov/open/license` or `https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits`.
- Wikipedia: the footer of the page says "Text is available under the Creative Commons Attribution-ShareAlike 4.0 License".
- The student's own writing: no page to open, so ask whether anyone else owns it, for example a publisher they signed over copyright to or an employer.

**3. Find and quote the license text.**
Copy the sentence that states the license or the copyright, word for word, and say which URL it came from.
A statement like "no license found on the page" is also a finding, and it means the work is still under default copyright, so it cannot go in the corpus.
Do not paraphrase the license into a quote.

**4. Decide with the student.**
Lay out what the page says and what each choice would mean, then let the student choose.
These are the usual cases:

- **The source states a license on the vocabulary list** (`cc-by-4.0`, `cc-by-sa-4.0`, `cc0-1.0`, `us-gov-public-domain`, `own-work`, `mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `public-domain`, and the other versions of CC BY and CC BY-SA from 2.0 to 4.0).
  Set the `license` column to that exact value, and record the quote if the document was flagged.
  For CC BY-SA, remind the student that the cleaned text stays BY-SA, and the manifest already records that.
- **The source says NC, ND, "all rights reserved", a publisher's own terms, or nothing at all.**
  The document cannot go in the repo.
  Remove it, and before you do, ask whether an openly licensed version of the same work exists, such as a CC BY copy on arXiv or in the author's institutional repository.
  If there is one, check the license of that version the same way, because it is a different source.
- **A flag that is a false alarm.**
  A publisher's name inside a citation, or "ISO" as a standard that the text mentions, does not make the document the publisher's.
  The reason should still point at the document's own rights statement, for example the agency page that says its employees' works are not under copyright.
- **A government report with third-party material, a contractor's notice or a "courtesy of" credit.**
  The US government's text is public domain, but photos, figures and quoted passages inside it may not be, and some reports are written by contractors.
  A contractor's notice ("prepared as an account of work sponsored by an agency of the United States Government", "under Contract No. DE-...") means the report is the contractor's work, which the brief does not allow, so the usual decision is to remove it unless the report itself states an open license.
  Check whether the flagged line is about material that is in the corpus text, or only about an image that ingest did not keep.
  If it is a figure or a photo that is not in the committed text, the reason can say that, with the report's own sentence about third-party material as the quote.
  If a long quoted passage from a copyrighted source is in the text, the student should cut that part or remove the document.
- **The student's own published paper with a copyright line.**
  If they signed copyright over to the publisher, `own-work` is not accurate, and the paper cannot go in unless a preprint or accepted manuscript has an open license of its own.
  Ask what they signed.
- **An online mismatch.**
  The service published a license different from the manifest.
  Look at the source page to see which one is right, then correct the manifest or remove the document.
  A `reviewed:` note does not clear a mismatch, because the fix is to make the manifest say what the source says.

**5. Record it.**
Edit `corpora/own/manifest.tsv`, which is tab-separated with the columns `docid`, `title`, `source`, `license`, `notes`.
Keep every row at exactly five columns and put no tab inside a cell, or the report will fail the row.
Append to `notes` and keep what ingest wrote there (such as `file: raw/paper.pdf`), separated by a semicolon.
Write one line of this shape:

```
reviewed: <what you found>; the page says "<exact quote>" (<URL>)
```

For example: `reviewed: author line only, the article itself is open access; the page says "This is an open access article distributed under the terms of the Creative Commons Attribution License" (https://pmc.ncbi.nlm.nih.gov/articles/PMC1182327/)`.
To remove a document, delete its row from the manifest and its file from `corpora/own/docs/`, run `uv run p2 ingest --report` so `corpora/own/INGEST.md` lists only the documents that are left, and tell the student you did it.
If `eval/own/qrels.txt` already mentions that docid, tell the student to take those judgments out too, because a judgment for a document that no longer exists fails `p2 check`.
Never reuse a removed docid for a different document.

**6. Check.**
Run `uv run p2 license` again after each document or small batch, and say what changed: for example "that was the last failing one, three flags left".

## Do not

- Do not edit a document's text to make a flag disappear.
  The flag is evidence about the source, and deleting the copyright line leaves the copyright in place.
- Do not change a `license` to a value you did not see on the source page just to make the tool pass.
- Do not write a quote from memory.
  If you remember what a site says, still fetch it, because licenses change and some sites say different things on different pages.
- Do not run `uv run p2 license --online` from CI or in a workflow file.
- Do not tell the student a document is "safe" or "legal".
  Say what the evidence shows, and that the tool and you gather evidence while they decide.

## When it is green

`uv run p2 license` exits 0 and `uv run p2 check` shows PASS on the license line.
Say it plainly: a green report means nothing here looked wrong to a program and to a careful read of the sources, and the student still answers for what they published.
Mention that every `reviewed:` note is public in the repo and in `LICENSES.md`, so each one should be a reason a stranger could follow.
If any rule here felt wrong for their corpus, they can tell the instructor, who wants to hear it.
