# The shared corpus: 30 CFR Chapter I

This folder holds the federal mine safety and health rules that the Mine Safety and Health Administration (MSHA) enforces: Title 30 of the Code of Federal Regulations, Chapter I, parts 1 to 104.
Each file in `docs/` is one section of those rules, and `manifest.tsv` records where each one came from and under what terms.

## Where the text comes from

The text was downloaded from the eCFR versioner API as the full Title 30 XML for one date, 2026-10-01: `https://www.ecfr.gov/api/versioner/v1/full/2026-10-01/title-30.xml`.
The download was made on 2026-10-04, when the eCFR reported Title 30 as up to date as of 2026-10-01.
Every document's `source` in `manifest.tsv` is the eCFR's point-in-time page for that section on that date, for example `https://www.ecfr.gov/on/2026-10-01/title-30/section-75.403`, so you can compare a file with the page it came from.
The rules change from time to time, and this copy stays fixed at 2026-10-01 so that everyone's scores are comparable.

## Why you may copy it

The rules are a work of the United States Government, and US copyright law puts such works in the public domain.
Title 17 of the United States Code, section 105, says: "Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise." (quoted from `https://www.govinfo.gov/about/policies`, read 2026-10-04).
That is why every row of `manifest.tsv` has the license `us-gov-public-domain`.
Some sections name an outside standard, such as an ASTM, ANSI, ISO or SAE document, that the rule incorporates by reference.
The corpus contains only the government's rule text that cites those standards by name, never the text of the standards themselves, and the 24 rows that name such a body say so in a `reviewed:` note.

## What it is not

The eCFR is not the official text of the rules.
In the words of the National Archives page about it: "The e-CFR is an editorial compilation of CFR material and Federal Register amendments, not an official legal edition of the CFR." (`https://www.archives.gov/federal-register/cfr/about-ecfr`, read 2026-10-04).
This corpus is a teaching dataset for building and measuring a search system.
It is not legal or compliance advice, and an answer your system gives from it is not a statement of what any mine must do.
Anyone who needs the rules for real work should read the official edition of the CFR and the Federal Register at `https://www.govinfo.gov/`.

## How the files were made

The course turned each section of the XML into one Markdown file, with these changes and nothing else:
- The document id is `cfr30-` plus the section number in lower case, for example `cfr30-75.403` and `cfr30-56.5001t`.
- The first line is `# ` plus the section number and heading without the section sign, for example `# 75.403 Maintenance of incombustible content of rock dust.`, followed by a blank line and the body.
- Each paragraph of the body is one line, with a blank line between paragraphs.
- Each table row is one line, with its cells joined by ` | `.
- History citations (the bracketed Federal Register references at the end of a section), section authority lines, links to pending amendments, effective-date notes, images and formulas that exist only as images are left out.
- Footnote marks are left out, and the text of each footnote is kept.
- Fractions are written with a slash (`1/2`), and exponents are written inline (`mg/m3`, `10^5`).
- Sections marked `[Reserved]` and one section with no text (75.160) are left out, as are the appendices, which are not sections.

The result is 2,484 documents with about 392,000 words: half of the sections have 68 words or fewer, and the longest has about 3,600.
Many rules appear more than once in nearly the same words, because parts 56 (surface metal and nonmetal mines), 57 (underground metal and nonmetal mines), 75 (underground coal mines) and 77 (surface coal mines) each have their own version of a rule.
The shared gold sets count all such parallel sections as relevant when a query does not say which kind of mine it means.
