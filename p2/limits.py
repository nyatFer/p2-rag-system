"""The size rules for your own corpus, in one place: `p2 check` enforces them and `p2 ingest` reports against them.

Why the token limit is 1,000,000:
- Bring-your-own-domain is the default, and long documents (papers, reports, manuals) are common, so a
  limit sized for short documents would push most of the class under the 200-document floor.
- The cost is CI time, and it is paid once. The first CI run after `dense` runs are committed encodes the
  whole corpus cold: the shared corpus, about 550,000 tokens, took 3 to 7 minutes on GitHub's
  ubuntu-latest runner (build-log.md, Wave 3), so a corpus near 1,000,000 tokens takes about twice that.
  Later runs reuse the vector cache (6 s for the shared corpus), and the check has 45 minutes
  (`timeout-minutes` in .github/workflows/p2-check.yml, which CI_MINUTES must match).

The floor and the limit meet at about 3,500 words a document: 200 documents of about 3,500 words is about
1,000,000 tokens. So a corpus of long documents reaches 200 by splitting them, which is what
`p2 ingest --part-pages 5` does for PDFs.

Only the Python standard library is used, so `p2 ingest` can import this without the rest of `p2`.
"""

OWN_MIN_DOCS = 200
OWN_MAX_BYTES = 25 * 1024 * 1024
OWN_MAX_FILE_BYTES = 10 * 1024 * 1024
TOKENS_PER_WORD = 1.4  # the estimate: words times 1.4
OWN_TOKEN_WARNING = 800_000  # p2 check says the first CI encode will be slow
OWN_TOKEN_LIMIT = 1_000_000  # p2 check fails above this
CI_MINUTES = 45  # the student CI job's timeout-minutes
SUGGESTED_PART_PAGES = 5  # what the brief and ingest suggest when long documents leave a corpus under the floor


def estimated_tokens(words: int) -> int:
    """The token estimate every size rule uses: words times 1.4, rounded down."""
    return int(words * TOKENS_PER_WORD)
