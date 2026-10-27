"""The BM25 tokenizers.

"lab" is the 12 lab's tokenizer: lower-case, runs of [a-z0-9], stopwords dropped. It splits a code
such as 75.403 into 75 and 403, so a query for "75.403" also matches every document that mentions
section 403 of any part, or the number 75.

"codes" (the default) keeps such codes whole and also emits their parts: 75.403 becomes the tokens
75.403, 75 and 403; syn-4127 becomes syn-4127, syn and 4127; sync.chunk_size_kb becomes
sync.chunk_size_kb, sync, chunk, size and kb. The whole code is a rare token, so an exact match
scores high, and the parts still match documents that write the code another way.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an and are as at be but by do does for from how i if in is it my of on or so that the this to was what when where which who why with you your".split()
)
PART = re.compile(r"[a-z0-9]+")
CODE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokens_lab(text: str) -> list[str]:
    """The 12 lab's tokens: [a-z0-9]+ runs of the lower-cased text, without stopwords."""
    return [t for t in PART.findall(text.lower()) if t not in STOPWORDS]


def tokens_codes(text: str) -> list[str]:
    """Codes joined by . _ or - kept whole, followed by their parts; stopwords dropped."""
    out = []
    for match in CODE.finditer(text.lower()):
        token = match.group()
        parts = PART.findall(token)
        if len(parts) > 1:
            out.append(token)
        out.extend(p for p in parts if p not in STOPWORDS)
    return out


TOKENIZERS = {"lab": tokens_lab, "codes": tokens_codes}


def tokens(text: str, tokenizer: str = "codes") -> list[str]:
    """Tokens of `text` with the named tokenizer ("codes" or "lab")."""
    try:
        return TOKENIZERS[tokenizer](text)
    except KeyError:
        raise ValueError(f'Unknown tokenizer {tokenizer!r}; use "codes" or "lab".') from None
