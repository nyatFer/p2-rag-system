"""p2 ingest: turn a folder of your own documents into a clean corpus.

    uv sync --group ingest
    uv run p2 ingest ../p2-raw --license cc-by-4.0 --source "arXiv, CC BY papers"
    uv run p2 ingest ../p2-raw/2024 --license us-gov-public-domain --source "https://pubs.usgs.gov/periodicals/mcs2024/{name}"
    uv run p2 ingest ../p2-raw --part-pages 5 --license cc-by-4.0 --source "..."   long PDFs as 5-page parts
    uv run p2 ingest --report      rewrite INGEST.md after you removed documents by hand

What it does, in order:

1. Walks the folder and reads each file.
   PDF goes through pypdf, HTML through trafilatura (favor_recall=True), and .md and .txt are read as they are.
   Word and PowerPoint files are not read: open them and use Save As or Export to make a PDF.
2. Runs the cleaning pass on every document.
   The pass is not optional, because the quote check in `p2 verify` compares what Claude copies with the text you stored.
   If the stored text still has Markdown marks, ligatures such as the single character "fi", or a hyphen left over from a line break, a perfectly honest quote fails the check.
   So the pass strips markup, replaces ligatures and soft hyphens, re-joins words hyphenated at a line break,
   drops lines repeated at the top or bottom of at least 30 percent of the pages (running headers, footers, page numbers),
   and drops a trailing reference list.
3. Splits a document of more than about 15,000 words into parts of about 10 pages and puts the page range in the id,
   for example `nasa-handbook__p041-050`, because a 300-page handbook is relevant to every query and so says nothing about any one of them.
   With --part-pages N, every PDF longer than N pages is split into parts of N pages instead, whatever its length in words,
   with the same kind of id; each part counts as a document, which is how a corpus of long papers reaches the 200-document floor
   while staying under the token limit (p2/limits.py has the arithmetic). Web pages and text files are not affected.
4. Gives each document an id built from its file name, for example `Taylor 1994 - TN1297.pdf` becomes `taylor-1994-tn1297`,
   and a title from the PDF's metadata, the first heading or the first line; when several PDFs of one run share a
   metadata title (every chapter of a volume, say), each title starts with its file name.
   Ids use only lower-case letters, digits, dots, underscores and hyphens, and an id never changes once your qrels mention it.
5. Skips a document whose normalized text is identical to one already in the corpus, so running ingest twice on the same folder adds nothing the second time.
   A file whose text is already in the corpus cut another way (ingested whole before, and now with --part-pages, say)
   is skipped too, with the ids to remove first if you want it split again.
6. Writes `corpora/<into>/docs/<docid>.md`, appends one row per document to `corpora/<into>/manifest.tsv`
   (license from --license, or `unknown` until you fix it; source from --source, where {name} and {stem}
   stand for each file's name, or the file name),
   and writes `corpora/<into>/INGEST.md` with one line per document: words, characters per page and flags.

A document under about 400 characters per page is probably a scan with no text layer, and is flagged.
This tool does not do OCR.
If you need it, the opt-in route is described at the end of INGEST.md (pymupdf4llm plus rapidocr, which are not installed by this template).

Only the Python standard library is needed to import this module.
pypdf and trafilatura are imported when the first PDF or HTML file is read.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from p2 import limits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
INTO_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

SPLIT_WORDS = 15_000  # a document above this is split into parts
PAGES_PER_PART = 10  # parts of about this many pages
MIN_LAST_PART_PAGES = 4  # a shorter remainder is folded into the previous part
WORDS_PER_TEXT_PART = 4_000  # parts of unpaged documents (HTML, Markdown, text)
SCANNED_CHARS_PER_PAGE = 400  # below this a PDF is probably a scan
VERY_SHORT_WORDS = 50
MAX_ID_LENGTH = 64
RUNNING_LINE_SHARE = 0.3  # a line at the edge of at least 30 percent of the pages is furniture
RUNNING_LINE_EDGE = 3  # the first and last three non-empty lines of a page count as its edge
TOKENS_PER_WORD = limits.TOKENS_PER_WORD
TOKEN_WARNING = limits.OWN_TOKEN_WARNING
TOKEN_LIMIT = limits.OWN_TOKEN_LIMIT  # p2 check fails above this
MIN_DOCS = limits.OWN_MIN_DOCS
MIN_ID_TEXT = 200  # a document shorter than this (normalized) is too short to recognize inside another file

OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".odt", ".odp", ".rtf", ".pages", ".key"}
SHEET_EXTENSIONS = {".xls", ".xlsx", ".csv", ".tsv", ".ods", ".numbers"}
PDF_EXTENSIONS = {".pdf"}
HTML_EXTENSIONS = {".html", ".htm"}
TEXT_EXTENSIONS = {".txt"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
SKIP_NAMES = {"thumbs.db", "desktop.ini"}

WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}

# Characters that a model cannot be expected to type back: ligatures, soft hyphens, odd spaces.
_CHAR_MAP = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
        "\xad": "",
        "\xa0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u200b": "",
        "\u2060": "",
        "\ufeff": "",
        "\t": " ",
        "\x0b": "\n",
        "\x0c": "\n",
    }
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")

_REF_HEAD = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:\d+\.?\s*)?"
    r"(references|reference list|bibliography|literature cited|works cited|references and notes)"
    r"\s*:?\s*(?:\*\*|__)?\s*$",
    re.I,
)
_AFTER_REFS_HEAD = re.compile(r"^\s*(?:appendix|appendices|supplement\w*|acknowledg\w+)\b.{0,70}$", re.I)
_REF_LIKE = re.compile(r"\b(?:19|20)\d\d\b|\bdoi\b|https?://|\bet al\b|\bpp?\.\s*\d|\bvol\.|\bpmid\b", re.I)
_BRACKET_ENTRY = re.compile(r"^\s*\[(\d{1,4})\]\s*\S")

ODD_GLYPHS = {"\ufffd": "the replacement character", "(cid:": "(cid:...) codes", "\xfe": "the letter thorn", "\xbc": "the fraction one quarter"}


# ---------------------------------------------------------------------------
# Cleaning pass (pure functions, no third-party imports)
# ---------------------------------------------------------------------------


def clean_chars(text: str) -> str:
    """Make the characters quote-safe: ligatures, soft hyphens and odd spaces become plain text.

    A soft hyphen at the end of a line is removed together with the line break, so the word is whole again.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xad\n", "")
    text = text.translate(_CHAR_MAP)
    text = _CONTROL_RE.sub("", text)
    text = re.sub(r"(?<=\S)[ ]{2,}", " ", text)  # runs of spaces inside a line; indentation stays
    text = re.sub(r"[ ]+\n", "\n", text)
    return text


def dehyphenate(text: str) -> str:
    """Re-join words that a line break split with a hyphen: 'inter-\\nnational' becomes 'international'."""
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)


def tidy_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_HTML_TAG_NAMES = (
    "p|div|span|br|hr|a|b|i|u|s|em|strong|sup|sub|small|big|code|pre|kbd|mark|del|ins|strike|details|summary|"
    "table|thead|tbody|tfoot|tr|td|th|caption|ul|ol|li|h[1-6]|img|center|font|section|article|header|footer|"
    "nav|aside|figure|figcaption|blockquote|abbr|cite|dl|dt|dd|var|samp|tt|nobr|wbr|main|label|input|button|iframe"
)
_HTML_BLOCK_TAGS = re.compile(r"</?(?:p|div|li|tr|h[1-6]|ul|ol|table|section|article|blockquote|figure|dl|dt|dd|details|summary)(?:\s[^<>]*)?/?>", re.I)
_HTML_BR = re.compile(r"<br\s*/?>", re.I)
_HTML_TAG = re.compile(rf"</?(?:{_HTML_TAG_NAMES})(?:\s[^<>]*)?/?>", re.I)
_MD_URL = r"(?:[^()\s]|\([^()]*\))*(?:\s+\"[^\"]*\")?"
_MD_IMAGE = re.compile(rf"!\[[^\]]*\]\({_MD_URL}\)")
_MD_LINK = re.compile(rf"\[([^\]]*)\]\({_MD_URL}\)")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n(.*?)(?:^[ \t]*\1[ \t]*$|\Z)", re.S | re.M)
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)
_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&apos;": "'"}


def strip_markup(text: str, *, source: str = "md") -> str:
    """Remove Markdown and HTML marks so that the stored text is what a reader sees.

    Code (fenced blocks and `inline code`) keeps its content untouched, only the fences and backticks go.
    source is "md" for Markdown files and "html" for what trafilatura produced from a web page.
    """
    saved: list[str] = []

    def keep(content: str) -> str:
        saved.append(content)
        return f"\ue000{len(saved) - 1}\ue001"

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FENCE.sub(lambda m: keep(m.group(2).rstrip("\n")) + "\n", text)
    text = _INLINE_CODE.sub(lambda m: keep(m.group(2)), text)

    if source == "html":
        text = re.sub(r"\\?\[edit\\?\]", "", text)
        text = re.sub(r"(?<=\S)\\?\[(?:\d{1,3}|[a-z]|[a-z]+ needed|note \d+)\\?\]", "", text)
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|<>~$])", lambda m: keep(m.group(1)), text)  # an escaped mark is literal text
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = re.sub(r"<(https?://[^>\s]+)>", r"\1", text)
    text = _HTML_BR.sub(" ", text)
    text = _HTML_BLOCK_TAGS.sub("\n", text)
    text = _HTML_TAG.sub("", text)
    for entity, plain in _ENTITIES.items():
        text = text.replace(entity, plain)
    if source == "md":
        text = re.sub(r"^[ \t]*\[[^\]\n]+\]:[ \t]+\S.*$", "", text, flags=re.M)  # link reference definitions

    text = re.sub(r"^[ \t]*(?:>[ \t]?)+", "", text, flags=re.M)  # block quotes
    text = re.sub(r"^[ \t]*(?:[-*_][ \t]*){3,}$", "", text, flags=re.M)  # horizontal rules
    text = re.sub(r"^[ \t]*=+[ \t]*$", "", text, flags=re.M)  # setext heading underlines
    text = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]+", "", text, flags=re.M)  # heading marks
    text = re.sub(r"(?<=\S)[ \t]+#+[ \t]*$", "", text, flags=re.M)
    text = re.sub(r"^[ \t]*[-*+][ \t]+", "", text, flags=re.M)  # bullet marks

    text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)", r"\1", text)
    text = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"\1", text)
    text = re.sub(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])", r"\1", text)
    text = re.sub(r"(?<![\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])", r"\1", text)

    # table separator rows (| --- | :---: |) and the outer pipes of table rows
    text = re.sub(r"^[ \t]*\|[ \t:|-]*-[ \t:|-]*$", "", text, flags=re.M)
    text = re.sub(r"^[ \t]*\|[ \t]*", "", text, flags=re.M)
    text = re.sub(r"(?<=\S)[ \t]*\|[ \t]*$", "", text, flags=re.M)


    text = re.sub(r"\ue000(\d+)\ue001", lambda m: saved[int(m.group(1))], text)
    return text


def _shape(line: str) -> str:
    """A line's identity for repeat detection: digits collapsed, case and spacing ignored."""
    s = re.sub(r"[*_`#|>]", "", line).strip().lower()
    s = re.sub(r"\d+", "#", s)
    return re.sub(r"\s+", " ", s)


def strip_running_lines(pages: list[str]) -> tuple[list[str], list[str]]:
    """Drop page headers, footers and page numbers.

    A line counts as furniture when, with digits collapsed, it sits among the first or last three
    non-empty lines of at least 30 percent of the pages (and at least 3 pages).
    Documents under 4 pages are left alone, because there is not enough repetition to be sure.
    Returns the pages and the list of line shapes that were removed.
    """
    if len(pages) < 4:
        return pages, []
    seen: collections.Counter[str] = collections.Counter()
    for page in pages:
        lines = [ln for ln in page.splitlines() if ln.strip()]
        edge = {_shape(ln) for ln in lines[:RUNNING_LINE_EDGE] + lines[-RUNNING_LINE_EDGE:]}
        for shape in edge:
            if shape:
                seen[shape] += 1
    cutoff = max(3, RUNNING_LINE_SHARE * len(pages))
    bad = {shape for shape, count in seen.items() if count >= cutoff and len(shape) < 160}
    if not bad:
        return pages, []
    out = []
    for page in pages:
        lines = page.splitlines()
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(non_empty[:RUNNING_LINE_EDGE] + non_empty[-RUNNING_LINE_EDGE:])
        out.append("\n".join(ln for i, ln in enumerate(lines) if not (i in edge_idx and _shape(ln) in bad)))
    return out, sorted(bad)


def _looks_like_references(lines: list[str]) -> bool:
    body = [ln for ln in lines if ln.strip()]
    if len(body) < 3:
        return False
    hits = sum(1 for ln in body if _REF_LIKE.search(ln) or _BRACKET_ENTRY.match(ln))
    return hits >= 0.4 * len(body)


def drop_reference_list(pages: list[str]) -> tuple[list[str], int, str]:
    """Remove a trailing reference list. Returns (pages, characters removed, how it was found).

    Two ways to find it:
      a heading line named References, Bibliography, Literature cited or Works cited in the last 45 percent of the text,
      or a run of numbered entries [1], [2], [3] ... in order, for papers that have no heading.
    The removed part must look like references (years, DOIs, URLs, "et al.") or it is kept, so a chapter
    that happens to be called References in the middle of a book is not thrown away.
    The reference list ends at an Appendix, Supplement or Acknowledgments heading when one follows it.
    """
    index: list[tuple[int, int]] = []  # (page, line) for every line
    flat: list[str] = []
    for p, page in enumerate(pages):
        for i, line in enumerate(page.splitlines()):
            index.append((p, i))
            flat.append(line)
    total = len(flat)
    if total < 20:
        return pages, 0, ""

    start = end = None
    how = ""
    for i, line in enumerate(flat):
        if i > 0.55 * total and _REF_HEAD.match(line):
            stop = next((k for k in range(i + 1, total) if _AFTER_REFS_HEAD.match(flat[k])), total)
            if _looks_like_references(flat[i + 1 : stop]):
                start, end, how = i, stop, "heading"
                break
    if start is None:
        for i, line in enumerate(flat):
            if i < 0.3 * total or not re.match(r"^\s*\[1\]\s*\S", line):
                continue
            expected, last, starts = 2, i, [i]
            for k in range(i + 1, total):
                m = _BRACKET_ENTRY.match(flat[k])
                if m and int(m.group(1)) == expected:
                    expected, last = expected + 1, k
                    starts.append(k)
            if expected >= 6:
                # the last entry runs on for about as many lines as the others do
                gaps = sorted(b - a for a, b in zip(starts, starts[1:], strict=False))
                extra = max(0, min(3, gaps[len(gaps) // 2] - 1))
                end = last + 1
                while extra and end < total and not _AFTER_REFS_HEAD.match(flat[end]) and not _REF_HEAD.match(flat[end]):
                    end, extra = end + 1, extra - 1
                start, how = i, "numbered entries"
                break
    if start is None:
        return pages, 0, ""

    removed = sum(len(flat[k]) + 1 for k in range(start, end))
    if removed > 0.45 * sum(len(ln) + 1 for ln in flat):
        return pages, 0, ""
    drop = {index[k] for k in range(start, end)}
    out = []
    for p, page in enumerate(pages):
        kept = [ln for i, ln in enumerate(page.splitlines()) if (p, i) not in drop]
        out.append("\n".join(kept))
    return out, removed, how


def normalized_text(text: str) -> str:
    """The text lower-cased, with punctuation and whitespace runs as single spaces."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def normalized_hash(text: str) -> str:
    """Hash of the lower-cased text with punctuation and whitespace ignored, for exact-duplicate detection."""
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def odd_glyph_flags(text: str) -> list[str]:
    found = []
    for glyph, what in ODD_GLYPHS.items():
        n = text.count(glyph)
        if glyph == "(cid:" or glyph == "\ufffd":
            if n:
                found.append(f"{what} x{n}")
        elif n >= 3:
            found.append(f"{what} x{n}")
    return found


# ---------------------------------------------------------------------------
# Ids, titles, parts
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """A file name becomes a stable id: lower-case ASCII, digits, dots, underscores and hyphens."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    s = re.sub(r"_{2,}", "_", s)
    s = re.sub(r"\.{2,}", ".", s)
    s = re.sub(r"-*\.-*", ".", s)
    s = s.strip("-._")
    s = s[:MAX_ID_LENGTH].rstrip("-._")
    if not s:
        return ""
    if s.split(".")[0] in WINDOWS_RESERVED:
        s = "doc-" + s
    return s


def make_id(path: Path, taken: set[str], root: Path) -> str:
    """The id for one file: its name, then the parent folder's name, then a number, until it is unused."""
    def used(candidate: str) -> bool:
        # an id is also used when a part of it exists: "report__p001-010" uses up "report"
        return candidate in taken or any(t.startswith(candidate + "__") for t in taken)

    base = slugify(path.stem)
    if not base:
        digest = hashlib.sha1(path.name.encode("utf-8", "replace")).hexdigest()[:8]
        base = f"doc-{digest}"
    if not used(base):
        return base
    parent = slugify(path.parent.name) if path.parent != root else ""
    if parent:
        candidate = slugify(f"{parent}-{base}")
        if candidate and not used(candidate):
            return candidate
    n = 2
    while used(f"{base}-{n}"):
        n += 1
    return f"{base}-{n}"


def _first_line_title(text: str, limit: int = 160) -> str:
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 4:
            return line[:limit].rstrip()
    return ""


_FILE_EXT = r"\.(?:docx?|pdf|tex|indd|pptx?|dvi|qxd)\b"


def _usable_pdf_title(raw) -> str:
    """The PDF's Title metadata when it reads like a title; a file name glued to it is cut off
    ("mcs2025.pdf - Mineral Commodity Summaries 2025" keeps the second half)."""
    title = " ".join(str(raw or "").split())
    title = re.sub(rf"^\S+{_FILE_EXT}\s*[-:|]\s*", "", title, flags=re.I)
    title = re.sub(rf"\s*[-:|]\s*\S+{_FILE_EXT}$", "", title, flags=re.I)
    if len(title) < 8 or title.lower() in {"untitled", "untitled document", "title"}:
        return ""
    if re.search(rf"\S{_FILE_EXT}", title, re.I) or title.lower().startswith("microsoft word"):
        return ""
    return title[:200]


def pdf_metadata_title(path: Path) -> str:
    """The usable Title metadata of a PDF, read without reading its pages ("" when there is none)."""
    try:
        from pypdf import PdfReader

        return _usable_pdf_title((PdfReader(str(path)).metadata or {}).get("/Title"))
    except Exception:  # noqa: BLE001 - a PDF that cannot be opened is reported when it is read
        return ""


def name_title(path: Path) -> str:
    """A file name as words: "mcs2024-aluminum.pdf" gives "mcs2024 aluminum"."""
    return " ".join(re.split(r"[-_\s]+", path.stem)).strip()


def one_line(text: str, limit: int = 200) -> str:
    return " ".join(str(text).split())[:limit]


def min_last_part(per_part: int) -> int:
    """The shortest remainder that stays a part of its own: 4 pages for 10-page parts, 2 for 5-page parts."""
    return max(1, math.ceil(per_part * MIN_LAST_PART_PAGES / PAGES_PER_PART))


def page_blocks(n_pages: int, per_part: int = PAGES_PER_PART) -> list[tuple[int, int]]:
    """Blocks of per_part pages as (first, last) 1-based page numbers; a short remainder joins the last block."""
    starts = list(range(1, n_pages + 1, per_part))
    blocks = [(s, min(s + per_part - 1, n_pages)) for s in starts]
    if len(blocks) > 1 and blocks[-1][1] - blocks[-1][0] + 1 < min_last_part(per_part):
        last = blocks.pop()
        blocks[-1] = (blocks[-1][0], last[1])
    return blocks


def split_paragraphs(text: str, words_per_part: int = WORDS_PER_TEXT_PART) -> list[str]:
    """Split unpaged text at paragraph breaks into parts of about words_per_part words."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts, current, count = [], [], 0
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > words_per_part * 1.25 and count >= words_per_part * 0.5:
            parts.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += n
        if count >= words_per_part:
            parts.append("\n\n".join(current))
            current, count = [], 0
    if current:
        if parts and count < words_per_part * 0.25:
            parts[-1] += "\n\n" + "\n\n".join(current)
        else:
            parts.append("\n\n".join(current))
    return parts


# ---------------------------------------------------------------------------
# Readers (third-party imports happen here, not at import time)
# ---------------------------------------------------------------------------


class MissingDependency(RuntimeError):
    pass


@dataclass
class Extracted:
    title: str
    pages: list[str]  # one entry per PDF page; a single entry for unpaged documents
    paged: bool
    notes: list[str] = field(default_factory=list)


def read_pdf(path: Path) -> Extracted:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MissingDependency("pypdf") from exc
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception:  # noqa: BLE001 - pypdf raises several types for bad passwords
            ok = 0
        if not ok:
            raise ValueError("the PDF is password protected")
    pages, failed = [], 0
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one unreadable page should not lose the document
            pages.append("")
            failed += 1
    if not pages:
        raise ValueError("the PDF has no pages")
    meta_title = ""
    try:
        meta_title = _usable_pdf_title((reader.metadata or {}).get("/Title"))
    except Exception:  # noqa: BLE001
        pass
    title = meta_title or _first_line_title(clean_chars(pages[0]))
    notes = [f"{failed} page(s) could not be read"] if failed else []
    return Extracted(title=title, pages=pages, paged=True, notes=notes)


def read_html(path: Path) -> Extracted:
    try:
        import trafilatura
    except ImportError as exc:
        raise MissingDependency("trafilatura") from exc
    raw = path.read_bytes()
    text = trafilatura.extract(
        raw,
        output_format="markdown",
        include_tables=True,
        include_links=False,
        include_images=False,
        include_comments=False,
        favor_recall=True,
    )
    title = ""
    try:
        meta = trafilatura.extract_metadata(raw)
        title = one_line(getattr(meta, "title", "") or "")
        title = re.sub(r"\s+[-|]\s+Wikipedia$", "", title)
    except Exception:  # noqa: BLE001
        pass
    text = text or ""
    first = _first_line_title(text)
    if first.startswith("# "):
        title = first[2:].strip()
    return Extracted(title=title or first, pages=[text], paged=False)


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def read_markdown(path: Path) -> Extracted:
    text = _read_text_file(path).replace("\r\n", "\n").replace("\r", "\n")
    title = ""
    front = re.match(r"^---\n(.*?)\n---[ \t]*\n", text, re.S)
    if front:
        m = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", front.group(1), re.M)
        title = m.group(1) if m else ""
        text = text[front.end() :]
    heading = re.search(r"^#{1,6}[ \t]+(.+?)[ \t#]*$", text, re.M)
    if not title and heading:
        title = heading.group(1)
    text = strip_markup(text, source="md")
    return Extracted(title=one_line(title) or _first_line_title(text), pages=[text], paged=False)


def read_plain(path: Path) -> Extracted:
    text = _read_text_file(path)
    return Extracted(title=_first_line_title(text, 120), pages=[text], paged=False)


# ---------------------------------------------------------------------------
# One file through the whole pass
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    """One document that will be written: a whole file, or one part of a long one."""

    docid: str
    title: str
    body: str
    words: int
    chars_per_page: int | None
    flags: list[str]
    cleaning: str
    notes: str  # for the manifest
    digest: str


@dataclass
class FileResult:
    rel: str
    status: str  # "ok", "refused", "unsupported", "error", "empty", "duplicate"
    message: str = ""
    units: list[Unit] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (docid that would have been, existing id)
    title: str = ""  # the title every unit's title starts with
    pages: int = 0  # the page count of a PDF, 0 for unpaged files


def clean_document(extracted: Extracted, kind: str) -> tuple[list[str], dict]:
    """Run the cleaning pass over the pages of one document. Returns (pages, what was removed)."""
    info = {"running_lines": 0, "refs_chars": 0, "refs_how": ""}
    pages = [clean_chars(p) for p in extracted.pages]
    if extracted.paged:
        pages, bad = strip_running_lines(pages)
        info["running_lines"] = len(bad)
    pages = [dehyphenate(p) for p in pages]
    pages, removed, how = drop_reference_list(pages)
    info["refs_chars"], info["refs_how"] = removed, how
    pages = [tidy_blank_lines(p) if p.strip() else "" for p in pages]
    return pages, info


def _describe_cleaning(info: dict) -> str:
    bits = []
    if info["running_lines"]:
        bits.append(f"{info['running_lines']} running line(s) removed")
    if info["refs_chars"]:
        bits.append(f"reference list dropped, {info['refs_chars']:,} characters ({info['refs_how']})")
    return "; ".join(bits)


def _make_unit(docid, title, pages_text, n_pages, cleaning, notes) -> Unit:
    body = tidy_blank_lines(dehyphenate("\n".join(pages_text)))
    words = len(body.split())
    cpp = round(len(body) / n_pages) if n_pages else None
    flags = []
    if cpp is not None and cpp < SCANNED_CHARS_PER_PAGE:
        flags.append(f"scanned-looking ({cpp} characters per page, under {SCANNED_CHARS_PER_PAGE})")
    if words < VERY_SHORT_WORDS:
        flags.append(f"very short ({words} words)")
    flags.extend(odd_glyph_flags(body))
    return Unit(docid, one_line(title) or docid, body, words, cpp, flags, cleaning, notes, normalized_hash(body))


def removed_license_lines(raw_pages: list[str], clean_pages: list[str]) -> list[str]:
    """License evidence that only the raw text had: a publisher footer on every page, say, which the
    cleaning pass drops as a running line. Each comes back as `<kind> "<line>"` for the manifest notes,
    where `p2 license` still sees it."""
    try:
        from p2.license import scan_text
    except ImportError:
        return []
    kept = {h["kind"] for h in scan_text("\n".join(clean_pages))}
    found: dict[str, str] = {}
    for hit in scan_text("\n".join(clean_chars(p) for p in raw_pages)):
        if hit["kind"] not in kept and hit["kind"] not in found:
            found[hit["kind"]] = hit["line"][:120].replace('"', "'")
    return [f'{kind} "{line}"' for kind, line in found.items()]


def _page_parts(docid, title, pages, n_pages, per_part, cleaning, base_note) -> list[Unit]:
    """A paged document as parts of per_part pages, each with its page range in the id: `report__p011-020`."""
    units: list[Unit] = []
    width = max(3, len(str(n_pages)))
    for first, last in page_blocks(n_pages, per_part):
        block = pages[first - 1 : last]
        if not any(p.strip() for p in block):
            continue
        part_id = f"{docid}__p{first:0{width}d}-{last:0{width}d}"
        note = f"{base_note}; pages {first}-{last} of {n_pages}"
        units.append(_make_unit(part_id, f"{title} (pages {first}-{last})", block, last - first + 1, cleaning, note))
    return units


def build_units(docid: str, rel: str, extracted: Extracted, kind: str, part_pages: int | None = None) -> tuple[list[Unit], str]:
    """Clean one document and cut it into the units to be written. Returns (units, problem).

    part_pages (from --part-pages) splits a PDF longer than that many pages into parts of that many pages,
    whatever its word count; without it, only documents over SPLIT_WORDS words are split.
    """
    pages, info = clean_document(extracted, kind)
    cleaning = _describe_cleaning(info)
    n_pages = len(pages) if extracted.paged else 0
    total_words = sum(len(p.split()) for p in pages)
    title = extracted.title or docid
    base_note = f"file: {rel}"
    removed = removed_license_lines(extracted.pages, pages)
    if removed:
        base_note += "; cleaning removed: " + " | ".join(removed)

    if total_words == 0:
        if extracted.paged:
            return [], f"no text found in {n_pages} page(s), so this looks like a scan; see the OCR notes in INGEST.md"
        return [], "no text found"
    if extracted.paged and total_words < VERY_SHORT_WORDS and sum(len(p) for p in pages) / n_pages < SCANNED_CHARS_PER_PAGE:
        # almost nothing came out of a PDF: not worth a document, and almost surely a scan
        return [], f"only {total_words} words from {n_pages} page(s), so this looks like a scan; see the OCR notes in INGEST.md"

    def whole() -> list[Unit]:
        return [_make_unit(docid, title, pages, n_pages, cleaning, base_note + (f"; {n_pages} pages" if n_pages else ""))]

    if part_pages and extracted.paged:
        if len(page_blocks(n_pages, part_pages)) > 1:
            return _page_parts(docid, title, pages, n_pages, part_pages, cleaning, base_note), ""
        return whole(), ""  # not longer than part_pages, or only by a remainder too short to be a part of its own

    if total_words <= SPLIT_WORDS:
        return whole(), ""

    if extracted.paged and n_pages > PAGES_PER_PART:
        return _page_parts(docid, title, pages, n_pages, PAGES_PER_PART, cleaning, base_note), ""

    parts = split_paragraphs("\n\n".join(pages))
    width = max(2, len(str(len(parts))))
    if len(parts) == 1:
        return [_make_unit(docid, title, pages, n_pages, cleaning, base_note)], ""
    units: list[Unit] = []
    for n, part in enumerate(parts, 1):
        part_id = f"{docid}__part{n:0{width}d}"
        units.append(_make_unit(part_id, f"{title} (part {n} of {len(parts)})", [part], 0, cleaning, f"{base_note}; part {n} of {len(parts)}"))
    return units, ""


def process_file(path: Path, rel: str, docid: str, part_pages: int | None = None) -> FileResult:
    """Read and clean one file. Does not touch the corpus."""
    ext = path.suffix.lower()
    if ext in OFFICE_EXTENSIONS:
        return FileResult(
            rel,
            "refused",
            "Word and PowerPoint files are not read directly, because their text and layout depend on the program that made them. "
            "Open the file, choose File > Save As (or Export) > PDF, put the PDF in this folder, and run ingest again.",
        )
    if ext in SHEET_EXTENSIONS:
        return FileResult(
            rel,
            "refused",
            "A spreadsheet is rows of data, not a document. Leave it out, or turn the rows into sentences yourself and say so in DECISIONS.md.",
        )
    try:
        if ext in PDF_EXTENSIONS:
            extracted, kind = read_pdf(path), "pdf"
        elif ext in HTML_EXTENSIONS:
            extracted, kind = read_html(path), "html"
            extracted.pages = [strip_markup(extracted.pages[0], source="html")]
        elif ext in MARKDOWN_EXTENSIONS:
            extracted, kind = read_markdown(path), "md"
        elif ext in TEXT_EXTENSIONS:
            extracted, kind = read_plain(path), "txt"
        else:
            return FileResult(rel, "unsupported", f"file type {ext or '(none)'} is not one ingest reads (it reads PDF, HTML, Markdown and plain text)")
        units, problem = build_units(docid, rel, extracted, kind, part_pages)
    except MissingDependency:
        raise
    except Exception as exc:  # noqa: BLE001 - report any failure against the file, keep going with the others
        return FileResult(rel, "error", f"{type(exc).__name__}: {one_line(exc, 160)}")
    if problem:
        return FileResult(rel, "empty", problem)
    return FileResult(rel, "ok", units=units, title=extracted.title or docid, pages=len(extracted.pages) if extracted.paged else 0)


# ---------------------------------------------------------------------------
# The corpus on disk
# ---------------------------------------------------------------------------

MANIFEST_HEADER = "docid\ttitle\tsource\tlicense\tnotes"


def _find_root(args) -> Path:
    root = getattr(args, "root", None)
    if root:
        return Path(root)
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / "p2.toml").exists():
            return candidate
    return here


def _cell(text: str) -> str:
    return " ".join(str(text).split())


def read_manifest_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        if line.strip():
            ids.add(line.split("\t", 1)[0].strip())
    return ids


def append_manifest(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if not existing.strip():
        existing = MANIFEST_HEADER + "\n"
    elif not existing.endswith("\n"):
        existing += "\n"
    lines = ["\t".join(_cell(c) for c in row) for row in rows]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(existing + "".join(line + "\n" for line in lines))


def doc_body(path: Path) -> str:
    """A corpus document's text without its `# title` line."""
    text = path.read_text(encoding="utf-8")
    head, _, rest = text.partition("\n")
    return rest if head.startswith("# ") else text


def existing_hashes(docs_dir: Path) -> dict[str, str]:
    """normalized hash of the body of every document already in the corpus -> its id"""
    seen: dict[str, str] = {}
    if not docs_dir.is_dir():
        return seen
    for p in sorted(docs_dir.glob("*.md")):
        body = doc_body(p)
        if body.strip():
            seen.setdefault(normalized_hash(body), p.stem)
    return seen


def manifest_file_notes(path: Path) -> list[tuple[str, str]]:
    """(docid, notes) for every manifest row whose notes start with ingest's `file: <path>`."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        cells = line.split("\t")
        if len(cells) >= 5 and cells[4].strip().startswith("file: "):
            rows.append((cells[0].strip(), cells[4].strip()))
    return rows


def cut_another_way(path: Path, units: list[Unit], file_notes: list[tuple[str, str]], docs_dir: Path) -> list[str]:
    """When this file is already in the corpus cut into documents another way (whole, and now with --part-pages,
    say, or parts of another size), every document it is in; otherwise [].

    A candidate is a document whose manifest notes name a file of the same name, and it came from this file
    when all of its text is inside this file's text. The file counts as cut another way only when one of
    those documents differs from every new unit; when all of them are new units, this is the same cut, and
    the duplicate check adds just the missing parts (after an interrupted run, say).
    """
    name = _cell(path.name)
    candidates = [docid for docid, notes in file_notes if notes[len("file: ") :].startswith(name) or f"/{name}" in notes]
    if not candidates:
        return []
    digests = {u.digest for u in units}
    full = normalized_text(dehyphenate("\n".join(u.body for u in units)))
    inside, other_cut = [], False
    for docid in candidates:
        doc = docs_dir / f"{docid}.md"
        if not doc.is_file():
            continue
        body = doc_body(doc)
        text = normalized_text(body)
        if len(text) >= MIN_ID_TEXT and text in full:
            inside.append(docid)
            other_cut = other_cut or normalized_hash(body) not in digests
    return inside if other_cut else []


def write_doc(docs_dir: Path, unit: Unit) -> None:
    text = f"# {unit.title}\n\n{unit.body}\n"
    with open(docs_dir / f"{unit.docid}.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# INGEST.md
# ---------------------------------------------------------------------------

_ROW = re.compile(r"^\|\s*`?([a-z0-9][a-z0-9._-]*)`?\s*\|\s*([\d,]+)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*$")


def _read_report_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            rows[m.group(1)] = {
                "words": int(m.group(2).replace(",", "")),
                "cpp": m.group(3),
                "flags": "" if m.group(4) == "-" else m.group(4),
                "cleaning": "" if m.group(5) == "-" else m.group(5),
            }
    return rows


def _cell_md(text: str) -> str:
    return (_cell(text) or "-").replace("|", "\\|")


OCR_HELP = """\
## If a document is flagged as scanned

A scan is a picture of a page, so there is no text for ingest to read.
This template does not install an OCR tool, because it is large and most students never need it.
If you do need one, the opt-in route is pymupdf4llm with rapidocr, run on its own outside the template's environment:

```
uv run --no-project --with pymupdf4llm --with rapidocr python -c "import pymupdf4llm, sys, pathlib; pathlib.Path(sys.argv[2]).write_text(pymupdf4llm.to_markdown(sys.argv[1], use_ocr=True, force_ocr=True), encoding='utf-8')" scan.pdf scan.md
```

Put the resulting `.md` file in your folder and ingest the folder again.
It costs about 2 seconds a page, and OCR text needs a second look: it misreads ligatures, hyphens and numbers.
pymupdf4llm is AGPL-3.0 software, so use it only as a tool on your own machine, and do not add it to this repo's dependencies or copy its code into the repo.
"""

CHECK_HELP = """\
## What to check by eye

Ingest cleans text by rules, and rules miss things.
Take five minutes to open the shortest, a middle one and the longest document next to its original and read one page of each.

- Search the documents for `þ`, `¼`, `(cid:` and the replacement character.
  A hit usually means the PDF's font map is wrong, which no tool can repair, so treat it as a defect and leave that document out.
- Look at the end of a few papers.
  Ingest drops a trailing reference list when it finds a References heading or a run of numbered entries.
  A paper whose list has neither can keep it, and then every author and journal name matches queries it should not.
- Open the first lines of a few documents.
  Author lists, copyright pages and tables of contents are not removed automatically, and you may want to cut them.
- Compare one table with the PDF.
  A PDF prints footnote marks as small raised digits, and the converter glues them onto the word or number next to them: `ALUMINUM1` is the heading ALUMINUM with footnote 1, and `11210,000` can be 210,000 with footnote 11.
  Quotes still match the stored text, but a number read from it can be wrong, so check before you trust a figure in a query, a judgment or an answer.
- Fix the `license` column in the manifest for every document, then run `uv run p2 license`.
  It is `unknown` until you do, and `unknown` does not pass.
"""


def write_report(path: Path, rows: dict[str, dict], notices: list[tuple[str, str, str]], totals: dict) -> None:
    lines = [
        "# Ingest report",
        "",
        "Written by `uv run p2 ingest`.",
        "It lists every document in this corpus that ingest added, with what the cleaning pass did and anything that looks wrong.",
        'Running ingest again adds to the table and replaces the "last run" section, and `uv run p2 ingest --report` rewrites it for the documents in docs/ now, after you removed some.',
        "",
        "## Summary",
        "",
        f"- Documents: {totals['documents']} (the final check needs at least {MIN_DOCS})",
        f"- Words: {totals['words']:,}",
        f"- Estimated tokens (words times {TOKENS_PER_WORD}): {int(totals['words'] * TOKENS_PER_WORD):,} (`p2 check` warns above {TOKEN_WARNING:,} and fails above {TOKEN_LIMIT:,})",
        f"- Flagged documents: {totals['flagged']}",
        "",
        "## Documents",
        "",
        "| docid | words | characters per page | flags | cleaning |",
        "|---|---:|---:|---|---|",
    ]
    for docid in sorted(rows):
        r = rows[docid]
        lines.append(f"| `{docid}` | {r['words']:,} | {_cell_md(r['cpp'])} | {_cell_md(r['flags'])} | {_cell_md(r['cleaning'])} |")
    lines += ["", "## Not added in the last run", ""]
    if notices:
        for rel, _status, message in notices:
            sentence = message[:1].upper() + message[1:]
            lines.append(f"- `{rel}`: {sentence}" + ("" if sentence.endswith((".", "!", "?")) else "."))
    else:
        lines.append("Nothing was left out.")
    lines += ["", CHECK_HELP, OCR_HELP]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines).rstrip("\n") + "\n")


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _say(text: str = "", end: str = "\n", flush: bool = False) -> None:
    """print() that never fails on a console that cannot show a character (an old Windows code page, a redirect)."""
    try:
        print(text, end=end, flush=flush)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "replace").decode(enc), end=end, flush=flush)


PART_PAGES_HELP = (
    f"split every PDF longer than N pages into parts of N pages, each its own document (try {limits.SUGGESTED_PART_PAGES} when long"
    f" documents leave you under {limits.OWN_MIN_DOCS}); web pages and text files are not affected"
)


def part_pages_problem(value) -> str:
    """Why a --part-pages value cannot be used, or "" when it can."""
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return f"--part-pages takes a whole number of pages, 1 or more, for example --part-pages {limits.SUGGESTED_PART_PAGES}"
    return ""


def add_arguments(parser) -> None:
    """Add the p2 ingest options to an argparse parser (cli.py may use this)."""
    parser.add_argument("src_dir", help="folder with your documents (PDF, HTML, Markdown, text)")
    parser.add_argument("--into", default="own", help="corpus to add to, under corpora/ (default: own)")
    parser.add_argument("--license", default=None, help="license id for every document in this run, for example cc-by-4.0 (default: unknown)")
    parser.add_argument("--source", default=None, help="where these documents came from, a URL or a citation (default: the file name); {name} and {stem} stand for each file's name")
    parser.add_argument("--part-pages", type=int, default=None, metavar="N", help=PART_PAGES_HELP)
    parser.add_argument("--report", action="store_true", help="rewrite INGEST.md for the documents in docs/ now, without reading a folder")


def _arg(args, *names, default=None):
    for name in names:
        value = getattr(args, name, None)
        if value not in (None, ""):
            return value
    return default


def _walk(src: Path, skip_under: Path | None) -> list[Path]:
    files = []
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(src).parts
        if any(part.startswith(".") or part.startswith("~$") for part in rel_parts):
            continue
        if p.name.lower() in SKIP_NAMES:
            continue
        if skip_under is not None:
            try:
                p.resolve().relative_to(skip_under)
                continue
            except ValueError:
                pass
        files.append(p)
    return sorted(files, key=lambda p: p.relative_to(src).as_posix().lower())


def source_for(template: str | None, path: Path, rel: str) -> str:
    """The manifest source of one file: --source with {name}, {stem} and {path} filled in, or the file's path.

    In a URL, {name} and {path} are percent-encoded, so a space in a file name becomes %20."""
    if not template:
        return rel
    is_url = re.match(r"^https?://", template.strip(), re.I) is not None
    values = {"name": path.name, "stem": path.stem, "path": rel}
    for key, value in values.items():
        template = template.replace("{" + key + "}", quote(value) if is_url and key != "stem" else value)
    return template


def _notices_from(report: Path) -> list[tuple[str, str, str]]:
    """The "not added in the last run" lines of an existing INGEST.md, so --report keeps them."""
    if not report.exists():
        return []
    text = report.read_text(encoding="utf-8")
    m = re.search(r"^## Not added in the last run\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    out = []
    for line in (m.group(1) if m else "").splitlines():
        item = re.match(r"^- `([^`]+)`: (.+)$", line)
        if item:
            out.append((item.group(1), "kept", item.group(2)))
    return out


def write_current_report(report: Path, docs_dir: Path, report_rows: dict[str, dict], notices: list[tuple[str, str, str]]) -> tuple[int, int, dict]:
    """Write INGEST.md for the documents in docs/ now. Returns (documents, words, totals)."""
    present = {p.stem: p for p in docs_dir.glob("*.md")}
    rows = {k: v for k, v in report_rows.items() if k in present}
    total_words = 0
    for docid, path in present.items():
        words = len(path.read_text(encoding="utf-8").split())
        total_words += words
        rows.setdefault(docid, {"words": words, "cpp": "-", "flags": "", "cleaning": "not added by p2 ingest"})
    totals = {"documents": len(present), "words": total_words, "flagged": sum(1 for v in rows.values() if v["flags"])}
    write_report(report, rows, notices, totals)
    return len(present), total_words, totals


def refresh_report(root: Path, into: str) -> int:
    """`p2 ingest --report`: rewrite INGEST.md for the documents in docs/ now, without reading a folder."""
    corpus_dir = root / "corpora" / into
    docs_dir, report = corpus_dir / "docs", corpus_dir / "INGEST.md"
    if not docs_dir.is_dir():
        _say(f"corpora/{into}/docs/ does not exist yet, so there is nothing to report.")
        return 1
    documents, words, totals = write_current_report(report, docs_dir, _read_report_rows(report), _notices_from(report))
    _say(f"Rewrote corpora/{into}/INGEST.md: {documents} document(s), about {words:,} words (about {int(words * TOKENS_PER_WORD):,} tokens), {totals['flagged']} flagged.")
    return 0


def run(args) -> int:
    into = str(_arg(args, "into", default="own"))
    if not INTO_RE.match(into):
        _say(f"{into!r} is not a corpus name. Use lower-case letters, digits, hyphens and underscores, for example: own")
        return 1
    if getattr(args, "report", False):
        return refresh_report(_find_root(args), into)
    src_arg = _arg(args, "src_dir", "src", "srcdir", "folder", "path", "directory")
    if not src_arg:
        _say("Tell me which folder to read, for example: uv run p2 ingest ../p2-raw")
        return 1
    src = Path(src_arg)
    if not src.is_dir():
        _say(f"I could not find a folder called {src} (I looked for {src.resolve()}).")
        if not src.is_absolute():
            _say(f"A path is read from the folder you run p2 in, which is your P2 repo, so a folder beside the repo is ../{src.name}.")
        return 1

    root = _find_root(args)
    corpus_dir = root / "corpora" / into
    docs_dir = corpus_dir / "docs"
    manifest = corpus_dir / "manifest.tsv"
    report = corpus_dir / "INGEST.md"

    license_value = str(_arg(args, "license", default="unknown")).strip().lower() or "unknown"
    source_text = _arg(args, "source", default=None)
    part_pages = getattr(args, "part_pages", None)
    problem = part_pages_problem(part_pages)
    if problem:
        _say(problem + ".")
        return 1

    try:
        from p2 import license as _license  # my own module; guarded so ingest still works alone

        if license_value not in _license.ALLOWED and license_value != "unknown":
            ok, why = _license.check_license_value(license_value)
            _say(f"Note: the license {license_value!r} will not pass `p2 license` ({why}). Ingest will write it anyway, and you can fix it in the manifest.")
    except Exception:  # noqa: BLE001
        pass

    files = _walk(src, docs_dir.resolve() if docs_dir.exists() else None)
    if not files:
        _say(f"There is nothing to read in {src}. Put PDF, HTML, Markdown or text files there and run ingest again.")
        return 1

    kinds = {p.suffix.lower() for p in files}
    needs = []
    if kinds & PDF_EXTENSIONS:
        needs.append("pypdf")
    if kinds & HTML_EXTENSIONS:
        needs.append("trafilatura")
    for module in needs:
        try:
            __import__(module)
        except ImportError:
            _say(f"Ingest needs {module} to read some of these files, and it is not installed.")
            _say("Install the ingest tools with:  uv sync --group ingest")
            _say("Then run ingest again.")
            return 1

    docs_dir.mkdir(parents=True, exist_ok=True)
    taken = {p.stem for p in docs_dir.glob("*.md")} | read_manifest_ids(manifest)
    seen_hash = existing_hashes(docs_dir)
    file_notes = manifest_file_notes(manifest)

    # Many publishers put one title on every chapter of a volume (the USGS Mineral Commodity Summaries
    # call every chapter "Mineral Commodity Summaries 2024"). A title several files share says nothing
    # about any one of them, so those files start their title with their file name.
    meta = {p: pdf_metadata_title(p) for p in files if p.suffix.lower() in PDF_EXTENSIONS}
    shared_titles = {t for t, n in collections.Counter(t for t in meta.values() if t).items() if n > 1}
    if shared_titles:
        sharing = sum(1 for t in meta.values() if t in shared_titles)
        shown = "; ".join(sorted(shared_titles)[:2])
        _say(f"{sharing} PDF files share {'a title' if len(shared_titles) == 1 else 'titles'} in their metadata ({shown}), so each of their titles starts with the file name.")

    rows_out: list[tuple[str, str, str, str, str]] = []
    report_rows = _read_report_rows(report)
    notices: list[tuple[str, str, str]] = []
    added = 0
    recut: list[str] = []  # files already in the corpus, cut into documents another way
    long_pdfs = 0  # PDFs longer than the suggested part size, for the hint at the end

    _say(f"Reading {len(files)} file(s) from {src} ...")
    try:
        for path in files:
            rel = path.relative_to(src).as_posix()
            docid = make_id(path, taken, src)
            _say(f"  {rel} ", end="", flush=True)
            result = process_file(path, rel, docid, part_pages)
            if result.pages > limits.SUGGESTED_PART_PAGES:
                long_pdfs += 1
            if result.status != "ok":
                _say(f"-> not added: {result.message}")
                notices.append((rel, result.status, result.message))
                continue
            fresh = []
            for unit in result.units:
                if unit.digest in seen_hash:
                    result.duplicates.append((unit.docid, seen_hash[unit.digest]))
                else:
                    fresh.append(unit)
            if not fresh:
                same = result.duplicates[0][1]
                _say(f"-> already in the corpus (same text as {same}), skipped")
                notices.append((rel, "duplicate", f"The text is the same as `{same}`, which is already in the corpus."))
                continue
            earlier = cut_another_way(path, result.units, file_notes, docs_dir)
            if earlier:
                more = f" and {len(earlier) - 1} more" if len(earlier) > 1 else ""
                _say(f"-> not added: its text is already in the corpus as {earlier[0]}{more}, cut into documents another way")
                named = ", ".join(f"`{d}`" for d in earlier)
                notices.append((rel, "recut", f"Its text is already in the corpus, cut into documents another way, as {named}, so it was not added again. "
                                              "To cut it again, remove those documents and their manifest rows first."))  # fmt: skip
                recut.append(rel)
                continue
            if meta.get(path) and meta[path] in shared_titles and result.title == meta[path]:
                better = f"{name_title(path)} - {result.title}"
                for unit in fresh:
                    if unit.title.startswith(result.title):
                        unit.title = one_line(better + unit.title[len(result.title) :])
            for unit in fresh:
                if unit.docid in taken:  # a part id that clashes with an existing document
                    n = 2
                    while f"{unit.docid}-{n}" in taken:
                        n += 1
                    unit.docid = f"{unit.docid}-{n}"
                taken.add(unit.docid)
                seen_hash[unit.digest] = unit.docid
                write_doc(docs_dir, unit)
                rows_out.append((unit.docid, unit.title, source_for(source_text, path, rel), license_value, unit.notes))
                report_rows[unit.docid] = {
                    "words": unit.words,
                    "cpp": "-" if unit.chars_per_page is None else str(unit.chars_per_page),
                    "flags": "; ".join(unit.flags),
                    "cleaning": unit.cleaning,
                }
                added += 1
            flagged = [u for u in fresh if u.flags]
            suffix = f" in {len(fresh)} parts" if len(fresh) > 1 else ""
            _say(f"-> {fresh[0].docid if len(fresh) == 1 else docid}{suffix} ({sum(u.words for u in fresh):,} words)")
            for unit in flagged[:3]:
                _say(f"      check {unit.docid}: {'; '.join(unit.flags)}")
            if len(flagged) > 3:
                _say(f"      ... and {len(flagged) - 3} more flagged part(s); see INGEST.md")
            if result.duplicates:
                _say(f"      {len(result.duplicates)} part(s) were already in the corpus and were skipped")
            if fresh[0].cleaning:
                _say(f"      cleaning: {fresh[0].cleaning}")
    finally:
        # even if the run is interrupted, every document that was written has its manifest row
        if rows_out:
            append_manifest(manifest, rows_out)
    if not manifest.exists():
        append_manifest(manifest, [])

    # the report lists the documents that exist now
    n_present, total_words, totals = write_current_report(report, docs_dir, report_rows, notices)

    est_tokens = int(total_words * TOKENS_PER_WORD)
    shown_report = report.relative_to(root).as_posix() if report.is_relative_to(root) else str(report)
    shown_manifest = manifest.relative_to(root).as_posix() if manifest.is_relative_to(root) else str(manifest)
    shown_docs = docs_dir.relative_to(root).as_posix() if docs_dir.is_relative_to(root) else str(docs_dir)
    _say()
    _say(f"Added {added} document(s). The corpus now has {n_present} document(s), about {total_words:,} words (about {est_tokens:,} tokens).")
    if notices:
        _say(f"{len(notices)} file(s) were not added; the reasons are in {shown_report}.")
    if totals["flagged"]:
        _say(f"{totals['flagged']} document(s) have flags. Open {shown_report} and look at them before you trust the text.")
    if recut:
        _say(f"{len(recut)} file(s) are already in the corpus, cut into documents another way, so they were not added again.")
        _say(f"To cut them again, delete their documents from {shown_docs}/ and their rows from {shown_manifest} (INGEST.md names them), then run ingest again.")
        _say("Do it before your qrels mention those documents, because a document's id must not change once a judgment names it.")
    if n_present < MIN_DOCS:
        _say(f"You need at least {MIN_DOCS} documents in the own corpus for the final check, so {MIN_DOCS - n_present} more to go (long documents split into parts count as parts).")
        if not part_pages and long_pdfs:
            n = limits.SUGGESTED_PART_PAGES
            _say(f"This run read {long_pdfs} PDF(s) longer than {n} pages. If long documents are what keeps you under {MIN_DOCS}, "
                 f"--part-pages {n} splits each one into parts of {n} pages that each count as a document; the README's size rules show the arithmetic.")  # fmt: skip
    if est_tokens > TOKEN_LIMIT:
        _say(f"That is above the limit of {TOKEN_LIMIT:,} tokens, so `p2 check` will fail, because CI could not encode the corpus within its {limits.CI_MINUTES} minutes. "
             "Remove documents, or keep only the parts your queries need.")  # fmt: skip
    elif est_tokens > TOKEN_WARNING:
        _say(f"Heads up: that is above {TOKEN_WARNING:,} tokens, near the limit of {TOKEN_LIMIT:,}, so the first CI run that encodes it can take up to about 15 minutes, once; later runs reuse the cache.")
    if added and license_value == "unknown":
        _say(f"The license column says unknown for these documents. Fix it in {shown_manifest}, then run: uv run p2 license")
    _say(f"Wrote {shown_report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(prog="p2 ingest", description=__doc__.split("\n")[0])
    add_arguments(ap)
    sys.exit(run(ap.parse_args()))
