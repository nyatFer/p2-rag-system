"""Load a corpus (its documents and manifest), cut documents into chunks, and lift chunk scores to documents.

A corpus lives in corpora/<name>/: docs/<docid>.md, one file per document whose first line is
"# <title>", and manifest.tsv with one row per document (docid, title, source, license, notes).

Chunks are fixed windows of words over a document's whole text, title line included, with a given
overlap; a chunk's id is "<docid>#<n>" with n counted from 0, and its text is the exact slice of
the document it covers, so a quote copied from a chunk is also a quote from the document.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from p2 import paths
from p2.runfile import DOCID_RE, data_lines, order

MANIFEST_COLUMNS = ("docid", "title", "source", "license", "notes")
WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Doc:
    docid: str
    title: str
    text: str  # the whole file, title line included; this is what the retrievers index
    path: Path


@dataclass(frozen=True)
class Chunk:
    id: str  # "<docid>#<n>"
    docid: str
    n: int
    text: str


def split_words(text: str, words: int, overlap: int) -> list[str]:
    """Windows of `words` words that share `overlap` words with the window before, as exact slices
    of `text`. words = 0, or a text no longer than one window, gives the whole text as one piece."""
    spans = [m.span() for m in WORD.finditer(text)]
    if words <= 0 or len(spans) <= words:
        return [text]
    step = words - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than words")
    pieces = []
    start = 0
    while True:
        end = min(start + words, len(spans))
        pieces.append(text[spans[start][0] : spans[end - 1][1]])
        if end == len(spans):
            return pieces
        start += step


def chunks(doc: Doc, words: int, overlap: int) -> list[Chunk]:
    """The chunks of one document."""
    return [Chunk(f"{doc.docid}#{n}", doc.docid, n, piece) for n, piece in enumerate(split_words(doc.text, words, overlap))]


def docid_of(chunk_id: str) -> str:
    """The document id inside a chunk id ("cfr30-75.403#2" gives "cfr30-75.403")."""
    return chunk_id.rsplit("#", 1)[0]


def doc_level(chunk_scores: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """Chunk scores lifted to documents: each document scores its best chunk (max), best document first,
    equal scores in document id order. Takes (chunk id, score) pairs."""
    best: dict[str, float] = {}
    for chunk_id, score in chunk_scores:
        docid = docid_of(chunk_id)
        score = float(score)
        if docid not in best or score > best[docid]:
            best[docid] = score
    return order(best.items())


@dataclass
class Corpus:
    """The documents of one corpus, in document id order, and their chunks."""

    name: str
    docs: dict[str, Doc]
    _chunks: dict[tuple[int, int], list[Chunk]] = field(default_factory=dict, repr=False)

    @property
    def ids(self) -> list[str]:
        return list(self.docs)

    def chunks(self, words: int, overlap: int) -> list[Chunk]:
        """Every chunk of every document, documents in id order (computed once per setting)."""
        key = (words, overlap)
        if key not in self._chunks:
            self._chunks[key] = [c for doc in self.docs.values() for c in chunks(doc, words, overlap)]
        return self._chunks[key]

    def chunk_texts(self, words: int, overlap: int) -> dict[str, str]:
        return {c.id: c.text for c in self.chunks(words, overlap)}

    def hash(self) -> str:
        """A short fingerprint of every document id and text."""
        h = hashlib.sha256()
        for docid, doc in self.docs.items():
            h.update(docid.encode("utf-8") + b"\0" + doc.text.encode("utf-8") + b"\0")
        return h.hexdigest()[:16]


def read_doc(path: Path) -> Doc:
    text = path.read_text(encoding="utf-8-sig")  # a byte order mark from a Windows editor is not part of the text
    first = text.split("\n", 1)[0]
    title = first[2:].strip() if first.startswith("# ") else ""
    return Doc(path.stem, title, text, path)


def doc_files(folder: Path) -> list[Path]:
    """The .md files of a docs folder, in file name order."""
    return sorted(folder.glob("*.md")) if folder.is_dir() else []


def load_dir(folder: Path, name: str) -> Corpus:
    """A corpus from a folder of .md files (or a folder with a docs/ folder inside)."""
    folder = Path(folder)
    if (folder / "docs").is_dir():
        folder = folder / "docs"
    docs = {}
    for path in doc_files(folder):
        doc = read_doc(path)
        docs[doc.docid] = doc
    return Corpus(name, dict(sorted(docs.items())))


def load(root: Path, name: str, extra: Path | None = None) -> Corpus:
    """The corpus corpora/<name>/docs, plus the documents of `extra` for this run only (--extra-corpus)."""
    corpus = load_dir(paths.docs_dir(root, name), name)
    if extra is not None:
        added = load_dir(Path(extra), "extra")
        clash = sorted(set(added.docs) & set(corpus.docs))
        if clash:
            raise ValueError(f"The extra corpus repeats document ids of {name}, such as {clash[0]}.")
        corpus = Corpus(name, dict(sorted({**corpus.docs, **added.docs}.items())))
    return corpus


# ---- format checks (used by p2 check) ----


def check_docs(folder: Path) -> list[str]:
    """Problems with the document files of a docs folder: ids, UTF-8, the title line, the blank line."""
    problems = []
    for path in doc_files(folder):
        if not DOCID_RE.match(path.stem):
            problems.append(f"{path.name} has a document id that is not allowed (lower-case letters, digits, . _ and -)")
            continue
        try:
            text = path.read_bytes().decode("utf-8-sig")
        except UnicodeDecodeError:
            problems.append(f"{path.name} is not UTF-8 text")
            continue
        lines = text.split("\n")
        if not lines[0].startswith("# ") or not lines[0][2:].strip():
            problems.append(f"{path.name} does not start with a '# title' line")
        elif len(lines) < 3 or lines[1].strip() or not text[len(lines[0]) :].strip():
            problems.append(f"{path.name} needs a blank line after the title, then the body")
    others = [p.name for p in folder.iterdir() if p.is_file() and p.suffix != ".md" and not p.name.startswith(".")] if folder.is_dir() else []
    if others:
        problems.append(f"docs/ holds files that are not .md documents, such as {sorted(others)[0]}")
    return problems


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """(rows, problems) from a manifest.tsv; rows are dicts keyed by the five column names."""
    rows: list[dict[str, str]] = []
    problems: list[str] = []
    if not path.is_file():
        return rows, ["manifest.tsv is missing"]
    lines = data_lines(path)
    if not lines or tuple(c.strip() for c in lines[0][1].split("\t")) != MANIFEST_COLUMNS:
        return rows, ["the first line of manifest.tsv should be the header docid, title, source, license, notes (tab-separated)"]
    seen = set()
    for n, line in lines[1:]:
        parts = line.split("\t")
        if len(parts) == 4:
            parts.append("")
        if len(parts) != 5:
            problems.append(f"manifest line {n} has {len(parts)} tab-separated fields, not 5")
            continue
        row = dict(zip(MANIFEST_COLUMNS, (p.strip() for p in parts)))
        if row["docid"] in seen:
            problems.append(f"manifest line {n} lists {row['docid']} a second time")
            continue
        seen.add(row["docid"])
        rows.append(row)
    return rows, problems


def check_manifest(root: Path, name: str) -> list[str]:
    """Problems with a corpus's manifest against its documents: a row per document and a document per row."""
    rows, problems = read_manifest(paths.manifest(root, name))
    docs = {p.stem for p in doc_files(paths.docs_dir(root, name))}
    if not docs and not rows:
        return problems if problems != ["manifest.tsv is missing"] else []
    listed = {r["docid"] for r in rows}
    missing = sorted(docs - listed)
    orphans = sorted(listed - docs)
    if missing:
        problems.append(f"{len(missing)} document(s) have no manifest row, such as {missing[0]}")
    if orphans:
        problems.append(f"{len(orphans)} manifest row(s) name no document in docs/, such as {orphans[0]}")
    empty = sorted(r["docid"] for r in rows if not r["source"])
    if empty:
        problems.append(f"{len(empty)} manifest row(s) have no source, such as {empty[0]}")
    return problems
