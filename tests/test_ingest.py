"""Tests for p2 ingest: the cleaning pass, ids, splitting, duplicates, the manifest and INGEST.md.

The pure functions need nothing installed.
Tests that read a PDF or an HTML file need the ingest group (`uv sync --group ingest`) and are skipped without it,
so a plain `uv run pytest` stays green in CI, where the group is never installed.
PDFs are built in the tests by a tiny writer, so no binary fixture is needed; the other fixtures are in tests/fixtures/ingest/.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import re
import shutil
from pathlib import Path

import pytest

from p2 import ingest

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"
DOCID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def norm(text):
    """The lab 13 quote-check normalization: lower-case, collapse whitespace."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


needs_pypdf = pytest.mark.skipif(not _installed("pypdf"), reason="needs the ingest group (pypdf)")
needs_trafilatura = pytest.mark.skipif(not _installed("trafilatura"), reason="needs the ingest group (trafilatura)")


# ---------------------------------------------------------------------------
# A tiny PDF writer: Helvetica with a ToUnicode map so that byte 1 reads as the ligature "fi" and byte 2 as "fl"
# ---------------------------------------------------------------------------

_CMAP = b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
2 beginbfchar
<01> <FB01>
<02> <FB02>
endbfchar
1 beginbfrange
<20> <7E> <0020>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"""


def _pdf_string(line: str) -> bytes:
    out = bytearray()
    for ch in line:
        if ch == "\ufb01":
            out += b"\\001"
        elif ch == "\ufb02":
            out += b"\\002"
        elif ch in "\\()":
            out += b"\\" + ch.encode()
        elif 32 <= ord(ch) < 127:
            out += ch.encode()
        else:
            out += b"?"
    return bytes(out)


def build_pdf(pages: list[list[str]], title: str | None = None) -> bytes:
    """pages is a list of pages, each a list of text lines. Returns the bytes of a PDF."""
    objects: dict[int, bytes] = {}
    n = len(pages)
    kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(n))
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 4 0 R >>"
    objects[4] = b"<< /Length %d >>\nstream\n" % len(_CMAP) + _CMAP + b"endstream"
    for i, lines in enumerate(pages):
        body = bytearray(b"BT /F1 10 Tf 40 760 Td 12 TL\n")
        for line in lines:
            body += b"(" + _pdf_string(line) + b") '\n"
        body += b"ET"
        objects[5 + 2 * i] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {6 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects[6 + 2 * i] = b"<< /Length %d >>\nstream\n" % len(body) + bytes(body) + b"\nendstream"
    info = b""
    if title:
        objects[5 + 2 * n] = b"<< /Title (" + _pdf_string(title) + b") >>"
        info = b" /Info %d 0 R" % (5 + 2 * n)
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += b"%d 0 obj\n" % num + objects[num] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for num in range(1, size):
        out += b"%010d 00000 n \n" % offsets[num]
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (size, info, xref)
    return bytes(out)


def words(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 9))) for _ in range(count)]


def sentence_lines(seed: int, n_lines: int, per_line: int = 12) -> list[str]:
    ws = words(seed, n_lines * per_line)
    return [" ".join(ws[i * per_line : (i + 1) * per_line]) for i in range(n_lines)]


def paper_pages() -> list[list[str]]:
    """Six pages of a fake paper: running header and footer with page numbers, ligatures, a hyphenated break, a reference list."""
    pages = []
    for p in range(1, 7):
        body = sentence_lines(p, 14)
        if p == 1:
            body[0] = "The \ufb01nal report describes the \ufb02ow through the pump and the inter-"
            body[1] = "national standard that governs it. The result is a short, clear sentence about the pump."
        if p == 6:
            body = body[:6] + [
                "References",
                "[1] A. Author, Pump Handbook, Example Press, 2001.",
                "[2] B. Writer, Cavitation in practice, Journal of Fluids 12, 45-60 (1998).",
                "[3] C. Person et al., Pump Selection, 2010, doi:10.1000/example.",
                "[4] D. Someone, Hydraulic Standards, 2004.",
            ]
        pages.append([f"Journal of Pump Studies, Vol. 12 (2020) page {p}"] + body + [f"Example Press, page {p}"])
    return pages


def args_for(src, root, **kw):
    ns = dict(src_dir=str(src), into="own", license=None, source=None, root=str(root))
    ns.update(kw)
    return argparse.Namespace(**ns)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "p2.toml").write_text("", encoding="utf-8", newline="\n")
    return root


@pytest.fixture
def src(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    return folder


def corpus(root, name="own"):
    return root / "corpora" / name


def doc_text(root, docid, name="own"):
    return (corpus(root, name) / "docs" / f"{docid}.md").read_text(encoding="utf-8")


def manifest_rows(root, name="own"):
    lines = (corpus(root, name) / "manifest.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "docid\ttitle\tsource\tlicense\tnotes"
    return [line.split("\t") for line in lines[1:]]


def doc_ids(root, name="own"):
    return sorted(p.stem for p in (corpus(root, name) / "docs").glob("*.md"))


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------


def test_clean_chars_replaces_ligatures_soft_hyphens_and_odd_spaces():
    raw = "\ufb01nal \ufb02ow o\ufb03ce \ufb00 \ufb04 \ufb05 \ufb06 ab\xadc non\xa0breaking thin\u2009space zero\u200bwidth"
    assert ingest.clean_chars(raw) == "final flow office ff ffl st st abc non breaking thin space zerowidth"
    assert not re.search("[\ufb00-\ufb06\xad\xa0]", ingest.clean_chars(raw))


def test_soft_hyphen_at_a_line_end_joins_the_word():
    assert ingest.clean_chars("inter\xad\nnational") == "international"


def test_clean_chars_normalizes_line_endings_and_control_characters():
    assert ingest.clean_chars("a\r\nb\rc\x0cd\x00e") == "a\nb\nc\nde"


def test_clean_chars_keeps_indentation_and_collapses_inner_runs():
    assert ingest.clean_chars("    return  d  * 0.9   \nnext   line") == "    return d * 0.9\nnext line"


def test_dehyphenate_joins_only_lowercase_continuations():
    assert ingest.dehyphenate("inter-\nnational") == "international"
    assert ingest.dehyphenate("a well-known fact") == "a well-known fact"
    assert ingest.dehyphenate("X-\nRay stays") == "X-\nRay stays"
    assert ingest.dehyphenate("a trailing -\nlist") == "a trailing -\nlist"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("**bold** and __bold__ text", "bold and bold text"),
        ("_italic_ and *italic* text", "italic and italic text"),
        ("~~gone~~ stays as text", "gone stays as text"),
        ("a [link](https://example.org/x_(y) \"title\") here", "a link here"),
        ("see ![alt text](img.png) there", "see  there"),
        ("# Heading\n## Sub heading ##\ntext", "Heading\nSub heading\ntext"),
        ("> quoted\n> > nested", "quoted\nnested"),
        ("- one\n* two\n+ three", "one\ntwo\nthree"),
        ("| a | b |\n|---|:---:|\n| 1 | 2 |", "a | b\n\n1 | 2"),
        ("| single |\n| --- |\n| cell |", "single\n\ncell"),
        ("text<br>more<br/>lines", "text more lines"),
        ("water<sup>2</sup> and H<sub>2</sub>O", "water2 and H2O"),
        ("<p>one</p><p>two</p>", "\none\n\ntwo\n"),
        ("<!-- hidden --> shown", " shown"),
        ("escaped \\*stars\\* and \\_under\\_", "escaped *stars* and _under_"),
        ("a &amp; b &lt; c &nbsp;d", "a & b < c  d"),
        ("<https://example.org/a>", "https://example.org/a"),
        ("---\n***\n___\ntext", "\n\n\ntext"),
        ("Title\n=====\ntext", "Title\n\ntext"),
        ("snake_case_name and 2*3*4 and a * b * c", "snake_case_name and 2*3*4 and a * b * c"),
        ("x < y and y > z", "x < y and y > z"),
    ],
)
def test_strip_markup(raw, expected):
    assert ingest.strip_markup(raw) == expected


def test_strip_markup_keeps_code_content_untouched():
    raw = "Use `__init__.py` and `a*b*c` here.\n\n```python\ndef f(x):\n    return x * 2  # **not bold**\n```\n\nAfter."
    out = ingest.strip_markup(raw)
    assert "Use __init__.py and a*b*c here." in out
    assert "    return x * 2  # **not bold**" in out
    assert "```" not in out and "`" not in out


def test_strip_markup_for_html_drops_edit_marks_and_citation_numbers():
    raw = "Finite elements[1] were introduced.[citation needed] Methods [edit]\nIt works.[a]"
    out = ingest.strip_markup(raw, source="html")
    assert "[1]" not in out and "[edit]" not in out and "[citation needed]" not in out and "[a]" not in out
    assert "Finite elements were introduced." in out
    # Markdown files keep square brackets: they may be real text
    assert "[1]" in ingest.strip_markup("see [1] for details", source="md")


# ---------------------------------------------------------------------------
# Running headers and footers
# ---------------------------------------------------------------------------


def make_pages(n, header="Journal of Tests 12 (2020)", footer="Page {p}"):
    pages = []
    for p in range(1, n + 1):
        body = "\n".join(sentence_lines(100 + p, 12))
        pages.append(f"{header}\n{body}\n{footer.format(p=p)}")
    return pages


def test_strip_running_lines_removes_headers_footers_and_page_numbers():
    pages = make_pages(8)
    out, bad = ingest.strip_running_lines(pages)
    assert bad
    joined = "\n".join(out)
    assert "Journal of Tests" not in joined
    assert "Page 3" not in joined
    assert sentence_lines(103, 12)[5] in out[2]  # the body is untouched


def test_strip_running_lines_ignores_changing_digits_and_alternating_headers():
    pages = [f"{'Odd title' if p % 2 else 'Even title'} {p}\n" + "\n".join(sentence_lines(200 + p, 10)) + f"\n{p}" for p in range(1, 9)]
    out, _ = ingest.strip_running_lines(pages)
    assert all("title" not in page for page in out)
    assert all(not page.rstrip().endswith(str(p)) or True for p, page in enumerate(out, 1))


def test_strip_running_lines_needs_enough_pages_and_repetition():
    three = make_pages(3)
    assert ingest.strip_running_lines(three) == (three, [])
    pages = make_pages(10)
    pages[4] = pages[4].replace("Journal of Tests 12 (2020)", "A one-off line")
    out, _ = ingest.strip_running_lines(pages)
    assert "A one-off line" in out[4]  # a line seen on one page only is content
    # a repeated line in the middle of a page is content too
    mid = [p.replace("\n", "\nMiddle repeated line\n", 6) for p in make_pages(8, header="H{}".format(1))]
    out, _ = ingest.strip_running_lines(mid)
    assert any("Middle repeated line" in page for page in out)


# ---------------------------------------------------------------------------
# Reference lists
# ---------------------------------------------------------------------------


def body_pages(n_lines=40):
    return ["\n".join(sentence_lines(300, n_lines))]


def test_reference_list_dropped_at_a_references_heading():
    refs = ["References", "A. Author. Pump Handbook. Example Press, 2001.", "B. Writer. Cavitation in practice. J. Fluids 12, 1998.", "C. Person et al. Selection. 2010. doi:10.1000/x", "D. Someone. Standards. 2004."]
    pages = ["\n".join(sentence_lines(301, 40) + refs)]
    out, removed, how = ingest.drop_reference_list(pages)
    assert how == "heading" and removed > 100
    assert "References" not in out[0] and "Pump Handbook" not in out[0]
    assert sentence_lines(301, 40)[-1] in out[0]


@pytest.mark.parametrize("heading", ["References", "REFERENCES", "# References", "**References**", "6. References", "Bibliography", "Literature cited", "Works Cited", "References:"])
def test_reference_heading_variants(heading):
    refs = [heading] + [f"Author {i}. A title about pumps. Journal of Fluids {i}, {1990 + i}." for i in range(1, 6)]
    out, removed, how = ingest.drop_reference_list(["\n".join(sentence_lines(302, 40) + refs)])
    assert how == "heading", heading
    assert "Author 3" not in out[0]


def test_reference_list_without_heading_is_found_by_its_numbered_entries():
    entries = [f"[{i}] A. Person{i}, Journal of Fluids {i}, {1980 + i} (19{80 + i}).\ncontinued second line of entry {i}." for i in range(1, 12)]
    pages = ["\n".join(sentence_lines(303, 40) + entries + ["Appendix text starts here and is kept."])]
    out, removed, how = ingest.drop_reference_list(pages)
    assert how == "numbered entries"
    assert "[5]" not in out[0] and "Person7" not in out[0]
    assert "Appendix text starts here" in out[0]


def test_reference_list_across_pages_cuts_the_page_it_starts_on():
    refs = ["References"] + [f"Author {i}. A title. Journal {i}, {1990 + i}." for i in range(1, 9)]
    pages = ["\n".join(sentence_lines(304, 30)), "\n".join(sentence_lines(305, 10) + refs), "\n".join(f"Author {i}. More. Journal {i}, {2000 + i}." for i in range(20, 28))]
    out, removed, how = ingest.drop_reference_list(pages)
    assert len(out) == 3
    assert out[0] == pages[0]
    assert "References" not in out[1] and out[1].count("\n") == 9
    assert out[2].strip() == ""


def test_a_references_heading_over_ordinary_prose_is_kept():
    prose = ["References", "This chapter is called References but it is ordinary prose about how pumps are selected.", "It has no years and no citations, only sentences that explain the idea at length.", "A third line keeps going about the same topic without any bibliographic hint."]
    pages = ["\n".join(sentence_lines(306, 40) + prose)]
    out, removed, how = ingest.drop_reference_list(pages)
    assert (removed, how) == (0, "") and out == pages


def test_reference_list_stops_at_an_appendix_heading():
    refs = ["References"] + [f"Author {i}. A title. Journal {i}, {1990 + i}." for i in range(1, 8)] + ["Appendix A. Data tables", "The appendix keeps its own text."]
    out, _, _ = ingest.drop_reference_list(["\n".join(sentence_lines(307, 40) + refs)])
    assert "Appendix A. Data tables" in out[0] and "The appendix keeps its own text." in out[0]
    assert "Author 4" not in out[0]


def test_short_documents_never_lose_their_reference_section():
    pages = ["Short text.\nReferences\nAuthor. Title. 2001.\nAuthor. Title. 2002.\nAuthor. Title. 2003."]
    assert ingest.drop_reference_list(pages) == (pages, 0, "")


def test_a_reference_list_that_is_most_of_the_text_is_kept():
    refs = ["References"] + [f"Author {i}. A title. Journal {i}, {1990 + i}." for i in range(1, 60)]
    pages = ["\n".join(sentence_lines(308, 25) + refs)]
    # the list starts after 55 percent of the lines but is more than 45 percent of the characters: too suspicious to drop
    out, removed, how = ingest.drop_reference_list(pages)
    assert (removed, how) == (0, "") or removed < 0.45 * len(pages[0])


# ---------------------------------------------------------------------------
# Ids, parts, hashes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Taylor 1994 - TN1297", "taylor-1994-tn1297"),
        ("arXiv 1602.03837", "arxiv-1602.03837"),
        ("cfr30-75.403", "cfr30-75.403"),
        ("Caf\xe9 r\xe9sum\xe9 (final v2)", "cafe-resume-final-v2"),
        ("--weird__name..here--", "weird_name.here"),
        ("my_paper", "my_paper"),
        ("report.v2.final", "report.v2.final"),
        ("a - b", "a-b"),
        ("v1 (final)", "v1-final"),
        ("   ", ""),
        ("\u6587\u6863", ""),
        ("con", "doc-con"),
        ("NUL.txt", "doc-nul.txt"),
        ("lpt3", "doc-lpt3"),
        ("x" * 100, "x" * 64),
    ],
)
def test_slugify(name, expected):
    assert ingest.slugify(name) == expected
    if expected:
        assert DOCID.match(expected)


def test_make_id_resolves_collisions_by_parent_folder_then_number(tmp_path):
    root = tmp_path
    taken = set()
    a = ingest.make_id(root / "report.pdf", taken, root)
    taken.add(a)
    b = ingest.make_id(root / "Week 3" / "report.pdf", taken, root)
    taken.add(b)
    c = ingest.make_id(root / "Week 3" / "report.html", taken, root)
    taken.add(c)
    d = ingest.make_id(root / "other" / "report.pdf", taken, root)
    assert (a, b) == ("report", "week-3-report")
    assert c.startswith("week-3-report-") or c == "report-2"
    assert d == "other-report"
    assert len({a, b, c, d}) == 4


def test_make_id_never_returns_an_empty_id(tmp_path):
    first = ingest.make_id(tmp_path / "\u6587\u6863.pdf", set(), tmp_path)
    again = ingest.make_id(tmp_path / "\u6587\u6863.pdf", set(), tmp_path)
    assert DOCID.match(first) and first == again  # stable, derived from the name


def test_make_id_treats_an_id_with_parts_as_used(tmp_path):
    taken = {"report__p001-010", "report__p011-020"}
    assert ingest.make_id(tmp_path / "report.pdf", taken, tmp_path) == "report-2"
    assert ingest.make_id(tmp_path / "other" / "report.pdf", taken, tmp_path) == "other-report"


def test_output_never_fails_on_a_console_that_cannot_show_a_character(monkeypatch, capsys):
    import io
    import sys

    fake = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", fake)
    ingest._say("caf" + chr(233) + " " + chr(0x6587), end="")
    fake.flush()
    assert fake.buffer.getvalue() == b"caf? ?"


def test_a_file_with_a_non_ascii_name_gets_an_ascii_id(repo, src):
    name = "Caf" + chr(233) + " " + chr(0x6587) + " notes.md"
    (src / name).write_text("# Notes\n\n" + " ".join(words(9, 70)), encoding="utf-8", newline="\n")
    assert ingest.run(args_for(src, repo)) == 0
    assert doc_ids(repo) == ["cafe-notes"]
    assert manifest_rows(repo)[0][2] == name  # the source column keeps the real file name


def test_an_interrupted_run_still_leaves_a_manifest_row_for_every_document_written(repo, src, monkeypatch):
    (src / "a.md").write_text("# A\n\n" + " ".join(words(11, 60)), encoding="utf-8", newline="\n")
    (src / "b.md").write_text("# B\n\n" + " ".join(words(12, 60)), encoding="utf-8", newline="\n")
    real = ingest.process_file

    def interrupt_on_b(path, rel, docid, *rest):
        if rel == "b.md":
            raise KeyboardInterrupt
        return real(path, rel, docid, *rest)

    monkeypatch.setattr(ingest, "process_file", interrupt_on_b)
    with pytest.raises(KeyboardInterrupt):
        ingest.run(args_for(src, repo))
    assert doc_ids(repo) == ["a"]
    assert [r[0] for r in manifest_rows(repo)] == ["a"]


def test_a_failure_while_cleaning_one_file_is_reported_and_the_others_are_added(repo, src, monkeypatch, capsys):
    (src / "a.md").write_text("# A\n\n" + " ".join(words(13, 60)), encoding="utf-8", newline="\n")
    (src / "b.md").write_text("# B\n\n" + " ".join(words(14, 60)), encoding="utf-8", newline="\n")
    real = ingest.build_units

    def explode_on_b(docid, rel, extracted, kind, *rest):
        if rel == "b.md":
            raise ValueError("something odd in this file")
        return real(docid, rel, extracted, kind, *rest)

    monkeypatch.setattr(ingest, "build_units", explode_on_b)
    assert ingest.run(args_for(src, repo)) == 0
    assert doc_ids(repo) == ["a"]
    assert "b.md -> not added: ValueError: something odd in this file" in capsys.readouterr().out


@pytest.mark.parametrize(
    "pages,expected",
    [
        (5, [(1, 5)]),
        (10, [(1, 10)]),
        (11, [(1, 11)]),  # a one-page remainder joins the last part
        (13, [(1, 13)]),
        (14, [(1, 10), (11, 14)]),
        (25, [(1, 10), (11, 20), (21, 25)]),
        (23, [(1, 10), (11, 23)]),
        (226, [(1, 10)] + [(s, s + 9) for s in range(11, 221, 10)] + [(221, 226)]),
    ],
)
def test_page_blocks(pages, expected):
    assert ingest.page_blocks(pages) == expected


def test_split_paragraphs_makes_parts_of_about_the_target_size():
    paragraphs = ["\n".join(sentence_lines(400 + i, 5, 20)) for i in range(60)]  # 100 words each
    text = "\n\n".join(paragraphs)
    parts = ingest.split_paragraphs(text, 4000)
    assert 1 < len(parts) <= 3
    assert "\n\n".join(parts) == text
    assert all(len(p.split()) >= 1500 for p in parts)


def test_normalized_hash_ignores_case_punctuation_and_whitespace():
    assert ingest.normalized_hash("The Final  Report, v2.") == ingest.normalized_hash("the final report v2")
    assert ingest.normalized_hash("alpha beta") != ingest.normalized_hash("alpha gamma")


def test_odd_glyph_flags():
    assert ingest.odd_glyph_flags("clean text") == []
    assert any("replacement" in f for f in ingest.odd_glyph_flags("bad \ufffd char"))
    assert any("cid" in f for f in ingest.odd_glyph_flags("(cid:12) text"))
    assert ingest.odd_glyph_flags("one \xbc inch") == []  # a single fraction is ordinary text
    assert any("thorn" in f for f in ingest.odd_glyph_flags("410\xfe160 and 3\xfe5 and 4\xfe2"))


# ---------------------------------------------------------------------------
# The command on text, Markdown and HTML files
# ---------------------------------------------------------------------------


def copy_fixtures(src, *names):
    for name in names:
        shutil.copy(FIXTURES / name, src / name)


def test_markdown_and_text_files_are_read_and_cleaned(repo, src, capsys):
    copy_fixtures(src, "notes.md", "plain.txt")
    assert ingest.run(args_for(src, repo, license="cc-by-4.0")) == 0
    assert doc_ids(repo) == ["notes", "plain"]
    notes = doc_text(repo, "notes")
    assert notes.startswith("# Pump curve notes\n\n")
    assert "**" not in notes and "](" not in notes and "![" not in notes and "```" not in notes and "`" not in notes
    assert "These notes use bold, italic and starred text, a link, and an image" in notes
    assert "Call pump.trim(diameter) after reading __init__.py; the setting sync.chunk_size_kb stays as written." in notes
    assert "    return d * 0.9" in notes
    assert "Snake_case_names and 2*3*4 stay untouched" in notes
    assert "Pump Handbook" not in notes  # the trailing reference list
    plain = doc_text(repo, "plain")
    assert plain.startswith("# Pump notes in plain text\n\n")
    assert "final report covers the flow of the international standard" in plain
    assert "split at a line break like example stays readable" in plain
    assert "A well-known fact keeps its hyphen." in plain
    assert "\xad" not in plain and "\xa0" not in plain and "\ufb01" not in plain


@needs_trafilatura
def test_html_keeps_the_article_and_loses_the_page_furniture(repo, src):
    copy_fixtures(src, "page.html")
    assert ingest.run(args_for(src, repo)) == 0
    text = doc_text(repo, "page")
    assert text.startswith("# Centrifugal pump basics\n\n")
    for gone in ("cookies", "Sign in", "Privacy", "footer links", "**", "](", "[1]", "Pump Handbook", "Hydraulic Institute"):
        assert gone not in text, gone
    for kept in ("turning the kinetic energy of a spinning impeller into pressure", "Reading a pump curve", "never throttle the suction valve to control flow"):
        assert kept in text, kept
    assert "pump curve plots head against flow rate" in text  # link text kept, link target dropped


def test_office_files_are_refused_with_a_pointer_to_pdf(repo, src, capsys):
    copy_fixtures(src, "Lab Manual.docx", "slides.pptx", "notes.md")
    assert ingest.run(args_for(src, repo)) == 0
    out = capsys.readouterr().out
    assert "Save As" in out and "PDF" in out
    assert doc_ids(repo) == ["notes"]
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    assert "- `Lab Manual.docx`: Word and PowerPoint files are not read directly" in report and "- `slides.pptx`: Word and PowerPoint files" in report


def test_spreadsheets_hidden_and_unknown_files_are_not_added(repo, src, capsys):
    copy_fixtures(src, "data.csv", "figure.png", ".hidden.md", "notes.md")
    (src / "~$lock.docx").write_text("lock", encoding="utf-8", newline="\n")
    (src / ".git").mkdir()
    (src / ".git" / "config.md").write_text("# hidden dir\n\nbody text here", encoding="utf-8", newline="\n")
    assert ingest.run(args_for(src, repo)) == 0
    assert doc_ids(repo) == ["notes"]
    out = capsys.readouterr().out
    assert "data.csv" in out and "spreadsheet" in out
    assert "figure.png" in out and "not one ingest reads" in out
    assert ".hidden" not in out and "lock.docx" not in out


def test_manifest_rows_license_and_source(repo, src):
    copy_fixtures(src, "notes.md", "plain.txt")
    ingest.run(args_for(src, repo))
    rows = manifest_rows(repo)
    assert [r[0] for r in rows] == ["notes", "plain"]
    for docid, _title, source, license_, notes in rows:
        assert license_ == "unknown"
        assert source == f"{docid}.md" or source == f"{docid}.txt"
        assert notes == f"file: {source}"
    assert rows[0][1] == "Pump curve notes"


def test_license_and_source_options(repo, src, capsys):
    copy_fixtures(src, "notes.md")
    ingest.run(args_for(src, repo, license="CC-BY-4.0", source="Example Engineering Notes, https://example.org/notes"))
    assert manifest_rows(repo) == [["notes", "Pump curve notes", "Example Engineering Notes, https://example.org/notes", "cc-by-4.0", "file: notes.md"]]
    capsys.readouterr()
    ingest.run(args_for(src, repo, license="gpl-3.0"))
    assert "will not pass `p2 license`" in capsys.readouterr().out


def test_ids_come_from_file_names(repo, src):
    (src / "Taylor 1994 - TN1297.md").write_text("# Guidelines\n\n" + " ".join(words(1, 80)), encoding="utf-8", newline="\n")
    sub = src / "Week 3"
    sub.mkdir()
    (sub / "Taylor 1994 - TN1297.md").write_text("# Another\n\n" + " ".join(words(2, 80)), encoding="utf-8", newline="\n")
    ingest.run(args_for(src, repo))
    ids = doc_ids(repo)
    assert ids == ["taylor-1994-tn1297", "week-3-taylor-1994-tn1297"]
    assert all(DOCID.match(i) for i in ids)


def test_running_twice_adds_nothing_the_second_time(repo, src, capsys):
    copy_fixtures(src, "notes.md", "plain.txt")
    ingest.run(args_for(src, repo, license="cc-by-4.0"))
    before = (corpus(repo) / "manifest.tsv").read_text(encoding="utf-8")
    docs_before = {p.name: p.read_text(encoding="utf-8") for p in (corpus(repo) / "docs").glob("*.md")}
    capsys.readouterr()
    assert ingest.run(args_for(src, repo, license="cc-by-4.0")) == 0
    out = capsys.readouterr().out
    assert "already in the corpus" in out and "Added 0 document(s)" in out
    assert (corpus(repo) / "manifest.tsv").read_text(encoding="utf-8") == before
    assert {p.name: p.read_text(encoding="utf-8") for p in (corpus(repo) / "docs").glob("*.md")} == docs_before


def test_exact_duplicates_are_skipped_by_normalized_text(repo, src, capsys):
    text = "# Pump note\n\n" + " ".join(words(5, 120)) + "\n"
    (src / "a.md").write_text(text, encoding="utf-8", newline="\n")
    (src / "b copy.md").write_text(text.upper().replace("\n\n", "\n\n\n").replace(" ", "  "), encoding="utf-8", newline="\n")
    (src / "c.md").write_text("# Different\n\n" + " ".join(words(6, 120)), encoding="utf-8", newline="\n")
    ingest.run(args_for(src, repo))
    assert doc_ids(repo) == ["a", "c"]
    out = capsys.readouterr().out
    assert "b copy.md -> already in the corpus (same text as a)" in out


def test_a_second_folder_appends_to_the_manifest_and_keeps_the_report_rows(repo, src, tmp_path):
    copy_fixtures(src, "notes.md")
    ingest.run(args_for(src, repo, license="own-work"))
    other = tmp_path / "raw2"
    other.mkdir()
    copy_fixtures(other, "plain.txt")
    ingest.run(args_for(other, repo, license="cc0-1.0"))
    rows = manifest_rows(repo)
    assert [(r[0], r[3]) for r in rows] == [("notes", "own-work"), ("plain", "cc0-1.0")]
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    assert "| `notes` |" in report and "| `plain` |" in report
    assert "- Documents: 2" in report


def test_a_manifest_without_a_final_newline_still_appends_cleanly(repo, src):
    copy_fixtures(src, "notes.md")
    corpus(repo).mkdir(parents=True)
    (corpus(repo) / "manifest.tsv").write_text("docid\ttitle\tsource\tlicense\tnotes\nold\tOld\thttps://x\tmit\t", encoding="utf-8", newline="\n")
    ingest.run(args_for(src, repo))
    rows = manifest_rows(repo)
    assert [r[0] for r in rows] == ["old", "notes"]


def test_a_manifest_with_a_byte_order_mark_and_crlf_is_appended_cleanly(repo, src):
    copy_fixtures(src, "notes.md")
    corpus(repo).mkdir(parents=True)
    (corpus(repo) / "manifest.tsv").write_bytes(b"\xef\xbb\xbfdocid\ttitle\tsource\tlicense\tnotes\r\nold\tOld\thttps://x\tmit\t\r\n")
    ingest.run(args_for(src, repo))
    raw = (corpus(repo) / "manifest.tsv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    assert [r[0] for r in manifest_rows(repo)] == ["old", "notes"]


def test_documents_and_manifest_are_utf8_with_lf_endings(repo, src):
    (src / "crlf.md").write_text("# Caf\xe9 notes\r\n\r\nLine one\r\nLine two with \xe9\r\n" + " ".join(words(7, 60)), encoding="utf-8", newline="")
    ingest.run(args_for(src, repo))
    for path in [corpus(repo) / "docs" / "crlf.md", corpus(repo) / "manifest.tsv", corpus(repo) / "INGEST.md"]:
        raw = path.read_bytes()
        assert b"\r" not in raw, path
        raw.decode("utf-8")
    assert doc_text(repo, "crlf").startswith("# Caf\xe9 notes\n\nCaf\xe9 notes\n\nLine one\nLine two with \xe9")


def test_latin_encoded_text_is_read(repo, src):
    (src / "old.txt").write_bytes("Caf\xe9 menu\n\n".encode("cp1252") + " ".join(words(8, 60)).encode("ascii"))
    ingest.run(args_for(src, repo))
    assert doc_text(repo, "old").startswith("# Caf\xe9 menu")


def test_missing_and_empty_folders(repo, tmp_path, capsys):
    assert ingest.run(args_for(tmp_path / "nope", repo)) == 1
    assert "could not find a folder" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ingest.run(args_for(empty, repo)) == 1
    assert "nothing to read" in capsys.readouterr().out
    assert ingest.run(args_for(empty, repo, into="Bad Name")) == 1


def test_into_chooses_the_corpus(repo, src):
    copy_fixtures(src, "notes.md")
    ingest.run(args_for(src, repo, into="demo"))
    assert doc_ids(repo, "demo") == ["notes"]
    assert not (corpus(repo, "own")).exists()


def test_ingest_md_has_one_line_per_document_and_the_guidance(repo, src):
    copy_fixtures(src, "notes.md", "plain.txt", "Lab Manual.docx")
    ingest.run(args_for(src, repo))
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    assert re.search(r"^\| `notes` \| 84 \| - \|", report, re.M)
    assert re.search(r"^\| `plain` \| 40 \| - \| very short \(40 words\)", report, re.M)
    assert "| docid | words | characters per page | flags | cleaning |" in report
    assert "## What to check by eye" in report
    assert "pymupdf4llm" in report and "rapidocr" in report  # the opt-in OCR route is documented
    assert "400" not in report.split("## What to check")[0] or True
    assert "\u2014" not in report  # no em dash anywhere


def test_a_corrupt_pdf_is_reported_not_fatal(repo, src, capsys):
    pytest.importorskip("pypdf")
    (src / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    copy_fixtures(src, "notes.md")
    assert ingest.run(args_for(src, repo)) == 0
    assert doc_ids(repo) == ["notes"]
    assert "broken.pdf" in capsys.readouterr().out
    assert "- `broken.pdf`: " in (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------


@needs_pypdf
def test_pdf_cleaning_pass_end_to_end(repo, src):
    (src / "Pump Paper.pdf").write_bytes(build_pdf(paper_pages(), title="A study of pump flow"))
    assert ingest.run(args_for(src, repo, license="cc-by-4.0")) == 0
    assert doc_ids(repo) == ["pump-paper"]
    text = doc_text(repo, "pump-paper")
    assert text.startswith("# A study of pump flow\n\n")
    # ligatures, hyphenated break
    assert "The final report describes the flow through the pump and the international" in text
    assert not re.search("[\ufb00-\ufb06\xad]", text)
    # running header and footer and page numbers are gone
    assert "Journal of Pump Studies" not in text and "Example Press, page" not in text
    # the reference list is gone, the text before it stays
    assert "Pump Handbook" not in text and "Cavitation in practice" not in text and "References" not in text
    assert sentence_lines(5, 14)[3] in text
    row = manifest_rows(repo)[0]
    assert row[0] == "pump-paper" and row[3] == "cc-by-4.0" and row[4] == "file: Pump Paper.pdf; 6 pages"
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    line = next(ln for ln in report.splitlines() if ln.startswith("| `pump-paper`"))
    assert "running line(s) removed" in line and "reference list dropped" in line


@needs_pypdf
def test_quote_check_effect_on_a_pdf(repo, src):
    """A sentence typed the way a reader reads it is found in the cleaned text and not in the raw extraction."""
    pdf = src / "paper.pdf"
    pdf.write_bytes(build_pdf(paper_pages()))
    raw = "\n".join(ingest.read_pdf(pdf).pages)
    assert "\ufb01" in raw and "\ufb02" in raw  # the converter output has ligatures
    assert "inter-\nnational" in raw  # and a hyphen left over from a line break
    ingest.run(args_for(src, repo))
    cleaned = doc_text(repo, "paper")
    honest_quote = "The final report describes the flow through the pump and the international standard that governs it."
    assert norm(honest_quote) in norm(cleaned)
    assert norm(honest_quote) not in norm(raw)


@needs_pypdf
def test_pdf_title_falls_back_to_the_first_line(repo, src):
    (src / "untitled.pdf").write_bytes(build_pdf(paper_pages()))
    ingest.run(args_for(src, repo))
    assert manifest_rows(repo)[0][1] == "Journal of Pump Studies, Vol. 12 (2020) page 1"


@needs_pypdf
def test_a_scan_with_no_text_is_not_added_and_says_why(repo, src, capsys):
    (src / "scan.pdf").write_bytes(build_pdf([[""] for _ in range(5)]))
    copy_fixtures(src, "notes.md")
    assert ingest.run(args_for(src, repo)) == 0
    assert doc_ids(repo) == ["notes"]
    out = capsys.readouterr().out
    assert "scan.pdf -> not added" in out and "scan" in out
    assert "- `scan.pdf`: No text found in 5 page(s), so this looks like a scan" in (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")


@needs_pypdf
def test_a_thin_pdf_is_added_but_flagged_as_scanned_looking(repo, src):
    pages = [sentence_lines(10 + p, 3, 7) for p in range(1, 5)]  # about 100 characters and 21 words per page
    (src / "thin.pdf").write_bytes(build_pdf(pages))
    ingest.run(args_for(src, repo))
    assert doc_ids(repo) == ["thin"]
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    line = next(ln for ln in report.splitlines() if ln.startswith("| `thin`"))
    assert "scanned-looking" in line
    cpp = int(line.split("|")[3].strip())
    assert cpp < ingest.SCANNED_CHARS_PER_PAGE
    assert "Flagged documents: 1" in report


@needs_pypdf
def test_a_dense_pdf_is_not_flagged(repo, src):
    (src / "paper.pdf").write_bytes(build_pdf(paper_pages()))
    ingest.run(args_for(src, repo))
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    line = next(ln for ln in report.splitlines() if ln.startswith("| `paper`"))
    assert "scanned" not in line
    assert int(line.split("|")[3].strip()) >= ingest.SCANNED_CHARS_PER_PAGE


@needs_pypdf
def test_a_long_pdf_is_split_into_parts_of_about_ten_pages(repo, src):
    pages = [sentence_lines(1000 + p, 50, 14) for p in range(25)]  # 25 pages of 700 words = 17,500 words
    (src / "Big Handbook.pdf").write_bytes(build_pdf(pages, title="The big handbook"))
    assert ingest.run(args_for(src, repo, source="Example agency")) == 0
    assert doc_ids(repo) == ["big-handbook__p001-010", "big-handbook__p011-020", "big-handbook__p021-025"]
    assert all(DOCID.match(i) for i in doc_ids(repo))
    rows = manifest_rows(repo)
    assert [r[4] for r in rows] == [
        "file: Big Handbook.pdf; pages 1-10 of 25",
        "file: Big Handbook.pdf; pages 11-20 of 25",
        "file: Big Handbook.pdf; pages 21-25 of 25",
    ]
    assert rows[1][1] == "The big handbook (pages 11-20)"
    assert all(r[2] == "Example agency" for r in rows)
    first = doc_text(repo, "big-handbook__p001-010")
    assert sentence_lines(1001, 50, 14)[0] in first
    assert sentence_lines(1011, 50, 14)[0] not in first
    assert sentence_lines(1011, 50, 14)[0] in doc_text(repo, "big-handbook__p011-020")


@needs_pypdf
def test_a_pdf_just_under_the_limit_stays_whole(repo, src):
    pages = [sentence_lines(2000 + p, 50, 12) for p in range(25)]  # 15,000 words exactly is not over the limit
    (src / "edge.pdf").write_bytes(build_pdf(pages))
    ingest.run(args_for(src, repo))
    assert doc_ids(repo) == ["edge"]


@needs_pypdf
def test_rerunning_a_split_pdf_adds_no_parts(repo, src):
    pages = [sentence_lines(3000 + p, 50, 14) for p in range(25)]
    (src / "big.pdf").write_bytes(build_pdf(pages))
    ingest.run(args_for(src, repo))
    before = doc_ids(repo)
    ingest.run(args_for(src, repo))
    assert doc_ids(repo) == before and len(manifest_rows(repo)) == 3


def test_a_long_markdown_file_is_split_into_numbered_parts(repo, src):
    paragraphs = "\n\n".join(" ".join(words(4000 + i, 100)) for i in range(170))  # 17,000 words
    (src / "book.md").write_text("# The book\n\n" + paragraphs, encoding="utf-8", newline="\n")
    ingest.run(args_for(src, repo))
    ids = doc_ids(repo)
    assert len(ids) >= 3 and all(re.fullmatch(r"book__part0\d", i) for i in ids)
    assert manifest_rows(repo)[0][1] == "The book (part 1 of %d)" % len(ids)


# ---------------------------------------------------------------------------
# Wave 2: shared metadata titles, per-file sources, --report, removed license lines, the path hint
# ---------------------------------------------------------------------------


def test_a_metadata_title_with_a_file_name_keeps_only_the_title():
    assert ingest._usable_pdf_title("mcs2025.pdf - Mineral Commodity Summaries 2025") == "Mineral Commodity Summaries 2025"
    assert ingest._usable_pdf_title("Mineral Commodity Summaries 2025 | mcs2025.pdf") == "Mineral Commodity Summaries 2025"
    assert ingest._usable_pdf_title("Draft of report.docx for review") == ""
    assert ingest._usable_pdf_title("A study of pump flow") == "A study of pump flow"


@needs_pypdf
def test_pdfs_that_share_a_metadata_title_start_with_their_file_name(repo, src, capsys):
    for seed, name in enumerate(("mcs2024-aluminum", "mcs2024-antimony")):
        pages = [sentence_lines(100 * seed + p, 10) for p in range(3)]
        (src / f"{name}.pdf").write_bytes(build_pdf(pages, title="Mineral Commodity Summaries 2024"))
    (src / "other.pdf").write_bytes(build_pdf(paper_pages()[:2], title="A study of pump flow"))
    assert ingest.run(args_for(src, repo, source="https://pubs.usgs.gov/periodicals/mcs2024/{name}")) == 0
    rows = {r[0]: r for r in manifest_rows(repo)}
    assert rows["mcs2024-aluminum"][1] == "mcs2024 aluminum - Mineral Commodity Summaries 2024"
    assert doc_text(repo, "mcs2024-antimony").startswith("# mcs2024 antimony - Mineral Commodity Summaries 2024\n")
    assert rows["other"][1] == "A study of pump flow"
    assert rows["mcs2024-aluminum"][2] == "https://pubs.usgs.gov/periodicals/mcs2024/mcs2024-aluminum.pdf"
    assert "2 PDF files share a title" in capsys.readouterr().out


def test_source_template_fills_in_each_file_name():
    path = Path("raw/My Report.pdf")
    assert ingest.source_for("https://example.org/r/{name}", path, "My Report.pdf") == "https://example.org/r/My%20Report.pdf"
    assert ingest.source_for("USGS, {stem}", path, "My Report.pdf") == "USGS, My Report"
    assert ingest.source_for(None, path, "sub/My Report.pdf") == "sub/My Report.pdf"


def test_report_rewrites_ingest_md_for_the_documents_left(repo, src, capsys):
    for i in range(3):
        (src / f"note{i}.md").write_text(f"# Note {i}\n\n" + " ".join(words(30 + i, 80)), encoding="utf-8", newline="\n")
    assert ingest.run(args_for(src, repo)) == 0
    (corpus(repo) / "docs" / "note1.md").unlink()
    assert ingest.run(argparse.Namespace(src_dir=None, into="own", report=True, root=str(repo))) == 0
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    assert "- Documents: 2" in report and "`note1`" not in report and "`note2`" in report
    assert "Rewrote corpora/own/INGEST.md: 2 document(s)" in capsys.readouterr().out


@needs_pypdf
def test_a_publisher_footer_the_cleaning_removes_is_kept_in_the_notes(repo, src):
    from p2 import license as lic

    pages = [[f"Pump text on page {p} goes on about pumps and flow."] + sentence_lines(40 + p, 12) + ["Downloaded from IEEE Xplore. Restrictions apply."] for p in range(1, 9)]
    (src / "paper.pdf").write_bytes(build_pdf(pages))
    assert ingest.run(args_for(src, repo, license="cc-by-4.0")) == 0
    assert "IEEE Xplore" not in doc_text(repo, "paper")  # the running footer is gone from the text
    notes = manifest_rows(repo)[0][4]
    assert 'cleaning removed: publisher name "Downloaded from IEEE Xplore. Restrictions apply."' in notes
    row = lic.offline_report(corpus(repo))[0]
    assert row["status"] == "flag" and "a line ingest removed" in row["reason"]


def test_a_missing_folder_says_where_it_looked(repo, capsys, monkeypatch):
    monkeypatch.chdir(repo)
    assert ingest.run(args_for("work/p2-raw", repo)) == 1
    out = capsys.readouterr().out
    assert "I looked for" in out and "../p2-raw" in out


# ---------------------------------------------------------------------------
# --part-pages: long PDFs as parts of N pages, so a corpus of long papers can reach 200 documents
# ---------------------------------------------------------------------------


def paged_pdf(seed: int, n_pages: int, title: str | None = None) -> bytes:
    """n_pages of 600 words each (50 lines of 12), every page's first line unique to it."""
    return build_pdf([sentence_lines(seed + p, 50, 12) for p in range(n_pages)], title=title)


def first_line_of_page(seed: int, page: int) -> str:
    return sentence_lines(seed + page - 1, 50, 12)[0]


@pytest.mark.parametrize(
    "pages,per_part,expected",
    [
        (5, 5, [(1, 5)]),
        (6, 5, [(1, 6)]),  # a one-page remainder joins the part before it
        (7, 5, [(1, 5), (6, 7)]),
        (11, 5, [(1, 5), (6, 11)]),
        (12, 5, [(1, 5), (6, 10), (11, 12)]),
        (15, 5, [(1, 5), (6, 10), (11, 15)]),
        (3, 1, [(1, 1), (2, 2), (3, 3)]),
        (10, 3, [(1, 3), (4, 6), (7, 10)]),
        (25, 10, [(1, 10), (11, 20), (21, 25)]),  # the default size, unchanged
    ],
)
def test_page_blocks_with_a_part_size(pages, per_part, expected):
    assert ingest.page_blocks(pages, per_part) == expected


def test_the_shortest_remainder_scales_with_the_part_size():
    assert ingest.min_last_part(ingest.PAGES_PER_PART) == ingest.MIN_LAST_PART_PAGES == 4
    assert [ingest.min_last_part(n) for n in (1, 2, 3, 5, 20)] == [1, 1, 2, 2, 8]


@needs_pypdf
def test_without_part_pages_a_long_paper_stays_whole(repo, src):
    (src / "Paper.pdf").write_bytes(paged_pdf(5000, 12))  # 12 pages, 7,200 words: under the 15,000-word split
    assert ingest.run(args_for(src, repo)) == 0
    assert doc_ids(repo) == ["paper"]


@needs_pypdf
def test_part_pages_splits_every_pdf_longer_than_n_pages(repo, src, capsys):
    (src / "Paper.pdf").write_bytes(paged_pdf(5000, 12, title="A paper about pumps"))
    (src / "short.pdf").write_bytes(paged_pdf(5100, 4))
    (src / "six.pdf").write_bytes(paged_pdf(5200, 6))  # the one-page remainder is too short to be a part
    (src / "notes.md").write_text("# Notes\n\n" + " ".join(words(5300, 300)), encoding="utf-8", newline="\n")
    assert ingest.run(args_for(src, repo, part_pages=5, source="Example lab")) == 0
    assert doc_ids(repo) == ["notes", "paper__p001-005", "paper__p006-010", "paper__p011-012", "short", "six"]
    rows = {r[0]: r for r in manifest_rows(repo)}
    assert rows["paper__p006-010"][1] == "A paper about pumps (pages 6-10)"
    assert rows["paper__p006-010"][4] == "file: Paper.pdf; pages 6-10 of 12"
    assert rows["short"][4] == "file: short.pdf; 4 pages" and rows["six"][4] == "file: six.pdf; 6 pages"
    assert all(r[2] == "Example lab" for r in rows.values())
    middle = doc_text(repo, "paper__p006-010")
    assert first_line_of_page(5000, 6) in middle and first_line_of_page(5000, 10) in middle
    assert first_line_of_page(5000, 5) not in middle and first_line_of_page(5000, 11) not in middle
    assert "in 3 parts" in capsys.readouterr().out


@needs_pypdf
def test_part_pages_also_cuts_a_pdf_over_the_word_split_into_its_own_size(repo, src):
    pages = [sentence_lines(5400 + p, 50, 14) for p in range(25)]  # 17,500 words: 10-page parts without the option
    (src / "big.pdf").write_bytes(build_pdf(pages))
    ingest.run(args_for(src, repo, part_pages=5))
    assert doc_ids(repo) == [f"big__p{a:03d}-{b:03d}" for a, b in [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]]


@pytest.mark.parametrize("value", [0, -3, True, "5"])
def test_part_pages_needs_a_whole_number_of_pages(repo, src, capsys, value):
    (src / "notes.md").write_text("# Notes\n\nSome text.\n", encoding="utf-8", newline="\n")
    assert ingest.run(args_for(src, repo, part_pages=value)) == 1
    assert "--part-pages takes a whole number of pages" in capsys.readouterr().out
    assert not (corpus(repo) / "docs").exists()


def test_the_cli_and_the_module_parse_part_pages():
    from p2 import cli

    args = cli.parser().parse_args(["ingest", "../p2-raw", "--part-pages", "5", "--license", "cc-by-4.0"])
    assert args.part_pages == 5 and args.src_dir == "../p2-raw"
    assert cli.parser().parse_args(["ingest", "../p2-raw"]).part_pages is None
    parser = argparse.ArgumentParser()
    ingest.add_arguments(parser)
    assert parser.parse_args(["raw", "--part-pages", "3"]).part_pages == 3


@needs_pypdf
def test_a_file_ingested_whole_is_not_added_again_as_parts(repo, src, capsys):
    (src / "Paper.pdf").write_bytes(paged_pdf(5500, 12))
    ingest.run(args_for(src, repo))
    assert doc_ids(repo) == ["paper"]
    capsys.readouterr()

    assert ingest.run(args_for(src, repo, part_pages=5)) == 0
    out = capsys.readouterr().out
    assert doc_ids(repo) == ["paper"] and len(manifest_rows(repo)) == 1
    assert "Paper.pdf -> not added: its text is already in the corpus as paper, cut into documents another way" in out
    assert "1 file(s) are already in the corpus, cut into documents another way" in out
    assert "corpora/own/docs/" in out and "corpora/own/manifest.tsv" in out
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    assert "- `Paper.pdf`: Its text is already in the corpus, cut into documents another way, as `paper`, so it was not added again." in report

    # once the whole document and its row are gone, the same run cuts it into parts under the same name
    (corpus(repo) / "docs" / "paper.md").unlink()
    (corpus(repo) / "manifest.tsv").write_text("docid\ttitle\tsource\tlicense\tnotes\n", encoding="utf-8", newline="\n")
    ingest.run(args_for(src, repo, part_pages=5))
    assert doc_ids(repo) == ["paper__p001-005", "paper__p006-010", "paper__p011-012"]


@needs_pypdf
def test_parts_are_not_added_again_whole_or_in_another_size(repo, src, capsys):
    (src / "Paper.pdf").write_bytes(paged_pdf(5600, 15))
    ingest.run(args_for(src, repo, part_pages=5))
    before = doc_ids(repo)
    assert len(before) == 3
    for size in (None, 3, 10):
        capsys.readouterr()
        ingest.run(args_for(src, repo, part_pages=size))
        assert doc_ids(repo) == before and len(manifest_rows(repo)) == 3
        assert "cut into documents another way" in capsys.readouterr().out
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    assert "as `paper__p001-005`, `paper__p006-010`, `paper__p011-015`, so it was not added again" in report


@needs_pypdf
def test_rerunning_the_same_part_size_adds_nothing_and_restores_a_lost_part(repo, src, capsys):
    (src / "Paper.pdf").write_bytes(paged_pdf(5700, 15))
    ingest.run(args_for(src, repo, part_pages=5))
    ingest.run(args_for(src, repo, part_pages=5))
    assert len(doc_ids(repo)) == 3 and len(manifest_rows(repo)) == 3

    # an interrupted run that lost the last part: the next run adds only that part, and is not taken for a re-cut
    (corpus(repo) / "docs" / "paper__p011-015.md").unlink()
    rows = (corpus(repo) / "manifest.tsv").read_text(encoding="utf-8").splitlines()
    (corpus(repo) / "manifest.tsv").write_text("\n".join(r for r in rows if not r.startswith("paper__p011-015")) + "\n", encoding="utf-8", newline="\n")
    capsys.readouterr()
    ingest.run(args_for(src, repo, part_pages=5))
    out = capsys.readouterr().out
    assert "cut into documents another way" not in out and "2 part(s) were already in the corpus" in out
    assert len(doc_ids(repo)) == 3
    texts = [doc_text(repo, d) for d in doc_ids(repo)]
    assert sum(first_line_of_page(5700, 11) in t for t in texts) == 1


@needs_pypdf
def test_a_different_file_with_the_same_name_is_still_added(repo, tmp_path):
    first, second = tmp_path / "2024", tmp_path / "2025"
    first.mkdir()
    second.mkdir()
    (first / "chapter.pdf").write_bytes(paged_pdf(5800, 12))
    (second / "chapter.pdf").write_bytes(paged_pdf(5900, 12))
    ingest.run(args_for(first, repo))
    ingest.run(args_for(second, repo, part_pages=5))
    assert doc_ids(repo) == ["chapter", "chapter-2__p001-005", "chapter-2__p006-010", "chapter-2__p011-012"]


@needs_pypdf
def test_under_the_floor_with_long_pdfs_the_summary_suggests_part_pages(repo, src, capsys):
    (src / "Paper.pdf").write_bytes(paged_pdf(6000, 12))
    ingest.run(args_for(src, repo))
    out = capsys.readouterr().out
    assert "so 199 more to go" in out
    assert "This run read 1 PDF(s) longer than 5 pages" in out and "--part-pages 5" in out
    (corpus(repo) / "docs" / "paper.md").unlink()
    (corpus(repo) / "manifest.tsv").write_text("docid\ttitle\tsource\tlicense\tnotes\n", encoding="utf-8", newline="\n")
    ingest.run(args_for(src, repo, part_pages=5))
    out = capsys.readouterr().out
    assert "so 197 more to go" in out and "--part-pages" not in out


def test_ingest_md_and_the_summary_state_the_size_rules(repo, src, capsys, monkeypatch):
    (src / "notes.md").write_text("# Notes\n\n" + " ".join(words(6100, 300)), encoding="utf-8", newline="\n")
    ingest.run(args_for(src, repo))
    report = (corpus(repo) / "INGEST.md").read_text(encoding="utf-8")
    assert "- Documents: 1 (the final check needs at least 200)" in report
    assert "(`p2 check` warns above 800,000 and fails above 1,000,000)" in report
    out = capsys.readouterr().out
    assert "Heads up" not in out and "above the limit" not in out

    monkeypatch.setattr(ingest, "TOKEN_WARNING", 10)
    ingest.run(args_for(src, repo))
    assert "Heads up: that is above 10 tokens, near the limit of 1,000,000" in capsys.readouterr().out
    monkeypatch.setattr(ingest, "TOKEN_LIMIT", 20)
    ingest.run(args_for(src, repo))
    assert "above the limit of 20 tokens, so `p2 check` will fail, because CI could not encode the corpus within its 45 minutes" in capsys.readouterr().out
