"""Tokenizers, chunking, document-level lifting, and corpus loading and checks."""

from pathlib import Path

import pytest
from core_repo import make_repo, write

from p2 import corpus, textproc


def test_lab_tokenizer_is_the_12_lab_regex():
    assert textproc.tokens_lab("What does error SYN-4127 mean?") == ["error", "syn", "4127", "mean"]


def test_codes_tokenizer_keeps_codes_whole_and_emits_parts():
    assert textproc.tokens_codes("See 30 CFR 75.403.") == ["see", "30", "cfr", "75.403", "75", "403"]
    assert textproc.tokens_codes("SYN-4127") == ["syn-4127", "syn", "4127"]
    assert textproc.tokens_codes("sync.chunk_size_kb") == ["sync.chunk_size_kb", "sync", "chunk", "size", "kb"]
    assert textproc.tokens_codes("the dust") == ["dust"]
    with pytest.raises(ValueError):
        textproc.tokens("x", "bogus")


def test_split_words_windows_are_exact_slices_with_overlap():
    text = "one two  three\nfour five six seven"
    pieces = corpus.split_words(text, words=3, overlap=1)
    assert pieces == ["one two  three", "three\nfour five", "five six seven"]
    assert all(p in text for p in pieces)
    assert corpus.split_words(text, words=0, overlap=0) == [text]
    assert corpus.split_words(text, words=50, overlap=10) == [text]


def test_chunk_ids_and_doc_level_max():
    doc = corpus.Doc("d1", "T", "# T\n\na b c d e f g", Path("d1.md"))
    chunks = corpus.chunks(doc, words=4, overlap=0)
    assert [c.id for c in chunks] == ["d1#0", "d1#1", "d1#2"]  # 9 words in windows of 4
    assert corpus.docid_of("cfr30-75.403#12") == "cfr30-75.403"
    lifted = corpus.doc_level([("b#0", 1.0), ("a#1", 2.0), ("a#0", 0.5), ("c#0", 2.0)])
    assert lifted == [("a", 2.0), ("c", 2.0), ("b", 1.0)]


def test_load_with_extra_corpus(tmp_path):
    root = make_repo(tmp_path)
    shared = corpus.load(root, "shared")
    assert shared.ids == sorted(shared.ids)
    assert shared.docs["cfr30-75.403"].title == "§ 75.403 Maintenance of incombustible content of rock dust"
    extra = tmp_path / "extra"
    write(extra / "canary-1.md", "# Canary\n\nA planted canary document.\n")
    both = corpus.load(root, "shared", extra=extra)
    assert "canary-1" in both.docs and len(both.docs) == len(shared.docs) + 1
    write(extra / "cfr30-75.403.md", "# Clash\n\nSame id.\n")
    with pytest.raises(ValueError, match="repeats document ids"):
        corpus.load(root, "shared", extra=extra)


def test_document_and_manifest_checks(tmp_path):
    root = make_repo(tmp_path)
    assert corpus.check_docs(root / "corpora" / "shared" / "docs") == []
    assert corpus.check_manifest(root, "shared") == []
    write(root / "corpora" / "shared" / "docs" / "no-title.md", "Body without a title.\n")
    write(root / "corpora" / "shared" / "docs" / "Bad_Case.md", "# T\n\nbody\n")
    problems = corpus.check_docs(root / "corpora" / "shared" / "docs")
    assert any("no-title.md does not start" in p for p in problems)
    assert any("Bad_Case.md has a document id" in p for p in problems)
    assert any("no manifest row" in p for p in corpus.check_manifest(root, "shared"))
