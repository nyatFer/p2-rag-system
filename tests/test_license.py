"""Tests for p2 license: the vocabulary, the text scan, the reviewed rule, and the online lookups.

The fixtures under tests/fixtures/license/ are:
  corpus/   one document per vocabulary class and per text-scan pattern, plus manifest problems
  online/   real answers recorded from the live services on 2026-10-04 (Crossref, arXiv OAI-PMH, the PMC Cloud Service),
            so the parsers are tested against the real response shapes without touching the network.
Set P2_ONLINE_TESTS=1 to also run the few tests that call the live services.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from p2 import license as lic

FIXTURES = Path(__file__).parent / "fixtures" / "license"
CORPUS = FIXTURES / "corpus"
ONLINE = FIXTURES / "online"


def by_id(rows):
    return {r["docid"]: r for r in rows}


@pytest.fixture(scope="module")
def report():
    rows = lic.offline_report(CORPUS)
    return rows


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

ALLOWED_VALUES = (
    ["us-gov-public-domain", "public-domain", "cc0-1.0", "own-work", "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"]
    + [f"cc-by-{v}" for v in ("2.0", "2.1", "2.5", "3.0", "4.0")]
    + [f"cc-by-sa-{v}" for v in ("2.0", "2.1", "2.5", "3.0", "4.0")]
)


@pytest.mark.parametrize("value", ALLOWED_VALUES)
def test_every_allowed_value_passes(value):
    assert lic.check_license_value(value) == (True, "")
    assert lic.ALLOWED >= {value}


def test_allowed_set_is_exactly_the_spec_vocabulary():
    assert set(ALLOWED_VALUES) == set(lic.ALLOWED)


@pytest.mark.parametrize(
    "value,needle",
    [
        ("cc-by-nc-4.0", "NonCommercial"),
        ("cc-by-nd-4.0", "NonCommercial or NoDerivatives"),
        ("cc-by-nc-sa-4.0", "NonCommercial"),
        ("cc-by-nc-nd-3.0", "NonCommercial"),
        ("CC-BY-ND-4.0", "NonCommercial"),
        ("by-nc", "NonCommercial"),
        ("unknown", "unknown"),
        ("UNKNOWN", "unknown"),
        ("all-rights-reserved", "all rights reserved"),
        ("", "empty"),
        ("   ", "empty"),
        (None, "empty"),
        ("gpl-3.0", "not in the vocabulary"),
        ("cc-by-1.0", "not in the vocabulary"),
        ("cc-by-5.0", "not in the vocabulary"),
        ("creative commons", "not in the vocabulary"),
        ("CC BY 4.0", "use `cc-by-4.0`"),
        ("cc_by_sa_4.0", "use `cc-by-sa-4.0`"),
        ("MIT License", "not in the vocabulary"),
    ],
)
def test_every_failing_class_fails_with_a_reason(value, needle):
    ok, reason = lic.check_license_value(value)
    assert not ok
    assert needle in reason


def test_nc_and_nd_are_tokens_not_substrings():
    # "second" and "undergraduate" contain nd, "snc" and "rsnc" contain nc; none of these is a license element
    assert lic.check_license_value("second-license")[1].count("NonCommercial") == 0
    assert lic.check_license_value("undergrad-work")[1].count("NonCommercial") == 0


# ---------------------------------------------------------------------------
# The text scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,kind",
    [
        ("\xa9 2019 Ada Example", "copyright line"),
        ("Copyright holder: (c) 2018 Ada", "copyright line"),
        ("(C) 1999 Ada", "copyright line"),
        ("Copyright 2015 by Ada", "copyright line"),
        ("copyright \xa9 2001", "copyright line"),
        ("Copyright (c) 1987", "copyright line"),
        ("All rights reserved.", "all rights reserved"),
        ("all rights\nreserved", "all rights reserved"),
        ("Published by Elsevier B.V.", "publisher name"),
        ("published by elsevier B.V.", "publisher name"),
        ("ELSEVIER. Reprints and permissions: see the journal site", "publisher name"),
        ("This chapter is licensed by Springer Nature", "publisher name"),
        ("Published by John Wiley & Sons, Wiley Online Library", "publisher name"),
        ("Downloaded from IEEE Xplore. Restrictions apply.", "publisher name"),
        ("Reprinted from the Journal of the ACS", "publisher name"),
        ("Published by Taylor & Francis", "publisher name"),
        ("Copyright Taylor and Francis", "publisher name"),
        ("published by taylor & francis", "publisher name"),
        ("Reprints and permissions: SAGE Publications", "publisher name"),
        ("Published by Cambridge University Press", "publisher name"),
        ("oxford university press, on behalf of the society", "publisher name"),
        ("ASTM E2500, a licensed copy", "publisher name"),
        ("ISO 9001, reproduced under license", "publisher name"),
        ("Copyright ACME Corporation. Reproduced under license.", "copyright line"),
        ("This report was prepared as an account of work sponsored by an agency of the United States Government.", "contractor notice"),
        ("Battelle Memorial Institute under Contract DE-AC05-76RL01830", "contractor notice"),
        ("operated by Battelle for the U.S. Department of Energy", "contractor notice"),
        ("Modified from Smith and others (2019), courtesy of the American Geophysical Union", "credit line"),
        ("licensed under CC BY-NC 4.0", "NC or ND notice"),
        ("CC BY-ND", "NC or ND notice"),
        ("cc-by-nc-sa", "NC or ND notice"),
        ("CC BY-NC-ND 4.0", "NC or ND notice"),
        ("https://creativecommons.org/licenses/by-nc-sa/4.0/", "NC or ND notice"),
        ("https://creativecommons.org/licenses/by-nd/3.0/", "NC or ND notice"),
        ("Creative Commons Attribution-NonCommercial", "NC or ND notice"),
        ("Attribution NoDerivatives", "NC or ND notice"),
        ("reprinted with permission from the authors", "permission"),
        ("Reprinted with the permission of ACS", "permission"),
        ("reproduced with permission", "permission"),
        ("Photograph used with permission from Freeport", "permission"),
        (
            "Although this report is in the public domain, permission must be secured from the individual copyright owners to reproduce any material.",
            "third-party material",
        ),
        ("Permission to reproduce copyrighted items must be secured from the copyright owner.", "third-party material"),
        ("This information product may also contain copyrighted material as noted in the text.", "third-party material"),
    ],
)
def test_every_scan_pattern_finds_its_hit(text, kind):
    kinds = {h["kind"] for h in lic.scan_text("Intro line.\n" + text + "\nClosing line.")}
    assert kind in kinds


@pytest.mark.parametrize(
    "text",
    [
        "The second candidate gave sage advice.",
        "An advisory group read the isolated isotope data.",
        "The undergraduate read section 75.403(c) 2000 pounds of rock.",
        "Wileyan style in Sagebrush station, ISOtope label.",
        "A copyright law course covers fair use.",
        "The Copyright Act of 1976 is a federal law.",
        "ordinary text with (a) and (b) lists",
        "The pump is rated at 1999 gallons.",
        "Section 75.403(c) applies to sage, iso-propyl and acs units.",
    ],
)
def test_scan_leaves_lookalikes_alone(text):
    assert lic.scan_text(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "See ASTM E2500 for the practice.",
        "a profile of ISO 8601 dates",
        "IEEE 754 double precision",
        "The American Community Survey (ACS) counts households.",
        "the Advanced Camera for Surveys (ACS) on Hubble",
        "Professor Wiley Post flew around the world.",
        "SAGE grouse habitat",
        "Smith, J., 2019, Nickel deposits: Springer, Berlin, 300 p.",
    ],
)
def test_a_publisher_name_without_words_about_rights_is_only_a_mention(text):
    hits = lic.scan_text(text)
    assert hits and {h["kind"] for h in hits} == {lic.MENTION}


def test_a_mention_is_listed_but_does_not_flag_the_document(report):
    row = by_id(report)["clean-publisher-mentions"]
    assert row["status"] == "ok" and not row["flags"]
    assert "mentions ASTM, ACS, IEEE without words about rights" in row["reason"] and "not a flag" in row["reason"]


def test_evidence_that_cleaning_removed_is_read_from_the_notes(tmp_path):
    corpus = tmp_path / "own"
    (corpus / "docs").mkdir(parents=True)
    (corpus / "docs" / "paper.md").write_text("# Paper\n\nPump notes.\n", encoding="utf-8", newline="\n")
    note = 'file: paper.pdf; cleaning removed: publisher name "Downloaded from IEEE Xplore. Restrictions apply."'
    (corpus / "manifest.tsv").write_text(f"docid\ttitle\tsource\tlicense\tnotes\npaper\tPaper\thttps://x\tcc-by-4.0\t{note}\n", encoding="utf-8", newline="\n")
    row = lic.offline_report(corpus)[0]
    assert row["status"] == "flag" and "a line ingest removed" in row["reason"]
    reviewed = note + '; reviewed: the page says "This article is licensed under CC BY 4.0" (https://x)'
    (corpus / "manifest.tsv").write_text(f"docid\ttitle\tsource\tlicense\tnotes\npaper\tPaper\thttps://x\tcc-by-4.0\t{reviewed}\n", encoding="utf-8", newline="\n")
    assert lic.offline_report(corpus)[0]["status"] == "ok"


def test_hit_carries_line_number_and_the_matching_line():
    hits = lic.scan_text("# Title\n\nSome text.\nPublished by Elsevier B.V. on behalf of the authors.\nMore text.\n")
    assert len(hits) == 1
    assert hits[0]["line_no"] == 4
    assert hits[0]["line"] == "Published by Elsevier B.V. on behalf of the authors."
    assert hits[0]["kind"] == "publisher name"


def test_reviewed_reason_rule():
    assert lic.reviewed_reason('reviewed: the footer says "CC BY 4.0"') == 'the footer says "CC BY 4.0"'
    assert lic.reviewed_reason('file: a.pdf; reviewed: author line only, "licensed under CC BY"') == 'author line only, "licensed under CC BY"'
    assert lic.reviewed_reason("Reviewed: fine, quote above") == "fine, quote above"
    assert lic.reviewed_reason("reviewed:") == ""
    assert lic.reviewed_reason("reviewed:   ") == ""
    assert lic.reviewed_reason("reviewed: -") == ""
    assert lic.reviewed_reason("I reviewed this") == ""
    assert lic.reviewed_reason("") == ""
    assert lic.reviewed_reason(None) == ""


# ---------------------------------------------------------------------------
# The offline report on the fixture corpus
# ---------------------------------------------------------------------------


def test_report_rows_are_sorted_and_have_the_promised_fields(report):
    ids = [r["docid"] for r in report]
    assert ids == sorted(ids)
    for r in report:
        assert r["status"] in {"ok", "flag", "fail"}
        assert isinstance(r["reason"], str)
        assert r.docid == r["docid"] and r.status == r["status"]  # attribute access works too
        assert r.get("reason") == r.reason


@pytest.mark.parametrize("value", ALLOWED_VALUES)
def test_clean_document_with_each_allowed_license_is_ok(report, value):
    row = by_id(report)[f"vocab-{value}"]
    assert row["status"] == "ok"
    assert row["reason"] == ""


@pytest.mark.parametrize(
    "docid,needle",
    [
        ("fail-cc-by-nc-4.0", "NonCommercial"),
        ("fail-cc-by-nd-4.0", "NonCommercial or NoDerivatives"),
        ("fail-cc-by-nc-sa-4.0", "NonCommercial"),
        ("fail-cc-by-nc-nd-4.0", "NonCommercial"),
        ("fail-uppercase-nd", "NonCommercial"),
        ("fail-unknown", "unknown"),
        ("fail-all-rights-reserved", "all rights reserved"),
        ("fail-empty", "empty"),
        ("fail-unlisted-gpl", "not in the vocabulary"),
        ("fail-unlisted-cc-by-1.0", "not in the vocabulary"),
        ("fail-spelled-with-spaces", "use `cc-by-4.0`"),
    ],
)
def test_each_failing_vocabulary_class_fails(report, docid, needle):
    row = by_id(report)[docid]
    assert row["status"] == "fail"
    assert needle in row["reason"]


SCAN_DOCS = {
    "scan-copyright-symbol": "copyright line",
    "scan-copyright-c-year": "copyright line",
    "scan-copyright-word-year": "copyright line",
    "scan-copyright-word-1999": "copyright line",
    "scan-all-rights-reserved": "all rights reserved",
    "scan-all-rights-reserved-wrapped": "all rights reserved",
    "scan-publisher-elsevier": "publisher name",
    "scan-publisher-springer": "publisher name",
    "scan-publisher-wiley": "publisher name",
    "scan-publisher-ieee": "publisher name",
    "scan-publisher-acs": "publisher name",
    "scan-publisher-taylor": "publisher name",
    "scan-publisher-taylor-and": "publisher name",
    "scan-publisher-sage": "publisher name",
    "scan-publisher-cup": "publisher name",
    "scan-publisher-oup": "publisher name",
    "scan-publisher-astm": "publisher name",
    "scan-publisher-iso": "publisher name",
    "scan-cc-by-nc": "NC or ND notice",
    "scan-cc-by-nd": "NC or ND notice",
    "scan-cc-by-nc-sa-url": "NC or ND notice",
    "scan-cc-attribution-noncommercial": "NC or ND notice",
    "scan-permission": "permission",
    "scan-usgs-sentence": "third-party material",
    "scan-usgs-sentence-gov": "third-party material",
    "scan-many-hits": "publisher name",
    "scan-contractor-notice": "contractor notice",
    "scan-credit-line": "credit line",
}


@pytest.mark.parametrize("docid,kind", sorted(SCAN_DOCS.items()))
def test_each_scan_pattern_flags_the_document_with_the_matching_line(report, docid, kind):
    row = by_id(report)[docid]
    assert row["status"] == "flag"
    assert kind in row["reason"]
    assert "line " in row["reason"]
    assert row["flags"], "the hits themselves are kept on the row"
    assert any(h["kind"] == kind for h in row["flags"])
    assert "reviewed:" in row["reason"]  # tells the student how to resolve it


def test_a_flag_names_the_actual_line(report):
    row = by_id(report)["scan-copyright-symbol"]
    assert "\xa9 2019 Ada Example" in row["reason"]


def test_many_hits_are_summarised_not_listed_in_full(report):
    row = by_id(report)["scan-many-hits"]
    assert "publisher name on 5 lines" in row["reason"]
    assert "(and 3 more)" in row["reason"]
    assert len(row["flags"]) >= 7  # all of them stay in flags


@pytest.mark.parametrize("docid", ["clean-lookalikes", "clean-iso-inside-word", "clean-publisher-mentions"])
def test_lookalike_text_is_not_flagged(report, docid):
    assert by_id(report)[docid]["status"] == "ok"


def test_a_flag_passes_only_with_a_written_reason(report):
    rows = by_id(report)
    assert rows["review-resolved"]["status"] == "ok"
    assert rows["review-resolved"]["reason"].startswith("reviewed: the page footer says")
    assert rows["review-with-other-notes"]["status"] == "ok"
    for docid in ("review-empty-reason", "review-only-punctuation", "review-word-without-colon"):
        assert rows[docid]["status"] == "flag", docid
    assert rows["review-on-clean-doc"]["status"] == "ok"


def test_reviewed_does_not_rescue_a_failing_license(report):
    row = by_id(report)["review-does-not-rescue-license"]
    assert row["status"] == "fail"
    assert "unknown" in row["reason"]
    assert "text scan also found" in row["reason"]


def test_manifest_problems_are_failures(report):
    rows = by_id(report)
    assert rows["orphan-doc"]["status"] == "fail" and "no manifest row" in rows["orphan-doc"]["reason"]
    assert rows["missing-file"]["status"] == "fail" and "does not exist" in rows["missing-file"]["reason"]
    assert rows["Bad_ID"]["status"] == "fail" and "not a valid document id" in rows["Bad_ID"]["reason"]
    assert rows["too-many-columns"]["status"] == "fail" and "columns" in rows["too-many-columns"]["reason"]
    dups = [r for r in report if r["docid"] == "vocab-mit"]
    assert sorted(r["status"] for r in dups) == ["fail", "ok"]


def test_every_vocabulary_class_and_scan_pattern_is_covered_by_a_fixture(report):
    statuses = {r["docid"]: r["status"] for r in report}
    assert sum(1 for k in statuses if k.startswith("vocab-")) >= len(ALLOWED_VALUES)
    kinds_in_fixtures = {kind for _, kind in SCAN_DOCS.items()}
    assert kinds_in_fixtures == {kind for kind, _ in lic.SCAN_PATTERNS}


def test_offline_report_accepts_the_docs_folder_too():
    assert len(lic.offline_report(CORPUS / "docs")) == len(lic.offline_report(CORPUS))


def test_empty_and_missing_corpora(tmp_path):
    assert lic.offline_report(tmp_path) == []
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n\nText.\n", encoding="utf-8", newline="\n")
    rows = lic.offline_report(tmp_path)  # a document and no manifest at all
    assert [(r["docid"], r["status"]) for r in rows] == [("a", "fail")]


def test_manifest_with_a_wrong_header_fails_once(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "manifest.tsv").write_text("id\tname\nx\ty\n", encoding="utf-8", newline="\n")
    rows = lic.offline_report(tmp_path)
    assert rows[0]["status"] == "fail"
    assert "header" in rows[0]["reason"]


# ---------------------------------------------------------------------------
# Online lookups, parsed from recorded real answers
# ---------------------------------------------------------------------------


def recorded(name):
    return (ONLINE / name).read_bytes()


def fake_get(table, calls=None):
    """A stand-in for the network: url fragment -> (status, bytes). Unknown urls answer 404."""

    def get(url):
        if calls is not None:
            calls.append(url)
        for fragment, answer in table.items():
            if fragment in url:
                return answer
        return 404, b"Resource not found."

    return get


def test_crossref_cc_license_is_parsed():
    get = fake_get({"10.1103/PhysRevLett.116.061102": (200, recorded("crossref-prl.json"))})
    found = lic.lookup_crossref("10.1103/PhysRevLett.116.061102", get)
    assert found == [{"family": "cc-by", "version": "3.0", "raw": "http://creativecommons.org/licenses/by/3.0/"}]


def test_crossref_publisher_license_is_restricted():
    get = fake_get({"10.1016/j.jmb.2005.01.025": (200, recorded("crossref-jmb.json"))})
    found = lic.lookup_crossref("10.1016/j.jmb.2005.01.025", get)
    assert [f["family"] for f in found] == ["restricted"]
    assert "elsevier.com/tdm" in found[0]["raw"]


def test_crossref_work_without_license_key_gives_nothing():
    get = fake_get({"science": (200, recorded("crossref-science.json"))})
    assert lic.lookup_crossref("10.1126/science.1127647", get) == []


def test_crossref_unknown_doi_could_not_check():
    get = fake_get({"nope": (404, recorded("crossref-notfound.txt"))})
    with pytest.raises(lic.CouldNotCheck, match="does not know the DOI"):
        lic.lookup_crossref("10.9999/nope", get)


def test_crossref_garbage_and_server_errors_could_not_check():
    with pytest.raises(lic.CouldNotCheck):
        lic.lookup_crossref("10.1/x", fake_get({"x": (200, b"<html>maintenance</html>")}))
    with pytest.raises(lic.CouldNotCheck, match="HTTP 503"):
        lic.lookup_crossref("10.1/x", fake_get({"x": (503, b"")}))


def test_crossref_url_encodes_the_doi():
    calls = []
    get = fake_get({"crossref": (200, recorded("crossref-prl.json"))}, calls)
    lic.lookup_crossref("10.1002/(SICI)1097-4571(199806)49:8<693::AID-ASI4>3.0.CO;2-O", get)
    assert "%3C" in calls[0] and "%3E" in calls[0] and "<" not in calls[0]
    assert calls[0].startswith("https://api.crossref.org/works/10.1002/")


def test_arxiv_cc_license_is_parsed():
    get = fake_get({"1602.03837": (200, recorded("arxiv-1602.03837.xml"))})
    found = lic.lookup_arxiv("1602.03837", get)
    assert found == [{"family": "cc-by", "version": "4.0", "raw": "http://creativecommons.org/licenses/by/4.0/"}]


def test_arxiv_default_license_is_restricted():
    get = fake_get({"1706.03762": (200, recorded("arxiv-1706.03762.xml"))})
    found = lic.lookup_arxiv("1706.03762", get)
    assert [f["family"] for f in found] == ["restricted"]
    assert "nonexclusive-distrib" in found[0]["raw"]


def test_arxiv_record_without_license_element_is_restricted():
    get = fake_get({"hep-th": (200, recorded("arxiv-hep-th-9711200.xml"))})
    found = lic.lookup_arxiv("hep-th/9711200", get)
    assert [f["family"] for f in found] == ["restricted"]
    assert "no license recorded" in found[0]["raw"]


def test_arxiv_unknown_id_could_not_check():
    get = fake_get({"9999.99999": (200, recorded("arxiv-notfound.xml"))})
    with pytest.raises(lic.CouldNotCheck, match="idDoesNotExist"):
        lic.lookup_arxiv("9999.99999", get)


def test_arxiv_uses_the_oai_host_and_the_id_in_the_url():
    calls = []
    get = fake_get({"arXiv.org:1602.03837": (200, recorded("arxiv-1602.03837.xml"))}, calls)
    lic.lookup_arxiv("1602.03837", get)
    assert calls[0].startswith("https://oaipmh.arxiv.org/oai?verb=GetRecord")
    assert "metadataPrefix=arXivRaw" in calls[0]


def test_xml_with_a_dtd_is_refused():
    evil = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><OAI-PMH>&a;</OAI-PMH>'
    with pytest.raises(lic.CouldNotCheck, match="DTD"):
        lic.lookup_arxiv("1602.03837", fake_get({"arXiv": (200, evil)}))


def pmc_get(pmcid, version, meta_file, list_file=None):
    return fake_get(
        {
            f"prefix={pmcid}.": (200, recorded(list_file or f"pmc-list-{pmcid}.xml")),
            f"metadata/{pmcid}.{version}.json": (200, recorded(meta_file)),
        }
    )


def test_pmc_cc_by_is_parsed_from_the_newest_version():
    found = lic.lookup_pmc("PMC1182327", pmc_get("PMC1182327", 2, "pmc-meta-PMC1182327.2.json"))
    assert found == [{"family": "cc-by", "version": "", "raw": "CC BY"}]


def test_pmc_nc_nd_and_tdm_codes():
    nc = lic.lookup_pmc("PMC11000000", pmc_get("PMC11000000", 1, "pmc-meta-PMC11000000.1.json"))
    assert nc[0]["family"] == "cc-by-nc-nd"
    tdm = lic.lookup_pmc("PMC11000027", pmc_get("PMC11000027", 1, "pmc-meta-PMC11000027.1.json"))
    assert tdm[0]["family"] == "restricted" and tdm[0]["raw"] == "TDM"


def test_pmc_article_outside_the_open_access_dataset_is_restricted():
    found = lic.lookup_pmc("PMC99999999", fake_get({"prefix=PMC99999999.": (200, recorded("pmc-list-none.xml"))}))
    assert found[0]["family"] == "restricted"
    assert "not in the PMC open-access dataset" in found[0]["raw"]


def test_pmc_server_error_could_not_check():
    with pytest.raises(lic.CouldNotCheck, match="HTTP 500"):
        lic.lookup_pmc("PMC1", fake_get({"prefix": (500, b"")}))


@pytest.mark.parametrize(
    "url,family,version",
    [
        ("http://creativecommons.org/licenses/by/4.0/", "cc-by", "4.0"),
        ("https://creativecommons.org/licenses/by/4.0", "cc-by", "4.0"),
        ("https://creativecommons.org/licenses/by-sa/3.0/de/", "cc-by-sa", "3.0"),
        ("https://creativecommons.org/licenses/by-nc/4.0/", "cc-by-nc", "4.0"),
        ("https://creativecommons.org/licenses/by-nc-sa/2.5/", "cc-by-nc-sa", "2.5"),
        ("https://creativecommons.org/licenses/by-nd/4.0/", "cc-by-nd", "4.0"),
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "cc-by-nc-nd", "4.0"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0", "1.0"),
        ("https://creativecommons.org/publicdomain/mark/1.0/", "public-domain", ""),
        ("https://www.springer.com/tdm", "restricted", ""),
        ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "restricted", ""),
    ],
)
def test_license_urls_map_to_families(url, family, version):
    found = lic.found_from_url(url)
    assert (found["family"], found["version"]) == (family, version)


def test_extract_ids():
    ids = lic.extract_ids("Abbott et al., doi:10.1103/PhysRevLett.116.061102. arXiv:1602.03837v2")
    assert ids == {"doi": ["10.1103/PhysRevLett.116.061102"], "arxiv": ["1602.03837"], "pmc": []}
    assert lic.extract_ids("https://doi.org/10.1016/j.jmb.2005.01.025")["doi"] == ["10.1016/j.jmb.2005.01.025"]
    assert lic.extract_ids("(see 10.1002/(SICI)1097-4571(199806)49:8<693::AID-ASI4>3.0.CO;2-O)")["doi"][0].startswith("10.1002/(SICI)1097-4571(199806)49:8")
    assert lic.extract_ids("https://arxiv.org/abs/1602.03837")["arxiv"] == ["1602.03837"]
    assert lic.extract_ids("https://arxiv.org/pdf/1602.03837v1.pdf")["arxiv"] == ["1602.03837"]
    assert lic.extract_ids("https://arxiv.org/abs/hep-th/9711200")["arxiv"] == ["hep-th/9711200"]
    assert lic.extract_ids("arXiv:math.GT/0309136")["arxiv"] == ["math.GT/0309136"]
    assert lic.extract_ids("https://pmc.ncbi.nlm.nih.gov/articles/PMC1182327/")["pmc"] == ["PMC1182327"]
    assert lic.extract_ids("PMCID: PMC4301750")["pmc"] == ["PMC4301750"]
    assert lic.extract_ids("My own lab notes, 2025") == {"doi": [], "arxiv": [], "pmc": []}
    assert lic.extract_ids("") == {"doi": [], "arxiv": [], "pmc": []}


# ---------------------------------------------------------------------------
# Comparing the declared license with what a service says
# ---------------------------------------------------------------------------


def found(family, version="", raw="x"):
    return {"family": family, "version": version, "raw": raw}


@pytest.mark.parametrize(
    "declared,says,verdict",
    [
        ("cc-by-4.0", [found("cc-by", "4.0")], "match"),
        ("cc-by-3.0", [found("cc-by", "4.0")], "match"),  # version differs: noted, not failed
        ("cc-by-4.0", [found("cc-by", "")], "match"),  # PMC gives no version
        ("cc-by-sa-4.0", [found("cc-by-sa", "4.0")], "match"),
        ("cc0-1.0", [found("cc0", "1.0")], "match"),
        ("cc0-1.0", [found("public-domain")], "match"),
        ("public-domain", [found("cc0", "1.0")], "match"),
        ("cc-by-4.0", [found("restricted", raw="https://x/tdm"), found("cc-by", "4.0")], "match"),
        ("cc-by-4.0", [found("cc-by-nc", "4.0")], "mismatch"),
        ("cc-by-4.0", [found("cc-by-sa", "4.0")], "mismatch"),
        ("cc-by-sa-4.0", [found("cc-by", "4.0")], "mismatch"),
        ("cc-by-4.0", [found("cc-by-nc-nd", "")], "mismatch"),
        ("cc-by-4.0", [found("restricted", raw="https://www.elsevier.com/tdm/userlicense/1.0/")], "mismatch"),
        ("cc0-1.0", [found("cc-by", "4.0")], "mismatch"),
        ("own-work", [found("cc-by", "4.0")], "info"),
        ("us-gov-public-domain", [found("cc-by", "4.0")], "info"),
        ("us-gov-public-domain", [found("restricted", raw="https://x/tdm")], "restricted"),
        ("mit", [found("restricted", raw="https://x/tdm")], "restricted"),
        ("own-work", [found("cc-by-nc", "4.0")], "restricted"),
        ("cc-by-4.0", [], "info"),
    ],
)
def test_compare_licenses(declared, says, verdict):
    assert lic.compare_licenses(declared, says)[0] == verdict


def test_version_difference_is_mentioned():
    verdict, detail = lic.compare_licenses("cc-by-4.0", [found("cc-by", "3.0")])
    assert verdict == "match"
    assert "3.0" in detail and "4.0" in detail


def test_mismatch_detail_names_both_sides():
    verdict, detail = lic.compare_licenses("cc-by-4.0", [found("cc-by-nc", "4.0")])
    assert "cc-by-4.0" in detail and "cc-by-nc-4.0" in detail


# ---------------------------------------------------------------------------
# online_check and the command, with the network faked
# ---------------------------------------------------------------------------


def make_corpus(tmp_path, rows):
    docs = tmp_path / "corpora" / "own" / "docs"
    docs.mkdir(parents=True)
    lines = ["docid\ttitle\tsource\tlicense\tnotes"]
    for docid, source, license_, notes in rows:
        lines.append(f"{docid}\tT\t{source}\t{license_}\t{notes}")
        (docs / f"{docid}.md").write_text("# T\n\nPlain text without any trouble.\n", encoding="utf-8", newline="\n")
    (tmp_path / "corpora" / "own" / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (tmp_path / "p2.toml").write_text("", encoding="utf-8", newline="\n")
    return tmp_path / "corpora" / "own"


def all_services():
    return fake_get(
        {
            "crossref.org/works/10.1103/PhysRevLett.116.061102": (200, recorded("crossref-prl.json")),
            "crossref.org/works/10.1016/j.jmb.2005.01.025": (200, recorded("crossref-jmb.json")),
            "crossref.org/works/10.1126/science.1127647": (200, recorded("crossref-science.json")),
            "arXiv.org:1602.03837": (200, recorded("arxiv-1602.03837.xml")),
            "arXiv.org:1706.03762": (200, recorded("arxiv-1706.03762.xml")),
            "prefix=PMC1182327.": (200, recorded("pmc-list-PMC1182327.xml")),
            "metadata/PMC1182327.2.json": (200, recorded("pmc-meta-PMC1182327.2.json")),
            "prefix=PMC11000000.": (200, recorded("pmc-list-PMC11000000.xml")),
            "metadata/PMC11000000.1.json": (200, recorded("pmc-meta-PMC11000000.1.json")),
        }
    )


def test_online_check_matches_mismatches_and_leaves_the_rest(tmp_path):
    corpus = make_corpus(
        tmp_path,
        [
            ("arxiv-ok", "arXiv:1602.03837", "cc-by-4.0", ""),
            ("pmc-ok", "PMC1182327", "cc-by-4.0", ""),
            ("doi-publisher", "https://doi.org/10.1016/j.jmb.2005.01.025", "cc-by-4.0", ""),
            ("arxiv-default", "https://arxiv.org/abs/1706.03762", "cc-by-4.0", ""),
            ("pmc-nc", "PMC11000000", "cc-by-4.0", ""),
            ("doi-version", "doi:10.1103/PhysRevLett.116.061102", "cc-by-4.0", ""),
            ("doi-silent", "10.1126/science.1127647", "cc-by-4.0", ""),
            ("no-ids", "my own lab notes", "own-work", ""),
            ("gov-publisher", "10.1016/j.jmb.2005.01.025", "us-gov-public-domain", ""),
            ("gov-publisher-reviewed", "10.1016/j.jmb.2005.01.025", "us-gov-public-domain", 'reviewed: NIST author, "Not subject to copyright"'),
        ],
    )
    rows = lic.online_check(lic.offline_report(corpus), get=all_services(), sleep=lambda s: None)
    r = by_id(rows)
    assert r["arxiv-ok"]["status"] == "ok" and r["arxiv-ok"]["online"][0]["verdict"] == "match"
    assert r["pmc-ok"]["status"] == "ok"
    assert r["doi-publisher"]["status"] == "fail" and "publisher license" in r["doi-publisher"]["reason"]
    assert r["arxiv-default"]["status"] == "fail"
    assert r["pmc-nc"]["status"] == "fail" and "cc-by-nc-nd" in r["pmc-nc"]["reason"]
    assert r["doi-version"]["status"] == "ok" and "differs" in r["doi-version"]["online"][0]["detail"]
    assert r["doi-silent"]["status"] == "ok" and r["doi-silent"]["online"][0]["verdict"] == "info"
    assert r["no-ids"]["status"] == "ok" and r["no-ids"]["online"] == []
    assert r["gov-publisher"]["status"] == "flag"
    assert r["gov-publisher-reviewed"]["status"] == "ok"  # a reviewed note resolves a flag but not a mismatch


def test_a_reviewed_note_does_not_rescue_a_mismatch(tmp_path):
    corpus = make_corpus(tmp_path, [("x", "10.1016/j.jmb.2005.01.025", "cc-by-4.0", 'reviewed: "free to read" on the page')])
    rows = lic.online_check(lic.offline_report(corpus), get=all_services(), sleep=lambda s: None)
    assert rows[0]["status"] == "fail"


def test_manifest_saved_by_excel_with_a_byte_order_mark_and_crlf_is_read(tmp_path):
    corpus = make_corpus(tmp_path, [("a", "x", "cc-by-4.0", "")])
    manifest = corpus / "manifest.tsv"
    manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    assert [(r["docid"], r["status"]) for r in lic.offline_report(corpus)] == [("a", "ok")]


def test_an_online_restriction_is_added_to_an_existing_flag(tmp_path):
    corpus = make_corpus(tmp_path, [("gov", "10.1016/j.jmb.2005.01.025", "us-gov-public-domain", "")])
    (corpus / "docs" / "gov.md").write_text("# T\n\nPublished by Elsevier.\n", encoding="utf-8", newline="\n")
    rows = lic.online_check(lic.offline_report(corpus), get=all_services(), sleep=lambda s: None)
    assert rows[0]["status"] == "flag"
    assert "publisher name" in rows[0]["reason"] and "online check:" in rows[0]["reason"]
    assert rows[0]["reason"].count("write `reviewed:") == 1


def test_network_errors_degrade_to_could_not_check(tmp_path):
    corpus = make_corpus(tmp_path, [("a", "arXiv:1602.03837", "cc-by-4.0", ""), ("b", "10.1016/j.jmb.2005.01.025", "cc-by-4.0", ""), ("c", "PMC1182327", "cc-by-4.0", "")])

    def down(url):
        raise lic.CouldNotCheck("the network did not answer (URLError: timed out)")

    rows = lic.online_check(lic.offline_report(corpus), get=down, sleep=lambda s: None)
    for row in rows:
        assert row["status"] == "ok"
        assert row["online"][0]["verdict"] == "could not check"
        assert "network" in row["online"][0]["detail"]


def test_each_service_id_is_asked_once_and_paced(tmp_path):
    corpus = make_corpus(
        tmp_path,
        [("a", "arXiv:1602.03837", "cc-by-4.0", ""), ("b", "https://arxiv.org/abs/1602.03837", "cc-by-4.0", ""), ("c", "arXiv:1706.03762", "own-work", "")],
    )
    calls, sleeps = [], []
    table = {"arXiv.org:1602.03837": (200, recorded("arxiv-1602.03837.xml")), "arXiv.org:1706.03762": (200, recorded("arxiv-1706.03762.xml"))}
    lic.online_check(lic.offline_report(corpus), get=fake_get(table, calls), sleep=sleeps.append)
    assert len(calls) == 2  # the same id is cached
    assert sleeps and sleeps[0] > 0  # arXiv asked for one request every three seconds


def test_unreadable_license_still_gets_a_hint_from_the_source(tmp_path):
    corpus = make_corpus(tmp_path, [("a", "arXiv:1602.03837", "unknown", "")])
    rows = lic.online_check(lic.offline_report(corpus), get=all_services(), sleep=lambda s: None)
    assert rows[0]["status"] == "fail"
    assert "source says cc-by-4.0" in rows[0]["online"][0]["detail"]


def run_args(root, online=False):
    return argparse.Namespace(online=online, root=root)


def test_run_writes_licenses_md_and_returns_nonzero_on_failures(tmp_path, capsys):
    make_corpus(tmp_path, [("good", "own notes", "own-work", ""), ("bad", "somewhere", "unknown", "")])
    assert lic.run(run_args(tmp_path)) == 1
    text = (tmp_path / "LICENSES.md").read_text(encoding="utf-8")
    assert "| `good` | own-work | ok |" in text
    assert "| `bad` | unknown | fail |" in text
    assert "fail: 1" in text and "Documents: 2" in text
    assert "cannot prove a license" in text
    assert "\r" not in text
    out = capsys.readouterr().out
    assert "FAIL  bad" in out


def test_run_returns_zero_when_everything_is_ok(tmp_path, capsys):
    make_corpus(tmp_path, [("good", "own notes", "own-work", ""), ("also", "x", "cc-by-4.0", "")])
    assert lic.run(run_args(tmp_path)) == 0
    assert "| `also` | cc-by-4.0 | ok |" in (tmp_path / "LICENSES.md").read_text(encoding="utf-8")
    assert "evidence, not proof" in capsys.readouterr().out


def test_run_returns_nonzero_for_an_unreviewed_flag_and_zero_once_reviewed(tmp_path):
    corpus = make_corpus(tmp_path, [("a", "x", "cc-by-4.0", "")])
    (corpus / "docs" / "a.md").write_text("# T\n\n\xa9 2019 Someone\n", encoding="utf-8", newline="\n")
    assert lic.run(run_args(tmp_path)) == 1
    manifest = corpus / "manifest.tsv"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("cc-by-4.0\t", 'cc-by-4.0\treviewed: footer says "CC BY 4.0"'), encoding="utf-8", newline="\n")
    assert lic.run(run_args(tmp_path)) == 0
    assert "of which 1 had a flag that you reviewed" in (tmp_path / "LICENSES.md").read_text(encoding="utf-8")


def test_run_with_no_documents_is_not_an_error(tmp_path, capsys):
    (tmp_path / "p2.toml").write_text("", encoding="utf-8", newline="\n")
    assert lic.run(run_args(tmp_path)) == 0
    assert (tmp_path / "LICENSES.md").exists()
    assert "nothing to check" in capsys.readouterr().out


def test_run_online_reports_mismatch_as_failure_and_network_loss_as_could_not_check(tmp_path, monkeypatch):
    make_corpus(tmp_path, [("a", "arXiv:1602.03837", "cc-by-4.0", ""), ("b", "10.1016/j.jmb.2005.01.025", "cc-by-4.0", "")])
    monkeypatch.setattr(lic, "fetch", all_services())
    for name in ("ARXIV_PAUSE_SECONDS", "PMC_PAUSE_SECONDS", "CROSSREF_PAUSE_SECONDS"):
        monkeypatch.setattr(lic, name, 0)
    assert lic.run(run_args(tmp_path, online=True)) == 1
    text = (tmp_path / "LICENSES.md").read_text(encoding="utf-8")
    assert "## Online checks" in text
    assert "| `a` | arxiv | 1602.03837 | match:" in text
    assert "| `b` | crossref | 10.1016/j.jmb.2005.01.025 | mismatch:" in text
    assert "| `b` | cc-by-4.0 | fail |" in text

    def down(url):
        raise lic.CouldNotCheck("the network did not answer")

    monkeypatch.setattr(lic, "fetch", down)
    assert lic.run(run_args(tmp_path, online=True)) == 0
    text = (tmp_path / "LICENSES.md").read_text(encoding="utf-8")
    assert "could not check: the network did not answer" in text
    assert "could not be checked" in text


def test_run_online_says_when_nothing_could_be_checked(tmp_path, monkeypatch, capsys):
    make_corpus(tmp_path, [("a", "arXiv:1602.03837", "cc-by-4.0", "")])

    def down(url):
        raise lic.CouldNotCheck("the network did not answer")

    monkeypatch.setattr(lic, "fetch", down)
    assert lic.run(run_args(tmp_path, online=True)) == 0
    assert "1 lookup(s), 1 could not be checked (is the internet reachable?)" in capsys.readouterr().out


def test_output_never_fails_on_a_console_that_cannot_show_a_character(monkeypatch):
    import io
    import sys

    fake = io.TextIOWrapper(io.BytesIO(), encoding="ascii", newline="\n")
    monkeypatch.setattr(sys, "stdout", fake)
    lic._say("quote: " + chr(169) + " 2019")
    fake.flush()
    assert fake.buffer.getvalue() == b"quote: ? 2019\n"


def test_licenses_md_is_stable_between_offline_runs(tmp_path):
    make_corpus(tmp_path, [("a", "x", "cc-by-4.0", ""), ("b", "y", "mit", "")])
    lic.run(run_args(tmp_path))
    first = (tmp_path / "LICENSES.md").read_text(encoding="utf-8")
    lic.run(run_args(tmp_path))
    assert (tmp_path / "LICENSES.md").read_text(encoding="utf-8") == first


def test_fetch_turns_network_failures_into_could_not_check(monkeypatch):
    import urllib.error
    import urllib.request

    def boom(*a, **k):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(lic.CouldNotCheck, match="network"):
        lic.fetch("https://api.crossref.org/works/10.1/x")

    def timeout(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", timeout)
    with pytest.raises(lic.CouldNotCheck, match="network"):
        lic.fetch("https://api.crossref.org/works/10.1/x")


def test_fetch_returns_http_errors_as_status(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    def not_found(*a, **k):
        raise urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b"Resource not found."))

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    assert lic.fetch("https://api.crossref.org/works/10.1/x") == (404, b"Resource not found.")


def test_user_agent_names_the_tool_and_takes_an_optional_contact(monkeypatch):
    monkeypatch.delenv("P2_MAILTO", raising=False)
    assert "p2-license-check" in lic._user_agent() and "mailto" not in lic._user_agent()
    monkeypatch.setenv("P2_MAILTO", "me@mines.edu")
    assert lic._user_agent().endswith("mailto:me@mines.edu")


# ---------------------------------------------------------------------------
# Live services (opt in: P2_ONLINE_TESTS=1)
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(not os.environ.get("P2_ONLINE_TESTS"), reason="set P2_ONLINE_TESTS=1 to call the live services")


@live
def test_live_arxiv_cc_by_paper():
    assert lic.lookup_arxiv("1602.03837")[0]["family"] == "cc-by"


@live
def test_live_pmc_cc_by_article():
    assert lic.lookup_pmc("PMC1182327")[0]["family"] == "cc-by"


@live
def test_live_crossref_publisher_license():
    found = lic.lookup_crossref("10.1016/j.jmb.2005.01.025")
    assert found and found[0]["family"] == "restricted"
