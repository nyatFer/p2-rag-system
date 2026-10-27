"""p2 license: gather evidence about the license of every document in your corpus.

    uv run p2 license             check the manifest and scan every document's text
    uv run p2 license --online    also ask Crossref, arXiv and PubMed Central what they say

The report goes to LICENSES.md, one row per document, and the command exits with 1 if any document fails or is flagged.

What the tool does and does not do.
It gathers evidence and flags problems.
It cannot prove a license, and you stay responsible for everything you commit to a public repo.
A green report means "nothing here looked wrong to a program", not "this is legal".

The three checks:

1. The manifest.
   Every document needs a `license` value from this list, written exactly like this:
   us-gov-public-domain, public-domain, cc0-1.0, cc-by-2.0 to cc-by-4.0, cc-by-sa-2.0 to cc-by-sa-4.0,
   own-work, mit, apache-2.0, bsd-2-clause, bsd-3-clause.
   Anything with NC (non-commercial) or ND (no derivatives), `unknown`, `all-rights-reserved`, an empty value, or a value not on the list fails.
2. The text scan.
   Each document is searched for things that contradict an open license: a copyright line, "All rights reserved",
   a publisher's name (Elsevier, Springer, Wiley, IEEE, ACS, Taylor & Francis, SAGE, Cambridge or Oxford University Press, ASTM, ISO)
   on a line with words about rights ("© 2020 Elsevier", "Published by Wiley", "IEEE Xplore. Restrictions apply."),
   a CC BY-NC or BY-ND notice, "reprinted with permission", the sentence US government reports use for third-party material,
   a contractor's notice ("prepared as an account of work sponsored by an agency of the United States Government"),
   and a "courtesy of" credit line. Evidence that p2 ingest saw in the raw text and its cleaning removed
   (a publisher footer on every page) is in the manifest notes, and counts the same.
   Every hit is a flag with the line it was found on.
   A publisher's name with no words about rights on its line is usually a citation, so it is listed as a mention and is not a flag.
   A flag passes only when the manifest's `notes` column for that document says `reviewed: <reason>`,
   and the reason should quote the license text you read on the source page.
3. The online lookup (--online, on your machine only, never in CI).
   If a document's `source` holds a DOI, an arXiv id or a PubMed Central id, the tool asks the service for the license and compares it with yours.
   A different license is a failure.
   If the service cannot be reached, the document is reported as "could not check" and that is not a failure.

The license-check skill, .claude/skills/license-check/SKILL.md, walks you through each failing or flagged document.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

CC_VERSIONS = ("2.0", "2.1", "2.5", "3.0", "4.0")
ALLOWED = frozenset(
    {
        "us-gov-public-domain",
        "public-domain",
        "cc0-1.0",
        "own-work",
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
    }
    | {f"cc-by-{v}" for v in CC_VERSIONS}
    | {f"cc-by-sa-{v}" for v in CC_VERSIONS}
)

_ARR_VALUES = {"all-rights-reserved", "all rights reserved", "copyrighted", "proprietary", "arr", "\xa9"}


def check_license_value(value) -> tuple[bool, str]:
    """Is this manifest license value allowed? Returns (ok, reason); the reason is empty when ok."""
    raw = str(value or "").strip()
    if not raw:
        return False, "the license is empty; find the license on the source page and write it in the manifest, or remove the document"
    low = raw.lower()
    if low in ALLOWED:
        return True, ""
    tokens = set(re.split(r"[^a-z0-9.]+", low))
    if tokens & {"nc", "nd"} or re.search(r"non-?commercial|no-?deriv", low):
        return False, f"`{raw}` has a NonCommercial or NoDerivatives term, which cannot go in a public repo; remove the document"
    if low == "unknown":
        return False, "the license is `unknown`, which means nobody has checked yet; find the license on the source page, or remove the document"
    if low in _ARR_VALUES:
        return False, "all rights reserved text cannot be redistributed; remove the document"
    hint = re.sub(r"[\s_]+", "-", low)
    if hint in ALLOWED:
        return False, f"`{raw}` is not written the way the vocabulary spells it; use `{hint}`"
    return False, f"`{raw}` is not in the vocabulary (allowed: {', '.join(sorted(ALLOWED))})"


# ---------------------------------------------------------------------------
# The text scan
# ---------------------------------------------------------------------------

# Names in any case, and the acronyms in capitals only ("sage advice" and "iso-propyl" are not publishers).
_PUBLISHERS = (
    r"(?i:Elsevier|Springer|Wiley|Taylor\s*(?:&|and)\s*Francis|Cambridge\s+University\s+Press|Oxford\s+University\s+Press)"
    r"|IEEE|ACS|SAGE|ASTM|ISO"
)
# A publisher's name counts as evidence only next to words about rights: "© 2021 Elsevier",
# "Published by Wiley", "Downloaded from IEEE Xplore. Restrictions apply." On its own a name is
# usually a citation or a mention ("see ASTM E2500", "ISO 8601", "the American Community Survey (ACS)"),
# so it is listed in LICENSES.md as a mention and is not a flag.
_RIGHTS_CUE = re.compile(
    r"\xa9|\(c\)\s*(?:19|20)\d\d|\bcopyright|\ball\s+rights\b|\bpublish(?:ed|er|ing)\b|\bpermissions?\b|\blicen[cs](?:e|ed|ing)\b"
    r"|\breprint|\breproduc|\bcourtesy\b|\brestrictions\s+apply\b|\bterms\s+of\s+use\b|\bxplore\b|\bon\s+behalf\s+of\b",
    re.I,
)
_PUBLISHER_RE = re.compile(rf"\b(?:{_PUBLISHERS})\b")
_PUBLISHER_MENTION_RE = re.compile(
    r"\b(?:Elsevier|ELSEVIER|Springer|SPRINGER|Wiley|WILEY|IEEE|ACS|SAGE|ASTM|ISO)\b"
    r"|(?i:\bTaylor\s*(?:&|and)\s*Francis\b|\bCambridge\s+University\s+Press\b|\bOxford\s+University\s+Press\b)"
)

SCAN_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "copyright line",
        re.compile(
            r"\xa9|(?<![\w)])\(c\)\s*(?:19|20)\d\d\b|\bcopyright\b\s*(?:\xa9|\(c\))?\s*(?:19|20)\d\d\b"
            r"|\bcopyright(?:ed)?\s+(?:by\s+)?(?-i:(?!Act\b|Office\b|Law\b|Clearance\b)[A-Z][A-Za-z&'-]+(?:\s+[A-Z][A-Za-z&'-]+)*)",
            re.I,
        ),
    ),
    ("all rights reserved", re.compile(r"\ball\s+rights\s+reserved\b", re.I)),
    ("publisher name", _PUBLISHER_RE),  # only on a line with rights wording; see scan_text
    (
        "NC or ND notice",
        re.compile(
            r"\bCC[\s-]*BY[\s-]*(?:SA[\s-]*)?(?:NC|ND)\b"
            r"|creativecommons\.org/licenses/by-(?:nc|nd)"
            r"|\bAttribution[\s-]+Non-?Commercial"
            r"|\bNo[\s-]?Deriv(?:ative)?s\b"
            r"|\bNon-?Commercial[\s-]+(?:ShareAlike|NoDerivs)",
            re.I,
        ),
    ),
    (
        "permission",
        re.compile(r"\b(?:reprinted|reproduced|adapted|used|republished)\s+(?:here\s+)?(?:with|by)\s+(?:the\s+)?(?:kind\s+)?permission\b", re.I),
    ),
    (
        "third-party material",
        re.compile(
            r"permission\s+(?:to\s+reproduce\s+[^.]{0,80}?\s+)?must\s+be\s+(?:secured|obtained)\s+from\s+the\s+(?:individual\s+)?copyright\s+(?:owners?|holders?)"
            r"|may\s+(?:also\s+)?contain\s+copyrighted\s+material",
            re.I,
        ),
    ),
    (
        "contractor notice",
        re.compile(
            r"prepared\s+as\s+an\s+account\s+of\s+work\s+sponsored\s+by\s+an\s+agency\s+of\s+the\s+United\s+States\s+Government"
            r"|\bunder\s+contract\s+(?:no\.?\s*|number\s+)?[A-Z]{2,}[-\s]?[A-Z0-9-]{4,}"
            r"|\boperated\s+by\s+.{1,80}?\s+for\s+the\s+(?:U\.\s?S\.|United\s+States)\s+Department\s+of\s+Energy",
            re.I,
        ),
    ),
    ("credit line", re.compile(r"\bcourtesy\s+of\b", re.I)),
]
# The kinds a reviewed note has to clear; a "publisher mentioned" hit is only listed.
FLAG_KINDS = {kind for kind, _ in SCAN_PATTERNS}
MENTION = "publisher mentioned"


def _line_at(text: str, start: int, end: int) -> tuple[int, str]:
    line_no = text.count("\n", 0, start) + 1
    first = text.rfind("\n", 0, start) + 1
    last = text.find("\n", end)
    last = len(text) if last < 0 else last
    return line_no, " ".join(text[first:last].split())


def scan_text(text: str) -> list[dict]:
    """Find lines that contradict an open license. Each hit: kind, line_no (1-based), line, match.

    A publisher's name is a flag ("publisher name") on a line that also has words about rights, and
    a "publisher mentioned" hit otherwise, which LICENSES.md lists but which needs no review."""
    hits: dict[tuple[int, str], dict] = {}
    for kind, rx in SCAN_PATTERNS:
        for m in rx.finditer(text):
            line_no, line = _line_at(text, m.start(), m.end())
            if kind == "publisher name" and not _RIGHTS_CUE.search(_PUBLISHER_RE.sub(" ", line)):
                continue
            if (line_no, kind) in hits:
                continue
            hits[(line_no, kind)] = {"kind": kind, "line_no": line_no, "line": line[:200], "match": " ".join(m.group(0).split())}
    flagged_lines = {n for n, k in hits if k == "publisher name"}
    for m in _PUBLISHER_MENTION_RE.finditer(text):
        line_no, line = _line_at(text, m.start(), m.end())
        if line_no not in flagged_lines and (line_no, MENTION) not in hits:
            hits[(line_no, MENTION)] = {"kind": MENTION, "line_no": line_no, "line": line[:200], "match": " ".join(m.group(0).split())}
    return sorted(hits.values(), key=lambda h: (h["line_no"], h["kind"]))


REMOVED_RE = re.compile(r"cleaning removed: (.*)$", re.I)


def removed_hits(notes) -> list[dict]:
    """Evidence `p2 ingest` saw in the raw text and its cleaning pass removed (a publisher footer on
    every page, say), as it wrote it in the manifest notes: `cleaning removed: <kind> "<line>" | ...`."""
    m = REMOVED_RE.search(str(notes or "").split("reviewed:")[0])
    out = []
    for kind, line in re.findall(r'([a-z -]+?) "([^"]*)"', m.group(1) if m else ""):
        kind = kind.strip(" |;")
        if kind in FLAG_KINDS:
            out.append({"kind": kind, "line_no": 0, "line": line, "match": line})
    return out


def reviewed_reason(notes) -> str:
    """The reason in a `reviewed: <reason>` note, or an empty string when there is none."""
    m = re.search(r"\breviewed:\s*(.+)", str(notes or ""), re.I | re.S)
    if not m:
        return ""
    reason = m.group(1).strip()
    return reason if re.search(r"\w", reason) else ""


# ---------------------------------------------------------------------------
# The offline report
# ---------------------------------------------------------------------------


class Row(dict):
    """A result row that works as a dict and with attribute access: row["status"] and row.status."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


MANIFEST_COLUMNS = ["docid", "title", "source", "license", "notes"]
_DOCID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _read_manifest(path: Path) -> tuple[list[dict], list[Row]]:
    """The manifest rows, and result rows for problems with the manifest itself."""
    problems: list[Row] = []
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8-sig")  # Excel adds a byte order mark when it saves a TSV
    lines = text.splitlines()
    if not lines or [c.strip() for c in lines[0].split("\t")] != MANIFEST_COLUMNS:
        problems.append(
            Row(
                docid=path.name,
                status="fail",
                reason="the first line must be the header: " + " <TAB> ".join(MANIFEST_COLUMNS),
                license="",
                flags=[],
                reviewed="",
                online=[],
            )
        )
        return rows, problems
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) > len(MANIFEST_COLUMNS):
            problems.append(
                Row(docid=cells[0].strip() or f"line {n}", status="fail", reason=f"manifest line {n} has {len(cells)} columns; a tab inside a field breaks the row", license="", flags=[], reviewed="", online=[])
            )
            continue
        cells += [""] * (len(MANIFEST_COLUMNS) - len(cells))
        rows.append({**dict(zip(MANIFEST_COLUMNS, (c.strip() for c in cells), strict=False)), "line": n})
    return rows, problems


def _where(h: dict) -> str:
    return f"line {h['line_no']}" if h["line_no"] else "a line ingest removed"


def _fmt_hits(hits: list[dict], per_kind: int = 2, quote: int = 110) -> str:
    """One short phrase per kind of hit: how many lines, and the first lines themselves."""
    groups: dict[str, list[dict]] = {}
    for h in hits:
        groups.setdefault(h["kind"], []).append(h)
    parts = []
    for kind, items in groups.items():
        shown = "; ".join(f"{_where(h)}: \"{h['line'][:quote]}\"" for h in items[:per_kind])
        more = f" (and {len(items) - per_kind} more)" if len(items) > per_kind else ""
        parts.append(f"{kind} on {len(items)} line{'s' if len(items) != 1 else ''} ({shown}{more})")
    return "; ".join(parts)


FLAG_ACTION = "read the source, then write `reviewed: <reason with the quote>` in the notes, or remove the document"


def offline_report(corpus_dir) -> list[Row]:
    """Check every document of one corpus without the network.

    corpus_dir holds `docs/` and `manifest.tsv`, for example corpora/own.
    Returns one Row per document, sorted by docid, with docid, status ("ok", "flag" or "fail") and reason,
    plus license, title, source, notes, flags (the text-scan hits) and reviewed (the reason, if any).
    """
    corpus = Path(corpus_dir)
    if corpus.name == "docs" and not (corpus / "manifest.tsv").exists() and (corpus.parent / "manifest.tsv").exists():
        corpus = corpus.parent
    docs_dir = corpus / "docs"
    manifest = corpus / "manifest.tsv"

    results: list[Row] = []
    entries: dict[str, dict] = {}
    if manifest.exists():
        rows, problems = _read_manifest(manifest)
        results.extend(problems)
        for r in rows:
            if not _DOCID.match(r["docid"]):
                results.append(Row(docid=r["docid"] or f"line {r['line']}", status="fail", reason=f"manifest line {r['line']}: `{r['docid']}` is not a valid document id", license=r["license"], flags=[], reviewed="", online=[]))
            elif r["docid"] in entries:
                results.append(Row(docid=r["docid"], status="fail", reason=f"manifest line {r['line']}: this docid already has a row, and each document needs exactly one", license=r["license"], flags=[], reviewed="", online=[]))
            else:
                entries[r["docid"]] = r
    doc_files = {p.stem: p for p in docs_dir.glob("*.md")} if docs_dir.is_dir() else {}

    for docid in sorted(set(entries) | set(doc_files)):
        entry = entries.get(docid)
        path = doc_files.get(docid)
        base = {"docid": docid, "title": "", "source": "", "license": "", "notes": "", "flags": [], "mentions": [], "reviewed": "", "online": []}
        if entry is None:
            results.append(Row(**base, status="fail", reason="there is a document but no manifest row, so no license is recorded; add a row, or remove the document"))
            continue
        base.update(title=entry["title"], source=entry["source"], license=entry["license"], notes=entry["notes"])
        if path is None:
            results.append(Row(**base, status="fail", reason=f"the manifest lists it but docs/{docid}.md does not exist"))
            continue
        ok, why = check_license_value(entry["license"])
        found = scan_text(path.read_text(encoding="utf-8")) + removed_hits(entry["notes"])
        hits = [h for h in found if h["kind"] != MENTION]
        mentions = [h for h in found if h["kind"] == MENTION]
        reviewed = reviewed_reason(entry["notes"])
        base.update(flags=hits, mentions=mentions, reviewed=reviewed)
        mentioned = f"mentions {', '.join(dict.fromkeys(h['match'] for h in mentions))} without words about rights on the same line, which is not a flag" if mentions else ""
        if not ok:
            reason = why + (f"; the text scan also found {len(hits)} thing(s) to look at" if hits else "")
            results.append(Row(**base, status="fail", reason=reason))
        elif hits and not reviewed:
            results.append(Row(**base, status="flag", reason=_fmt_hits(hits) + "; " + FLAG_ACTION, evidence=_fmt_hits(hits, per_kind=1, quote=60)))
        elif hits:
            results.append(Row(**base, status="ok", reason=f"reviewed: {reviewed}"))
        else:
            results.append(Row(**base, status="ok", reason=mentioned))
    return sorted(results, key=lambda r: r["docid"])


# ---------------------------------------------------------------------------
# Online lookups
# ---------------------------------------------------------------------------

# Verified against the live services on 2026-10-04 (see the notes in each lookup).

CROSSREF_URL = "https://api.crossref.org/works/{doi}"
SKILL_FILE = ".claude/skills/license-check/SKILL.md"
ARXIV_URL = "https://oaipmh.arxiv.org/oai?verb=GetRecord&identifier=oai:arXiv.org:{id}&metadataPrefix=arXivRaw"
PMC_LIST_URL = "https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix={pmcid}.&delimiter=/"
PMC_META_URL = "https://pmc-oa-opendata.s3.amazonaws.com/metadata/{pmcid}.{version}.json"
TIMEOUT_SECONDS = 20
ARXIV_PAUSE_SECONDS = 3.0  # arXiv asks for no more than one request every three seconds
PMC_PAUSE_SECONDS = 0.4  # NCBI asks for no more than three requests a second
CROSSREF_PAUSE_SECONDS = 0.3


class CouldNotCheck(Exception):
    """The lookup could not be completed (network error, odd response). Never a failure of the document."""


def _parse_xml(body: bytes):
    """Parse a service's XML answer.

    ElementTree does not fetch external entities, and the answers we expect never carry a DTD,
    so a body that declares one is refused instead of parsed (it could only be an attack or a broken proxy page).
    """
    head = body[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in head:
        raise CouldNotCheck("the service sent XML with a DTD, which a license lookup never needs")
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise CouldNotCheck("the service sent an answer I could not read") from exc


def _user_agent() -> str:
    ua = "p2-license-check/1.0 (Mines CSCI 498E course tool; reads license metadata only)"
    contact = os.environ.get("P2_MAILTO", "").strip()
    return f"{ua}; mailto:{contact}" if contact else ua


def _curl_fetch(url: str) -> tuple[int, bytes]:
    """Fallback when Python cannot verify HTTPS certificates (a python.org build on macOS before 'Install Certificates').

    curl ships with macOS, Windows 10 and later, and Linux, and it uses the operating system's own certificate store.
    """
    curl = shutil.which("curl")
    if not curl:
        raise CouldNotCheck("Python could not verify the site's HTTPS certificate and curl is not installed")
    try:
        proc = subprocess.run(
            [curl, "-sS", "-L", "--max-time", str(TIMEOUT_SECONDS), "-A", _user_agent(), "-w", "\n%{http_code}", url],
            capture_output=True,
            timeout=TIMEOUT_SECONDS + 5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CouldNotCheck(f"curl failed ({type(exc).__name__})") from exc
    body, _, code = proc.stdout.rpartition(b"\n")
    if not code.strip().isdigit() or int(code) == 0:
        raise CouldNotCheck("curl could not reach the site")
    return int(code), body


def fetch(url: str) -> tuple[int, bytes]:
    """GET a URL. Returns (status, body); raises CouldNotCheck on a network failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent(), "Accept": "application/json, application/xml, text/xml, */*"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 - fixed https hosts only
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001
            body = b""
        return exc.code, body
    except (urllib.error.URLError, ssl.SSLError) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLCertVerificationError) or isinstance(exc, ssl.SSLCertVerificationError):
            return _curl_fetch(url)
        raise CouldNotCheck(f"the network did not answer ({type(exc).__name__}: {str(reason)[:80]})") from exc
    except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
        raise CouldNotCheck(f"the network did not answer ({type(exc).__name__}: {str(exc)[:80]})") from exc


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;]+", re.I)
_ARXIV_ID = r"(?:[a-z\-]+(?:\.[A-Za-z]{2})?/\d{7}|\d{4}\.\d{4,5})"
_ARXIV_RES = [
    re.compile(rf"arxiv\.org/(?:abs|pdf)/({_ARXIV_ID})(?:v\d+)?(?:\.pdf)?", re.I),
    re.compile(rf"\barxiv\s*[:\s]\s*({_ARXIV_ID})(?:v\d+)?", re.I),
]
_PMC_RE = re.compile(r"\bPMC\d{4,9}\b")


def _trim_doi(doi: str) -> str:
    doi = doi.rstrip(".:]}'\"")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip(".:")
    return doi


def extract_ids(source: str) -> dict[str, list[str]]:
    """Pull DOIs, arXiv ids and PubMed Central ids out of a manifest `source` cell."""
    source = str(source or "")
    dois = list(dict.fromkeys(_trim_doi(m.group(0)) for m in _DOI_RE.finditer(source)))
    arxiv: list[str] = []
    for rx in _ARXIV_RES:
        for m in rx.finditer(source):
            if m.group(1) not in arxiv:
                arxiv.append(m.group(1))
    pmc = list(dict.fromkeys(_PMC_RE.findall(source)))
    return {"doi": [d for d in dois if d], "arxiv": arxiv, "pmc": pmc}


# A "found" license is a dict: family, version, raw.
# family is one of cc0, public-domain, cc-by, cc-by-sa, cc-by-nc, cc-by-nd, cc-by-nc-sa, cc-by-nc-nd, or "restricted" for anything else
# (a publisher's own license, arXiv's default license, PMC's text-mining-only label).
_OPEN_FAMILIES = {"cc0", "public-domain", "cc-by", "cc-by-sa"}
_CC_URL = re.compile(r"creativecommons\.org/licenses/(by(?:-nc)?(?:-nd|-sa)?)/(\d\.\d)", re.I)
_CC0_URL = re.compile(r"creativecommons\.org/publicdomain/zero/(\d\.\d)", re.I)
_PDM_URL = re.compile(r"creativecommons\.org/publicdomain/mark/", re.I)


def found_from_url(url: str) -> dict:
    m = _CC_URL.search(url)
    if m:
        return {"family": "cc-" + m.group(1).lower(), "version": m.group(2), "raw": url}
    m = _CC0_URL.search(url)
    if m:
        return {"family": "cc0", "version": m.group(1), "raw": url}
    if _PDM_URL.search(url):
        return {"family": "public-domain", "version": "", "raw": url}
    if "arxiv.org/licenses/nonexclusive-distrib" in url:
        return {"family": "restricted", "version": "", "raw": url, "label": "arXiv's default license, which does not let others redistribute the paper"}
    return {"family": "restricted", "version": "", "raw": url}


def found_from_label(label: str) -> dict:
    """PMC's license_code: "CC BY", "CC BY-NC-ND", "CC0", "TDM" and so on."""
    low = re.sub(r"[\s_]+", "-", str(label or "").strip().lower())
    if low in {"cc0", "cc0-1.0"}:
        return {"family": "cc0", "version": "", "raw": label}
    if re.fullmatch(r"cc-by(?:-nc)?(?:-nd|-sa)?", low):
        return {"family": low, "version": "", "raw": label}
    if low == "tdm":
        return {"family": "restricted", "version": "", "raw": label, "label": "PMC's text-mining-only code (TDM), which does not let others redistribute the article"}
    return {"family": "restricted", "version": "", "raw": label}


def lookup_crossref(doi: str, get=None) -> list[dict]:
    """Licenses Crossref lists for a DOI.

    GET https://api.crossref.org/works/10.1103/PhysRevLett.116.061102 answers (verified 2026-10-04)
        {"status": "ok", "message-type": "work", "message": {..., "license": [
            {"URL": "http://creativecommons.org/licenses/by/3.0/", "content-version": "vor",
             "delay-in-days": 0, "start": {"date-time": "2016-02-11T00:00:00Z", ...}}], ...}}
    `license` is a list; `content-version` is vor (published version), tdm (text mining), am (accepted manuscript) or unspecified.
    A publisher license looks like {"URL": "https://www.elsevier.com/tdm/userlicense/1.0/", "content-version": "tdm"}.
    A work with no license data has no `license` key at all.
    An unknown DOI answers HTTP 404 with the plain text "Resource not found.".
    """
    get = get or fetch
    status, body = get(CROSSREF_URL.format(doi=quote(doi, safe="/")))
    if status == 404:
        raise CouldNotCheck(f"Crossref does not know the DOI {doi}")
    if status != 200:
        raise CouldNotCheck(f"Crossref answered HTTP {status}")
    try:
        message = json.loads(body.decode("utf-8"))["message"]
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        raise CouldNotCheck("Crossref sent an answer I could not read") from exc
    return [found_from_url(str(item.get("URL", ""))) for item in message.get("license", []) if item.get("URL")]


def lookup_arxiv(arxiv_id: str, get=None) -> list[dict]:
    """The license arXiv records for a paper.

    GET https://oaipmh.arxiv.org/oai?verb=GetRecord&identifier=oai:arXiv.org:1602.03837&metadataPrefix=arXivRaw
    answers OAI-PMH XML (verified 2026-10-04; the older export.arxiv.org/oai2 address redirects here):
        <OAI-PMH><GetRecord><record><metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
            <id>1602.03837</id> ... <license>http://creativecommons.org/licenses/by/4.0/</license> ...
    A paper under arXiv's default license has <license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>,
    and some older papers have no <license> element at all; both mean no Creative Commons license was granted.
    An unknown id answers HTTP 200 with <error code="idDoesNotExist">.
    """
    get = get or fetch
    status, body = get(ARXIV_URL.format(id=arxiv_id))
    if status != 200:
        raise CouldNotCheck(f"arXiv answered HTTP {status}")
    root = _parse_xml(body)
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "error":
            raise CouldNotCheck(f"arXiv does not know the id {arxiv_id} ({el.attrib.get('code', 'error')})")
    urls = [el.text.strip() for el in root.iter() if el.tag.rsplit("}", 1)[-1] == "license" and el.text and el.text.strip()]
    if not urls:
        return [{"family": "restricted", "version": "", "raw": "no license recorded, so arXiv's default license applies and others may not redistribute the paper"}]
    return [found_from_url(u) for u in urls]


def lookup_pmc(pmcid: str, get=None) -> list[dict]:
    """The license PubMed Central records for an article, from the PMC Cloud Service (the open-access dataset on S3).

    The older "OA Web Service" (https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC...) is gone: it answers HTTP 404, and
    PMC's own page says "the PMC OA Web Service is no longer available" (read 2026-10-04). The Cloud Service is the route PMC allows now.
    Step 1, list the versions of the article (verified 2026-10-04):
        GET https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix=PMC1182327.&delimiter=/
        <ListBucketResult><KeyCount>1</KeyCount><CommonPrefixes><Prefix>PMC1182327.2/</Prefix></CommonPrefixes></ListBucketResult>
      KeyCount 0 means the article is not in the open-access dataset.
    Step 2, read the metadata of the newest version:
        GET https://pmc-oa-opendata.s3.amazonaws.com/metadata/PMC1182327.2.json
        {"pmcid": "PMC1182327", "version": 2, "doi": "10.1371/journal.pmed.0020124", "is_pmc_openaccess": true,
         "license_code": "CC BY", ...}
      license_code is a Creative Commons code ("CC BY", "CC BY-NC-ND", "CC0", ...) or "TDM" for author manuscripts
      that may be mined but not redistributed. Some articles have no code at all, and the lookup then reports no license.
    """
    get = get or fetch
    status, body = get(PMC_LIST_URL.format(pmcid=pmcid))
    if status != 200:
        raise CouldNotCheck(f"the PMC open-access dataset answered HTTP {status}")
    root = _parse_xml(body)
    versions = []
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "Prefix" and el.text:
            m = re.fullmatch(rf"{re.escape(pmcid)}\.(\d+)/", el.text.strip())
            if m:
                versions.append(int(m.group(1)))
    if not versions:
        return [{"family": "restricted", "version": "", "raw": f"{pmcid} is not in the PMC open-access dataset"}]
    status, body = get(PMC_META_URL.format(pmcid=pmcid, version=max(versions)))
    if status != 200:
        raise CouldNotCheck(f"the PMC open-access dataset answered HTTP {status} for the article's metadata")
    try:
        meta = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CouldNotCheck("the PMC open-access dataset sent metadata I could not read") from exc
    code = str(meta.get("license_code") or "").strip()
    if not code:
        return [{"family": "restricted", "version": "", "raw": f"{pmcid} has no license code in PMC"}]
    return [found_from_label(code)]


def declared_family(license_id: str) -> str:
    """The family of a declared license: cc0, public-domain, cc-by, cc-by-sa, or "other" (gov, own work, mit, ...)."""
    low = str(license_id or "").strip().lower()
    if low == "cc0-1.0":
        return "cc0"
    if low == "public-domain":
        return "public-domain"
    m = re.fullmatch(r"(cc-by(?:-sa)?)-\d\.\d", low)
    return m.group(1) if m else "other"


def _declared_version(license_id: str) -> str:
    m = re.search(r"(\d\.\d)$", str(license_id or "").strip().lower())
    return m.group(1) if m else ""


_EQUIVALENT = {"cc0": {"cc0", "public-domain"}, "public-domain": {"cc0", "public-domain"}}


def _describe(found: dict) -> str:
    fam = found["family"]
    if fam == "restricted":
        if found.get("label"):
            return found["label"]
        return f"a publisher license ({found['raw']})" if str(found["raw"]).startswith("http") else str(found["raw"])
    return fam + (f"-{found['version']}" if found["version"] else "")


def compare_licenses(declared: str, found: list[dict]) -> tuple[str, str]:
    """Compare the declared license with what a service says.

    Returns (verdict, detail). verdict is one of
      "match"      the families agree (a different version is only mentioned in the detail)
      "mismatch"   you declared a Creative Commons or public-domain license and the source says something else: a failure
      "restricted" you declared something else (government, own work, mit...) and the source lists only a restrictive license: a flag
      "info"       nothing to compare, or nothing contradicts you
    """
    if not found:
        return "info", "the service lists no license"
    families = {f["family"] for f in found}
    said = ", ".join(dict.fromkeys(_describe(f) for f in found))
    dfam = declared_family(declared)
    if dfam == "other":
        if not families & _OPEN_FAMILIES:
            return "restricted", f"the source says {said}, which is more restrictive than your `{declared}`; check that the text really is {declared}"
        return "info", f"the source says {said}"
    accepted = _EQUIVALENT.get(dfam, {dfam})
    if families & accepted:
        match = next(f for f in found if f["family"] in accepted)
        dv = _declared_version(declared)
        if dv and match["version"] and dv != match["version"]:
            return "match", f"the source says {said}; your version ({dv}) differs from the source's ({match['version']}), so make sure you used the version the text itself carries"
        return "match", f"the source says {said}"
    if families <= {"restricted"}:
        return "mismatch", f"you declared `{declared}`, but the source gives no Creative Commons license: it says {said}"
    return "mismatch", f"you declared `{declared}`, but the source says {said}"


def online_check(rows: list[Row], get=None, sleep=time.sleep) -> list[Row]:
    """Look up each document's source online and fold the verdict into its row.

    A row gains `online`, a list of {service, id, verdict, detail}. A mismatch makes the row fail,
    a restrictive source for a non-Creative-Commons declaration makes it flag (unless it was reviewed),
    and "could not check" changes nothing.
    """
    last_call: dict[str, float] = {}
    pauses = {"arxiv": ARXIV_PAUSE_SECONDS, "pmc": PMC_PAUSE_SECONDS, "crossref": CROSSREF_PAUSE_SECONDS}
    cache: dict[tuple[str, str], tuple[str, list[dict] | str]] = {}

    def call(service: str, ident: str):
        key = (service, ident)
        if key in cache:
            return cache[key]
        wait = pauses[service] - (time.monotonic() - last_call.get(service, -1e9))
        if wait > 0 and service in last_call:
            sleep(wait)
        last_call[service] = time.monotonic()
        try:
            fn = {"crossref": lookup_crossref, "arxiv": lookup_arxiv, "pmc": lookup_pmc}[service]
            cache[key] = ("ok", fn(ident, get))
        except CouldNotCheck as exc:
            cache[key] = ("could not check", str(exc))
        return cache[key]

    for row in rows:
        ids = extract_ids(row.get("source", ""))
        notes = []
        verdicts = []
        for service, key in (("crossref", "doi"), ("arxiv", "arxiv"), ("pmc", "pmc")):
            for ident in ids[key]:
                state, value = call(service, ident)
                if state != "ok":
                    notes.append({"service": service, "id": ident, "verdict": "could not check", "detail": value})
                    continue
                valid, _ = check_license_value(row["license"])
                if valid:
                    verdict, detail = compare_licenses(row["license"], value)
                else:
                    said = ", ".join(dict.fromkeys(_describe(f) for f in value)) if value else "no license"
                    verdict, detail = "info", f"the source says {said}; use this to fix the manifest's license value"
                notes.append({"service": service, "id": ident, "verdict": verdict, "detail": detail})
                verdicts.append((verdict, f"{service} {ident}: {detail}"))
        row["online"] = notes
        for verdict, text in verdicts:
            if verdict == "mismatch":
                row["status"] = "fail"
                row["reason"] = (row["reason"] + "; " if row["reason"] and not row["reason"].startswith("reviewed:") else "") + "online check: " + text
            elif verdict == "restricted" and row["status"] in ("ok", "flag") and not row["reviewed"]:
                earlier = row["reason"] if row["status"] == "flag" else ""
                row["status"] = "flag"
                row["reason"] = (earlier + "; " if earlier else "") + "online check: " + text
                if "write `reviewed:" not in row["reason"]:
                    row["reason"] += "; read the source, then write `reviewed: <reason with the quote>` in the notes, or remove the document"
    return rows


# ---------------------------------------------------------------------------
# LICENSES.md and the command
# ---------------------------------------------------------------------------


def _md(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def render_report(corpus_name: str, rows: list[Row], online: bool) -> str:
    ok = sum(1 for r in rows if r["status"] == "ok")
    flag = sum(1 for r in rows if r["status"] == "flag")
    fail = sum(1 for r in rows if r["status"] == "fail")
    reviewed = sum(1 for r in rows if r["status"] == "ok" and r["reviewed"])
    lines = [
        "# License report",
        "",
        f"Written by `uv run p2 license{' --online' if online else ''}` for the `{corpus_name}` corpus.",
        "The tool gathers evidence and flags problems.",
        "It cannot prove a license, and you remain responsible for every document you commit to this public repo.",
        "",
        "## Summary",
        "",
        f"- Documents: {len(rows)}",
        f"- ok: {ok} (of which {reviewed} had a flag that you reviewed)",
        f"- flag, not yet reviewed: {flag}",
        f"- fail: {fail}",
    ]
    if online:
        checks = [c for r in rows for c in r["online"]]
        lines += [
            f"- Online checked on {_dt.date.today().isoformat()}: {sum(1 for c in checks if c['verdict'] in ('match', 'info'))} agreed or had nothing to compare,"
            f" {sum(1 for c in checks if c['verdict'] in ('mismatch', 'restricted'))} disagreed, {sum(1 for c in checks if c['verdict'] == 'could not check')} could not be checked",
        ]
    else:
        lines.append("- Online lookups: not run (use `--online` on your own machine)")
    lines += ["", "## Documents", "", "| docid | license | status | why |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{_md(r['docid'])}` | {_md(r['license']) or '-'} | {r['status']} | {_md(r['reason']) or '-'} |")
    if online:
        lines += ["", "## Online checks", "", "| docid | service | id | result |", "|---|---|---|---|"]
        any_check = False
        for r in rows:
            for c in r["online"]:
                any_check = True
                lines.append(f"| `{_md(r['docid'])}` | {c['service']} | {_md(c['id'])} | {c['verdict']}: {_md(c['detail'])} |")
        if not any_check:
            lines.append("| - | - | - | no document has a DOI, arXiv id or PubMed Central id in its source |")
    if fail or flag:
        lines += [
            "",
            "## Next",
            "",
            f"Ask Claude Code to read `{SKILL_FILE}` and walk you through each failing or flagged document with it.",
            "For each one you open the source page, quote the license text, and then either fix the manifest, write `reviewed: <reason with the quote>` in the notes, or remove the document.",
        ]
    return "\n".join(lines) + "\n"


def add_arguments(parser) -> None:
    """Add the p2 license options to an argparse parser (cli.py may use this)."""
    parser.add_argument("--online", action="store_true", help="also ask Crossref, arXiv and PubMed Central (your machine only, never CI)")


def _find_root(args) -> Path:
    root = getattr(args, "root", None)
    if root:
        return Path(root)
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / "p2.toml").exists():
            return candidate
    return here


def _say(text: str = "") -> None:
    """print() that never fails on a console that cannot show a character (an old Windows code page, a redirect)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "replace").decode(enc))


def run(args) -> int:
    root = _find_root(args)
    corpus_name = str(getattr(args, "corpus", None) or getattr(args, "into", None) or "own")
    corpus_dir = root / "corpora" / corpus_name
    online = bool(getattr(args, "online", False))
    rows = offline_report(corpus_dir)
    if online and rows:
        lookups = sum(1 for r in rows if any(extract_ids(r.get("source", "")).values()))
        if lookups:
            _say(f"Asking Crossref, arXiv and PubMed Central about the {lookups} document(s) whose source has a DOI, an arXiv id or a PubMed Central id; this can take a minute ...")
        else:
            _say("None of your sources has a DOI, an arXiv id or a PubMed Central id, so there is nothing to look up online.")
        online_check(rows)
    report = render_report(corpus_name, rows, online)
    out = root / "LICENSES.md"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)

    if online:
        checks = [c for r in rows for c in r["online"]]
        unreachable = sum(1 for c in checks if c["verdict"] == "could not check")
        if checks:
            _say(f"Online: {len(checks)} lookup(s), {unreachable} could not be checked" + (" (is the internet reachable?)" if unreachable == len(checks) else "."))
    bad = [r for r in rows if r["status"] != "ok"]
    ok = len(rows) - len(bad)
    if not rows:
        _say(f"There are no documents in corpora/{corpus_name} yet, so there is nothing to check. Wrote {out.name}.")
        return 0
    for r in bad[:25]:
        reason = r["reason"]
        if r["status"] == "flag" and r.get("evidence"):
            reason = f"{r['evidence']}; {FLAG_ACTION}"  # the quotes are cut short so the next step always shows
        elif len(reason) > 240:
            reason = reason[:237].rsplit(" ", 1)[0] + " ..."
        _say(f"{r['status'].upper():<5} {r['docid']}: {reason}")
    if len(bad) > 25:
        _say(f"... and {len(bad) - 25} more; they are all in {out.name}")
    _say(f"{len(rows)} document(s): {ok} ok, {sum(1 for r in bad if r['status'] == 'flag')} flagged, {sum(1 for r in bad if r['status'] == 'fail')} failing. Wrote {out.name}.")
    if bad:
        _say(f"Ask Claude Code to read {SKILL_FILE} and walk you through them with it. The tool gathers evidence; the decision, and the responsibility, are yours.")
        return 1
    _say("Nothing looked wrong to the tool. That is evidence, not proof: you remain responsible for what you commit.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="p2 license", description=__doc__.split("\n")[0])
    add_arguments(ap)
    sys.exit(run(ap.parse_args()))
