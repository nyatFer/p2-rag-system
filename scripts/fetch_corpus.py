"""Build the raw stage 2 corpus: Yokubi's lessons and Wiktionary's Japanese grammar entries.

    uv run python scripts/fetch_corpus.py

It writes only outside the repo, into ../p2-raw/own/, one folder per license, for `p2 ingest`:

- yokubi/      the 64 lessons of Yokubi (github.com/Morgawr/yokubi, CC BY 4.0), copied from a clone at
               ../p2-raw/yokubi as yokubi-lesson-NN.md, with the furigana markup {f|翌|よく} turned into its base text
- wiktionary/  the Japanese section of Wiktionary entries (CC BY-SA 4.0) in four categories, one <revid>.txt each,
               so `--source "https://en.wiktionary.org/w/index.php?oldid={stem}"` names the exact revision

Wiktionary entries are filtered in two passes: the wikitext first (cheap, 50 pages a call), dropping titles with no
kana or kanji, soft redirects and near-empty sections; then the rendered Japanese section, keeping it only when it
still has MIN_WORDS English words. Quotations from published works are dropped, because the CC BY-SA license of
Wiktionary's own text may not cover them; Wiktionary's usage examples are kept. So are tables, references and
the sections that are lists or sound rather than prose (pronunciation, conjugation, derived terms and the like).

The record of what was fetched goes to ../p2-raw/own/sources.tsv, outside the folders ingest reads.
Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT.parent / "p2-raw"
OUT = RAW / "own"
YOKUBI_CLONE = RAW / "yokubi"

API = "https://en.wiktionary.org/w/api.php"
HEADERS = {"User-Agent": "p2-rag-system corpus fetch (https://github.com/nyatFer/p2-rag-system)"}
CATEGORIES = ["Japanese particles", "Japanese auxiliary verbs", "Japanese suffixes", "Japanese conjunctions"]
EXTRA_CATEGORY = "Japanese prefixes"  # added only when the four above give too few entries
# Most kanji suffixes are vocabulary (丼, 星, 帽), not grammar; the grammatical ones are written in kana (がる, げ, ごっこ)
KANA_ONLY_CATEGORIES = {"Japanese suffixes"}
SINGULAR = {"Japanese particles": "particle", "Japanese auxiliary verbs": "auxiliary verb", "Japanese suffixes": "suffix",
            "Japanese conjunctions": "conjunction", "Japanese prefixes": "prefix"}
MIN_ENTRIES = 140
MIN_WIKITEXT_WORDS = 10
MIN_WORDS = 25  # ので, a core particle, has 29
PAUSE_S = 0.2

JAPANESE_CHAR = re.compile(r"[぀-ヿ㐀-鿿]")
KANA_ONLY = re.compile(r"^[぀-ヿ・ー〜～]+$")
ENGLISH_WORD = re.compile(r"[A-Za-z]{2,}")
SOFT_REDIRECT = re.compile(r"\{\{\s*(ja-see|ja-see-kango|alt of|alternative (form|spelling) of)\b", re.I)
REDIRECT_TEXT = re.compile(r"see the following entry|alternative (form|spelling) of|(hiragana|katakana|kyūjitai) form of", re.I)
DROP_SECTIONS = {
    "Pronunciation", "Conjugation", "Inflection", "References", "Further_reading", "Derived_terms", "Related_terms",
    "See_also", "Anagrams", "Descendants", "Alternative_forms", "Synonyms", "Antonyms", "Coordinate_terms",
    "Compounds", "Kanji", "Hypernyms", "Hyponyms", "Idioms", "Proverbs",
}
DROP_CLASSES = {"citation-whole", "cited-source", "wiktQuote", "reflist", "references", "mw-editsection", "NavFrame",
                "inflection-table-wrapper", "wikitable", "mw-empty-elt", "maintenance-line"}
SKIP_TAGS = {"style", "script", "table", "sup", "rt", "rp", "figure"}
BLOCK_TAGS = {"p", "li", "dd", "dt", "div", "h3", "h4", "h5", "h6", "ol", "ul", "dl", "br"}


# --- Yokubi ------------------------------------------------------------------

def furigana_base(text: str) -> str:
    """{f|翌|よく} becomes 翌: the pattern of Yokubi's own scripts/preprocess-furigana.py, keeping the base text."""
    return re.sub(r"\{f\|([^{}|\n]+)\|([^{}|\n]+)\}", r"\1", text)


def copy_yokubi() -> list[tuple[str, str, str, str, int]]:
    if not YOKUBI_CLONE.is_dir():
        sys.exit(f"No Yokubi clone at {YOKUBI_CLONE}; run: git clone https://github.com/Morgawr/yokubi.git {YOKUBI_CLONE}")
    dest = OUT / "yokubi"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((YOKUBI_CLONE / "src").glob("Section*/Part*/Lesson*.md")):
        n = int(re.search(r"Lesson(\d+)", path.stem).group(1))
        name = f"yokubi-lesson-{n:02d}.md"
        if name in {r[0] for r in rows}:
            sys.exit(f"Two Yokubi files are lesson {n}")
        text = furigana_base(path.read_text(encoding="utf-8"))
        (dest / name).write_text(text, encoding="utf-8", newline="\n")
        rows.append((name, "yokubi", str(path.relative_to(YOKUBI_CLONE)), "", len(ENGLISH_WORD.findall(text))))
    print(f"Yokubi: {len(rows)} lessons into {dest}")
    return rows


# --- Wiktionary ----------------------------------------------------------------

def api(**params) -> dict:
    params = {"format": "json", "formatversion": 2, **params}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as reply:
                data = json.load(reply)
            time.sleep(PAUSE_S)
            return data
        except OSError as err:
            if attempt == 3:
                raise
            print(f"  retrying after {err}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def members(category: str) -> list[tuple[int, str]]:
    out, more = [], {}
    while True:
        data = api(action="query", list="categorymembers", cmtitle="Category:" + category, cmnamespace=0, cmlimit=500, **more)
        out += [(m["pageid"], m["title"]) for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            return out
        more = {"cmcontinue": data["continue"]["cmcontinue"]}


def japanese_wikitext(text: str) -> str:
    m = re.search(r"^==\s*Japanese\s*==[ \t]*\n(.*?)(?=^==[^=]|\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


def wikitext_words(section: str) -> int:
    plain = section
    for _ in range(3):  # templates nest
        plain = re.sub(r"\{\{[^{}]*\}\}", " ", plain)
    return len(ENGLISH_WORD.findall(plain))


class SectionText(HTMLParser):
    """Rendered HTML to plain text: block tags become line breaks, and the subtrees in SKIP_TAGS or DROP_CLASSES go."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.stack: list[tuple[str, bool]] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("br", "img", "meta", "link", "wbr", "hr", "input"):
            if tag == "br" and not self.skip_depth:
                self.parts.append("\n")
            return
        classes = set((dict(attrs).get("class") or "").split())
        # a list nested in a definition or a usage note (#* and ** in wikitext) holds quotations;
        # Wiktionary's own usage examples (#:) are dl/dd instead
        quotations = tag == "ul" and any(open_tag == "li" for open_tag, _ in self.stack)
        skipping = tag in SKIP_TAGS or bool(classes & DROP_CLASSES) or quotations
        self.stack.append((tag, skipping))
        if skipping:
            self.skip_depth += 1
        elif tag in BLOCK_TAGS and not self.skip_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if all(open_tag != tag for open_tag, _ in self.stack):
            return  # a stray end tag closes nothing
        while self.stack:
            open_tag, skipping = self.stack.pop()
            if skipping:
                self.skip_depth -= 1
            if open_tag == tag:
                break
        if tag in BLOCK_TAGS and not self.skip_depth:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in raw.split("\n")]
        out, blank = [], False
        for line in lines:
            if line:
                out.append(line)
                blank = False
            elif not blank and out:
                out.append("")
                blank = True
        return "\n".join(out).strip()


def japanese_html(html: str) -> str:
    """The rendered Japanese section, without the subsections in DROP_SECTIONS."""
    start = html.find('id="Japanese"')
    if start < 0:
        return ""
    start = html.find("</div>", start)  # the end of the h2 heading block
    nxt = html.find("mw-heading2", start)
    section = html[start + len("</div>") : nxt if nxt > 0 else None]
    if nxt > 0:
        section = section[: section.rfind("<div")]  # the next h2 block opens just before its class
    pieces = re.split(r'(?=<div class="mw-heading mw-heading[3-6]")', section)
    kept = []
    for piece in pieces:
        m = re.match(r'<div class="mw-heading mw-heading[3-6]"><h[3-6] id="([^"]+)"', piece)
        name = re.sub(r"_\d+$", "", m.group(1)) if m else ""
        if name not in DROP_SECTIONS:
            kept.append(piece)
    return "".join(kept)


def fetch_wiktionary(categories: list[str], seen: set[int]) -> list[tuple[str, str, str, str, int]]:
    dest = OUT / "wiktionary"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for category in categories:
        listed = [(pid, title) for pid, title in members(category) if pid not in seen]
        seen.update(pid for pid, _ in listed)
        counts = {"listed": len(listed), "no Japanese script": 0, "not kana": 0, "redirect": 0, "short wikitext": 0, "short text": 0, "kept": 0}
        candidates = []
        for pid, title in listed:
            if not JAPANESE_CHAR.search(title):
                counts["no Japanese script"] += 1
            elif category in KANA_ONLY_CATEGORIES and not KANA_ONLY.match(title):
                counts["not kana"] += 1
            else:
                candidates.append((pid, title))
        survivors = []
        for i in range(0, len(candidates), 50):
            batch = candidates[i : i + 50]
            data = api(action="query", prop="revisions", rvprop="content", rvslots="main", pageids="|".join(str(p) for p, _ in batch))
            for page in data["query"]["pages"]:
                content = page["revisions"][0]["slots"]["main"]["content"] if page.get("revisions") else ""
                section = japanese_wikitext(content)
                if SOFT_REDIRECT.search(section) and wikitext_words(section) < MIN_WORDS:
                    counts["redirect"] += 1
                elif wikitext_words(section) < MIN_WIKITEXT_WORDS:
                    counts["short wikitext"] += 1
                else:
                    survivors.append((page["pageid"], page["title"]))
        for pid, title in survivors:
            parsed = api(action="parse", pageid=pid, prop="text|revid")["parse"]
            converter = SectionText()
            converter.feed(japanese_html(parsed["text"]))
            body = converter.text()
            words = len(ENGLISH_WORD.findall(body))
            if words < MIN_WORDS or (REDIRECT_TEXT.search(body) and words < 2 * MIN_WORDS):
                counts["short text"] += 1
                continue
            name = f"{parsed['revid']}.txt"
            heading = f"{title} (Japanese, {SINGULAR[category]}, Wiktionary)"
            (dest / name).write_text(heading + "\n\n" + body + "\n", encoding="utf-8", newline="\n")
            rows.append((name, "wiktionary", title, category, words))
            counts["kept"] += 1
        print(f"Wiktionary, {category}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = copy_yokubi()
    seen: set[int] = set()
    wikt = fetch_wiktionary(CATEGORIES, seen)
    if len(wikt) < MIN_ENTRIES:
        print(f"Only {len(wikt)} Wiktionary entries; adding Category:{EXTRA_CATEGORY}.")
        wikt += fetch_wiktionary([EXTRA_CATEGORY], seen)
    rows += wikt
    with open(OUT / "sources.tsv", "w", encoding="utf-8", newline="\n") as out:
        out.write("file\torigin\ttitle\tcategory\tenglish_words\n")
        for row in rows:
            out.write("\t".join(str(v) for v in row) + "\n")
    print(f"Done: {len(rows)} files ({len(rows) - len(wikt)} Yokubi, {len(wikt)} Wiktionary); record in {OUT / 'sources.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
