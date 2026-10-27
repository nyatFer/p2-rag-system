"""The size rules and the CI timeout are stated in several places; these tests keep them equal to p2/limits.py."""

import re
from pathlib import Path

from p2 import check, ingest, limits

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_check_and_ingest_use_the_same_numbers():
    assert check.OWN_TOKEN_LIMIT == ingest.TOKEN_LIMIT == limits.OWN_TOKEN_LIMIT == 1_000_000
    assert check.OWN_TOKEN_WARNING == ingest.TOKEN_WARNING == limits.OWN_TOKEN_WARNING == 800_000
    assert check.OWN_MIN_DOCS == ingest.MIN_DOCS == limits.OWN_MIN_DOCS == 200
    assert check.TOKENS_PER_WORD == ingest.TOKENS_PER_WORD == limits.TOKENS_PER_WORD == 1.4


def test_the_student_ci_timeout_is_the_one_the_messages_quote():
    workflow = read(".github/workflows/p2-check.yml")
    assert re.findall(r"timeout-minutes: (\d+)", workflow) == [str(limits.CI_MINUTES)]


def test_the_floor_and_the_limit_meet_at_about_3500_words_a_document():
    # the brief's arithmetic: 200 documents of about 3,500 words is about 1,000,000 tokens
    tokens = limits.estimated_tokens(limits.OWN_MIN_DOCS * 3_500)
    assert abs(tokens - limits.OWN_TOKEN_LIMIT) / limits.OWN_TOKEN_LIMIT < 0.03


def test_the_brief_states_the_rules_the_code_enforces():
    readme = " ".join(read("README.md").split())
    assert f"`p2 check` warns above {limits.OWN_TOKEN_WARNING:,} and fails above {limits.OWN_TOKEN_LIMIT:,}" in readme
    assert f"At least **{limits.OWN_MIN_DOCS} documents**" in readme
    assert f"{limits.CI_MINUTES} minutes" in readme
    assert f"--part-pages {limits.SUGGESTED_PART_PAGES}" in readme
    assert "200 documents of about 3,500 words is about 700,000 words, or about 1,000,000 tokens" in readme
    for stale in ("500,000", "600,000", "30 minutes", "personal data"):
        assert stale not in readme, stale
    claude_md = " ".join(read("CLAUDE.md").split())
    assert f"{limits.OWN_TOKEN_LIMIT:,}" in claude_md and "--part-pages" in claude_md
