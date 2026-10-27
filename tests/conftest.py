"""Shared test setup.

The tests build tiny repositories of their own, whose shared corpus is not the course's, so the check
that pins corpora/shared/ and eval/shared/ to p2/shared.sha256 is switched off for them; the tests
that need it switch it back on with pins of their own (see test_core_check.py), and
test_template_files.py checks the real list against the real files.
"""

import pytest

from p2 import check


@pytest.fixture(autouse=True)
def no_course_pins(monkeypatch):
    monkeypatch.setattr(check, "load_pins", lambda: None)


@pytest.fixture
def add_system():
    """Add a system file to p2/retrievers/ for one test, the way a student adds one, and remove it after."""
    import importlib
    import sys
    import textwrap
    from pathlib import Path

    from p2 import retrievers

    folder = Path(retrievers.__file__).parent
    created = []

    def add(name: str, source: str) -> str:
        path = folder / f"{name}.py"
        path.write_text(textwrap.dedent(source), encoding="utf-8", newline="\n")
        created.append(path)
        importlib.invalidate_caches()
        return name

    yield add
    for path in created:
        path.unlink(missing_ok=True)
        sys.modules.pop(f"p2.retrievers.{path.stem}", None)
        for cached in (folder / "__pycache__").glob(f"{path.stem}.*"):
            cached.unlink(missing_ok=True)
    if created:
        retrievers._built.clear()
        importlib.invalidate_caches()
