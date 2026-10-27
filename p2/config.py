"""Read p2.toml, the student's settings, with a default for every value.

The file has a top-level `section` and the tables [run], [chunking], [bm25], [answer] and [claude]
(see the comments in p2.toml). Any other table is yours: read it in a retriever with cfg.table("name").
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SECTIONS = ("498E", "598E")
TOKENIZERS = ("codes", "lab")
SETTINGS_FILE = "p2.toml"


class ConfigError(Exception):
    """p2.toml is missing or holds a value p2 cannot use; the message says which and how to fix it."""


@dataclass(frozen=True)
class Config:
    """Every setting a p2 command or a retriever reads.

    A retriever gets one of these in build(corpus, cfg). To try a variant without touching p2.toml
    (for an ablation, say), make a changed copy: cfg.replace(chunk_words=100).
    """

    root: Path
    section: str = "498E"
    k: int = 10
    chunk_words: int = 300
    chunk_overlap: int = 50
    tokenizer: str = "codes"
    bm25_k1: float = 0.9
    bm25_b: float = 0.4
    answer_top: int = 5
    model: str = "sonnet"
    # Set by the command line, not by p2.toml: False when Claude replies must not come from the cache
    # (`--fresh`, and every `--repeat`).
    cache_claude: bool = True
    tables: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def table(self, name: str) -> dict:
        """A copy of one table of p2.toml, for example cfg.table("dense"); empty if there is none."""
        value = self.tables.get(name, {})
        return dict(value) if isinstance(value, dict) else {}

    def replace(self, **changes) -> Config:
        """A copy of these settings with some values changed."""
        return dataclasses.replace(self, **changes)

    def fingerprint(self) -> str:
        """A string that differs whenever any setting differs (used to reuse a built retriever)."""
        values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in ("root", "tables")}
        values["tables"] = self.tables
        return json.dumps(values, sort_keys=True, default=str)


def find_root(start: Path | None = None) -> Path:
    """The nearest folder at or above `start` (default: the current folder) that holds p2.toml."""
    here = (start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / SETTINGS_FILE).is_file():
            return folder
    raise ConfigError(f"Could not find {SETTINGS_FILE} here or in any folder above; run p2 from inside your P2 repository.")


def _get(data: dict, table: str, key: str, kind, default):
    section = data.get(table, {})
    if not isinstance(section, dict):
        raise ConfigError(f"[{table}] in {SETTINGS_FILE} should be a table; put it back as in the template.")
    value = section.get(key, default)
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise ConfigError(f"{key} in [{table}] of {SETTINGS_FILE} should be a {kind.__name__}, not {value!r}; fix the value.")
    return value


def load(root: Path | None = None) -> Config:
    """Read p2.toml from `root` (default: found from the current folder) and fill in the defaults."""
    root = Path(root).resolve() if root else find_root()
    path = root / SETTINGS_FILE
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{SETTINGS_FILE} is missing from {root}; restore it from the template.") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{SETTINGS_FILE} is not valid TOML ({error}); fix that line.") from None
    defaults = Config(root=root)
    section = data.get("section", defaults.section)
    if section not in SECTIONS:
        raise ConfigError(f'section in {SETTINGS_FILE} should be "498E" or "598E", not {section!r}; set your section.')
    cfg = Config(
        root=root,
        section=section,
        k=_get(data, "run", "k", int, defaults.k),
        chunk_words=_get(data, "chunking", "words", int, defaults.chunk_words),
        chunk_overlap=_get(data, "chunking", "overlap", int, defaults.chunk_overlap),
        tokenizer=_get(data, "bm25", "tokenizer", str, defaults.tokenizer),
        bm25_k1=_get(data, "bm25", "k1", float, defaults.bm25_k1),
        bm25_b=_get(data, "bm25", "b", float, defaults.bm25_b),
        answer_top=_get(data, "answer", "top", int, defaults.answer_top),
        model=_get(data, "claude", "model", str, defaults.model),
        tables={k: v for k, v in data.items() if isinstance(v, dict)},
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Raise ConfigError if a value is out of range."""
    if cfg.k < 1:
        raise ConfigError(f"k in [run] of {SETTINGS_FILE} should be at least 1, not {cfg.k}; fix the value.")
    if cfg.chunk_words < 0 or cfg.chunk_overlap < 0:
        raise ConfigError(f"words and overlap in [chunking] of {SETTINGS_FILE} cannot be negative; fix the value.")
    if cfg.chunk_words and cfg.chunk_overlap >= cfg.chunk_words:
        raise ConfigError(f"overlap in [chunking] of {SETTINGS_FILE} should be smaller than words; lower it.")
    if cfg.tokenizer not in TOKENIZERS:
        raise ConfigError(f'tokenizer in [bm25] of {SETTINGS_FILE} should be "codes" or "lab", not {cfg.tokenizer!r}; fix the value.')
    if cfg.answer_top < 1:
        raise ConfigError(f"top in [answer] of {SETTINGS_FILE} should be at least 1, not {cfg.answer_top}; fix the value.")
