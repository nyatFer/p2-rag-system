"""Helpers for dense retrievers: load an embedding model, encode texts, and cache document vectors.

The two models of the 12 lab load the way the lab loads them:
- potion-retrieval-32M (POTION) through model2vec, a static model: fast, no neural network at query time.
- bge-small-en-v1.5 (BGE) through fastembed, a small contextual model run with ONNX.
Both download once into the Hugging Face cache (~/.cache/huggingface/hub), the folder CI caches.
Another model name goes to fastembed if fastembed lists it, and to model2vec otherwise.

Every vector comes back with length 1, so a dot product is the cosine similarity.

doc_vectors() encodes a list of texts (your chunks) and keeps the vectors in
.cache/vectors/<model>.npz, keyed by each text's hash: a second run, a changed chunking setting or
the autograder's extra documents only encode the texts that are new. It encodes the texts sorted
by length in batches of 8, which measured 2 to 3 times faster than the original order and keeps
memory near 600 MB instead of several GB (the largest vector difference it caused was 0.0004).
It prints a progress line every 15 seconds or so and saves what it has every minute, so a long
first encode is visibly moving and an interrupted one is not lost.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

POTION = "minishlab/potion-retrieval-32M"
BGE = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 8
BLOCK = 256  # texts per progress step; a multiple of BATCH_SIZE, so the batches are the same as one big call
SAVE_EVERY_S = 60
NOTE_EVERY_S = 15
LONG_ENCODE = 500  # from this many new texts on, say that it takes a while


def unit(x) -> np.ndarray:
    """Rows scaled to length 1 (float32)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


class Embedder:
    """A loaded model: encode(texts) gives one unit-length row per text."""

    def __init__(self, name: str, encode: Callable[[list[str]], Sequence]):
        self.name = name
        self._encode = encode

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return unit(self._encode(texts))


def _hf_cache() -> str:
    from huggingface_hub.constants import HF_HUB_CACHE

    return HF_HUB_CACHE


def _load_model2vec(name: str) -> Embedder:
    from huggingface_hub import snapshot_download
    from model2vec import StaticModel

    # model2vec needs the weights and tokenizer only; skip the unused ONNX copy.
    path = snapshot_download(name, ignore_patterns=["onnx/*", "README.md"])
    model = StaticModel.from_pretrained(path)
    return Embedder(name, lambda texts: model.encode(texts))


def _load_fastembed(name: str) -> Embedder:
    from fastembed import TextEmbedding

    # fastembed's default cache is a temporary folder that the system can empty; use the Hugging Face cache.
    model = TextEmbedding(name, cache_dir=_hf_cache())
    return Embedder(name, lambda texts: np.stack(list(model.embed(texts, batch_size=BATCH_SIZE))))


def _fastembed_names() -> set[str]:
    from fastembed import TextEmbedding

    return {m["model"] for m in TextEmbedding.list_supported_models()}


@lru_cache(maxsize=None)
def load(name: str = BGE) -> Embedder:
    """Load an embedding model by its Hugging Face name (once per process)."""
    if name == POTION or name.startswith("minishlab/"):
        return _load_model2vec(name)
    if name == BGE or name in _fastembed_names():
        return _load_fastembed(name)
    return _load_model2vec(name)


def encode_sorted(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    """Encode texts in order of length (short first), and return the rows in the original order."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    by_length = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    rows = embedder.encode([texts[i] for i in by_length])
    out = np.empty_like(rows)
    out[by_length] = rows
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def store_path(root: Path, name: str) -> Path:
    return Path(root) / ".cache" / "vectors" / f"{_slug(name)}.npz"


def _minutes(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def doc_vectors(embedder: Embedder, texts: Sequence[str], root: Path) -> np.ndarray:
    """One unit vector per text, from the cache where possible; new texts are encoded and saved.

    New texts are encoded shortest first in blocks, with a progress line now and then, and the cache
    is saved every minute or so and when the run stops early (Ctrl-C, a cancelled CI job), so the
    next run starts from where this one got to.
    """
    texts = list(texts)
    path = store_path(root, embedder.name)
    keys = [_key(t) for t in texts]
    known: dict[str, int] = {}
    stored: list[np.ndarray] = []
    all_keys: list[str] = []
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as saved:
                all_keys, vectors = saved["keys"].tolist(), saved["vectors"]
            known = {k: i for i, k in enumerate(all_keys)}
            stored = [np.asarray(vectors, dtype=np.float32)] if len(all_keys) else []
        except (OSError, ValueError, KeyError):
            known, stored, all_keys = {}, [], []  # a damaged cache is rebuilt
    missing = list(dict.fromkeys(k for k in keys if k not in known))
    if missing:
        wanted = set(missing)
        first = {k: t for k, t in zip(keys, texts) if k in wanted}
        missing.sort(key=lambda k: len(first[k]))  # shortest first: batches of similar length encode fastest
        total_chars = sum(len(first[k]) for k in missing) or 1
        texts_word = "text" if len(missing) == 1 else "texts"
        note = f"Encoding {len(missing):,} new {texts_word} with {embedder.name}, saved in .cache/vectors/ for next time"
        if len(missing) >= LONG_ENCODE:
            note += "; a whole corpus takes a few minutes on a laptop, and later runs skip it"
        print(note + ".", file=sys.stderr, flush=True)
        started = last_note = last_save = time.perf_counter()
        done_chars = 0
        unsaved = 0
        try:
            for start in range(0, len(missing), BLOCK):
                block = missing[start : start + BLOCK]
                rows = embedder.encode([first[k] for k in block])
                stored.append(np.asarray(rows, dtype=np.float32))
                for k in block:
                    known[k] = len(all_keys)
                    all_keys.append(k)
                unsaved += len(block)
                done_chars += sum(len(first[k]) for k in block)
                now = time.perf_counter()
                if now - last_save >= SAVE_EVERY_S:
                    _save(path, all_keys, np.concatenate(stored))
                    last_save, unsaved = now, 0
                if now - last_note >= NOTE_EVERY_S and start + BLOCK < len(missing):
                    left = (now - started) * (total_chars - done_chars) / max(done_chars, 1)
                    print(f"  encoded {start + len(block):,} of {len(missing):,} texts in {_minutes(now - started)}, about {_minutes(left)} to go", file=sys.stderr, flush=True)
                    last_note = now
        finally:
            if unsaved and stored:
                _save(path, all_keys, np.concatenate(stored))  # keep the work done so far, even on Ctrl-C
        if len(missing) >= LONG_ENCODE:
            print(f"  encoded {len(missing):,} texts in {_minutes(time.perf_counter() - started)}", file=sys.stderr, flush=True)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = stored[0] if len(stored) == 1 else np.concatenate(stored)
    return np.asarray(vectors[[known[k] for k in keys]], dtype=np.float32)


def _save(path: Path, keys: list[str], vectors: np.ndarray) -> None:
    """Write the cache in one step, so an interrupted run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            np.savez(out, keys=np.array(keys), vectors=np.asarray(vectors, dtype=np.float32))
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # On Windows a virus scanner or a second p2 process can hold the old file for a moment.
                if attempt == 4:
                    print(f"Note: could not update {path.name} (the file is in use); the vectors stay in memory for this run.", file=sys.stderr)
                    break
                time.sleep(0.5 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
