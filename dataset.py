"""CSV-driven data loading for the BVP corpus.

Reads ``bvp_full.csv`` and provides filtered, optionally-deduped, optionally-cached
array loading plus train/test splitting. Train.py consumes this module instead
of directly globbing the extracted-zip tree.

Conceptual model:
    The corpus has 35,918 physical recordings spanning 17,897 unique natural keys
    (userid, gesture, location, face orientation, repetition). Most natural keys
    were re-recorded across days/batches — those are **independent measurements**
    of the same intended sample, not file-system duplicates. Treat them as natural
    session-augmentation (``dedup_mode="keep-all"``, the default) or collapse to one
    canonical recording per key (``dedup_mode="one-per-key"``).

    The 875 byte-identical files under ``BVP-Data-Batch1/20181208/20181208/`` are
    a zip-packaging artifact (a doubled date subdirectory in the source archive).
    They are dropped by default via ``drop_byte_dups=True``.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


NATURAL_KEY = ("userid", "gesture", "location", "face orientation", "repetition")
BYTE_DUP_PATTERN = "BVP-Data-Batch1/20181208/20181208/"


# ──────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────
def _norm_filter(values):
    return None if values is None else {str(v).strip() for v in values}


def _internal_date(file_path):
    parts = file_path.split("/")
    return parts[1] if len(parts) >= 2 else ""


def load_manifest(csv_path, *, data_dir="data",
                  users=None, gestures=None, locations=None, orientations=None,
                  drop_byte_dups=True, dedup_mode="keep-all"):
    """Read ``bvp_full.csv``, apply filters, return list of row dicts.

    Each returned row has the original CSV columns plus a resolved ``path`` key
    pointing to the absolute file on disk under ``data_dir``.

    Args:
        csv_path: Manifest CSV path.
        data_dir: Root where extracted-zip lives; ``data_dir/<row['file']>``
            must resolve to a real file.
        users, gestures, locations, orientations: Iterable of values to keep
            (exact string match against the CSV column). ``None`` keeps all.
        drop_byte_dups: Drop the 875 doubled-date zip artifacts.
        dedup_mode: ``"keep-all"`` keeps every physical recording (multi-session
            augmentation). ``"one-per-key"`` keeps one row per natural key,
            with the earliest internal date dir as the deterministic tiebreaker.

    Returns:
        List of dicts ordered by resolved path.
    """
    users = _norm_filter(users)
    gestures = _norm_filter(gestures)
    locations = _norm_filter(locations)
    orientations = _norm_filter(orientations)

    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if drop_byte_dups and r["file"].startswith(BYTE_DUP_PATTERN):
                continue
            if users is not None and r["userid"] not in users:
                continue
            if gestures is not None and r["gesture"] not in gestures:
                continue
            if locations is not None and r["location"] not in locations:
                continue
            if orientations is not None and r["face orientation"] not in orientations:
                continue
            r["path"] = os.path.join(data_dir, r["file"])
            rows.append(r)

    if dedup_mode == "one-per-key":
        rows = _dedup_one_per_key(rows)
    elif dedup_mode != "keep-all":
        raise ValueError(f"unknown dedup_mode: {dedup_mode!r}")

    rows.sort(key=lambda r: r["path"])
    return rows


def _dedup_one_per_key(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[c] for c in NATURAL_KEY)].append(r)
    kept = []
    for members in groups.values():
        members.sort(key=lambda r: (_internal_date(r["file"]), r["file"]))
        kept.append(members[0])
    return kept


def summarize(manifest, label="manifest"):
    """Print a quick count breakdown of a manifest."""
    from collections import Counter
    n = len(manifest)
    print(f"  {label}: {n} rows")
    if n == 0:
        return
    def _sorted_user_items(c):
        return dict(sorted(c.items(), key=lambda kv: int(kv[0].replace("user", ""))))
    print(f"    users:    {_sorted_user_items(Counter(r['userid'] for r in manifest))}")
    print(f"    gestures: {dict(sorted(Counter(int(r['gesture']) for r in manifest).items()))}")
    keys = {tuple(r[c] for c in NATURAL_KEY) for r in manifest}
    print(f"    unique natural keys: {len(keys)}")


# ──────────────────────────────────────────────────────────
# Array loading + optional cache
# ──────────────────────────────────────────────────────────
def _load_one(path):
    try:
        return np.load(path)["velocity_spectrum_ro"]
    except Exception:
        return None


def load_arrays(manifest, *, parallel=True, cache_path=None):
    """Load raw BVP arrays for every row in ``manifest``.

    If ``cache_path`` is given and the file exists, load from cache (fast,
    skips all file I/O). On a cache miss, read all files and write the cache.
    The user picks the cache filename — pick distinct names for distinct
    filter/dedup setups, e.g. ``cache/users_1-3_keep-all.npz``.

    Returns:
        (raw_list, motion_labels, user_labels, T_MAX)
    """
    if cache_path and Path(cache_path).exists():
        print(f"  cache hit: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return list(d["raw"]), d["motion"], d["user"], int(d["T_MAX"])

    paths = [r["path"] for r in manifest]
    motion = np.array([int(r["gesture"]) for r in manifest])
    user = np.array([r["userid"] for r in manifest])

    workers = min(16, os.cpu_count() or 4) if parallel else 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        arrays = list(ex.map(_load_one, paths))

    keep = [i for i, a in enumerate(arrays) if a is not None]
    if len(keep) < len(arrays):
        print(f"  WARNING: {len(arrays) - len(keep)} files failed to load; dropped")
        arrays = [arrays[i] for i in keep]
        motion = motion[keep]
        user = user[keep]

    T_MAX = max((a.shape[2] for a in arrays), default=0)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path,
                 raw=np.array(arrays, dtype=object),
                 motion=motion, user=user, T_MAX=T_MAX)
        print(f"  cache wrote: {cache_path}")

    return arrays, motion, user, T_MAX


# ──────────────────────────────────────────────────────────
# Train/test split
# ──────────────────────────────────────────────────────────
def make_split_indices(manifest, *, strategy="by-key", test_frac=0.1, seed=42):
    """Return ``(train_idx, test_idx)`` as positional indices into ``manifest``.

    Strategies:
        ``by-key``     — group rows by natural key; whole groups go to one side.
                         Prevents session leakage when the same recording was
                         captured on multiple days.
        ``random``     — independent row-level split (legacy).
        ``by-session`` — group rows by collection date dir; entire sessions
                         go to one side. Tests cross-session generalization.
    """
    rng = np.random.default_rng(seed)
    N = len(manifest)
    if N == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if strategy == "random":
        idx = np.arange(N)
        rng.shuffle(idx)
        n_test = int(round(N * test_frac))
        return np.sort(idx[n_test:]), np.sort(idx[:n_test])

    if strategy == "by-key":
        key_fn = lambda r: tuple(r[c] for c in NATURAL_KEY)
    elif strategy == "by-session":
        key_fn = lambda r: _internal_date(r["file"])
    else:
        raise ValueError(f"unknown split strategy: {strategy!r}")

    buckets = defaultdict(list)
    for i, r in enumerate(manifest):
        buckets[key_fn(r)].append(i)

    keys = sorted(buckets)
    order = rng.permutation(len(keys))

    test_target = N * test_frac
    test_idx, test_set = [], set()
    for j in order:
        if len(test_idx) >= test_target:
            break
        members = buckets[keys[j]]
        test_idx.extend(members)
        test_set.update(members)

    train_idx = [i for i in range(N) if i not in test_set]
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))
