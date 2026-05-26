"""Plot dataset statistics for the BVP corpus from ``bvp_full.csv``.

Reads the manifest and writes four PNGs under ``analysis/dataset/``:
    * distributions.png       — 2x2 bar panel (user, gesture, location, orientation)
    * user_gesture_heatmap.png
    * user_location_heatmap.png
    * dup_group_sizes.png     — natural-key group-size histogram (cross-batch dups)

Usage:
    python visualize_dataset.py
    python visualize_dataset.py --csv bvp_full.csv --out-dir analysis/dataset
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NATURAL_KEY = ("userid", "gesture", "location", "face orientation", "repetition")
BAR_COLUMNS = ("userid", "gesture", "location", "face orientation")


def load(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _user_sort_key(u: str) -> int:
    try:
        return int(u.replace("user", ""))
    except ValueError:
        return 10_000


def _sorted_keys(values, col: str) -> list[str]:
    if col == "userid":
        return sorted(values, key=_user_sort_key)
    try:
        return sorted(values, key=int)
    except ValueError:
        return sorted(values)


def _counts(rows: list[dict], col: str) -> tuple[list[str], list[int]]:
    c = Counter(row[col] for row in rows)
    keys = _sorted_keys(c, col)
    return keys, [c[k] for k in keys]


def plot_bar_panel(rows: list[dict], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.flatten(), BAR_COLUMNS):
        keys, counts = _counts(rows, col)
        ax.bar(range(len(keys)), counts, color="#4c72b0")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45 if col == "userid" else 0)
        ax.set_title(col)
        ax.set_ylabel("clip count")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(counts):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"BVP corpus distributions  (N = {len(rows)} clips)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_heatmap(rows: list[dict], y_col: str, x_col: str, out_path: str, title: str):
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        counts[(r[y_col], r[x_col])] += 1
    ys = _sorted_keys({r[y_col] for r in rows}, y_col)
    xs = _sorted_keys({r[x_col] for r in rows}, x_col)

    grid = np.zeros((len(ys), len(xs)), dtype=int)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            grid[i, j] = counts.get((y, x), 0)

    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(xs), 1 + 0.5 * len(ys)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(xs))); ax.set_xticklabels(xs)
    ax.set_yticks(np.arange(len(ys))); ax.set_yticklabels(ys)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title)

    threshold = grid.max() * 0.5 if grid.max() else 0
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = grid[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v < threshold else "black", fontsize=8)

    fig.colorbar(im, ax=ax, label="clips")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_dup_group_sizes(rows: list[dict], out_path: str):
    """Histogram of natural-key group sizes — how many clips share each (user, gesture, location, orientation, repetition)."""
    keys: dict[tuple, int] = defaultdict(int)
    for r in rows:
        keys[tuple(r[c] for c in NATURAL_KEY)] += 1
    sizes = Counter(keys.values())
    xs = sorted(sizes); ys = [sizes[s] for s in xs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, ys, color="#4c72b0")
    for x, y in zip(xs, ys):
        ax.text(x, y, str(y), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("group size (# physical clips per natural key)")
    ax.set_ylabel("# natural keys")
    ax.set_title(f"Duplicate-group sizes  ({len(keys)} unique keys, {len(rows)} rows)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def print_summary(rows: list[dict]):
    print(f"\n=== {len(rows)} clips ===")
    for col in BAR_COLUMNS:
        keys, counts = _counts(rows, col)
        breakdown = ", ".join(f"{k}={v}" for k, v in zip(keys, counts))
        print(f"  {col:<18} ({len(keys)} unique): {breakdown}")
    unique = len({tuple(r[c] for c in NATURAL_KEY) for r in rows})
    print(f"  natural keys: {unique} unique / {len(rows)} rows  ({len(rows) - unique} duplicate rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="bvp_full.csv", help="Manifest CSV (default: bvp_full.csv)")
    ap.add_argument("--out-dir", default="analysis/dataset", help="Where to write figures (default: analysis/dataset)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"manifest CSV not found: {args.csv}")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load(args.csv)
    print_summary(rows)

    plot_bar_panel(rows, os.path.join(args.out_dir, "distributions.png"))
    plot_heatmap(rows, "userid", "gesture", os.path.join(args.out_dir, "user_gesture_heatmap.png"),
                 "User x Gesture clip counts")
    plot_heatmap(rows, "userid", "location", os.path.join(args.out_dir, "user_location_heatmap.png"),
                 "User x Location clip counts")
    plot_dup_group_sizes(rows, os.path.join(args.out_dir, "dup_group_sizes.png"))

    print(f"\nDone. Figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
