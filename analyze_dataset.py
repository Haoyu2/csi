#!/usr/bin/env python3
"""
Dataset breakdown for data/bvp/.

Parses the Widar3.0 filename convention <user>-<motion>-<position>-<orientation>-<repetition>_bvp.npz
and produces:
  1. Text summary (printed)
  2. analysis/per_user.png        — bar chart of files per user, motions 1-6 vs 7-10
  3. analysis/user_motion.png     — heatmap of (user, motion) sample counts
  4. analysis/user_pos_ori.png    — per-user (position, orientation) coverage heatmap
  5. analysis/T_distribution.png  — sequence length histogram, by user
  6. analysis/summary.csv         — machine-readable per-user table

Usage: python analyze_dataset.py [--bvp-dir data/bvp]
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLIP_RE = re.compile(r"^(user\d+)-(\d+)-(\d+)-(\d+)-(\d+)_bvp\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvp-dir", default="data/bvp")
    ap.add_argument("--out-dir", default="analysis")
    ap.add_argument("--sample-T", type=int, default=2000,
                    help="Random-sample N files to read T (sequence length)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.bvp_dir, "**/*_bvp.npz"), recursive=True))
    print(f"Scanning {len(files)} files under {args.bvp_dir}\n")

    # parse metadata
    rows = []
    skipped = 0
    for fp in files:
        m = CLIP_RE.match(os.path.basename(fp))
        if not m:
            skipped += 1
            continue
        user, motion, pos, ori, rep = m.groups()
        rows.append({
            "path": fp,
            "user": user,
            "user_id": int(user.replace("user", "")),
            "motion": int(motion),
            "pos": int(pos),
            "ori": int(ori),
            "rep": int(rep),
        })
    if skipped:
        print(f"  ! skipped {skipped} files with non-matching names")

    users = sorted(set(r["user"] for r in rows), key=lambda u: int(u.replace("user", "")))
    motions = sorted(set(r["motion"] for r in rows))
    positions = sorted(set(r["pos"] for r in rows))
    orientations = sorted(set(r["ori"] for r in rows))

    # ---- text summary ----
    print(f"Total parseable: {len(rows)}")
    print(f"Users:        {users}")
    print(f"Motions:      {motions}  (motions 1-6 are kept by train.py default)")
    print(f"Positions:    {positions}")
    print(f"Orientations: {orientations}\n")

    # per-user counts split by motion 1-6 vs 7-10
    per_user_used = defaultdict(int)
    per_user_dropped = defaultdict(int)
    for r in rows:
        if r["motion"] in {1, 2, 3, 4, 5, 6}:
            per_user_used[r["user"]] += 1
        else:
            per_user_dropped[r["user"]] += 1

    print(f"{'user':8s}  {'used (m1-6)':>13s}  {'dropped (m7-10)':>17s}  {'total':>7s}  {'%dataset':>9s}")
    total_used = sum(per_user_used.values())
    total_drop = sum(per_user_dropped.values())
    for u in users:
        used = per_user_used[u]
        drop = per_user_dropped[u]
        tot = used + drop
        pct = 100 * used / total_used if total_used else 0
        print(f"{u:8s}  {used:>13d}  {drop:>17d}  {tot:>7d}  {pct:>8.1f}%")
    print(f"{'TOTAL':8s}  {total_used:>13d}  {total_drop:>17d}  {total_used+total_drop:>7d}\n")

    # ---- CSV ----
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user", "motion_1_6_count", "motion_7_10_count", "total"])
        for u in users:
            w.writerow([u, per_user_used[u], per_user_dropped[u],
                        per_user_used[u] + per_user_dropped[u]])
    print(f"  wrote {out/'summary.csv'}")

    # ---- 1. per-user bar chart ----
    x = np.arange(len(users))
    used_arr = [per_user_used[u] for u in users]
    drop_arr = [per_user_dropped[u] for u in users]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, used_arr, label="motions 1-6 (used by train.py)", color="#2c7fb8")
    ax.bar(x, drop_arr, bottom=used_arr, label="motions 7-10 (dropped)", color="#cccccc")
    for i, u in enumerate(users):
        ax.text(i, used_arr[i] + drop_arr[i] + 100, str(used_arr[i] + drop_arr[i]),
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(users, rotation=0)
    ax.set_ylabel("BVP files")
    ax.set_title(f"BVP files per user  (total {len(rows)}, kept {total_used}, dropped {total_drop})")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "per_user.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {out/'per_user.png'}")

    # ---- 2. user × motion heatmap ----
    M = np.zeros((len(users), len(motions)), dtype=int)
    user_ix = {u: i for i, u in enumerate(users)}
    motion_ix = {m: i for i, m in enumerate(motions)}
    for r in rows:
        M[user_ix[r["user"]], motion_ix[r["motion"]]] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(motions))); ax.set_xticklabels([f"m{m}" for m in motions])
    ax.set_yticks(range(len(users))); ax.set_yticklabels(users)
    ax.set_xlabel("motion id")
    ax.set_title("Sample count: user × motion  (motions 7-10 dropped by train.py)")
    # divider line between m6 and m7
    if 6 in motions and 7 in motions:
        ax.axvline(motions.index(7) - 0.5, color="red", linewidth=1.5, linestyle="--")
    for i in range(len(users)):
        for j in range(len(motions)):
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] < M.max() * 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="files")
    fig.tight_layout()
    fig.savefig(out / "user_motion.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {out/'user_motion.png'}")

    # ---- 3. user × (pos, ori) coverage ----
    # Show, for each user, how many position/orientation combos they cover.
    fig, axs = plt.subplots(1, len(users), figsize=(2.0 * len(users) + 1.5, 4),
                            sharey=True)
    if len(users) == 1:
        axs = [axs]
    pos_uni = sorted(positions)
    ori_uni = sorted(orientations)
    for ax, u in zip(axs, users):
        grid = np.zeros((len(ori_uni), len(pos_uni)), dtype=int)
        for r in rows:
            if r["user"] != u or r["motion"] not in {1,2,3,4,5,6}:
                continue
            grid[ori_uni.index(r["ori"]), pos_uni.index(r["pos"])] += 1
        ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_title(u, fontsize=10)
        ax.set_xticks(range(len(pos_uni))); ax.set_xticklabels(pos_uni, fontsize=8)
        ax.set_yticks(range(len(ori_uni))); ax.set_yticklabels(ori_uni, fontsize=8)
        ax.set_xlabel("pos")
        if u == users[0]:
            ax.set_ylabel("ori")
    fig.suptitle("Position × orientation coverage per user (motions 1-6 only)")
    fig.tight_layout()
    fig.savefig(out / "user_pos_ori.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {out/'user_pos_ori.png'}")

    # ---- 4. T distribution ----
    rng = np.random.default_rng(0)
    sample_paths = rng.choice([r["path"] for r in rows],
                              size=min(args.sample_T, len(rows)), replace=False)
    Ts = []
    Ts_per_user = defaultdict(list)
    for p in sample_paths:
        try:
            T = np.load(p)["velocity_spectrum_ro"].shape[2]
            Ts.append(T)
            user = CLIP_RE.match(os.path.basename(p)).group(1)
            Ts_per_user[user].append(T)
        except Exception:
            pass
    Ts = np.array(Ts)
    print(f"\nSampled {len(Ts)} files for T distribution: "
          f"min={Ts.min()}, p50={int(np.percentile(Ts, 50))}, "
          f"p90={int(np.percentile(Ts, 90))}, max={Ts.max()}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].hist(Ts, bins=range(int(Ts.min()), int(Ts.max()) + 2),
                color="#2c7fb8", edgecolor="white")
    axs[0].set_xlabel("T (sequence length, frames)")
    axs[0].set_ylabel("count")
    axs[0].set_title(f"T distribution (n={len(Ts)} sampled)")
    axs[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    box_data = [Ts_per_user[u] for u in users if Ts_per_user[u]]
    box_labels = [u for u in users if Ts_per_user[u]]
    axs[1].boxplot(box_data, labels=box_labels)
    axs[1].set_ylabel("T (sequence length)")
    axs[1].set_title("T per user")
    axs[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out / "T_distribution.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {out/'T_distribution.png'}")

    print(f"\nDone. Artifacts in {out}/")


if __name__ == "__main__":
    main()
