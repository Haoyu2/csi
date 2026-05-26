#!/usr/bin/env python3
"""experiments.py — sweep BAP vs BVP across three configurations.

All configs use users 1-3, gestures 1-6, ``keep-all`` dedup.

Config 1 — Motion classification:
    Single experiment per mode. Predict gesture (6-class).
    Split: by-key, test_frac=0.1.

Config 2 — Per-motion user classification:
    One experiment per motion x mode (12 total). Predict user (3-class).
    Split: by-key, test_frac=0.1.

Config 3 — Per-cell user classification (ideal-settings test):
    For each (motion, orientation, location) cell where all 3 users have
    data, predict user (3-class). Hundreds of experiments.
    Split: random, test_frac=0.2 (cells are small).

Usage:
    python experiments.py --gpu 0
    python experiments.py --configs 1,2
    python experiments.py --cache cache/u1-3_g1-6.npz --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from dataset import load_arrays, load_manifest, make_split_indices
from train import RANDOM_SEED, build_model, encode_labels, prepare_features


# ──────────────────────────────────────────────────────────
# Single-experiment runner
# ──────────────────────────────────────────────────────────
def run_one(name, X, y, train_idx, test_idx, n_class, n_channels, *,
            epochs, batch_size, verbose=0):
    T_MAX = X.shape[1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_oh = np.eye(n_class)[y_train]

    val_split = 0.1 if len(X_train) >= 10 else 0.0

    model = build_model(input_shape=(T_MAX, 20, 20, n_channels), n_class=n_class)
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t0 = time.time()
    history = model.fit(
        X_train, y_train_oh,
        batch_size=batch_size, epochs=epochs,
        verbose=verbose, validation_split=val_split, shuffle=True,
    )
    train_time = time.time() - t0

    pred = np.argmax(model.predict(X_test, verbose=0), axis=-1)
    acc = float(np.mean(pred == y_test))

    # Free GPU memory between experiments
    tf.keras.backend.clear_session()

    final_val = (float(history.history["val_accuracy"][-1])
                 if val_split > 0 else None)
    return {
        "name": name,
        "accuracy": acc,
        "train_time_s": round(train_time, 1),
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": final_val,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_class": int(n_class),
    }


def prepare_subset(raw_bvp, motion_lbls, user_lbls, idx, task, mode):
    raw_subset = [raw_bvp[i] for i in idx]
    T_MAX = max(a.shape[2] for a in raw_subset)
    target = (user_lbls if task == "user" else motion_lbls)[idx]
    y, n_class, classes = encode_labels(target)
    X = prepare_features(raw_subset, T_MAX, mode)
    n_ch = 2 if mode == "bvp+bap" else 1
    return X, y, n_class, classes, n_ch


# ──────────────────────────────────────────────────────────
# Config 1 — Motion classification
# ──────────────────────────────────────────────────────────
def run_config1(raw_bvp, motion_lbls, user_lbls, manifest, args):
    print("\n" + "=" * 60)
    print("  Config 1: motion classification  (users 1-3, gestures 1-6)")
    print("=" * 60)

    idx = np.arange(len(motion_lbls))
    sub_manifest = [manifest[i] for i in idx]
    train_local, test_local = make_split_indices(
        sub_manifest, strategy="by-key", test_frac=0.1, seed=RANDOM_SEED,
    )
    print(f"  samples={len(idx)}  train={len(train_local)}  test={len(test_local)}")

    results = []
    for mode in ("bvp", "bap"):
        X, y, n_class, classes, n_ch = prepare_subset(
            raw_bvp, motion_lbls, user_lbls, idx, task="motion", mode=mode,
        )
        res = run_one(
            f"C1|{mode.upper()}", X, y, train_local, test_local,
            n_class, n_ch, epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        )
        res.update({"config": 1, "mode": mode, "task": "motion",
                    "classes": [str(c) for c in classes]})
        results.append(res)
        print(f"  {res['name']}: test={res['accuracy']:.4f}  ({res['train_time_s']}s)")
    return results


# ──────────────────────────────────────────────────────────
# Config 2 — Per-motion user classification
# ──────────────────────────────────────────────────────────
def run_config2(raw_bvp, motion_lbls, user_lbls, manifest, args):
    print("\n" + "=" * 60)
    print("  Config 2: per-motion user classification  (users 1-3)")
    print("=" * 60)

    results = []
    for m in [1, 2, 3, 4, 5, 6]:
        idx = np.where(motion_lbls == m)[0]
        if len(idx) == 0:
            print(f"  motion {m}: no samples; skip")
            continue
        sub_manifest = [manifest[i] for i in idx]
        train_local, test_local = make_split_indices(
            sub_manifest, strategy="by-key", test_frac=0.1, seed=RANDOM_SEED,
        )

        for mode in ("bvp", "bap"):
            X, y, n_class, classes, n_ch = prepare_subset(
                raw_bvp, motion_lbls, user_lbls, idx, task="user", mode=mode,
            )
            res = run_one(
                f"C2|m{m}|{mode.upper()}", X, y, train_local, test_local,
                n_class, n_ch, epochs=args.epochs, batch_size=args.batch_size, verbose=2,
            )
            res.update({"config": 2, "mode": mode, "task": "user", "motion": m,
                        "classes": [str(c) for c in classes]})
            results.append(res)
            print(f"  {res['name']}: test={res['accuracy']:.4f}  "
                  f"({res['n_train']}/{res['n_test']}, {res['train_time_s']}s)")
    return results


# ──────────────────────────────────────────────────────────
# Config 3 — Per-cell user classification
# ──────────────────────────────────────────────────────────
def find_valid_cells(manifest, user_lbls):
    """Return list of ((motion, orientation, location), indices) where all 3 users have data."""
    cells = defaultdict(list)
    for i, r in enumerate(manifest):
        cells[(int(r["gesture"]), int(r["face orientation"]), int(r["location"]))].append(i)
    required = {"user1", "user2", "user3"}
    valid = []
    for cell, indices in sorted(cells.items()):
        users_present = {user_lbls[i] for i in indices}
        if required.issubset(users_present):
            valid.append((cell, indices))
    return valid


def run_config3(raw_bvp, motion_lbls, user_lbls, manifest, args):
    print("\n" + "=" * 60)
    print("  Config 3: per-cell user classification  (users 1-3, ideal settings)")
    print("=" * 60)

    valid = find_valid_cells(manifest, user_lbls)
    print(f"  found {len(valid)} valid cells (all 3 users present)")
    print(f"  total experiments: {len(valid) * 2}\n")

    results = []
    for cell_idx, (cell, indices) in enumerate(valid):
        m, o, l = cell
        idx_arr = np.array(indices)

        rng = np.random.default_rng(RANDOM_SEED)
        perm = rng.permutation(len(idx_arr))
        n_test = max(int(round(len(idx_arr) * 0.2)), 3)
        test_local = np.sort(perm[:n_test])
        train_local = np.sort(perm[n_test:])

        for mode in ("bvp", "bap"):
            X, y, n_class, classes, n_ch = prepare_subset(
                raw_bvp, motion_lbls, user_lbls, idx_arr, task="user", mode=mode,
            )
            if n_class < 3:
                continue
            res = run_one(
                f"C3|m{m}o{o}l{l}|{mode.upper()}", X, y, train_local, test_local,
                n_class, n_ch, epochs=args.epochs, batch_size=args.batch_size, verbose=0,
            )
            res.update({"config": 3, "mode": mode, "task": "user",
                        "motion": m, "orientation": o, "location": l,
                        "classes": [str(c) for c in classes]})
            results.append(res)

        if (cell_idx + 1) % 10 == 0 or cell_idx + 1 == len(valid):
            bvp_accs = [r["accuracy"] for r in results if r["mode"] == "bvp"]
            bap_accs = [r["accuracy"] for r in results if r["mode"] == "bap"]
            print(f"  [{cell_idx + 1}/{len(valid)}]  "
                  f"BVP mean={np.mean(bvp_accs):.4f}  BAP mean={np.mean(bap_accs):.4f}")
    return results


# ──────────────────────────────────────────────────────────
# Summary writer
# ──────────────────────────────────────────────────────────
def _fmt_val(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"


def write_summary(out_dir, results, manifest_size):
    lines = [f"# Experiment Sweep Results\n",
             f"Generated: {datetime.now().isoformat(timespec='seconds')}\n",
             f"Source manifest: users 1-3, gestures 1-6, keep-all dedup ({manifest_size} samples)\n"]

    if "config1" in results:
        lines.append("\n## Config 1 — Motion classification\n\n")
        lines.append("Predict gesture (6 classes). Split: by-key, test_frac=0.1.\n\n")
        lines.append("| Mode | Test acc | Train acc | Val acc | Time |\n|---|---|---|---|---|\n")
        for r in results["config1"]:
            lines.append(f"| {r['mode'].upper()} | {r['accuracy']:.4f} | "
                         f"{r['final_train_acc']:.4f} | {_fmt_val(r['final_val_acc'])} | "
                         f"{r['train_time_s']}s |\n")

    if "config2" in results:
        lines.append("\n## Config 2 — Per-motion user classification\n\n")
        lines.append("Predict user (3 classes). Split: by-key, test_frac=0.1.\n\n")
        lines.append("| Motion | Mode | Test acc | n_train | n_test |\n|---|---|---|---|---|\n")
        for r in results["config2"]:
            lines.append(f"| {r['motion']} | {r['mode'].upper()} | "
                         f"{r['accuracy']:.4f} | {r['n_train']} | {r['n_test']} |\n")

    if "config3" in results:
        rs = results["config3"]
        lines.append("\n## Config 3 — Per-cell user classification (ideal settings)\n\n")
        lines.append("Predict user (3 classes). One experiment per "
                     "(motion, orientation, location) cell. Split: random, test_frac=0.2.\n\n")
        bvp = [r["accuracy"] for r in rs if r["mode"] == "bvp"]
        bap = [r["accuracy"] for r in rs if r["mode"] == "bap"]
        lines.append(f"**Overall mean across {len(bvp)} cells:**\n\n")
        lines.append(f"- BVP: **{np.mean(bvp):.4f}** (std {np.std(bvp):.4f})\n")
        lines.append(f"- BAP: **{np.mean(bap):.4f}** (std {np.std(bap):.4f})\n\n")
        lines.append("### Per-motion mean\n\n| Motion | BVP | BAP | Cells |\n|---|---|---|---|\n")
        for m in range(1, 7):
            bvp_m = [r["accuracy"] for r in rs if r["mode"] == "bvp" and r["motion"] == m]
            bap_m = [r["accuracy"] for r in rs if r["mode"] == "bap" and r["motion"] == m]
            if bvp_m:
                lines.append(f"| {m} | {np.mean(bvp_m):.4f} | "
                             f"{np.mean(bap_m):.4f} | {len(bvp_m)} |\n")

    (out_dir / "summary.md").write_text("".join(lines))


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--configs", default="1,2,3", help="Comma-separated config IDs to run (default: all)")
    p.add_argument("--epochs", type=int, default=30, help="Epochs per experiment (default: 30)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--gpu", default=None, help="GPU device ID; omit for CPU")
    p.add_argument("--cache", default="cache/u1-3_g1-6.npz",
                   help="Path to raw-array cache (default: cache/u1-3_g1-6.npz)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.gpu is not None else "-1"
    configs = [int(c) for c in args.configs.split(",")]

    print("Loading manifest (users 1-3, gestures 1-6)...")
    manifest = load_manifest(
        "bvp_full.csv", data_dir="data",
        users=["user1", "user2", "user3"],
        gestures=[1, 2, 3, 4, 5, 6],
        dedup_mode="keep-all",
    )
    print(f"  {len(manifest)} rows")

    print("\nLoading raw arrays...")
    raw_bvp, motion_lbls, user_lbls, T_MAX = load_arrays(
        manifest, parallel=True, cache_path=args.cache,
    )
    print(f"  {len(raw_bvp)} samples, T_MAX={T_MAX}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"exp_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be written to {out_dir}/\n")

    all_results = {}
    t_start = time.time()

    if 1 in configs:
        r = run_config1(raw_bvp, motion_lbls, user_lbls, manifest, args)
        all_results["config1"] = r
        (out_dir / "config1_motion.json").write_text(json.dumps(r, indent=2))

    if 2 in configs:
        r = run_config2(raw_bvp, motion_lbls, user_lbls, manifest, args)
        all_results["config2"] = r
        (out_dir / "config2_per_motion.json").write_text(json.dumps(r, indent=2))

    if 3 in configs:
        r = run_config3(raw_bvp, motion_lbls, user_lbls, manifest, args)
        all_results["config3"] = r
        (out_dir / "config3_per_cell.json").write_text(json.dumps(r, indent=2))
        # Flat CSV for easy downstream analysis
        with open(out_dir / "config3_all_cells.csv", "w") as f:
            f.write("motion,orientation,location,mode,accuracy,n_train,n_test,train_time_s\n")
            for r_ in r:
                f.write(f"{r_['motion']},{r_['orientation']},{r_['location']},"
                        f"{r_['mode']},{r_['accuracy']:.4f},{r_['n_train']},"
                        f"{r_['n_test']},{r_['train_time_s']}\n")

    write_summary(out_dir, all_results, len(manifest))
    print(f"\n\nAll done in {(time.time() - t_start) / 60:.1f} min")
    print(f"Results in {out_dir}/")


if __name__ == "__main__":
    main()
