#!/usr/bin/env python3
"""
Widar3.0 gesture recognition training script.

Supports four feature modes:
  bvp       Single-channel Body Velocity Profile
  bap       Single-channel Body Acceleration Profile (computed as dBVP/dt)
  bvp+bap   Dual-channel BVP + BAP
  all       Run bvp, bap, bvp+bap sequentially and compare

BAP is always computed on-the-fly from BVP as: BAP[t] = BVP[t] - BVP[t-1].
Pre-extracted BAP files are NOT required.

Data loading is CSV-driven (see dataset.py): filters operate on bvp_full.csv,
the train/test split defaults to ``by-key`` (no session leakage), and the
875 doubled-date zip artifacts are dropped automatically.

Usage:
    python train.py --mode all
    python train.py --mode bap --users user1,user4 --gestures 1,2,3
    python train.py --mode bvp+bap --gpu 0 --epochs 50 --cache cache/all.npz
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, GRU, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, TimeDistributed,
)
from tensorflow.keras.models import Model

from dataset import load_arrays, load_manifest, make_split_indices, summarize


# ──────────────────────────────────────────────────────────
# Hyperparameters (Widar3.0 paper defaults)
# ──────────────────────────────────────────────────────────
DEFAULT_GESTURES = [1, 2, 3, 4, 5, 6]   # legacy; motions 7-10 selectable via --gestures
N_EPOCHS = 30
DROPOUT = 0.5
GRU_UNITS = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_FRACTION = 0.1
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────
def normalize_data(data):
    """Per-frame min-max normalization to [0, 1].

    Each time frame is normalized independently. Frames with no signal
    (all-zero) remain zero. This avoids the original Widar3.0 bug where
    the entire sample is skipped when any frame is constant.

    Args:
        data: array of shape (M, M, T).
    Returns:
        Normalized array of same shape.
    """
    result = np.zeros_like(data)
    for t in range(data.shape[2]):
        frame = data[:, :, t]
        fmin, fmax = frame.min(), frame.max()
        if fmax - fmin > 0:
            result[:, :, t] = (frame - fmin) / (fmax - fmin)
    return result


# ──────────────────────────────────────────────────────────
# Data loading — see dataset.py for the manifest layer
# ──────────────────────────────────────────────────────────
def encode_labels(labels):
    """Map arbitrary labels to 0..N-1. Returns (encoded, n_class, classes)."""
    classes = sorted(set(labels.tolist()))
    mapping = {c: i for i, c in enumerate(classes)}
    encoded = np.array([mapping[v] for v in labels], dtype=np.int64)
    return encoded, len(classes), classes


# ──────────────────────────────────────────────────────────
# Feature preparation
# ──────────────────────────────────────────────────────────
def compute_bap(bvp):
    """BAP[t] = BVP[t] - BVP[t-1]. First frame is zero."""
    bap = np.zeros_like(bvp)
    if bvp.shape[2] > 1:
        bap[:, :, 1:] = bvp[:, :, 1:] - bvp[:, :, :-1]
    return bap


def zero_pad(data_list, T_MAX):
    """Zero-pad each sample along the time axis to T_MAX."""
    padded = []
    for arr in data_list:
        t = arr.shape[2]
        if arr.ndim == 4:
            pw = ((0, 0), (0, 0), (T_MAX - t, 0), (0, 0))
        else:
            pw = ((0, 0), (0, 0), (T_MAX - t, 0))
        padded.append(np.pad(arr, pw, "constant", constant_values=0))
    return np.array(padded)


def prepare_features(raw_bvp, T_MAX, mode):
    """Build model input tensor for a given feature mode.

    Returns: np.ndarray of shape (N, T_MAX, 20, 20, C) where C=1 or 2.
    """
    samples = []

    if mode == "bvp":
        samples = [normalize_data(b) for b in raw_bvp]
    elif mode == "bap":
        samples = [normalize_data(compute_bap(b)) for b in raw_bvp]
    elif mode == "bvp+bap":
        for b in raw_bvp:
            bvp_n = normalize_data(b)
            bap_n = normalize_data(compute_bap(b))
            samples.append(np.stack([bvp_n, bap_n], axis=-1))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    data = zero_pad(samples, T_MAX)
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)  # (N,M,M,T[,C]) -> (N,T,M,M[,C])
    if data.ndim == 4:
        data = np.expand_dims(data, axis=-1)
    return data


# ──────────────────────────────────────────────────────────
# Model (Widar3.0 paper architecture)
# ──────────────────────────────────────────────────────────
def build_model(input_shape, n_class):
    """CNN + GRU classifier identical to the Widar3.0 paper.

    Architecture:
        TimeDistributed(Conv2D 16 × 5×5) → MaxPool 2×2 → Flatten
        → Dense 64 → Dropout → Dense 64
        → GRU 128 → Dropout → Dense(softmax)
    """
    inp = Input(shape=input_shape, dtype="float32")
    x = TimeDistributed(Conv2D(16, (5, 5), activation="relu", data_format="channels_last"))(inp)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    x = TimeDistributed(Dropout(DROPOUT))(x)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    x = GRU(GRU_UNITS, return_sequences=False)(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(n_class, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────
def run_experiment(name, data, labels, train_idx, test_idx, n_channels, n_class):
    """Train and evaluate one experiment using a pre-computed split.

    Labels must be 0-indexed in [0, n_class). The split (``train_idx``,
    ``test_idx``) is computed at the manifest level by ``make_split_indices``
    so all modes within a group see the same paired split.
    """
    T_MAX = data.shape[1]
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  Data: {data.shape}  |  train={len(train_idx)}  test={len(test_idx)}  n_class: {n_class}")
    print(f"{'=' * 60}")

    data_train, data_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    y_train_oh = np.eye(n_class)[y_train]

    model = build_model(input_shape=(T_MAX, 20, 20, n_channels), n_class=n_class)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t0 = time.time()
    history = model.fit(
        data_train, y_train_oh,
        batch_size=BATCH_SIZE, epochs=N_EPOCHS,
        verbose=1, validation_split=0.1, shuffle=True,
    )
    train_time = time.time() - t0

    pred = np.argmax(model.predict(data_test, verbose=0), axis=-1)
    acc = np.mean(pred == y_test)

    cm = confusion_matrix(y_test, pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(
        cm.astype(float), row_sums,
        out=np.zeros_like(cm, dtype=float), where=row_sums != 0,
    )

    print(f"\nConfusion matrix (normalized):\n{np.around(cm_norm, 2)}")
    print(f"\n>>> {name} — Test Accuracy: {acc:.4f}  ({train_time:.1f}s)\n")

    return {
        "name": name,
        "accuracy": float(acc),
        "train_time_s": round(train_time, 1),
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
    }


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def _csv_list(s):
    return [v.strip() for v in s.split(",") if v.strip()] if s else None


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Widar3.0 gesture classifier with BVP/BAP features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --mode all
  python train.py --mode bap --users user1,user4 --gestures 1,2,3
  python train.py --mode bvp+bap --gpu 0 --epochs 50 --cache cache/all.npz
  python train.py --mode all --dedup-mode one-per-key --split-strategy by-session
        """,
    )
    # Data source
    p.add_argument("--csv", default="bvp_full.csv", help="Manifest CSV (default: bvp_full.csv)")
    p.add_argument("--data-dir", default="data", help="Extracted-zip root (default: data)")

    # Filters (exact match against CSV columns; None keeps all)
    p.add_argument("--users", default=None,
                   help="Comma-separated user tokens, e.g. user1,user4 (exact match)")
    p.add_argument("--gestures", default=",".join(map(str, DEFAULT_GESTURES)),
                   help="Comma-separated gesture IDs (default: 1,2,3,4,5,6; pass 1..10 to include 7-10)")
    p.add_argument("--locations", default=None,
                   help="Comma-separated location IDs (default: all)")
    p.add_argument("--orientations", default=None,
                   help="Comma-separated face-orientation IDs (default: all)")

    # Manifest options
    p.add_argument("--dedup-mode", default="keep-all", choices=["keep-all", "one-per-key"],
                   help="Treat multi-session re-recordings: keep-all (default) or one-per-key")
    p.add_argument("--split-strategy", default="by-key",
                   choices=["by-key", "random", "by-session"],
                   help="Train/test split strategy (default: by-key; no session leakage)")
    p.add_argument("--cache", default=None,
                   help="Optional .npz path for the raw-array cache; speeds repeated runs")

    # Experiment
    p.add_argument("--mode", default="all", choices=["bvp", "bap", "bvp+bap", "all"],
                   help="Feature mode (default: all)")
    p.add_argument("--task", default="motion", choices=["motion", "user"],
                   help="Classification target (default: motion)")
    p.add_argument("--per-motion", action="store_true",
                   help="Run one experiment per gesture (useful for --task user)")
    p.add_argument("--gpu", default=None, help="GPU device ID. Omit for CPU.")
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return p.parse_args()


def main():
    global N_EPOCHS, BATCH_SIZE
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.gpu is not None else "-1"
    if args.epochs:
        N_EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size

    users = _csv_list(args.users)
    gestures = [int(g) for g in _csv_list(args.gestures)]
    locations = _csv_list(args.locations)
    orientations = _csv_list(args.orientations)
    modes = ["bvp", "bap", "bvp+bap"] if args.mode == "all" else [args.mode]
    groups = ([(f"motion{m}", [m]) for m in gestures] if args.per_motion
              else [("all", list(gestures))])

    print(f"CSV:      {args.csv}  (data_dir={args.data_dir})")
    print(f"Users:    {users or 'all'}")
    print(f"Gestures: {gestures}")
    print(f"Locations:    {locations or 'all'}")
    print(f"Orientations: {orientations or 'all'}")
    print(f"Dedup:    {args.dedup_mode}     Split: {args.split_strategy}")
    print(f"Modes:    {modes}     Task: {args.task}     Groups: {[g[0] for g in groups]}")
    print(f"Epochs:   {N_EPOCHS}  Batch: {BATCH_SIZE}")

    print("\nLoading manifest...")
    manifest = load_manifest(
        args.csv, data_dir=args.data_dir,
        users=users, gestures=gestures,
        locations=locations, orientations=orientations,
        dedup_mode=args.dedup_mode,
    )
    summarize(manifest, "filtered")
    if not manifest:
        print("No rows match the filters.")
        sys.exit(1)

    print("\nLoading BVP arrays...")
    raw_bvp, motion_lbls, user_lbls, T_MAX = load_arrays(
        manifest, parallel=True, cache_path=args.cache,
    )
    print(f"  loaded {len(raw_bvp)} samples | T_MAX: {T_MAX}")

    results = []
    for grp_name, motion_set in groups:
        grp_idx = np.where(np.isin(motion_lbls, motion_set))[0]
        if len(grp_idx) == 0:
            print(f"\n[{grp_name}] no samples; skip.")
            continue

        grp_manifest = [manifest[i] for i in grp_idx]
        train_local, test_local = make_split_indices(
            grp_manifest, strategy=args.split_strategy,
            test_frac=TEST_FRACTION, seed=RANDOM_SEED,
        )

        raw_grp = [raw_bvp[i] for i in grp_idx]
        raw_target = (user_lbls if args.task == "user" else motion_lbls)[grp_idx]
        y, n_class, classes = encode_labels(raw_target)
        print(f"\n[{grp_name}] samples={len(grp_idx)}  "
              f"train={len(train_local)}  test={len(test_local)}  classes={classes}")

        for mode in modes:
            n_ch = 2 if mode == "bvp+bap" else 1
            data = prepare_features(raw_grp, T_MAX, mode)
            res = run_experiment(
                f"{grp_name}|{mode.upper()}", data, y,
                train_local, test_local, n_ch, n_class,
            )
            res["group"] = grp_name
            res["mode"] = mode
            res["n_samples"] = int(len(grp_idx))
            res["n_train"] = int(len(train_local))
            res["n_test"] = int(len(test_local))
            res["classes"] = [str(c) for c in classes]
            results.append(res)

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS  task={args.task}  split={args.split_strategy}  dedup={args.dedup_mode}")
    print("-" * 60)
    for r in results:
        print(f"  {r['name']:30s}  test={r['accuracy']:.4f}  "
              f"val={r['final_val_acc']:.4f}  ({r['train_time_s']}s)")
    print("=" * 60)

    # Save
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id": run_id,
        "task": args.task,
        "per_motion": args.per_motion,
        "filters": {
            "users": users or "all", "gestures": gestures,
            "locations": locations or "all", "orientations": orientations or "all",
        },
        "dedup_mode": args.dedup_mode,
        "split_strategy": args.split_strategy,
        "n_samples_total": len(raw_bvp),
        "epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "results": results,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {run_dir}/results.json")


if __name__ == "__main__":
    main()
