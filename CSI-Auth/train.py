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

Usage:
    python train.py --mode all --bvp-dir data/bvp
    python train.py --mode bap --bvp-dir data/bvp --users user10,user11,user12,user13
    python train.py --mode bvp+bap --bvp-dir data/bvp --gpu 0
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as scio
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, GRU, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, TimeDistributed,
)
from tensorflow.keras.models import Model


# ──────────────────────────────────────────────────────────
# Hyperparameters (Widar3.0 paper defaults)
# ──────────────────────────────────────────────────────────
ALL_MOTION = [1, 2, 3, 4, 5, 6]
N_MOTION = len(ALL_MOTION)
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
# Data loading
# ──────────────────────────────────────────────────────────
def _load_single(file_path):
    """Load one BVP .npz or .mat file. Returns (array, label, user) or None."""
    try:
        fname = os.path.basename(file_path)
        if file_path.endswith(".npz"):
            arr = np.load(file_path)["velocity_spectrum_ro"]
            clip = fname.replace("_bvp.npz", "").replace(".npz", "")
        else:
            arr = scio.loadmat(file_path)["velocity_spectrum_ro"]
            clip = fname.replace(".mat", "")
        parts = clip.split("-")
        return arr, int(parts[1]), parts[0]
    except Exception:
        return None


def load_raw_bvp(bvp_dir, motion_sel, user_filter=None):
    """Load all raw (unnormalized) BVP arrays.

    Returns:
        raw_data: list of arrays, each (20, 20, T)
        labels:   np.ndarray of gesture labels
        T_MAX:    max time length across samples
    """
    files = sorted(
        glob.glob(os.path.join(bvp_dir, "**/*.npz"), recursive=True)
        + glob.glob(os.path.join(bvp_dir, "**/*.mat"), recursive=True)
    )
    raw, labels, T_MAX = [], [], 0
    users = set()
    workers = min(16, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_load_single, files):
            if res is None:
                continue
            arr, label, user = res
            if label not in motion_sel:
                continue
            if user_filter and user not in user_filter:
                continue
            raw.append(arr)
            labels.append(label)
            users.add(user)
            T_MAX = max(T_MAX, arr.shape[2])

    print(f"  Loaded {len(raw)} samples | users: {sorted(users)} | T_MAX: {T_MAX}")
    return raw, np.array(labels), T_MAX


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
def run_experiment(name, data, labels, n_channels):
    """Train and evaluate one experiment. Returns result dict."""
    T_MAX = data.shape[1]
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  Data: {data.shape}  |  Labels: {labels.shape}")
    print(f"{'=' * 60}")

    data_train, data_test, y_train, y_test = train_test_split(
        data, labels, test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )
    y_train_oh = np.eye(N_MOTION)[y_train - 1]

    model = build_model(input_shape=(T_MAX, 20, 20, n_channels), n_class=N_MOTION)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t0 = time.time()
    history = model.fit(
        data_train, y_train_oh,
        batch_size=BATCH_SIZE, epochs=N_EPOCHS,
        verbose=1, validation_split=0.1, shuffle=True,
    )
    train_time = time.time() - t0

    pred = np.argmax(model.predict(data_test, verbose=0), axis=-1) + 1
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
def parse_args():
    p = argparse.ArgumentParser(
        description="Train Widar3.0 gesture classifier with BVP/BAP features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --mode all --bvp-dir data/bvp
  python train.py --mode bap --bvp-dir data/bvp --users user10,user11,user12,user13
  python train.py --mode bvp+bap --bvp-dir data/bvp --gpu 0 --epochs 50
        """,
    )
    p.add_argument("--bvp-dir", default="data/bvp", help="Path to BVP data directory")
    p.add_argument("--mode", default="all", choices=["bvp", "bap", "bvp+bap", "all"],
                   help="Feature mode (default: all)")
    p.add_argument("--users", default=None,
                   help="Comma-separated user filter (e.g. user10,user11)")
    p.add_argument("--gpu", default=None, help="GPU device ID. Omit for CPU.")
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return p.parse_args()


def main():
    global N_EPOCHS, BATCH_SIZE
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.epochs:
        N_EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size

    user_filter = set(args.users.split(",")) if args.users else None
    modes = ["bvp", "bap", "bvp+bap"] if args.mode == "all" else [args.mode]

    print(f"Users:    {sorted(user_filter) if user_filter else 'all'}")
    print(f"Gestures: {ALL_MOTION}")
    print(f"Modes:    {modes}")
    print(f"Epochs:   {N_EPOCHS}  Batch: {BATCH_SIZE}")

    print("\nLoading BVP data...")
    raw_bvp, labels, T_MAX = load_raw_bvp(args.bvp_dir, ALL_MOTION, user_filter)

    if len(raw_bvp) == 0:
        print("No data found. Check --bvp-dir and --users.")
        sys.exit(1)

    results = []
    for mode in modes:
        n_ch = 2 if mode == "bvp+bap" else 1
        data = prepare_features(raw_bvp, T_MAX, mode)
        res = run_experiment(mode.upper(), data, labels, n_ch)
        results.append(res)

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS  ({len(raw_bvp)} samples, {N_MOTION} gestures, "
          f"users: {sorted(user_filter) if user_filter else 'all'})")
    print("-" * 60)
    for r in results:
        print(f"  {r['name']:12s}  test={r['accuracy']:.4f}  "
              f"val={r['final_val_acc']:.4f}  ({r['train_time_s']}s)")
    print("=" * 60)

    # Save
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id": run_id,
        "users": sorted(user_filter) if user_filter else "all",
        "n_samples": len(raw_bvp),
        "n_gestures": N_MOTION,
        "epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "results": results,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {run_dir}/results.json")


if __name__ == "__main__":
    main()
