#!/usr/bin/env python3
"""
Advanced comparative Widar3.0 training script for running pure BVP vs dual BVP+BAP models.
Automatically generates isolated training runs saving models, hyperparameters, and timing statistics securely.
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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def normalize_data(data_1: np.ndarray) -> np.ndarray:
    data_1_max = np.concatenate((data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if np.any((data_1_max - data_1_min) == 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    return (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep + 1e-12)


def zero_padding(data, T_MAX):
    data_pad = []
    for sample in data:
        t = np.array(sample).shape[2]
        if len(np.array(sample).shape) == 4:
            # Dual Channel: (20, 20, t, 2)
            data_pad.append(np.pad(sample, ((0, 0), (0, 0), (0, T_MAX - t), (0, 0)), "constant", constant_values=0).tolist())
        else:
            # Single Channel: (20, 20, t)
            data_pad.append(np.pad(sample, ((0, 0), (0, 0), (T_MAX - t, 0)), "constant", constant_values=0).tolist())
    return np.array(data_pad)


def load_data(bvp_path: Path, bap_path: Path, motion_sel, feature_mode):
    T_MAX = 0
    data, label = [], []
    
    bvp_files = glob.glob(str(bvp_path / "**/*_bvp.npz"), recursive=True) + glob.glob(str(bvp_path / "**/*.mat"), recursive=True)
    
    if not bvp_files:
        print(f"No BVP sequences found recursively natively in {bvp_path}")
        return np.array([]), np.array([])
        
    for file_path in bvp_files:
        p = Path(file_path)
        try:
            if p.suffix == ".npz":
                mat_bvp = np.load(p)
                data_bvp = mat_bvp["velocity_spectrum_ro"]
            else:
                mat_bvp = scio.loadmat(file_path)
                data_bvp = mat_bvp["velocity_spectrum_ro"]

            clip_name = p.stem.replace("_bvp", "")
            label_1 = int(clip_name.split("-")[1])
            if label_1 not in motion_sel:
                continue

            data_bvp_normed = normalize_data(data_bvp)
            T_MAX = max(T_MAX, np.array(data_bvp).shape[2])

            if feature_mode == "bvp+bap":
                bap_file = bap_path / p.relative_to(bvp_path).parent / f"{clip_name}_bap.npz"
                if bap_file.exists():
                    mat_bap = np.load(bap_file)
                    data_bap = mat_bap["acceleration_spectrum_ro"]
                else:
                    data_bap = np.zeros_like(data_bvp)
                
                data_bap_normed = normalize_data(data_bap)
                sample = np.stack([data_bvp_normed, data_bap_normed], axis=-1)
            else:
                sample = data_bvp_normed

            data.append(sample.tolist())
            label.append(label_1)
        except Exception:
            continue

    data = zero_padding(data, T_MAX)
    
    if feature_mode == "bvp+bap":
        data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
    else:
        data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
        data = np.expand_dims(data, axis=-1)
        
    return data, np.array(label)


def assemble_model(input_shape, n_class, dropout=0.5, n_gru_hidden_units=128, learning_rate=1e-3):
    model_input = layers.Input(shape=input_shape, dtype="float32", name="name_model_input")
    x = layers.TimeDistributed(layers.Conv2D(16, kernel_size=(5, 5), activation="relu", data_format="channels_last"))(model_input)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout))(x)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)
    x = layers.GRU(n_gru_hidden_units, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    model_output = layers.Dense(n_class, activation="softmax", name="name_model_output")(x)

    model = keras.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train comparative models safely plotting stats configurations natively.")
    parser.add_argument("--bvp-dir", default="bvp_data", help="Directory linking for BVP arrays.")
    parser.add_argument("--bap-dir", default="bap_data", help="Directory linking for BAP arrays.")
    parser.add_argument("--mode", choices=["bvp", "bvp+bap"], default="bvp+bap", help="Feature extraction mapping mode.")
    parser.add_argument("--motions", default="1,2,3,4,5,6", help="Target motions sequence.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gpu", default=None, help="Set GPU Device explicitly (e.g., 0).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        tf.random.set_seed(1)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ALL_MOTION = [int(m) for m in args.motions.split(",") if m]
    N_MOTION = len(ALL_MOTION)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.mode.upper()}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Firing Isolated Execution Run: {run_id} ---")
    
    t_load_start = time.time()
    data, label = load_data(Path(args.bvp_dir), Path(args.bap_dir), ALL_MOTION, args.mode)
    t_load_end = time.time()
    
    if data.shape[0] == 0:
        print("No valid spatial-temporal clips found comprehensively.")
        sys.exit(1)
        
    print(f"Natively cached {label.shape[0]} tracking samples perfectly. Feature Tensor Dimensionality: {data[0].shape}")

    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.1, stratify=label, random_state=42)
    label_train_1h = np.eye(N_MOTION)[label_train - 1]

    model = assemble_model(
        input_shape=data.shape[1:],
        n_class=N_MOTION,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    
    t_train_start = time.time()
    history = model.fit(
        {"name_model_input": data_train},
        {"name_model_output": label_train_1h},
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_split=0.1,
        shuffle=True,
    )
    t_train_end = time.time()
    
    model_path = run_dir / "trained_classifier.h5"
    model.save(str(model_path))

    label_test_pred = model.predict(data_test, verbose=0)
    label_test_pred = np.argmax(label_test_pred, axis=-1) + 1
    test_acc = np.mean(label_test_pred == label_test)
    print(f"Native Analytical Test Accuracy Mapping: {test_acc:.4f}")
    
    stats = {
        "run_id": run_id,
        "feature_mode": args.mode,
        "dataset_samples": len(label),
        "motions": args.motions,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate
        },
        "performance": {
            "test_accuracy": float(test_acc),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1])
        },
        "execution_time_seconds": {
            "data_loading": t_load_end - t_load_start,
            "training": t_train_end - t_train_start,
            "total_elapsed": time.time() - t_load_start
        }
    }
    
    with open(run_dir / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"--- Run successfully concluded! Output directory natively integrated exactly at: {run_dir} ---")

if __name__ == "__main__":
    main()
