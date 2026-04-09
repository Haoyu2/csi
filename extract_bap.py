#!/usr/bin/env python3
"""
Extract BAP (Body Acceleration Profile) from BVP (Body Velocity Profile) data.

BAP is the temporal first-difference of BVP:
    BAP[t] = BVP[t] - BVP[t-1]

This captures how the velocity power distribution changes over time —
when a body part accelerates, its power peak shifts between velocity bins.

Usage:
    python extract_bap.py --bvp-dir data/bvp --bap-dir data/bap
    python extract_bap.py --bvp-dir data/bvp --bap-dir data/bap --users user10,user11
"""
import argparse
import glob
import os
import time

import numpy as np
from tqdm import tqdm


def extract_bap(bvp: np.ndarray) -> np.ndarray:
    """Compute BAP as temporal first-difference of BVP.

    Args:
        bvp: array of shape (M, M, T), the velocity spectrum.

    Returns:
        bap: array of shape (M, M, T), the acceleration spectrum.
             bap[:, :, 0] = 0 (no previous frame).
             bap[:, :, t] = bvp[:, :, t] - bvp[:, :, t-1] for t >= 1.
    """
    bap = np.zeros_like(bvp)
    if bvp.shape[2] > 1:
        bap[:, :, 1:] = bvp[:, :, 1:] - bvp[:, :, :-1]
    return bap


def process_file(bvp_path: str, bap_path: str) -> bool:
    """Load a BVP file, compute BAP, save to bap_path. Returns True on success."""
    try:
        data = np.load(bvp_path)
        bvp = data["velocity_spectrum_ro"]
        bap = extract_bap(bvp)
        os.makedirs(os.path.dirname(bap_path), exist_ok=True)
        np.savez_compressed(bap_path, acceleration_spectrum_ro=bap)
        return True
    except Exception as e:
        print(f"  Error processing {bvp_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract BAP from BVP data.")
    parser.add_argument("--bvp-dir", required=True, help="Input directory containing BVP .npz files")
    parser.add_argument("--bap-dir", required=True, help="Output directory for BAP .npz files")
    parser.add_argument("--users", default=None, help="Comma-separated user filter (e.g. user10,user11)")
    args = parser.parse_args()

    user_filter = set(args.users.split(",")) if args.users else None

    # Find all BVP files
    bvp_files = sorted(glob.glob(os.path.join(args.bvp_dir, "**/*_bvp.npz"), recursive=True))
    if not bvp_files:
        # Try without _bvp suffix (some files may just be .npz)
        bvp_files = sorted(glob.glob(os.path.join(args.bvp_dir, "**/*.npz"), recursive=True))

    if not bvp_files:
        print(f"No .npz files found in {args.bvp_dir}")
        return

    # Apply user filter
    if user_filter:
        bvp_files = [f for f in bvp_files if any(u in f for u in user_filter)]

    print(f"Found {len(bvp_files)} BVP files to process")
    print(f"Output directory: {args.bap_dir}")

    t0 = time.time()
    success, fail = 0, 0

    for bvp_path in tqdm(bvp_files, desc="Extracting BAP"):
        # Mirror directory structure: bvp_dir/userX/file_bvp.npz -> bap_dir/userX/file_bap.npz
        rel_path = os.path.relpath(bvp_path, args.bvp_dir)
        bap_filename = os.path.basename(rel_path).replace("_bvp.npz", "_bap.npz")
        if "_bap" not in bap_filename:
            bap_filename = bap_filename.replace(".npz", "_bap.npz")
        bap_path = os.path.join(args.bap_dir, os.path.dirname(rel_path), bap_filename)

        if process_file(bvp_path, bap_path):
            success += 1
        else:
            fail += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {success} extracted, {fail} failed")


if __name__ == "__main__":
    main()
