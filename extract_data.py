#!/usr/bin/env python3
"""Extract Widar3.0 BVP archives into ``data/`` with zip-native layout.

Each archive already contains a top-level directory:
    BVP-Data-From_Batch1.zip  ->  BVP-Data-Batch1/<date>/<user>/<clip>_bvp.npz
    BVP-Data-From-Batch2.zip  ->  BVP-Data/<date>/<user>/<clip>_bvp.npz

Unpacked side-by-side under ``data/`` so that paths in ``bvp_full.csv``
resolve directly with ``os.path.join(out_dir, row['file'])``.

Usage:
    python extract_data.py
    python extract_data.py --out-dir data --force
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


# zip filename -> expected top-level directory inside the archive
ZIPS = {
    "BVP-Data-From_Batch1.zip": "BVP-Data-Batch1",
    "BVP-Data-From-Batch2.zip": "BVP-Data",
}


def extract_one(zip_path: Path, out_root: Path, expected_dir: str, force: bool) -> int:
    target = out_root / expected_dir
    if target.exists() and not force:
        n = sum(1 for _ in target.rglob("*_bvp.npz"))
        print(f"  skip {zip_path.name}: {target} already present ({n} files) — use --force to re-extract")
        return n

    print(f"  extract {zip_path.name} -> {out_root}/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_root)
    n = sum(1 for _ in target.rglob("*_bvp.npz"))
    print(f"    {n} _bvp.npz files under {target}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip-dir", default=".", help="Directory containing the zip archives (default: cwd)")
    ap.add_argument("--out-dir", default="data", help="Extraction root (default: data)")
    ap.add_argument("--force", action="store_true", help="Re-extract even if target directory already exists")
    args = ap.parse_args()

    zip_dir = Path(args.zip_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    missing = [name for name in ZIPS if not (zip_dir / name).exists()]
    if missing:
        sys.exit(f"missing archive(s) in {zip_dir}/: {', '.join(missing)}")

    total = 0
    for name, expected_dir in ZIPS.items():
        total += extract_one(zip_dir / name, out_root, expected_dir, args.force)

    print(f"\nDone: {total} BVP files under {out_root}/")


if __name__ == "__main__":
    main()
