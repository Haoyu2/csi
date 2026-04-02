#!/usr/bin/env python
"""
Batch extraction of BVP matrices from precomputed Doppler Spectrograms.

This script recursively searches a directory for `*_doppler.npz` files, computes their 
Body-Velocity Profile (BVP) maps, and saves the results into a structured output folder.
It strictly logs the processing time iteratively for performance tracking and monitoring.
"""

import argparse
import logging
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from run_pipeline import doppler_to_bvp, MappingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Batch process Doppler data directly into spatial BVP matrices.")
    parser.add_argument("--doppler-dir", type=str, default="doppler_data", help="Root directory containing doppler.npz files.")
    parser.add_argument("--out-dir", type=str, default="bvp_data", help="Structured output directory for BVP matrices.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    
    in_root = pathlib.Path(args.doppler_dir).resolve()
    out_root = pathlib.Path(args.out_dir).resolve()
    
    if not in_root.exists():
        print(f"Input directory not found: {in_root}")
        sys.exit(1)
        
    out_root.mkdir(parents=True, exist_ok=True)
    
    log_file_path = out_root / "processing_time_log.txt"
    
    # Find all recursively processed doppler.npz files
    doppler_files = list(in_root.rglob("*_doppler.npz"))
    
    if not doppler_files:
        print(f"No valid Doppler files (*_doppler.npz) found cleanly in {in_root}.")
        sys.exit(0)
        
    print(f"Found {len(doppler_files)} clean Doppler sequence files. Starting intensive BVP optimization...")
    map_cfg = MappingConfig()
    
    successful = 0
    failed = 0
    
    with open(log_file_path, "a") as log_f:
        log_f.write(f"\n--- Batch Structural Run Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        for p in tqdm(doppler_files, desc="Batch Extracting BVP Matrices"):
            t_start = time.time()
            try:
                rel_path = p.relative_to(in_root)
                clip_name = p.stem.replace("_doppler", "")
                
                target_dir = out_root / rel_path.parent
                target_dir.mkdir(parents=True, exist_ok=True)
                
                out_file = target_dir / f"{clip_name}_bvp.npz"
                
                if out_file.exists():
                    successful += 1
                    t_end = time.time()
                    log_f.write(f"SKIPPED (File already exists): {rel_path} - Computation time bypassed: [{t_end - t_start:.2f}s]\n")
                    continue
                
                # Load the strictly tracked existing doppler spectrum
                data = np.load(p)
                doppler_spectrum = data["doppler"]
                freq_bin = data["freq_bin"]
                
                # Metadata must be dynamically decoded directly from the clip filename.
                # Widar Convention: <user>-<motion>-<position>-<orientation>-<repetition>
                parts = clip_name.split("-")
                rx_cnt = 6 # Set rigidly by Widar hardware configuration
                
                # Extract position and orientation for projection math and rotational mapping
                pos_sel = int(parts[2]) if len(parts) >= 3 else 1
                ori_sel = int(parts[3]) if len(parts) >= 4 else 1
                
                # Invoke the heavy SLSQP mapping math execution
                bvp = doppler_to_bvp(doppler_spectrum, freq_bin, rx_cnt=rx_cnt, pos_sel=pos_sel, ori_sel=ori_sel, map_cfg=map_cfg)
                
                # Save purely structural results
                np.savez_compressed(out_file, velocity_spectrum_ro=bvp)
                successful += 1
                
                t_end = time.time()
                elapsed = t_end - t_start
                log_f.write(f"SUCCESS:  {rel_path} processed into {out_file.name} | Execution Duration: [{elapsed:.2f}s]\n")
                
            except Exception as e:
                t_end = time.time()
                elapsed = t_end - t_start
                logging.error(f"Failed to cleanly optimize {p}: {e}")
                log_f.write(f"FAILED:   {p.relative_to(in_root)} - Critical Error encountered: {e} | Dumped after [{elapsed:.2f}s]\n")
                failed += 1
                
        # Final tally marker log
        log_f.write(f"--- BVP Run Cleanly Terminated! Total Success: {successful}, Total Failed: {failed} ---\n")
            
    print(f"\nMatrix Map Processing entirely finished operations!")
    print(f"Successfully tracked & processed: {successful} data files")
    print(f"Map Failures: {failed}")
    print(f"Strict time execution logs cleanly appended to: {log_file_path}")

if __name__ == "__main__":
    main()
