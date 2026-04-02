#!/usr/bin/env python
"""
Batch extraction of BAP matrices from precomputed Doppler Spectrograms.

This script recursively searches a directory for `*_doppler.npz` files, computes their 
Body Acceleration Profile (BAP) maps natively using Method B, and saves the results 
directly into a structured output folder. It features enhanced statistical evaluation 
for recording cumulative execution time combined with discrete file computation speeds.
"""

import argparse
import logging
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from run_pipeline_bap import doppler_to_bvp_bap, MappingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Batch process Doppler data directly into BAP matrices.")
    parser.add_argument("--doppler-dir", type=str, default="doppler_data", help="Root directory containing *_doppler.npz files.")
    parser.add_argument("--out-dir", type=str, default="bap_data", help="Structured output directory specifically targeting BAP tensors.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    
    in_root = pathlib.Path(args.doppler_dir).resolve()
    out_root = pathlib.Path(args.out_dir).resolve()
    
    if not in_root.exists():
        print(f"Input directory not found comprehensively: {in_root}")
        sys.exit(1)
        
    out_root.mkdir(parents=True, exist_ok=True)
    
    log_file_path = out_root / "bap_processing_time_log.txt"
    
    doppler_files = list(in_root.rglob("*_doppler.npz"))
    
    if not doppler_files:
        print(f"No Doppler sequence files (*_doppler.npz) found cleanly inside {in_root}.")
        sys.exit(0)
        
    print(f"Found {len(doppler_files)} structured Doppler sequence files. Starting Method B optimization execution...")
    map_cfg = MappingConfig()
    
    successful = 0
    failed = 0
    
    # Store global execution timestamp to compute total cumulative runtime
    global_start_time = time.time()
    
    with open(log_file_path, "a") as log_f:
        log_f.write(f"\n--- BAP Native Batch Execution Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        for p in tqdm(doppler_files, desc="Batch Extracting Native BAP Matrices"):
            file_start_time = time.time()
            try:
                rel_path = p.relative_to(in_root)
                clip_name = p.stem.replace("_doppler", "")
                
                target_dir = out_root / rel_path.parent
                target_dir.mkdir(parents=True, exist_ok=True)
                
                out_file = target_dir / f"{clip_name}_bap.npz"
                
                if out_file.exists():
                    successful += 1
                    file_end_time = time.time()
                    elapsed_file = file_end_time - file_start_time
                    cumulative_time = file_end_time - global_start_time
                    log_f.write(f"SKIPPED (Already exists): {rel_path} | File Time: [{elapsed_file:.2f}s] | Cumulative Runtime: [{cumulative_time:.2f}s]\n")
                    continue
                
                data = np.load(p)
                doppler_spectrum = data["doppler"]
                freq_bin = data["freq_bin"]
                
                parts = clip_name.split("-")
                rx_cnt = 6
                pos_sel = int(parts[2]) if len(parts) >= 3 else 1
                ori_sel = int(parts[3]) if len(parts) >= 4 else 1
                
                # Utilizing run_pipeline_bap.py to execute the double optimization via Method B
                # It correctly evaluates both the physical velocity constraint bounded by 0 and the differentiated Doppler bounds.
                bvp_res, bap_res = doppler_to_bvp_bap(doppler_spectrum, freq_bin, rx_cnt=rx_cnt, pos_sel=pos_sel, ori_sel=ori_sel, map_cfg=map_cfg)
                
                np.savez_compressed(out_file, acceleration_spectrum_ro=bap_res)
                successful += 1
                
                file_end_time = time.time()
                elapsed_file = file_end_time - file_start_time
                cumulative_time = file_end_time - global_start_time
                
                log_f.write(f"SUCCESS: {rel_path} -> {out_file.name} | File Time: [{elapsed_file:.2f}s] | Cumulative Runtime: [{cumulative_time:.2f}s]\n")
                
            except Exception as e:
                file_end_time = time.time()
                elapsed_file = file_end_time - file_start_time
                cumulative_time = file_end_time - global_start_time
                logging.error(f"Failed to cleanly process spatial optimization for {p}: {e}")
                log_f.write(f"FAILED:  {p.relative_to(in_root)} - Error encountered: {e} | File Time: [{elapsed_file:.2f}s] | Cumulative Runtime: [{cumulative_time:.2f}s]\n")
                failed += 1
                
        total_batch_time = time.time() - global_start_time
        log_f.write(f"--- BAP Execution Cleanly Terminated! Complete Successes: {successful}, Total Failures: {failed} | Total Batch Run Window: [{total_batch_time:.2f}s] ---\n")
            
    print(f"\nMatrix BAP mapping entirely finished operations!")
    print(f"Successfully tracked & processed: {successful} profile sequences")
    print(f"Execution Failures: {failed}")
    print(f"Total Cumulative time statistics natively securely written to: {log_file_path}")

if __name__ == "__main__":
    main()
