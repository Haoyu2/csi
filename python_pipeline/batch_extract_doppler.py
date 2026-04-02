#!/usr/bin/env python
"""
Batch extraction of Doppler Spectrograms from Widar3.0 CSI data.

This script recursively searches a given CSI directory for raw CSI gesture clips, 
computes their Doppler spectrograms, and saves the results into a dedicated structured 
subfolder while strictly maintaining the original nested directory structure.
It strictly logs the processing time iteratively for performance tracking.
"""
import argparse
import logging
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from run_pipeline import compute_doppler_spectrum, DopplerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Batch process CSI data to natively extract Doppler Spectra.")
    parser.add_argument("--csi-dir", type=str, default="../CSI", help="Root directory containing raw Widar3.0 CSI data.")
    parser.add_argument("--out-dir", type=str, default="doppler_data", help="Output directory inside python_pipeline for Doppler data.")
    parser.add_argument("--rx-cnt", type=int, default=6, help="Number of Receivers (typically 6)")
    parser.add_argument("--rx-acnt", type=int, default=3, help="Number of Antennas (typically 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    
    csi_root = pathlib.Path(args.csi_dir).resolve()
    out_root = pathlib.Path(args.out_dir).resolve()
    
    if not csi_root.exists():
        print(f"CSI directory not found: {csi_root}")
        sys.exit(1)
        
    out_root.mkdir(parents=True, exist_ok=True)
    
    log_file_path = out_root / "doppler_processing_time_log.txt"
    
    prefixes = []
    print(f"Scanning '{csi_root}' for unzipped CSI clips...")
    for p in csi_root.rglob("*-r1.dat"):
        prefix_str = str(p)[:-7]
        prefixes.append(pathlib.Path(prefix_str))
        
    if not prefixes:
        print(f"No CSI clips (ending in *-r1.dat) were found in the folder '{csi_root}'.")
        print("Note: Ensure your CSI data matrices are fully unzipped before processing!")
        sys.exit(0)
        
    print(f"Successfully found {len(prefixes)} CSI clips. Setting up the extraction geometry...")
    config = DopplerConfig()
    
    successful = 0
    failed = 0
    
    global_start_time = time.time()
    
    with open(log_file_path, "a") as log_f:
        log_f.write(f"\n--- Batch Doppler Sequence Run Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        for prefix in tqdm(prefixes, desc="Extracting Doppler Spectrograms"):
            file_start_time = time.time()
            try:
                rel_path = prefix.relative_to(csi_root).parent
                clip_name = prefix.name
                
                target_dir = out_root / rel_path
                target_dir.mkdir(parents=True, exist_ok=True)
                
                out_file = target_dir / f"{clip_name}_doppler.npz"
                
                if out_file.exists():
                    successful += 1
                    file_end_time = time.time()
                    elapsed_file = file_end_time - file_start_time
                    cumulative_time = file_end_time - global_start_time
                    log_f.write(f"SKIPPED (Already exists): {rel_path} | File Time: [{elapsed_file:.2f}s] | Cumulative Runtime: [{cumulative_time:.2f}s]\n")
                    continue
                    
                doppler, freq_bin = compute_doppler_spectrum(prefix, rx_cnt=args.rx_cnt, rx_acnt=args.rx_acnt, cfg=config)
                
                np.savez_compressed(out_file, doppler=doppler, freq_bin=freq_bin)
                successful += 1
                
                file_end_time = time.time()
                elapsed_file = file_end_time - file_start_time
                cumulative_time = file_end_time - global_start_time
                
                log_f.write(f"SUCCESS: {rel_path} -> {out_file.name} | File Time: [{elapsed_file:.2f}s] | Cumulative Runtime: [{cumulative_time:.2f}s]\n")
                
            except Exception as e:
                file_end_time = time.time()
                elapsed_file = file_end_time - file_start_time
                cumulative_time = file_end_time - global_start_time
                logging.error(f"Failed to systematically process {prefix}: {e}")
                log_f.write(f"FAILED:  {prefix.relative_to(csi_root)} - Critical Error encountered: {e} | File Time: [{elapsed_file:.2f}s] | Cumulative Runtime: [{cumulative_time:.2f}s]\n")
                failed += 1
                
        total_batch_time = time.time() - global_start_time
        log_f.write(f"--- Doppler Sequences Cleanly Terminated! Total Success: {successful}, Failed: {failed} | Total Batch Execution Time: [{total_batch_time:.2f}s] ---\n")
            
    print(f"\nProcessing completely finished!")
    print(f"Successfully processed and stored: {successful} clips")
    print(f"Processing Failures: {failed}")
    print(f"Structural Doppler logs securely saved to: {log_file_path}")

if __name__ == "__main__":
    main()
