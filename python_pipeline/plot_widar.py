#!/usr/bin/env python3
"""
plot_widar.py

Visualization script for Widar3.0 processed data across three primary domains:
1. CSI (Channel State Information) in the Time Domain
2. DFS (Doppler Frequency Shift) in the Time-Frequency Domain
3. BVP (Body Velocity Profile) in the Spatial (Velocity) Domain

Run this script with:
  python plot_widar.py --csi path/to/csi.dat
  python plot_widar.py --doppler path/to/doppler.npz
  python plot_widar.py --bvp path/to/bvp.npz
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def plot_csi_amplitude(dat_path, subcarrier=0):
    """
    Plots the amplitude of the raw CSI stream over time.
    Domain: Time Domain.
    X-axis: Packet Index (Time)
    Y-axis: CSI Amplitude
    """
    print(f"[*] Visualizing CSI from: {dat_path}")
    try:
        import csiread
    except ImportError:
        print("[!] Error: 'csiread' package is required to read Intel 5300 .dat CSI files.")
        print("    Please install it inside your environment via: pip install csiread")
        return

    # Using csiread to parse Wi-Fi packets
    # Support older and newer API signatures just like run_pipeline.py
    try:
        reader = csiread.Intel(dat_path, nrx=3, ntx=1, if_report=False)
    except TypeError:
        reader = csiread.Intel(dat_path, nrxnum=3, ntxnum=1, if_report=False)

    reader.read()
    csi = reader.csi  # shape depends on API version, e.g., (frames, nsub, nrx)
    
    # Flatten the antenna/subcarrier dimensions to simplify extraction
    csi_flat = csi.reshape(csi.shape[0], -1) 
    
    if csi_flat.shape[1] == 0:
        print("[!] No packets found or invalid CSI format.")
        return

    # Select a single specific channel (antenna/subcarrier pair) to plot
    # Subcarrier 0 means the very first stream extracted.
    channel_idx = subcarrier % csi_flat.shape[1]
    
    # Compute amplitude (sqrt(real^2 + imag^2))
    amp = np.abs(csi_flat[:, channel_idx])

    plt.figure(figsize=(10, 4))
    plt.plot(amp, color='#1f77b4', linewidth=1.0)
    plt.title(f'CSI Amplitude - Time Domain\nFile: {os.path.basename(dat_path)}')
    plt.xlabel('Packet Index (Discrete Time)')
    plt.ylabel('Signal Amplitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_dfs_spectrogram(npz_path, rx_index=0):
    """
    Plots the Doppler Frequency Shift Spectrogram.
    Domain: Time-Frequency Domain.
    X-axis: Time Segments (STFT windows)
    Y-axis: Doppler Frequency (Hz)
    Color: Signal Power (Magnitude) at that frequency
    """
    print(f"[*] Visualizing DFS Spectrogram from: {npz_path}")
    try:
        data = np.load(npz_path)
        doppler = data['doppler']   # Shape: (rx_cnt, freq_bins, time_segments)
        freq_bin = data['freq_bin'] # Shape: (freq_bins,)
    except (KeyError, FileNotFoundError) as e:
        print(f"[!] Error loading DFS npz: {e}")
        return

    if rx_index >= doppler.shape[0]:
        print(f"[!] Warning: requested RX index {rx_index} exceeds available {doppler.shape[0]}. Falling back to Rx 0.")
        rx_index = 0

    # Extract the spectrogram for the chosen receiver link
    spec = doppler[rx_index]

    plt.figure(figsize=(10, 6))
    # pcolormesh maps the 2D grid
    plt.pcolormesh(np.arange(spec.shape[1]), freq_bin, spec, shading='auto', cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Normalized Signal Power')

    plt.title(f'Doppler Spectrogram (DFS) - RX {rx_index+1}\nFile: {os.path.basename(npz_path)}')
    plt.xlabel('Time Segment (STFT Window)')
    plt.ylabel('Doppler Frequency (Hz)')
    plt.tight_layout()
    plt.show()


def plot_bvp_snapshot(path, time_segment=None):
    """
    Plots the Body Velocity Profile (BVP).
    Domain: Spatial (Velocity) Domain over one snapshot.
    X-axis: Velocity in X direction (body coordinate)
    Y-axis: Velocity in Y direction (body coordinate)
    Color: Energy accumulated in each velocity bin.
    """
    print(f"[*] Visualizing BVP from: {path}")
    if path.endswith('.npz'):
        data = np.load(path)
        bvp = data['velocity_spectrum_ro'] # Shape: (Vx, Vy, TimeSegments)
    elif path.endswith('.mat'):
        data = sio.loadmat(path)
        bvp = data['velocity_spectrum_ro']
    else:
        print(f"[!] Unsupported format for BVP: {path}. Use .npz or .mat")
        return

    # Usually the grid resolves to M x M velocity boxes
    v_bins = bvp.shape[0]
    segments = bvp.shape[2]

    # If no specific time snapshot is given, we pick the one with the maximum motion energy
    if time_segment is None:
        energy_per_segment = np.sum(bvp, axis=(0, 1))
        time_segment = int(np.argmax(energy_per_segment))
    elif time_segment >= segments:
        print(f"[!] Target segment {time_segment} out of range ({segments}). Using last segment.")
        time_segment = segments - 1

    snapshot = bvp[:, :, time_segment]

    # Default speeds used by Widar3.0 MappingConfig are -2.0 to 2.0 m/s
    v_min, v_max = -2.0, 2.0

    plt.figure(figsize=(8, 7))
    plt.imshow(snapshot, origin='lower', extent=[v_min, v_max, v_min, v_max], cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Motion Energy P(V)')

    plt.title(f'Body Velocity Profile (BVP) Snapshot - Frame {time_segment}\nFile: {os.path.basename(path)}')
    plt.xlabel('Velocity X (m/s)')
    plt.ylabel('Velocity Y (m/s)')
    
    # Adding origin lines for clarity
    plt.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Widar3.0 Visualization Scripts")
    parser.add_argument("--csi", type=str, help="Path to raw CSI .dat file")
    parser.add_argument("--doppler", type=str, help="Path to DFS .npz file (e.g. *_doppler.npz)")
    parser.add_argument("--bvp", type=str, help="Path to BVP .npz or .mat file (e.g. *_bvp.npz)")
    parser.add_argument("--rx", type=int, default=0, help="Receiver index (0-5) to plot for DFS")
    parser.add_argument("--bvp-frame", type=int, default=None, help="Specific time segment to plot for BVP")
    args = parser.parse_args()

    # Determine if the user provided anything at all
    if not (args.csi or args.doppler or args.bvp):
        parser.print_help()
        print("\n[!] Please provide a file to visualize using --csi, --doppler, or --bvp")
        sys.exit(1)

    # Dispatch to the respective visualization methods
    if args.csi:
        plot_csi_amplitude(args.csi)

    if args.doppler:
        plot_dfs_spectrogram(args.doppler, rx_index=args.rx)

    if args.bvp:
        plot_bvp_snapshot(args.bvp, time_segment=args.bvp_frame)


if __name__ == "__main__":
    main()
