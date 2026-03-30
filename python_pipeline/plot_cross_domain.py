#!/usr/bin/env python3
"""
plot_cross_domain.py

Recreates the cross-domain comparison figures shown in the Widar3.0 paper and website.
By inputting two different files (from Domain 1 vs Domain 2), this script plots
them side-by-side to visually demonstrate how BVP remains domain-independent while
DFS (Doppler Spectrogram) is highly sensitive to domain changes.

Usage:
  # Compare two BVP snapshots from different domains
  python plot_cross_domain.py --bvp1 domain1_bvp.mat --bvp2 domain2_bvp.mat
  
  # Compare two DFS spectrograms from different domains
  python plot_cross_domain.py --dfs1 domain1_doppler.npz --dfs2 domain2_doppler.npz
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio


def load_bvp(path):
    if path.endswith('.npz'):
        data = np.load(path)
        return data['velocity_spectrum_ro']
    elif path.endswith('.mat'):
        data = sio.loadmat(path)
        return data['velocity_spectrum_ro']
    else:
        raise ValueError(f"Unsupported format: {path}")


def load_dfs(path):
    data = np.load(path)
    return data['doppler'], data['freq_bin']


def plot_side_by_side_bvp(bvp1_path, bvp2_path, time_segment=None):
    """
    Plots two BVP snapshots side-by-side to show domain-independence.
    """
    bvp1 = load_bvp(bvp1_path)
    bvp2 = load_bvp(bvp2_path)

    def get_snapshot(bvp, segment):
        # Default to segment with max energy
        if segment is None:
            energy_per_segment = np.sum(bvp, axis=(0, 1))
            segment = int(np.argmax(energy_per_segment))
        return bvp[:, :, segment], segment

    snap1, seg1 = get_snapshot(bvp1, time_segment)
    snap2, seg2 = get_snapshot(bvp2, time_segment)

    v_min, v_max = -2.0, 2.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cross-Domain Comparison: Body Velocity Profile (BVP)', fontsize=16)

    # Subplot 1 (Domain 1)
    im1 = axes[0].imshow(snap1, origin='lower', extent=[v_min, v_max, v_min, v_max], cmap='jet')
    axes[0].set_title(f"Domain 1\n{os.path.basename(bvp1_path)}")
    axes[0].set_xlabel("Velocity X (m/s)")
    axes[0].set_ylabel("Velocity Y (m/s)")
    axes[0].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

    # Subplot 2 (Domain 2)
    im2 = axes[1].imshow(snap2, origin='lower', extent=[v_min, v_max, v_min, v_max], cmap='jet')
    axes[1].set_title(f"Domain 2\n{os.path.basename(bvp2_path)}")
    axes[1].set_xlabel("Velocity X (m/s)")
    axes[1].set_ylabel("Velocity Y (m/s)")
    axes[1].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Motion Energy P(V)')

    plt.show()


def plot_animated_bvp(bvp1_path, bvp2_path):
    """
    Plots two BVP sequences as an animation over time segments.
    """
    bvp1 = load_bvp(bvp1_path)
    bvp2 = load_bvp(bvp2_path)

    v_min, v_max = -2.0, 2.0
    num_frames = min(bvp1.shape[2], bvp2.shape[2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cross-Domain Comparison: Body Velocity Profile (BVP) Animation', fontsize=16)

    # Use a fixed vmax based on individual max energy to prevent flickering
    vmax1 = np.max(bvp1) * 0.8
    vmax2 = np.max(bvp2) * 0.8

    im1 = axes[0].imshow(bvp1[:, :, 0], origin='lower', extent=[v_min, v_max, v_min, v_max], cmap='jet', vmin=0, vmax=vmax1)
    axes[0].set_title(f"Domain 1\n{os.path.basename(bvp1_path)}")
    axes[0].set_xlabel("Velocity X (m/s)")
    axes[0].set_ylabel("Velocity Y (m/s)")
    axes[0].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

    im2 = axes[1].imshow(bvp2[:, :, 0], origin='lower', extent=[v_min, v_max, v_min, v_max], cmap='jet', vmin=0, vmax=vmax2)
    axes[1].set_title(f"Domain 2\n{os.path.basename(bvp2_path)}")
    axes[1].set_xlabel("Velocity X (m/s)")
    axes[1].set_ylabel("Velocity Y (m/s)")
    axes[1].axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Motion Energy P(V)')
    
    time_text = fig.text(0.5, 0.04, '', ha='center', fontsize=12)

    def update(frame):
        im1.set_data(bvp1[:, :, frame])
        im2.set_data(bvp2[:, :, frame])
        time_text.set_text(f'Time Segment: {frame}/{num_frames-1}')
        return im1, im2, time_text

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True, repeat=True)
    
    # Keeping a reference to prevent garbage collection just in case
    fig._ani = ani
    
    print("[*] Saving animation to 'bvp_animation.gif' in the current directory...")
    ani.save('bvp_animation.gif', writer='pillow')
    print("[*] Animation saved successfully.")
    
    plt.show()


def plot_side_by_side_dfs(dfs1_path, dfs2_path, rx=0):
    """
    Plots two Doppler spectrograms side-by-side to demonstrate domain dependence.
    """
    doppler1, freq_bin1 = load_dfs(dfs1_path)
    doppler2, freq_bin2 = load_dfs(dfs2_path)

    spec1 = doppler1[rx]
    spec2 = doppler2[rx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cross-Domain Comparison: Doppler Frequency Shift (DFS)', fontsize=16)

    # Subplot 1 (Domain 1)
    im1 = axes[0].pcolormesh(np.arange(spec1.shape[1]), freq_bin1, spec1, shading='auto', cmap='jet')
    axes[0].set_title(f"Domain 1 - RX {rx+1}\n{os.path.basename(dfs1_path)}")
    axes[0].set_xlabel("Time Segment")
    axes[0].set_ylabel("Doppler Frequency (Hz)")

    # Subplot 2 (Domain 2)
    im2 = axes[1].pcolormesh(np.arange(spec2.shape[1]), freq_bin2, spec2, shading='auto', cmap='jet')
    axes[1].set_title(f"Domain 2 - RX {rx+1}\n{os.path.basename(dfs2_path)}")
    axes[1].set_xlabel("Time Segment")
    axes[1].set_ylabel("Doppler Frequency (Hz)")

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Normalized Signal Power')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Widar3.0 Cross-Domain Figure Generator")
    parser.add_argument("--bvp1", type=str, help="First BVP file (Domain 1)")
    parser.add_argument("--bvp2", type=str, help="Second BVP file (Domain 2)")
    parser.add_argument("--dfs1", type=str, help="First DFS file (Domain 1)")
    parser.add_argument("--dfs2", type=str, help="Second DFS file (Domain 2)")
    parser.add_argument("--rx", type=int, default=0, help="Receiver index for DFS comparison (default: 0)")
    parser.add_argument("--bvp-frame", type=int, default=None, help="Specific frame index for BVP (defaults to max energy frame)")
    parser.add_argument("--animate-bvp", action="store_true", default=True,  help="Play the BVP sequences as an animation over time")
    
    args = parser.parse_args()

    # If no files provided, run in demo mode with default BVP files
    if args.bvp1 is None and args.bvp2 is None and args.dfs1 is None and args.dfs2 is None:
        print("[!] No arguments provided. Running in demo mode with default BVP files...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        args.bvp1 = os.path.join(base_dir, "BVP", "BVP", "BVP", "20181117-VS", "6-link", "user4", "user4-4-2-4-3-1-1e-07-100-20-100000-L0.mat")
        args.bvp2 = os.path.join(base_dir, "BVP", "BVP", "BVP", "20181127-VS", "6-link", "user2", "user2-1-3-5-5-1-1e-07-100-20-100000-L0.mat")
        
        # Check if default files exist
        if not os.path.exists(args.bvp1) or not os.path.exists(args.bvp2):
            print(f"[!] Default BVP files not found. Please provide valid file paths via arguments.")
            sys.exit(1)

    if args.bvp1 and args.bvp2:
        if args.animate_bvp:
            print(f"[*] Generating Cross-Domain BVP Animation for:")
            print(f"    1) {args.bvp1}")
            print(f"    2) {args.bvp2}")
            plot_animated_bvp(args.bvp1, args.bvp2)
        else:
            print(f"[*] Generating Cross-Domain BVP Figure for:")
            print(f"    1) {args.bvp1}")
            print(f"    2) {args.bvp2}")
            plot_side_by_side_bvp(args.bvp1, args.bvp2, time_segment=args.bvp_frame)
    elif args.dfs1 and args.dfs2:
        print(f"[*] Generating Cross-Domain DFS Figure for:")
        print(f"    1) {args.dfs1}")
        print(f"    2) {args.dfs2}")
        plot_side_by_side_dfs(args.dfs1, args.dfs2, rx=args.rx)
    else:
        parser.print_help()
        print("\n[!] Please provide exactly two files of the same type to compare.")
        print("    Example: --bvp1 file1.mat --bvp2 file2.mat")
        sys.exit(1)


if __name__ == "__main__":
    main()
