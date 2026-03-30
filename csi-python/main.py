"""
Intel 5300 CSI Reader & Explorer
=================================
Reads .dat files produced by the Linux CSI Tool for the Intel Wi-Fi Link 5300.

The Linux CSI Tool (by Daniel Halperin, University of Washington) captures
Channel State Information (CSI) from the Intel 5300 NIC.  The log_to_file
utility writes a binary log whose format is:

  For each record:
    2 bytes : field length  (big-endian uint16)  — includes the code byte
    1 byte  : code          (0xBB = 187 for CSI / beamforming records)
    N bytes : payload       (N = field_length - 1)

  Payload layout for code == 0xBB (from read_bfee.c):
    Byte   Size  Field
    -----  ----  -----
    0-3    4     timestamp_low   (uint32 LE, lower 32 bits of 1 MHz NIC clock)
    4-5    2     bfee_count      (uint16 LE, running beamforming event counter)
    6-7    2     (reserved)
    8      1     Nrx             (number of receive antennas, 1-3)
    9      1     Ntx             (number of transmit antennas / spatial streams)
    10     1     rssi_a          (RSSI on antenna A, unsigned)
    11     1     rssi_b          (RSSI on antenna B)
    12     1     rssi_c          (RSSI on antenna C)
    13     1     noise           (noise floor, signed dBm)
    14     1     agc             (automatic gain control)
    15     1     antenna_sel     (antenna permutation encoding)
    16-17  2     len             (uint16 LE, byte length of CSI payload)
    18-19  2     fake_rate_n_flags (uint16 LE, MCS rate + flags)
    20+    var   CSI payload     (compressed beamforming feedback matrix)

Reference:
  https://dhalperi.github.io/linux-80211n-csitool/
  https://github.com/dhalperi/linux-80211n-csitool-supplementary
"""

import struct
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The code byte that marks a CSI record in the binary log
BFEE_CODE = 0xBB          # 187 decimal

# Number of OFDM subcarrier groups reported by the Intel 5300
NUM_SUBCARRIERS = 30

# Subcarrier index mapping — the 30 subcarrier groups that the Intel 5300
# reports, out of 56 possible subcarriers in a 20 MHz 802.11n channel.
SUBCARRIER_INDICES = np.array([
    -28, -26, -24, -22, -20, -18, -16, -14, -12, -10,
     -8,  -6,  -4,  -2,  -1,   1,   3,   5,   7,   9,
     11,  13,  15,  17,  19,  21,  23,  25,  27,  28
])

# For verifying the antenna permutation: perm[0]+perm[1]+...+perm[Nrx-1]
# should equal triangle[Nrx-1] when using 1-based indices.
TRIANGLE = [1, 3, 6]  # sum of 1..1, 1..2, 1..3


# ---------------------------------------------------------------------------
# Core parsing functions
# ---------------------------------------------------------------------------


def read_bfee(payload: bytes) -> dict:
    """
    Parse one CSI record payload.

    This is a faithful Python translation of read_bfee.c from the Linux
    CSI Tool supplementary code by Daniel Halperin.

    Parameters
    ----------
    payload : bytes
        Raw bytes of the record, starting right after the 1-byte code.
        (i.e., field_len - 1 bytes.)

    Returns
    -------
    dict with fields:
        timestamp_low : int       — lower 32 bits of the NIC's 1 MHz clock
        bfee_count    : int       — running beamforming event counter
        Nrx           : int       — number of receive antennas (1-3)
        Ntx           : int       — number of transmit antennas (1-3)
        rssi_a        : int       — RSSI on antenna A (unsigned)
        rssi_b        : int       — RSSI on antenna B
        rssi_c        : int       — RSSI on antenna C
        noise         : int       — noise floor (signed dBm)
        agc           : int       — automatic gain control setting
        antenna_sel   : int       — raw antenna selection byte
        perm          : list[int] — antenna permutation, 1-based (length 3)
        rate          : int       — MCS rate index
        csi           : np.ndarray — complex CSI, shape (Ntx, Nrx, 30)
    """
    # ---------- Fixed-size header fields (matches read_bfee.c) --------------
    # timestamp_low: bytes 0-3, little-endian unsigned 32-bit
    timestamp_low = (payload[0]
                     | (payload[1] << 8)
                     | (payload[2] << 16)
                     | (payload[3] << 24))

    # bfee_count: bytes 4-5, little-endian unsigned 16-bit
    bfee_count = payload[4] | (payload[5] << 8)

    # bytes 6-7 are reserved / unused

    Nrx  = payload[8]       # number of RX antennas
    Ntx  = payload[9]       # number of TX spatial streams

    rssi_a = payload[10]    # per-antenna RSSI (unsigned)
    rssi_b = payload[11]
    rssi_c = payload[12]

    # noise is a signed byte (int8)
    noise = struct.unpack_from("b", payload, 13)[0]

    agc          = payload[14]
    antenna_sel  = payload[15]

    # len: bytes 16-17, little-endian unsigned 16-bit — length of CSI data
    csi_len = payload[16] | (payload[17] << 8)

    # fake_rate_n_flags: bytes 18-19
    fake_rate_n_flags = payload[18] | (payload[19] << 8)

    # ---------- Validate CSI length ----------------------------------------
    # Expected length: (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8
    # This formula accounts for 30 subcarriers, each with Nrx*Ntx complex
    # coefficients (8 bits real + 8 bits imaginary) plus 3 preamble bits
    # per subcarrier, all packed into bytes.
    calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) // 8
    if csi_len != calc_len:
        raise ValueError(
            f"CSI length mismatch: got {csi_len}, expected {calc_len} "
            f"for {Ntx}Tx×{Nrx}Rx"
        )

    # ---------- Antenna permutation ----------------------------------------
    # Each 2-bit field of antenna_sel encodes which physical antenna maps
    # to each logical antenna index.  The reference code uses 1-based indices.
    perm = [
        ((antenna_sel     ) & 0x3) + 1,
        ((antenna_sel >> 2) & 0x3) + 1,
        ((antenna_sel >> 4) & 0x3) + 1,
    ]

    # ---------- Unpack the compressed CSI matrix ---------------------------
    csi = _unpack_csi_matrix(payload[20:20 + csi_len], Ntx, Nrx)

    return {
        "timestamp_low": timestamp_low,
        "bfee_count":    bfee_count,
        "Nrx":           Nrx,
        "Ntx":           Ntx,
        "rssi_a":        rssi_a,
        "rssi_b":        rssi_b,
        "rssi_c":        rssi_c,
        "noise":         noise,
        "agc":           agc,
        "antenna_sel":   antenna_sel,
        "perm":          perm,
        "rate":          fake_rate_n_flags,
        "csi":           csi,
    }


def _unpack_csi_matrix(payload: bytes, Ntx: int, Nrx: int) -> np.ndarray:
    """
    Unpack the compressed beamforming feedback matrix into a complex
    numpy array of shape (Ntx, Nrx, 30).

    This is a direct translation of the inner loop of read_bfee.c.

    The data is stored as a bit-stream.  A running bit-index (`index`)
    advances through the payload:
      - At the start of each subcarrier, `index` advances by 3 (preamble bits).
      - For each of the Nrx*Ntx antenna pairs, an 8-bit signed real part and
        an 8-bit signed imaginary part are extracted (total 16 bits per pair),
        so `index` advances by 16 per pair.

    Extraction uses byte-aligned reads with bit-shifting:
        remainder = index % 8
        value = (payload[index//8] >> remainder) | (payload[index//8 + 1] << (8 - remainder))
    The result is then interpreted as a signed 8-bit integer (char).

    Parameters
    ----------
    payload : bytes
        The CSI data portion of the record (starts at byte 20 of the full
        record payload).
    Ntx     : int
        Number of transmit antennas / spatial streams.
    Nrx     : int
        Number of receive antennas.

    Returns
    -------
    np.ndarray of complex128, shape (Ntx, Nrx, 30)
    """
    # Output array — column-major order matching the C code:
    # C code fills ptrR/ptrI sequentially, which in MATLAB's column-major
    # order means: Ntx varies fastest, then Nrx, then subcarrier.
    # We'll fill a flat array and reshape.
    n_elements = Ntx * Nrx * NUM_SUBCARRIERS
    real_parts = np.zeros(n_elements, dtype=np.float64)
    imag_parts = np.zeros(n_elements, dtype=np.float64)

    index = 0    # bit-level cursor into the payload
    flat_idx = 0 # sequential index into output arrays

    for sc in range(NUM_SUBCARRIERS):
        # 3 preamble bits per subcarrier (skip them)
        index += 3
        remainder = index % 8

        for _ in range(Nrx * Ntx):
            # Extract 8-bit signed real part
            byte_idx = index // 8
            # Combine two adjacent bytes and shift to extract the 8-bit value
            raw_real = ((payload[byte_idx] >> remainder)
                        | (payload[byte_idx + 1] << (8 - remainder))) & 0xFF
            # Interpret as signed int8
            if raw_real >= 128:
                raw_real -= 256

            # Extract 8-bit signed imaginary part (next byte pair)
            raw_imag = ((payload[byte_idx + 1] >> remainder)
                        | (payload[byte_idx + 2] << (8 - remainder))) & 0xFF
            if raw_imag >= 128:
                raw_imag -= 256

            real_parts[flat_idx] = raw_real
            imag_parts[flat_idx] = raw_imag
            flat_idx += 1

            index += 16  # advance past this 16-bit complex coefficient

    # Reshape to (Ntx, Nrx, 30) — column-major (Fortran) order to match
    # the C code's sequential filling of the MATLAB array
    csi = (real_parts + 1j * imag_parts).reshape(
        (Ntx, Nrx, NUM_SUBCARRIERS), order='F'
    )

    return csi


# ---------------------------------------------------------------------------
# File reader
# ---------------------------------------------------------------------------


def read_bf_file(filepath: str) -> list[dict]:
    """
    Read an entire .dat file produced by the Linux CSI Tool and return
    a list of parsed CSI records.

    This is a Python translation of read_bf_file.m.

    Parameters
    ----------
    filepath : str
        Path to the .dat binary log file.

    Returns
    -------
    list of dict
        Each element is one CSI measurement (see `read_bfee` for the dict
        layout).  Non-CSI records (code != 0xBB) are silently skipped.
    """
    records = []
    filesize = os.path.getsize(filepath)
    broken_perm_warned = False

    with open(filepath, "rb") as f:
        cur = 0  # current byte offset

        while cur < (filesize - 3):
            # --- Read the 2-byte big-endian field length ---
            field_len_raw = f.read(2)
            if len(field_len_raw) < 2:
                break
            field_len = struct.unpack(">H", field_len_raw)[0]

            # --- Read the 1-byte code ---
            code = f.read(1)
            if len(code) < 1:
                break
            code = code[0]
            cur += 3

            remaining = field_len - 1  # payload size (code byte counted in field_len)

            if code == BFEE_CODE:
                # Read the full payload for CSI records
                payload = f.read(remaining)
                cur += remaining
                if len(payload) != remaining:
                    break  # truncated file

                try:
                    entry = read_bfee(payload)
                except Exception as e:
                    print(f"[WARN] Skipping malformed CSI record at byte "
                          f"{cur}: {e}")
                    continue

                # --- Apply antenna permutation (from read_bf_file.m) --------
                # Re-order the RX dimension of the CSI matrix so that the
                # columns correspond to the physical antennas in order.
                Nrx = entry["Nrx"]
                perm = entry["perm"]
                if Nrx > 1:
                    if sum(perm[:Nrx]) != TRIANGLE[Nrx - 1]:
                        if not broken_perm_warned:
                            broken_perm_warned = True
                            print(f"[WARN] Found CSI with Nrx={Nrx} and "
                                  f"invalid perm={perm[:Nrx]}")
                    else:
                        # Apply permutation: csi[:, perm[0:Nrx], :] = csi[:, 0:Nrx, :]
                        # perm is 1-based, so convert to 0-based indices
                        perm_0 = [p - 1 for p in perm[:Nrx]]
                        entry["csi"][:, perm_0, :] = entry["csi"][:, :Nrx, :]

                records.append(entry)

            else:
                # Skip non-CSI records
                f.seek(remaining, 1)  # seek forward from current position
                cur += remaining

    return records


# ---------------------------------------------------------------------------
# Exploration / analysis helpers
# ---------------------------------------------------------------------------


def compute_csi_amplitude(csi_matrix: np.ndarray) -> np.ndarray:
    """Return the amplitude (magnitude) of each CSI coefficient."""
    return np.abs(csi_matrix)


def compute_csi_phase(csi_matrix: np.ndarray) -> np.ndarray:
    """Return the phase (in radians) of each CSI coefficient."""
    return np.angle(csi_matrix)


def get_rssi_dbm(entry: dict) -> list[int]:
    """Return a list of per-antenna RSSI values for a single CSI record."""
    return [entry["rssi_a"], entry["rssi_b"], entry["rssi_c"]][:entry["Nrx"]]


def summary(records: list[dict]) -> None:
    """Print a quick summary of the parsed CSI dataset."""
    if not records:
        print("No records found.")
        return

    print(f"Total CSI records : {len(records)}")
    print(f"First timestamp   : {records[0]['timestamp_low']}")
    print(f"Last  timestamp   : {records[-1]['timestamp_low']}")

    # Antenna configurations seen
    configs = set()
    for r in records:
        configs.add((r["Ntx"], r["Nrx"]))
    print(f"Antenna configs   : "
          f"{', '.join(f'{t}Tx×{r}Rx' for t, r in sorted(configs))}")

    # Noise floor range
    noises = [r["noise"] for r in records]
    print(f"Noise floor       : min={min(noises)} dBm, max={max(noises)} dBm")

    # AGC range
    agcs = [r["agc"] for r in records]
    print(f"AGC               : min={min(agcs)}, max={max(agcs)}")

    # Average CSI amplitude across all records (first TX-RX pair, all 30 subcarriers)
    amps = np.array([np.abs(r["csi"][0, 0, :]) for r in records])
    mean_amp = amps.mean(axis=0)
    print(f"\nMean |CSI| (TX0-RX0) per subcarrier group:")
    for i, (sc_idx, amp) in enumerate(zip(SUBCARRIER_INDICES, mean_amp)):
        print(f"  subcarrier {sc_idx:+3d} (group {i:2d}): {amp:.2f}")


def plot_csi(records: list[dict],
             tx: int = 0,
             rx: int = 0,
             max_packets: int = 200) -> None:
    """
    Plot CSI amplitude and phase heatmaps for a specific TX-RX antenna pair
    over time (packet index).

    Parameters
    ----------
    records     : list of parsed CSI dicts
    tx, rx      : which transmit/receive antenna pair to plot (0-indexed)
    max_packets : limit the number of packets to plot for clarity
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting.  Install it with:")
        print("  pip install matplotlib")
        return

    subset = records[:max_packets]
    n = len(subset)

    # Build amplitude and phase matrices: (n_packets, 30 subcarriers)
    amp_matrix   = np.zeros((n, NUM_SUBCARRIERS))
    phase_matrix = np.zeros((n, NUM_SUBCARRIERS))

    for i, rec in enumerate(subset):
        csi_slice = rec["csi"][tx, rx, :]   # shape (30,)
        amp_matrix[i, :]   = np.abs(csi_slice)
        phase_matrix[i, :] = np.angle(csi_slice)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Amplitude heatmap ---
    im0 = axes[0].imshow(amp_matrix.T, aspect="auto", origin="lower",
                          cmap="viridis",
                          extent=[0, n, 0, NUM_SUBCARRIERS])
    axes[0].set_ylabel("Subcarrier group index")
    axes[0].set_title(f"CSI Amplitude  (TX {tx}, RX {rx})")
    fig.colorbar(im0, ax=axes[0], label="|H|")

    # --- Phase heatmap ---
    im1 = axes[1].imshow(phase_matrix.T, aspect="auto", origin="lower",
                          cmap="twilight",
                          extent=[0, n, 0, NUM_SUBCARRIERS])
    axes[1].set_xlabel("Packet index")
    axes[1].set_ylabel("Subcarrier group index")
    axes[1].set_title(f"CSI Phase  (TX {tx}, RX {rx})")
    fig.colorbar(im1, ax=axes[1], label="∠H (rad)")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "csi_heatmap.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved heatmap plot to {out_path}")
    plt.show()


def plot_amplitude_over_time(records: list[dict],
                              tx: int = 0,
                              rx: int = 0,
                              subcarrier: int = 15,
                              max_packets: int = 500) -> None:
    """
    Plot the CSI amplitude of a single subcarrier over time (packet index).
    Useful for observing temporal fading patterns.

    Parameters
    ----------
    subcarrier : int
        Index into the 30 reported subcarrier groups (0-29).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting.  pip install matplotlib")
        return

    subset = records[:max_packets]
    amps = [np.abs(r["csi"][tx, rx, subcarrier]) for r in subset]

    plt.figure(figsize=(12, 4))
    plt.plot(amps, linewidth=0.8)
    plt.xlabel("Packet index")
    plt.ylabel("|H|")
    plt.title(f"CSI Amplitude over time — Subcarrier group {subcarrier} "
              f"(index {SUBCARRIER_INDICES[subcarrier]:+d})  "
              f"TX {tx}, RX {rx}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "csi_amplitude_time.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved amplitude plot to {out_path}")
    plt.show()


def plot_subcarrier_snapshot(records: list[dict],
                              packet_idx: int = 0,
                              tx: int = 0,
                              rx: int = 0) -> None:
    """
    Plot the CSI amplitude and phase across all 30 subcarrier groups for
    a single packet.  This shows the frequency-domain channel response
    at one instant in time.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required.  pip install matplotlib")
        return

    rec = records[packet_idx]
    csi_slice = rec["csi"][tx, rx, :]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.stem(SUBCARRIER_INDICES, np.abs(csi_slice), basefmt=" ")
    ax1.set_ylabel("|H|")
    ax1.set_title(f"Channel Frequency Response — Packet {packet_idx} "
                  f"(TX {tx}, RX {rx})")
    ax1.grid(True, alpha=0.3)

    ax2.stem(SUBCARRIER_INDICES, np.angle(csi_slice), basefmt=" ")
    ax2.set_xlabel("Subcarrier index")
    ax2.set_ylabel("∠H (rad)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "csi_snapshot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved snapshot plot to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main entry point — run this script directly to explore your .dat file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default to the .dat file in the same directory
    DEFAULT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "user1-1-1-1-1-r1.dat"
    )

    dat_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

    if not os.path.isfile(dat_file):
        print(f"Error: file not found: {dat_file}")
        sys.exit(1)

    print(f"Reading CSI data from: {dat_file}")
    print("=" * 60)

    # ---- 1. Parse all records ----
    csi_records = read_bf_file(dat_file)

    # ---- 2. Print a summary of the dataset ----
    summary(csi_records)

    # ---- 3. Show details of the first record as an example ----
    if csi_records:
        print("\n" + "=" * 60)
        print("First CSI record details:")
        first = csi_records[0]
        print(f"  Timestamp      : {first['timestamp_low']}")
        print(f"  BFEE count     : {first['bfee_count']}")
        print(f"  Config         : {first['Ntx']}Tx × {first['Nrx']}Rx")
        print(f"  RSSI (A/B/C)   : {first['rssi_a']}, {first['rssi_b']}, "
              f"{first['rssi_c']}")
        print(f"  Noise          : {first['noise']} dBm")
        print(f"  AGC            : {first['agc']}")
        print(f"  Ant. perm      : {first['perm']}")
        print(f"  CSI shape      : {first['csi'].shape}")
        print(f"  CSI[0,0,:5]    : {first['csi'][0, 0, :5]}")
        print(f"  |CSI[0,0,:5]|  : {np.abs(first['csi'][0, 0, :5])}")

    # ---- 4. Generate plots (requires matplotlib) ----
    print("\n" + "=" * 60)
    print("Generating plots...")
    plot_csi(csi_records)
    plot_amplitude_over_time(csi_records)
    if csi_records:
        plot_subcarrier_snapshot(csi_records, packet_idx=0)
