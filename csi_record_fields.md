# Intel 5300 CSI Record — Field Reference

The Linux CSI Tool captures **Channel State Information (CSI)** from the Intel Wi-Fi Link 5300 NIC. Each received 802.11n packet produces one CSI record containing metadata about the reception and the complex-valued channel frequency response.

This document explains every field in a parsed CSI record, then walks through a **real-world example** from the file `user1-1-1-1-1-r1.dat`.

---

## Field Descriptions

### `timestamp_low` — NIC Hardware Timestamp

| Property | Value |
|---|---|
| **Type** | `uint32` (unsigned 32-bit integer) |
| **Unit** | Microseconds (μs) — the NIC's 1 MHz clock |
| **Wraps** | Every 2³² μs ≈ 4295 seconds ≈ **71.6 minutes** |

The lower 32 bits of the Intel 5300's internal clock at the moment the packet was received. This is **not** wall-clock time — it's a monotonically increasing hardware counter that wraps around roughly every 72 minutes.

**Use cases:**
- Computing inter-packet timing (Δt between consecutive records)
- Detecting packet loss (gaps in timing)
- Synchronizing CSI with other sensor data on the same clock

---

### `bfee_count` — Beamforming Event Counter

| Property | Value |
|---|---|
| **Type** | `uint16` (unsigned 16-bit integer) |
| **Range** | 0 – 65535 (wraps around) |

A running counter incremented by the NIC driver for every beamforming measurement sent to userspace via the Netlink channel. If `bfee_count` increments by more than 1 between consecutive records, it means packets were **dropped** in the kernel-to-userspace path.

**Use cases:**
- Detecting dropped measurements: if record *i* has `bfee_count = 100` and record *i+1* has `bfee_count = 103`, then 2 packets were lost
- Evaluating the reliability of your measurement setup

---

### `Nrx` — Number of Receive Antennas

| Property | Value |
|---|---|
| **Type** | `uint8` |
| **Range** | 1, 2, or 3 |

The number of physical antennas the Intel 5300 used to receive this packet. The Intel 5300 has **3 antenna ports**, but the driver may report fewer if antennas are disabled or signal quality is low.

---

### `Ntx` — Number of Transmit Spatial Streams

| Property | Value |
|---|---|
| **Type** | `uint8` |
| **Range** | 1, 2, or 3 |

The number of spatial streams (antennas) the **transmitter** used when sending this packet. The Intel 5300 can resolve up to 3 streams. Combined with `Nrx`, this determines the MIMO configuration:

| Ntx × Nrx | MIMO Type | CSI Matrix Size |
|---|---|---|
| 1 × 1 | SISO | 1 × 1 × 30 |
| 1 × 3 | SIMO | 1 × 3 × 30 |
| 2 × 3 | 2×3 MIMO | 2 × 3 × 30 |
| 3 × 3 | Full 3×3 MIMO | 3 × 3 × 30 |

---

### `rssi_a`, `rssi_b`, `rssi_c` — Per-Antenna RSSI

| Property | Value |
|---|---|
| **Type** | `uint8` (unsigned) |
| **Unit** | Raw register value (not dBm directly) |

Received Signal Strength Indicator as read from each of the three antenna chains (A, B, C). These are **raw values** from the NIC's AGC circuitry. To convert to approximate dBm:

```
RSSI_dBm = rssi_X - 44 - agc
```

where `44` is a hardware-specific offset and `agc` is the automatic gain control setting (see below).

> [!NOTE]
> If `Nrx < 3`, only the first `Nrx` RSSI values are meaningful. The remaining values may be stale or garbage.

---

### `noise` — Noise Floor

| Property | Value |
|---|---|
| **Type** | `int8` (signed byte) |
| **Unit** | dBm |

The noise floor reported by the NIC at the moment of reception. Typical indoor values range from **-90 to -95 dBm**. A value of **-127 dBm** is a sentinel indicating the NIC did not provide a valid measurement (common on some firmware versions).

**Use cases:**
- Computing per-packet SNR: `SNR ≈ RSSI_dBm - noise`
- Characterizing the RF environment

---

### `agc` — Automatic Gain Control

| Property | Value |
|---|---|
| **Type** | `uint8` |
| **Unit** | dB (gain applied by the receiver) |

The gain the NIC's analog front-end applied before digitizing the signal. Higher AGC means the signal was weaker (the NIC had to amplify more). This value is needed to compute absolute RSSI in dBm (see `rssi_a/b/c` above) and to scale CSI to absolute power.

> [!IMPORTANT]
> When comparing CSI amplitudes across packets, you **must** account for AGC. Two identical channel conditions can produce different raw CSI amplitudes if AGC differs. Use `csiread`'s `get_scaled_csi()` or apply the AGC/RSSI correction manually.

---

### `antenna_sel` — Antenna Selection Byte

| Property | Value |
|---|---|
| **Type** | `uint8` |

A packed byte encoding which physical antenna maps to which logical antenna index. Each 2-bit field gives one mapping:

```
Bits [1:0]  → physical antenna for logical index 0
Bits [3:2]  → physical antenna for logical index 1
Bits [5:4]  → physical antenna for logical index 2
```

This value is used to compute the `perm` array.

---

### `perm` — Antenna Permutation

| Property | Value |
|---|---|
| **Type** | List of 3 integers (1-based) |
| **Example** | `[1, 2, 3]` (identity — no reordering needed) |

Derived from `antenna_sel`. Maps logical antenna indices to physical antenna positions. The CSI matrix columns are reordered using this permutation so that column *k* always corresponds to physical antenna *k*.

If `perm = [1, 2, 3]`, the antennas are already in natural order.
If `perm = [2, 1, 3]`, the NIC reported antenna 2 first and antenna 1 second, so the columns need swapping.

---

### `rate` — MCS Rate Index

| Property | Value |
|---|---|
| **Type** | `uint16` |

The `fake_rate_n_flags` field from the NIC. The lower byte encodes the 802.11n MCS index, and the upper byte contains flags (e.g., HT40 mode, short guard interval). Common values:

| Lower byte | MCS | Modulation | Streams |
|---|---|---|---|
| 0x00 | MCS 0 | BPSK 1/2 | 1 |
| 0x01 | MCS 1 | QPSK 1/2 | 1 |
| 0x07 | MCS 7 | 64-QAM 5/6 | 1 |
| 0x08 | MCS 8 | BPSK 1/2 | 2 |

---

### `csi` — Channel State Information Matrix

| Property | Value |
|---|---|
| **Type** | `np.ndarray` of `complex128` |
| **Shape** | `(Ntx, Nrx, 30)` |

The core measurement. Each element `csi[tx, rx, sc]` is a complex number representing the channel frequency response between transmit antenna `tx` and receive antenna `rx` at OFDM subcarrier group `sc`.

The **30 subcarrier groups** are sampled from the 56 usable OFDM subcarriers in a 20 MHz 802.11n channel:

```
Group:  0    1    2    3    4   ...  14   15  ...  28   29
Index: -28  -26  -24  -22  -20 ...  -1   +1  ...  +27  +28
```

Each complex coefficient `H = a + bj` encodes:
- **Amplitude** `|H| = √(a² + b²)` — channel attenuation at that frequency
- **Phase** `∠H = atan2(b, a)` — phase shift at that frequency

**Use cases:**
- **WiFi sensing / activity recognition:** CSI amplitude fluctuations caused by human movement
- **Indoor localization:** Phase differences across antennas for angle-of-arrival estimation
- **Channel characterization:** Frequency-selective fading patterns, coherence bandwidth
- **Beamforming research:** Studying the spatial structure of the wireless channel

---

## Real-World Example

Below is the **first record** parsed from `user1-1-1-1-1-r1.dat`, annotated to show how each field relates to the physical measurement.

### Scenario

An Intel 5300 NIC with **3 receive antennas** captured a packet from a single-antenna transmitter (1×3 SIMO) in an indoor environment.

### Raw Record Values

```
timestamp_low  : 1,202,803,040
bfee_count     : 29,462
Nrx            : 3
Ntx            : 1
rssi_a         : 36
rssi_b         : 37
rssi_c         : 34
noise          : -127 dBm
agc            : 35
antenna_sel    : 36   (binary: 00 10 01 00)
perm           : [1, 2, 3]
rate           : 257  (0x0101 → MCS 1, with HT flag)
csi            : shape (1, 3, 30), complex128
```

### Interpreting the Values

**Timing.** `timestamp_low = 1,202,803,040 μs ≈ 1202.8 seconds` since the NIC's clock last wrapped. This tells us roughly where in the capture session this packet fell.

**Packet loss.** `bfee_count = 29,462`. If the previous record had `bfee_count = 29,461`, no packets were dropped. A gap of *n* indicates *n − 1* lost measurements.

**MIMO configuration.** `Ntx=1, Nrx=3` — a **1×3 SIMO** link. The transmitter sent one spatial stream and the receiver captured it on all 3 antennas. This gives us 3 independent channel observations (spatial diversity) at each subcarrier.

**Signal strength.** Converting the raw RSSI values:

| Antenna | Raw RSSI | Formula | Effective (dBm) |
|---|---|---|---|
| A | 36 | 36 − 44 − 35 | **-43 dBm** |
| B | 37 | 37 − 44 − 35 | **-42 dBm** |
| C | 34 | 34 − 44 − 35 | **-45 dBm** |

All three antennas see a similar signal level around -43 dBm, which is a **strong** indoor WiFi signal (typical for same-room communication).

**Noise floor.** `noise = -127 dBm` — this is the sentinel value indicating the firmware didn't report a real noise measurement. On other firmware versions, you'd see values like -92 dBm.

**AGC.** `agc = 35 dB` — the receiver applied 35 dB of gain. This moderate gain is consistent with a medium-strength signal around -40 to -50 dBm.

**Antenna order.** `perm = [1, 2, 3]` — the identity permutation, meaning `csi[:, 0, :]` already corresponds to physical antenna A, `csi[:, 1, :]` to B, and `csi[:, 2, :]` to C.

### CSI Matrix — Channel Frequency Response

Since this is a 1×3 SIMO link, the CSI matrix has shape `(1, 3, 30)`. Here are the first 5 subcarrier coefficients for `TX 0 → RX 0` (antenna A):

| Subcarrier Group | Index | CSI (complex) | Amplitude \|H\| | Phase ∠H (rad) |
|---|---|---|---|---|
| 0 | -28 | -2 + 23j | 23.09 | +1.66 |
| 1 | -26 | 22 + 16j | 27.20 | +0.63 |
| 2 | -24 | 27 − 11j | 29.15 | -0.39 |
| 3 | -22 | 9 − 29j | 30.36 | -1.27 |
| 4 | -20 | -16 − 26j | 30.53 | -2.12 |

**What this tells us:**
- The **amplitude** varies across subcarriers (23 to 32), revealing **frequency-selective fading** — some frequencies experience more attenuation than others due to multipath reflections in the indoor environment.
- The **phase** rotates rapidly across subcarriers, which is expected — it encodes the propagation delay and multipath structure of the channel.

### Full Amplitude Profile (TX 0 → RX 0)

```
Subcarrier:  -28  -26  -24  -22  -20  -18  -16  -14  -12  -10
Amplitude:  23.1 27.2 29.2 30.4 30.5 31.6 30.8 30.8 32.3 29.8

Subcarrier:   -8   -6   -4   -2   -1   +1   +3   +5   +7   +9
Amplitude:  29.2 28.3 27.7 28.0 28.2 29.2 27.9 26.9 27.0 27.1

Subcarrier:  +11  +13  +15  +17  +19  +21  +23  +25  +27  +28
Amplitude:  28.3 27.0 28.3 26.3 29.7 29.6 29.1 27.7 27.7 24.1
```

The characteristic **roll-off at the band edges** (subcarriers ±28) and a roughly flat response in the center is typical of a line-of-sight or near-line-of-sight indoor channel. Deep frequency-selective nulls would indicate stronger multipath interference.

### Practical Applications

With this single record you can already:

1. **Estimate link quality:** Average |CSI| ≈ 28.5 with AGC = 35 dB indicates a solid link
2. **Observe spatial diversity:** Compare amplitudes across the 3 RX antennas to see which antenna has the strongest channel
3. **Track time variation:** Comparing this record to subsequent records reveals how the channel changes — e.g., due to a person walking through the propagation path
4. **Estimate angle-of-arrival:** Phase differences `∠csi[0,0,sc] − ∠csi[0,1,sc]` across antennas relate to the signal's direction of arrival (requires known antenna spacing)
