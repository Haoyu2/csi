# CSI-Auth

Wi-Fi CSI-based gesture recognition using Body Velocity Profile (BVP) and Body Acceleration Profile (BAP) features, built on the Widar3.0 system.

## Project Structure

```
├── README.md
├── requirements.txt
├── bvp_full.csv             # Manifest: 35,918 BVP rows (source of truth for filters)
├── extract_data.py          # Unpack BVP zip archives into data/
├── extract_bap.py           # Optional: pre-compute BAP files for inspection
├── dataset.py               # CSV-driven manifest loading, filters, dedup, split
├── train.py                 # Train and evaluate gesture classifier
├── visualize_dataset.py     # Plot per-user/gesture/location/orientation distributions
├── data/                    # Extracted BVP tree (populated by extract_data.py)
│   ├── BVP-Data-Batch1/<date>/<user>/...
│   └── BVP-Data/<date>/<user>/...
├── analysis/dataset/        # Tracked corpus distribution figures
└── runs/                    # Training outputs (results.json per run)
```

## Setup

```bash
conda activate csi-auth
pip install -r requirements.txt
```

## Step 1: Extract the BVP archives

```bash
python extract_data.py
```

Unpacks `BVP-Data-From_Batch1.zip` and `BVP-Data-From-Batch2.zip` (both expected in the repo root) into `data/`. The on-disk layout mirrors the CSV's `file` column, so `data/<row['file']>` resolves directly.

The CSV `bvp_full.csv` is the canonical source of truth — training and visualization filter the CSV, then load only what's needed.

## Step 2: (Optional) inspect the corpus

```bash
python visualize_dataset.py
```

Writes 4 PNGs under `analysis/dataset/`: per-user/gesture/location/orientation bars, user × gesture heatmap, user × location heatmap, and a duplicate-group-size histogram.

## Step 3: Train and evaluate

```bash
# All three modes (BVP, BAP, BVP+BAP) with default filters (gestures 1-6)
python train.py --mode all

# Subset by users and gestures (exact-match filters — no substring bugs)
python train.py --mode all --users user1,user4 --gestures 1,2,3

# Single mode + GPU + custom epochs
python train.py --mode bap --gpu 0 --epochs 50

# Cache the loaded arrays for fast repeated runs
python train.py --mode all --cache cache/all.npz

# Collapse multi-session re-recordings to one canonical clip per natural key
python train.py --mode all --dedup-mode one-per-key

# Stricter split: hold out entire recording sessions
python train.py --mode all --split-strategy by-session
```

**Modes:**

| Mode | Input | Channels | Description |
|---|---|---|---|
| `bvp` | BVP only | 1 | Baseline velocity profile |
| `bap` | BAP only | 1 | Acceleration profile (dBVP/dt) |
| `bvp+bap` | BVP + BAP | 2 | Dual-channel: velocity + acceleration |
| `all` | — | — | Runs bvp, bap, bvp+bap sequentially |

**Filters** (all exact-match against CSV columns):

| Flag | Default | Notes |
|---|---|---|
| `--users` | all | e.g. `user1,user4` |
| `--gestures` | `1,2,3,4,5,6` | Legacy default; pass `1,...,10` to include sparse motions 7-10 |
| `--locations` | all | Rooms 1-8 |
| `--orientations` | all | Body-facing 1-5 |
| `--dedup-mode` | `keep-all` | Use `one-per-key` to collapse multi-session re-recordings |
| `--split-strategy` | `by-key` | Or `random` (legacy) or `by-session` |
| `--cache` | — | Path to `.npz` cache; first run writes, subsequent runs read |

Results are saved to `runs/<timestamp>/results.json` (filters and split strategy are recorded for reproducibility).

## Dataset structure (important)

The corpus has **35,918 physical BVP recordings** spanning **17,897 unique natural keys** `(user, gesture, location, orientation, repetition)`. Most natural keys were re-recorded across days and batches — these are **independent measurements of the same intended sample**, not duplicate files. By default (`--dedup-mode keep-all`), training treats them as natural session-augmentation.

Specifics:
- **875 byte-identical files** under `BVP-Data-Batch1/20181208/20181208/` are a zip-packaging artifact (doubled date subdirectory). Always dropped by `dataset.load_manifest`.
- The remaining ~17K duplicate-key rows are **different sensor data** for the same natural key, captured on different days. Naive natural-key dedup discards ~50% of valid signal.
- Default `--split-strategy by-key` keeps all sessions of a natural key on one side of the train/test split, eliminating session-level leakage.

## Results

Tested on 4 users (user10–13), 6 gesture classes, 3000 samples, 30 epochs, CPU.

| Feature | Test Accuracy |
|---|---|
| BVP (velocity) | 43.3% |
| BAP (acceleration) | **49.0%** |
| BVP + BAP (dual-channel) | 46.3% |

Random chance = 16.7% (6 classes).

**Key finding:** BAP alone outperforms BVP. Temporal transitions between velocity states (acceleration/deceleration patterns) are more gesture-specific than instantaneous velocity snapshots. The dual-channel result sits between the two, likely because the simple 2-channel Conv2D doesn't fully exploit both features — a more specialized fusion architecture could improve this.

## What is BVP and BAP?

### BVP — Body Velocity Profile

From the [Widar3.0 paper](https://tns.thss.tsinghua.edu.cn/widar3.0/data/TPAMI_Widar3.0_paper.pdf): BVP is a 20×20 grid representing the power distribution over 2D velocity components in body coordinates (range: ±2.0 m/s). It is recovered from multi-link Doppler spectra via sparse optimization (SLSQP with L0/L1 regularization).

The mapping from velocity to Doppler frequency on link `i` is:

```
f_D^(i) = (a_x^(i) · v_x + a_y^(i) · v_y) / λ
```

where `a_x, a_y` are geometric coefficients from Tx/Rx positions and `λ` is the Wi-Fi wavelength.

### BAP — Body Acceleration Profile

BAP captures how the velocity distribution changes over time:

```
BAP[t] = BVP[t] - BVP[t-1]
```

This is mathematically grounded: since the Doppler-velocity mapping is linear (`D = A · V`), differentiating both sides gives `dD/dt = A · dV/dt`. So the temporal difference of BVP directly represents velocity changes (acceleration/deceleration) in body coordinates.

When a hand accelerates, its power peak shifts from one velocity bin to another — BAP captures exactly this transition. Positive values indicate power appearing at a velocity (body part accelerating to that speed); negative values indicate power leaving (body part moving away from that speed).

## Bugs Fixed from Original Codebase

### 1. Normalization skipping sparse samples

The original `normalize_data()` skips the entire sample when any time frame is all-zero. Since BVP is very sparse (most frames are zero), nearly all samples went unnormalized (values stuck in [0, 0.07]).

**Fix:** Per-frame normalization — each frame is independently scaled to [0, 1].

**Impact:** BVP accuracy 14.7% → 43.3%.

### 2. BAP optimization using invalid loss function

The original BAP extraction (`run_pipeline_bap.py`) fed `doppler_diff` (a signed signal) into an EMD loss function that only works for non-negative distributions. The optimizer produced near-zero output.

**Fix:** Compute BAP as `BVP[t] - BVP[t-1]` directly, bypassing the broken optimization.

**Impact:** BAP accuracy ~17% (random) → 49.0%.

## Model Architecture

Identical to the Widar3.0 paper (CNN + GRU):

```
Input (T, 20, 20, C)            # C=1 for single-channel, C=2 for dual
→ TimeDistributed(Conv2D 16, 5×5, ReLU)
→ TimeDistributed(MaxPool 2×2)
→ TimeDistributed(Flatten)
→ TimeDistributed(Dense 64, ReLU)
→ TimeDistributed(Dropout 0.5)
→ TimeDistributed(Dense 64, ReLU)
→ GRU 128
→ Dropout 0.5
→ Dense(N_class, Softmax)
```

## Data Format

Each `.npz` file contains one gesture sample:

- **BVP files** (`*_bvp.npz`): key `velocity_spectrum_ro`, shape `(20, 20, T)`
- **BAP files** (`*_bap.npz`): key `acceleration_spectrum_ro`, shape `(20, 20, T)`

Filename convention: `{user}-{gesture}-{location}-{orientation}-{repetition}_bvp.npz`

- `gesture`: 1–10 (1–6 are dense across users; 7–10 are sparsely sampled — see distribution figures)
- `location`: 1–8 (rooms; 1–5 dense, 6–8 sparse)
- `orientation`: 1–5 (body facing direction; uniformly sampled)
