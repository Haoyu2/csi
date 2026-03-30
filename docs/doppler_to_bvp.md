# Doppler Spectrum to BVP in Widar3.0

This note explains how Widar3.0 computes the body-coordinate velocity profile (BVP) from the Doppler spectrum, and how both are represented in the dataset and saved files.

The implementation exists in two places in this repo:

- Matlab release code:
  - `BVPExtractionCode/Widar3.0Release-Matlab/get_doppler_spectrum.m`
  - `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_main.m`
  - `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/get_velocity2doppler_mapping_matrix.m`
  - `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_target_func.m`
- Python port:
  - `python_pipeline/run_pipeline.py`

## Short answer

- The Doppler spectrum is a per-receiver time-frequency tensor.
- The BVP is a per-segment 2D velocity distribution over `(vx, vy)`.
- BVP is not a direct relabeling of Doppler bins.
- Instead, BVP is recovered by solving an optimization problem: find a 2D velocity image `P(vx, vy)` whose induced Doppler pattern best matches the measured Doppler spectrum across all receivers.

## Where each representation lives

### 1. Raw CSI

Input data is a set of six CSI files for one clip:

```text
<prefix>-r1.dat
<prefix>-r2.dat
...
<prefix>-r6.dat
```

Each receiver file contains complex CSI frames. In the Python port, one receiver file is loaded and reshaped to:

```text
(frames, 90)
```

because there are 3 receive antennas x 30 subcarriers.

### 2. Doppler spectrum

The Doppler output is saved as:

- `doppler`: shape `(rx_cnt, F, T)`
- `freq_bin`: shape `(F,)`

Meaning:

- `rx_cnt`: number of receivers, usually `6`
- `F`: number of Doppler frequency bins kept after filtering, usually about `121` for `-60..60 Hz`
- `T`: number of time snapshots from STFT

So the tensor is:

```text
doppler[receiver, doppler_frequency_bin, time]
```

In the Python pipeline this is saved as:

```text
python_pipeline/output/<prefix>_doppler.npz
```

with keys `doppler` and `freq_bin`.

### 3. BVP

The BVP output is saved as:

- `velocity_spectrum_ro`: shape `(M, M, segments)`

Meaning:

- first axis: `vx` bin
- second axis: `vy` bin
- third axis: segment index

So the tensor is:

```text
velocity_spectrum_ro[vx_bin, vy_bin, segment]
```

With the default settings:

- `V_min = -2 m/s`
- `V_max = 2 m/s`
- `V_bins = 20`

the BVP shape is usually:

```text
(20, 20, number_of_segments)
```

In the Python pipeline this is saved as:

```text
python_pipeline/output/<prefix>_bvp.npz
```

with key `velocity_spectrum_ro`.

The Matlab-compatible `.mat` file also stores only:

```text
velocity_spectrum_ro
```

I inspected one released sample and it contained only that field with shape `(20, 20, 14)`.

## Step 1: CSI to Doppler spectrum

This stage is implemented in:

- Matlab: `get_doppler_spectrum.m`
- Python: `compute_doppler_spectrum()` in `python_pipeline/run_pipeline.py`

### 1.1 Load one receiver CSI stream

For each receiver `r1` to `r6`, the code loads complex CSI samples across time.

For one receiver, the data is flattened to:

```text
csi[time, antenna_subcarrier] with shape (frames, 90)
```

### 1.2 Select a reference antenna

The code computes a mean-to-variance ratio on CSI amplitudes and picks the antenna with the largest ratio as the reference antenna.

This happens in the Matlab code before reference replication, and in the Python port via `_select_reference()`.

Intuition:

- the reference antenna should be relatively stable
- that helps isolate motion after conjugate multiplication

### 1.3 Amplitude adjustment

For each CSI column:

1. compute its amplitude
2. subtract the smallest nonzero amplitude `alpha`
3. rebuild the complex signal with the original phase

Then construct an adjusted reference copy using a shared `beta` offset.

This suppresses static amplitude bias and stabilizes the next step.

### 1.4 Conjugate multiplication

The core motion-sensitive signal is:

```text
conj_mult = adjusted_signal * conj(adjusted_reference)
```

After this, the code removes the reference antenna's own 30 subcarriers, so the remaining channels emphasize cross-antenna motion effects rather than self-products.

### 1.5 Bandpass filtering

Each conjugate-multiplied channel is filtered:

- low-pass at `60 Hz`
- high-pass at `2 Hz`

This keeps the motion-related Doppler region and suppresses static and high-frequency noise.

### 1.6 PCA

The filtered channels are reduced with PCA and only the first component is kept:

```text
principal[time]
```

This fuses the multiple antenna/subcarrier channels into one dominant motion trace per receiver.

### 1.7 STFT

The principal time series is transformed with STFT using roughly a 250 ms Gaussian window. The resulting spectrogram is a frequency-time matrix:

```text
spec[freq, time]
```

Only Doppler bins within `[-60, 60] Hz` are kept.

Then each time column is normalized so that its sum is 1:

```text
spec[:, t] = spec[:, t] / sum(spec[:, t])
```

This means each time snapshot becomes a normalized Doppler distribution.

### 1.8 Stack all receivers

After repeating the same process for all six receivers, the spectrograms are padded to the same time length and stacked:

```text
doppler_spectrum.shape = (rx_cnt, F, T)
```

This is the Doppler representation used by the next stage.

## Step 2: Doppler spectrum to BVP

This stage is implemented in:

- Matlab: `DVM_main.m`
- Python: `doppler_to_bvp()` in `python_pipeline/run_pipeline.py`

This is an inverse problem, not a direct transform.

### 2.1 Segment the Doppler spectrum in time

The Doppler tensor is split into fixed-length time segments.

Default:

```text
seg_length = 100
```

So if the Doppler tensor has shape:

```text
(6, 121, 1432)
```

then:

```text
segments = floor(1432 / 100) = 14
```

and only the first `1400` time snapshots are used for BVP generation.

For segment `k`, the Doppler target is the average over time inside that segment:

```text
doppler_tgt = mean(doppler[:, :, k*100:(k+1)*100], axis=time)
```

Its shape is:

```text
(rx_cnt, F)
```

For the example above:

```text
(6, 121)
```

This averaging step is important: one BVP slice corresponds to one time segment, not one STFT frame.

### 2.2 Build the velocity grid

The body velocity is represented on a 2D grid:

```text
vx in [-2, 2]
vy in [-2, 2]
```

with `20` bins per axis by default.

So one BVP slice is a `20 x 20` image:

```text
P[vx_bin, vy_bin]
```

Interpretation:

- each cell stores how much motion energy is assigned to that body-coordinate velocity
- bright cells indicate likely motion directions and speeds

### 2.3 Build the geometry matrix A

Widar uses transmitter position, receiver positions, and torso position to model how a body velocity projects to each receiver's Doppler shift.

The matrix:

```text
A.shape = (rx_cnt, 2)
```

maps a 2D body velocity vector `[vx, vy]` to a scalar Doppler shift for each receiver.

Each row depends on:

- transmitter location
- one receiver location
- subject torso position

This is what ties the physical room geometry to the signal model.

### 2.4 Build the velocity-to-Doppler mapping matrix VDM

This is the key object.

For every:

- receiver `r`
- x-velocity bin `i`
- y-velocity bin `j`

the code computes which Doppler frequency bin that velocity would produce:

```text
plcr_hz = round(A[r] dot [vx_i, vy_j] / wavelength)
```

Then it creates a one-hot assignment into the Doppler axis.

The full tensor is:

```text
VDM[receiver, vx_bin, vy_bin, doppler_freq_bin]
```

with shape:

```text
(rx_cnt, M, M, F)
```

For the default setup:

```text
(6, 20, 20, 121)
```

Semantically:

- fix a receiver and a velocity cell `(vx_i, vy_j)`
- `VDM` tells you which Doppler bin that velocity should activate for that receiver

So `VDM` is not data from the measurement. It is the forward model that predicts Doppler from velocity.

### 2.5 Solve for the BVP image P

For each segment, Widar solves for a nonnegative 2D image `P` such that its predicted Doppler spectrum matches the measured segment Doppler.

Unknown:

```text
P.shape = (M, M)
```

Predicted Doppler from `P`:

```text
approx[r, f] = sum over i, j of P[i, j] * VDM[r, i, j, f]
```

This gives:

```text
approx.shape = (rx_cnt, F)
```

which is directly comparable to `doppler_tgt`.

### Loss function

The Matlab code in `DVM_target_func.m` uses an earth-mover-like cumulative mismatch along frequency, plus a regularization term.

In simplified form:

```text
loss = sum_r sum_f | cumulative_sum_f(approx[r, f] - target[r, f]) |
       + lambda * regularization(P)
```

where the regularization is:

- approximately L0 in the default release code: number of nonzero cells in `P`
- optionally L1 in the Python port if configured

The default settings use:

```text
lambda = 1e-7
```

and nonnegative bounds:

```text
0 <= P[i, j] <= max(doppler_spectrum)
```

The solver is:

- Matlab: `fmincon(..., 'Algorithm', 'sqp')`
- Python: `scipy.optimize.minimize(..., method='SLSQP')`

So BVP is the optimizer's best-fit 2D velocity distribution for that segment.

### 2.6 Rotate into body coordinates

After solving `P` for each segment, the code rotates the 2D velocity image according to the subject orientation:

```text
velocity_spectrum_ro = rotate(velocity_spectrum, orientation)
```

This is why the saved variable is called:

```text
velocity_spectrum_ro
```

where `ro` means rotated.

This makes the final BVP aligned with the subject's body coordinate system rather than just the room coordinate system.

## End-to-end tensor example

Assume one clip produces the following Doppler output:

```text
doppler.shape = (6, 121, 1432)
freq_bin.shape = (121,)
```

Interpretation:

- `6` receivers
- `121` Doppler bins, roughly `-60..60 Hz`
- `1432` STFT time snapshots

With `seg_length = 100`:

```text
segments = floor(1432 / 100) = 14
```

For each of the 14 segments:

1. take a slice of shape `(6, 121, 100)`
2. average over time to get `(6, 121)`
3. solve for a `20 x 20` velocity image `P`
4. rotate it by the subject orientation

Final BVP shape:

```text
velocity_spectrum_ro.shape = (20, 20, 14)
```

That matches the shape seen in the released sample `.mat`.

## What is and is not stored in the files

### Doppler file

Saved in the Python pipeline as:

```text
<prefix>_doppler.npz
```

Keys:

- `doppler`
- `freq_bin`

### BVP file

Saved in the Python pipeline as:

```text
<prefix>_bvp.npz
```

Key:

- `velocity_spectrum_ro`

If `--save-mat` is used, the `.mat` file also stores only:

- `velocity_spectrum_ro`

### Important implication

The precomputed BVP files do not contain the Doppler tensor inside them.

So:

- Doppler and BVP are separate saved representations
- BVP is derived from Doppler
- but once saved, a BVP file usually contains only the final rotated velocity spectrum

## Mental model

It helps to think of the two representations like this:

- Doppler: "What frequency shifts did each receiver observe over time?"
- BVP: "What 2D body velocity distribution best explains those frequency shifts?"

So Doppler is measurement space, while BVP is latent velocity space.

## Source pointers

### Matlab release

- Doppler construction starts in `BVPExtractionCode/Widar3.0Release-Matlab/get_doppler_spectrum.m:1`
- Conjugate multiplication is at `BVPExtractionCode/Widar3.0Release-Matlab/get_doppler_spectrum.m:59`
- PCA is at `BVPExtractionCode/Widar3.0Release-Matlab/get_doppler_spectrum.m:69`
- Doppler column normalization is at `BVPExtractionCode/Widar3.0Release-Matlab/get_doppler_spectrum.m:90`
- Doppler to BVP starts in `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_main.m:2`
- VDM construction is used at `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_main.m:56`
- Segment averaging is at `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_main.m:68`
- Optimization call is at `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_main.m:80`
- Rotation is at `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_main.m:94`
- Saving BVP is at `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_main.m:97`
- VDM definition starts in `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/get_velocity2doppler_mapping_matrix.m:6`
- One-hot Doppler assignment is at `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/get_velocity2doppler_mapping_matrix.m:28`
- Loss definition starts in `BVPExtractionCode/Widar3.0Release-Matlab/Doppler2VelocityMapping/DVM_target_func.m:5`

### Python port

- Doppler computation starts at `python_pipeline/run_pipeline.py:141`
- VDM construction starts at `python_pipeline/run_pipeline.py:230`
- Loss function starts at `python_pipeline/run_pipeline.py:249`
- Doppler-to-BVP mapping starts at `python_pipeline/run_pipeline.py:267`
- Doppler save is at `python_pipeline/run_pipeline.py:346`
- BVP save is at `python_pipeline/run_pipeline.py:351`
- Matlab-compatible BVP save is at `python_pipeline/run_pipeline.py:356`
