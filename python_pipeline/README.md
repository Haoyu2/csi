# Python Pipeline Guide

This folder holds the pure-Python flow that mirrors the Widar3.0 Matlab release.

## Raw CSI structure (input)
- Source: unzip one date archive (e.g., `CSI/20181109.zip`).
- Each gesture clip is identified by a prefix: `userA-<motion>-<pos>-<ori>-<ges>`.
- For every prefix there are 6 receiver files (3 antennas each), named:
  - `userA-...-r1.dat`, `userA-...-r2.dat`, ..., `userA-...-r6.dat`
- Files contain Intel 5300 CSI frames (30 subcarriers x 3 Rx antennas), already converted to `.dat` format via `log_to_dat`.
- Relationship to released BVP:
  - those 6 synchronized `-r#.dat` files are one logical recording of the same clip;
  - the released BVP dataset stores the processed result for that clip as one `.mat` file containing `velocity_spectrum_ro`;
  - for example, `CSI/.../user2-1-1-1-1-r1.dat` ... `-r6.dat` map to `BVP/.../user2-1-1-1-1-20181208.mat`.

## Processing pipeline (run_pipeline.py)
1) Load CSI for each receiver (`load_csi` with `csiread`).
2) Preprocess per `get_doppler_spectrum.m`:
   - Bandpass (2–60 Hz), antenna selection, amplitude adjustment, conjugate multiply, PCA (1st component).
   - STFT to form Doppler spectrogram per receiver.
3) Normalize spectra and align frequency bins across receivers.
4) Map Doppler -> body-coordinate velocity profile (BVP) with SciPy SLSQP, matching `DVM_main.m`.
5) Rotate BVP according to orientation and save.

## Outputs
- Doppler spectrogram (per run): `python_pipeline/output/<prefix>_doppler.npz`
  - Keys: `doppler` (rx_cnt x F x T), `freq_bin` (F,)
- Body-coordinate velocity profile: `<prefix>_bvp.npz` (and `.mat` if `--save-mat`)
  - Key: `velocity_spectrum_ro` (M x M x segments), default grid 20x20, speed range [-2, 2] m/s

## CSI data in memory
- Loader: `csiread.Intel` returns `csi` shaped `(frames, rx, tx, subcarriers)`; we reshape to `(frames, 90)` for 3 Rx x 1 Tx x 30 subcarriers.
- Each element is complex64/complex128: amplitude and phase per subcarrier per antenna.

## Doppler spectrum computation (step by step)
1) Load one receiver file `*-rK.dat`, reshape to `(frames, 90)`.
2) Select reference antenna: compute mean/variance ratio over subcarriers per antenna, pick the max.
3) Amplitude adjustment:
   - For every column, subtract the smallest nonzero amplitude (alpha) to suppress DC bias.
   - Add a shared beta offset to the reference copy to stabilize conjugate multiplication.
4) Form conjugate multiplication: adjusted signal * conj(reference) to isolate motion-induced phase.
5) Drop the reference antenna’s own 30 subcarriers to avoid self-multiplication.
6) Bandpass filter each column: high-pass at 2 Hz, low-pass at 60 Hz (Butterworth orders 3 and 6).
7) PCA (1 component) across columns to fuse antennas/subcarriers into a single motion trace.
8) Short-time Fourier transform (Gaussian window, ~250 ms, 50% overlap, 1 Hz FFT):
   - Frequency axis is centered; keep |f| ≤ 60 Hz.
   - Normalize each time column so it sums to 1.
9) Repeat steps 1–8 for all 6 receivers, pad to the longest time axis, stack to shape `(rx_cnt, F, T)`.

### Diagram: CSI -> Doppler (per receiver)
```
Intel 5300 .dat
    |
    v
reshape (frames, 90)
    |
ref antenna pick
    |
amp adjust + conj mult
    |
drop ref subcarriers
    |
bandpass 2-60 Hz
    |
PCA (1 component)
    |
STFT (Gaussian window)
    |
freq mask |f|<=60, column normalize
    v
Doppler spectrogram (F x T)
```

## Quick start
```
python run_pipeline.py \
  --base-prefix D:/projects/Widar3.0ReleaseData/CSI/20181109/userA-1-1-1-1 \
  --people userA --motion 1 --pos 1 --ori 1 --ges 1 \
  --out-dir python_pipeline/output --save-mat
```

Use the resulting `_bvp` files as input to `train_model.py` for classification.
