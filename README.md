# Widar3.0 Python Pipeline (CSI -> Doppler -> BVP -> Classifier)

Python scripts that mirror the Widar3.0 Matlab release: raw CSI -> Doppler spectrogram -> body-coordinate velocity profile (BVP) -> CNN+GRU classifier.

## Prerequisites
- Python 3.10+
- Optional CUDA GPU
- Install: `pip install -r python_pipeline/requirements.txt`
- Raw CSI extracted from a date archive (e.g., `20181109.zip`) into `CSI/<date>/`. Each clip needs six files: `userA-1-1-1-1-r1.dat` ... `-r6.dat` (6 receivers x 3 antennas).

## End-to-end example
1) Choose a clip prefix  
   Example: `D:/projects/Widar3.0ReleaseData/CSI/20181109/userA-1-1-1-1`.

2) Run CSI -> Doppler -> BVP  
   ```bash
   python python_pipeline/run_pipeline.py \
     --base-prefix D:/projects/Widar3.0ReleaseData/CSI/20181109/userA-1-1-1-1 \
     --people userA --motion 1 --pos 1 --ori 1 --ges 1 \
     --out-dir python_pipeline/output --save-mat
   ```
   Outputs:
   - `python_pipeline/output/userA-1-1-1-1_doppler.npz`
   - `python_pipeline/output/userA-1-1-1-1_bvp.npz` (and `.mat`)

   Parity with Matlab:
   - Filtering, antenna selection, amplitude adjust, PCA, STFT match `get_doppler_spectrum.m`.
   - Doppler->velocity mapping uses SciPy SLSQP with the same loss/bounds as `DVM_main.m`.
   - Defaults: segment length 100, lambda=1e-7, velocity grid 20x20, speed range [-2, 2] m/s.

3) Train the CNN+GRU classifier  
   ```bash
   python python_pipeline/train_model.py \
     --data-dir python_pipeline/output \
     --gpu 0 --epochs 30 --batch-size 32
   ```
   Loads all `*_bvp.npz` (or `.mat`), normalizes, zero-pads time, trains, saves `model_widar3_trained.h5`, prints test accuracy.

## Files added
- `[python_pipeline/run_pipeline.py](/D:/projects/Widar3.0ReleaseData/python_pipeline/run_pipeline.py)` - CSI->Doppler->BVP pipeline.
- `[python_pipeline/train_model.py](/D:/projects/Widar3.0ReleaseData/python_pipeline/train_model.py)` - CNN+GRU training/testing.
- `[python_pipeline/requirements.txt](/D:/projects/Widar3.0ReleaseData/python_pipeline/requirements.txt)` - dependencies.

## Common tweaks
- Change position/orientation: set `--pos` / `--ori` on `run_pipeline.py` (rotation table matches the Matlab `torso_ori`).
- Inspect Doppler only: run `run_pipeline.py` and open `_doppler.npz`.
- Adjust velocity resolution: edit `MappingConfig` in `run_pipeline.py` (`V_max`, `V_min`, `V_bins`).

## Equivalence to the paper
- Preprocessing matches `get_doppler_spectrum.m` and `DVM_main.m`.
- Classifier architecture is the same as `widar3_keras.py` (TimeDistributed Conv2D -> GRU -> softmax).
- Filenames keep motion/position/orientation/repetition tokens for reproducible splits.

## Notes
- Doppler->BVP optimization is CPU-heavy; it can take seconds to minutes per clip. Lower `seg_length` or `MaxFunctionEvaluations` in `MappingConfig` if needed.
- `csiread` expects Intel 5300 `.dat` CSI. If parsing fails, verify each `-r#.dat` is CSI (not PCAP) and uses 3-antenna, 30-subcarrier frames.

## Data inventory and usage
- `CSI/` (zipped per date, e.g., `20181109.zip`): Raw CSI logs from 6 receivers, 3 antennas each. Used as the starting point for Doppler extraction (`run_pipeline.py` or Matlab `get_doppler_spectrum.m`).
- `BVP/` (zipped): Precomputed body-coordinate velocity profiles released by the authors. Can be loaded directly for model training (skip Doppler/BVP extraction).
- `Gait_Dataset/CSI_Gait/` (folder): Subset organized for gait-related tasks (still CSI-based); can be processed with the same pipeline or used for gait-specific experiments described in the paper.
- `DFSExtractionCode.zip`: Matlab scripts for Doppler frequency shift extraction only (similar to `get_doppler_spectrum.m`), useful if you need Doppler but not full BVP mapping.
- `BVPExtractionCode.zip`: Matlab baseline for full pipeline (CSI -> Doppler -> BVP) that our Python code mirrors.
- `DNN_Model.zip`: Keras reference model from the release; our `train_model.py` matches its architecture.
- `csi_tool_box.zip`: CSI reader utilities (Matlab) used by the original scripts.
- `README_baidudisk.pdf`: Official download/readme from the release site; describes dataset naming and layout.

### How the pieces connect
```
Raw CSI (.dat, 6x3)         Precomputed BVP (.mat)
       |                                |
       | (run_pipeline.py)              | (optional direct load)
       v                                v
 Doppler Spectrogram (npz)      Velocity Spectrum (npz/mat)
       |                                |
       | (SciPy SLSQP mapping)          |
       v                                v
 Body-coordinate Velocity Profile (BVP)
       |
       | (train_model.py / widar3_keras.py)
       v
 CNN + GRU classifier --> gesture/motion labels
```
