# Implementation Plan: BAP Extraction and Training Integration

## Goal Description
Implement "Method B" (Pre-Mapping) to extract the Body Acceleration Profile (BAP) directly from the differentiated Doppler spectrum in the processing pipeline. Furthermore, intelligently combine the BVP (velocity) and BAP (acceleration) profiles to train the neural network model efficiently with 2-channel spatial-temporal data.

## Proposed Changes

### [run_pipeline.py](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/run_pipeline.py)
- #### [MODIFY] [run_pipeline.py](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/run_pipeline.py)
  - Refactor [doppler_to_bvp()](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/run_pipeline.py#339-425) to return a tuple [(rotated_bvp, rotated_bap)](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/run_pipeline.py#445-466).
  - Inside the segment loop, store `prev_doppler_tgt`.
  - Calculate `doppler_diff = doppler_tgt - prev_doppler_tgt` (for the first segment, this diff is zero).
  - Add a second `optimize.minimize` call per segment specifically targeting `doppler_diff`.
  - Because acceleration represents a *change* in velocity, it can be negative. We will relax `bounds` for the BAP optimization to `optimize.Bounds(lb=-U_bound.flatten(), ub=U_bound.flatten())`.
  - Apply the torso orientation rotation identically to the resulting BAP sequence.
  - Update [main()](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/run_pipeline.py#445-466) to save `acceleration_spectrum_ro=bap` inside the `.npz` archives.

### [train_model.py](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/train_model.py)
- #### [MODIFY] [train_model.py](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/train_model.py)
  - During [load_data()](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/train_model.py#46-79), load both BVP and BAP tensors. If a file is missing BAP, provide a fallback zero-matrix.
  - Normalize BVP and BAP independently using the existing [normalize_data()](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/train_model.py#22-30) function to constrain them between 0 and 1 before combination.
  - Stack the two features into a 4D tensor `[X, Y, T, 2]`.
  - Update [zero_padding()](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/train_model.py#32-38) to handle arrays with a 4th dimension: [((0, 0), (0, 0), (T_MAX - t, 0), (0, 0))](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/run_pipeline.py#445-466).
  - Eliminate the `np.expand_dims(data, axis=-1)` line because the stacked array inherently has the final channel dimension (`2`).
  - The model input will organically adapt to `[T_MAX, 20, 20, 2]` given `channels_last` layout and dynamic input shape mapping.

## Verification Plan

### Automated/Mock Tests
We will synthesize a dummy `.npz` file mimicking the pipeline output to strictly verify that [train_model.py](file:///home/haoyu/Widar3.0ReleaseData/python_pipeline/train_model.py) ingests the new 2-channel shape and begins training flawlessly.
```bash
# Create a dummy data directory and script to generate a fake .npz file
mkdir -p /home/haoyu/Widar3.0ReleaseData/python_pipeline/tests/mock_data
```

### Manual Verification
1. I will provide you with the updated scripts.
2. If you have CSI `.dat` files available, you can perform an end-to-end extraction using: `python run_pipeline.py --base-prefix [path]`.
3. You can then launch training on the generated output using: `python train_model.py --epochs 1`.
