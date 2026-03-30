# Setting up Data and Plotting

This document provides instructions on how to use `plot_cross_domain.py` and how to supply the data it needs to generate these plots.

## Running in Demo Mode

If you just want to see a demonstration of the script, you can run the `plot_cross_domain.py` script without any arguments:

```bash
cd python_pipeline
python plot_cross_domain.py
```

The script will automatically pick up two differing pre-calculated BVP files from the provided `/BVP/BVP/BVP/...` directory and plot them side-by-side.

### Viewing BVP Animations
To see a time series animation of the BVP sequence instead of a static image (a single frame snippet), add the `--animate-bvp` flag:

```bash
python plot_cross_domain.py --animate-bvp
```

## Using Custom BVP Files (Pre-calculated)

The Widar3.0 release dataset includes pre-computed Body Velocity Profile (BVP) `.mat` files in the `BVP/` directory. You can specify any two of these files to see the domain-independency characteristics. The script will automatically pick the segment with the maximum energy to display.

```bash
# Compare BVP data from User 4 vs User 2
python plot_cross_domain.py \\
    --bvp1 ../BVP/BVP/BVP/20181117-VS/6-link/user4/user4-4-2-4-3-1-1e-07-100-20-100000-L0.mat \\
    --bvp2 ../BVP/BVP/BVP/20181127-VS/6-link/user2/user2-1-3-5-5-1-1e-07-100-20-100000-L0.mat
```

## Creating Doppler (DFS) Spectrograms for Plotting

Unlike BVP, which is pre-computed in the `BVP/` folder, the extracted Doppler Spectrogram matrices needed for the script to show DFS are **not** present in the repo by default. To visualize DFS, you need to first generate them from Raw CSI data using `run_pipeline.py`.

### 1. Identify Raw CSI Files
Locate a raw CSI directory in the `CSI` folder (extract the zip files if needed). For example, `CSI/20181109/userA/userA-1-1-1-1/`. Let's pick two clips from different environments or orientations.

### 2. Run Pipeline to Extract DFS
Run the `run_pipeline.py` script on these clips. The script will output the needed `_doppler.npz` files into the `python_pipeline/output/` directory.

Example command for Clip 1 (e.g. userA):
```bash
python run_pipeline.py \\
    --base-prefix /full/path/to/CSI/20181109/userA-1-1-1-1 \\
    --people userA --motion 1 --pos 1 --ori 1 --ges 1 \\
    --out-dir python_pipeline/output
```

Example command for Clip 2 (e.g. userB):
```bash
python run_pipeline.py \\
    --base-prefix /full/path/to/CSI/20181109/userB-1-1-1-1 \\
    --people userB --motion 1 --pos 1 --ori 1 --ges 1 \\
    --out-dir python_pipeline/output
```

### 3. Plot the DFS Comparisons
Now you can pass the generated `_doppler.npz` files to the plot script:

```bash
python plot_cross_domain.py \\
    --dfs1 output/userA-1-1-1-1_doppler.npz \\
    --dfs2 output/userB-1-1-1-1_doppler.npz
```
