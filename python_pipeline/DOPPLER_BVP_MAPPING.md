# Widar3.0: Doppler to BVP Mapping Explanation

This document explains how the Widar3.0 system defines the Doppler Spectrogram data structure and how it transforms that into a Body-coordinate Velocity Profile (BVP) via Doppler-Velocity Mapping (DVM).

## 1. Doppler Spectrogram Data Structure

When preprocessing the raw CSI data, the pipeline performs filtering and Short-Time Fourier Transform (STFT) on the signals received. 

It generates two main outputs representing the Doppler Spectrogram:

1. **`doppler_spectrum` Tensor**: A 3D NumPy array of shape `(R, F, T)`.
   - **`R`**: Number of receivers, typically `rx_cnt = 6`.
   - **`F`**: Number of extracted frequency bins. The STFT calculates components between `-60 Hz` and `+60 Hz` (`uppe_stop = 60.0`). With a `1 Hz` resolution (the sample rate is `1000 Hz` and FFT size `nfft = 1000`), there are approximately `121` frequency bins on this axis.
   - **`T`**: The time axis, representing overlapping sliding windows across the gesture clip duration. The values represent the normalized signal power at that time and frequency.
2. **`freq_bin` Vector**: A 1D array of shape `(F,)` that simply lists the exact frequency value (in Hz) corresponding to each index in the `F` dimension.

## 2. Calculating the Body Velocity Profile (BVP)

The step that converts the Doppler Spectrogram into a BVP matrix is highly non-trivial. It uses a constraint-based optimization function based on Earth Mover's Distance (EMD) to map Doppler frequency shifts measured at the receivers back to human body movement in 2D space.

The process, encapsulated by the `doppler_to_bvp()` function, is as follows:

### Concept: The 2D Spatial Velocity Vector

Before detailing the calculation steps, it is essential to understand the **2D spatial velocity vector**, denoted as $v = [v_x, v_y]$. This vector serves as the foundational unit for mapping physical human movement to the measured Doppler shifts.

- **$v_x$**: The physical speed of movement along the X-axis (e.g., in meters per second).
- **$v_y$**: The physical speed of movement along the Y-axis.

In the context of Widar3.0, the Body Velocity Profile (BVP) is essentially a discrete 2D resolution grid (e.g., a $20 \times 20$ matrix) of these spatial velocity vectors, covering a range of possible movement speeds (e.g., from $-2.0 \text{ m/s}$ to $+2.0 \text{ m/s}$). Each "pixel" in the BVP represents the "energy" or intensity of movement at that specific $(v_x, v_y)$ combination. The system's underlying goal is to reconstruct the true distribution of these vector energies across the human body based solely on the 1D Doppler frequency measurements.

### A. Geometrical Initialization (The `A` and `VDM` Matrices)
The Doppler shift measured at any receiver depends intimately on the spatial geometry between the transmitter ($Tx$), the receiver ($Rx$), and the moving human body (referred to as the target torso).

The script computes two key structures to handle this geometric mapping:

**1. The Projection Matrix `A`:**
A continuous wave reflecting off a moving point creates a Doppler frequency shift. The change in path length involves both the path from the Transmitter to the body, and the body to the Receiver.

The script models this by creating an `A` tensor of shape `(R, 2)`. For each receiver $i$, the vector $a_i$ is computed as the sum of two normalized unit vectors:
$$ a_i = \frac{P_{torso} - P_{Tx}}{||P_{torso} - P_{Tx}||} + \frac{P_{torso} - P_{Rx_i}}{||P_{torso} - P_{Rx_i}||} $$
Where $P$ denotes the 2D Cartesian spatial coordinates. This unit vector $a_i$ determines how much of the target's true velocity gets "projected" onto the propagation path of receiver $i$.

**2. The Velocity-to-Doppler Mapping (`VDM`) Matrix:**
The Doppler frequency shift $f_D$ (in Hz) at receiver $i$ for any 2D spatial velocity vector $v = [v_x, v_y]$ can be directly computed using the dot product:
$$ f_{D_i} = \frac{a_i \cdot v}{\lambda} $$
Where $\lambda$ is the corresponding wavelength of the Wi-Fi signal (e.g., $c / 5.825 \text{ GHz}$).

The script precomputes a massive 4D mapping tensor called `VDM` (Velocity-to-Doppler Mapping) of shape `(R, M, M, F)` to avoid calculating this on the fly:
- `M` is the resolution grid of the BVP (e.g., $20 \times 20$ velocity bins covering $-2.0 \text{ m/s}$ to $+2.0 \text{ m/s}$).
- The script iterates through every receiver and every possible discrete velocity bin $(v_x, v_y)$. 
- It evaluates the formula above. The calculated continuous frequency $f_{D_i}$ is rounded to the nearest discrete frequency bin `idx`.
- The corresponding entry `VDM[R, Vx, Vy, idx]` is activated (set to `1.0`), mapping that spatial velocity to that receiver's measured frequency. If a velocity creates a frequency shift outside the measured $-60\text{Hz}$ to $+60\text{Hz}$ bounds, it is assigned a massive penalty value (`1e10`).

### B. Segmenting and Averaging the Doppler 
The `T` time axis is grouped into discrete, non-overlapping segments (e.g., `seg_length = 100` frames). For each segment, it averages the Doppler spectrum over time, yielding a static `doppler_tgt` shape of `(R, F)` representing the average Doppler profile across all receivers during that specific sub-window of time.

### C. The Optimization Solver (SciPy SLSQP)
Because multiple body parts are moving (creating multiple reflections mapping to the same frequency bins), the problem of reconstructing the exact 2D velocity profile `P(Vx, Vy)` is underdetermined.

The system frames this as an optimization problem:
- **Decision Variable:** A flattened vector representing the 2D `MxM` BVP matrix (we want to find the velocity energy at each `[x, y]` pixel).
- **Target:** The averaged `doppler_tgt`.
- **Loss Function:** `dvm_loss` computes the expected Doppler projection if our guess of the BVP was correct (`approx = P * VDM`). It minimizes the cumulative difference between this `approx` tensor and the true `doppler_tgt` measured by the hardware. This behaves like an Earth Mover's Distance (EMD).
- **Sparsity Regularization:** The human body movement is sparse (only a few velocities exist at once). A sparsity penalty `\lambda` (e.g., `1e-7`) is added to the loss function using either L1 or pseudo-L0 norm.

A non-linear constrained solver (SciPy's `SLSQP` - Sequential Least SQuares Programming) minimizes the `dvm_loss`. The output bounds are strictly positive, yielding an `MxM` matrix representing the energy intensity of different translational speeds.

### D. Coordinate Rotation
Finally, the resulting BVP matrices are rotated by the subject's torso orientation (`torso_ori`). Because the underlying calculation assumes the person faces "forward," adding a digital rotation normalizes the BVP to be strictly local and orientation-independent—a core contribution of the Widar3.0 paper. The final output is reshaped into the `(M, M, segs)` BVP structure.
