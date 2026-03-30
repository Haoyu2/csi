# Extracting Body Acceleration Profile (BAP) for Widar3.0

This document outlines two mathematical procedures for deriving a **Body Acceleration Profile (BAP)** from Wi-Fi sensing data. Extracting a direct acceleration feature ($a = dv/dt$) is an advanced research extension. Specifically, it can be highly useful for testing deep learning model generalizability and accuracy beyond the standard Body Velocity Profile (BVP).

Because the physical phenomenon of the Doppler effect ($f_D = v / \lambda$) directly measures velocity, not acceleration, deriving the BAP requires computing temporal derivatives either *before* or *after* the computationally heavy spatial optimization mapping step.

## Method A: The Spatial-Derivative BAP (Post-Mapping)

The simplest and most computationally efficient method is to calculate the acceleration profile mathematically from the fully computed BVP sequence.

1. **Calculate Standard BVP**: Run the standard `doppler_to_bvp()` pipeline to convert the 1D Multi-Receiver Doppler Spectrogram into a 3D sequence of BVP matrices `(M, M, T)`, where $M$ is the spatial resolution and $T$ is the number of time frames.
2. **Apply Temporal Differentiation**: Acceleration is the rate of change of velocity. Calculate the frame-by-frame discrete derivative along the time axis ($T$).
   $$BAP[t] = BVP[t] - BVP[t-1]$$
   For more robustness against high-frequency noise, a central difference method can be employed:
   $$BAP[t] = \frac{BVP[t+1] - BVP[t-1]}{2 \cdot \Delta t}$$
3. **Usage**: The resulting $M \times M$ grid represents where the velocity energy is actively increasing or decreasing in 2D space. You can stack this BAP tensor alongside the standard BVP sequence as an additional multi-channel input (e.g., shape `[T, M, M, 2]`) for your neural network.

*Pros: Extremely fast to compute, fully utilizes the existing Widar3.0 codebase.*
*Cons: Mathematical differentiation amplifies high-frequency noise and artifacts inherent in the BVP optimization process.*

## Method B: The Direct Doppler-Derivative BAP (Pre-Mapping)

A purely novel research approach is to force the optimization solver to map the *change in the Doppler spectrum* directly into an acceleration grid, effectively predicting BAP without ever computing BVP.

1. **Obtain the Averaged Spectrogram**: As usual, calculate the `doppler_tgt` shape `(R, F, T)` representing the averaged Doppler spectrum across all receivers $R$ for frequency bins $F$.
2. **Differentiate the Spectrogram**: Calculate the temporal difference of the Doppler target *before* the optimization step.
   $$doppler\_diff[t] = doppler\_tgt[t] - doppler\_tgt[t-1]$$
   This step isolates the *shift* in the frequencies (the physical manifestation of acceleration), filtering out bodies or body parts moving at constant velocity.
3. **Run the Optimizer**: Feed the differential spectrum `doppler_diff` directly into the standard SciPy `SLSQP` optimizer using the *exact same* Geometric Projection Matrix ($A$) and Mapping tensor ($VDM$).
   - *Why this works mathematically:* The mapping projection $f_D = \frac{a_i \cdot v}{\lambda}$ is a linear operator. Taking the derivative of both sides implies $\frac{df_D}{dt} = \frac{a_i \cdot \frac{dv}{dt}}{\lambda}$. Thus, mapping the temporal derivative of the Doppler spectrum directly yields the target's acceleration vector in space.
4. **Coordinate Rotation**: Finally, apply the corresponding digital subject torso-rotation to normalize the orientation of the acceleration profile.
5. **Output**:  The optimized output matrix will uniquely represent the intensity of true acceleration in the $M \times M$ 2D spatial grid.

*Pros: Potentially filters out static clutter and constant-velocity noise entirely before the expensive spatial mapping optimization process.*
*Cons: Requires modifying the core `doppler_to_bvp()` function. Differentiating raw RF spectrograms is highly susceptible to phase noise, potentially requiring heavy smoothing.*
