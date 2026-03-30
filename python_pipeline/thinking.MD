Based on the Widar3.0 pipeline documentation, here is an explanation of what the 2D spatial velocity vector is and how it is created and used in the system:

What is the 2D Spatial Velocity Vector?
The 2D spatial velocity vector, denoted as $v = [v_x, v_y]$, is a mathematical representation of how fast and in what direction a human body part is moving through a 2-dimensional space (typically the plane parallel to the floor).

$v_x$: The physical speed of movement along the X-axis (e.g., in meters per second).
$v_y$: The physical speed of movement along the Y-axis.
In the context of Widar3.0, the goal of the system is to figure out the distribution of these 2D velocity vectors across the human body (which is called the Body Velocity Profile or BVP). Since different parts of the body (arms, torso, legs) might be moving at different speeds and in different directions during a gesture, the BVP represents the "energy" or intensity of movement at various possible $(v_x, v_y)$ combinations.

How it is Created and Used
The actual velocity vector of the person isn't directly measured; rather, the Wi-Fi receivers measure Doppler frequency shifts. The system has to reconstruct the 2D spatial velocity vectors from these 1D frequency shifts. Here is how that process works:

1. Setting up a Velocity Resolution Grid Instead of calculating continuous, infinite possibilities of movement, the system discretizes the possible velocities into a grid (e.g., a $20 \times 20$ matrix). This grid might represent velocities ranging from $-2.0 \text{ m/s}$ to $+2.0 \text{ m/s}$ on both the X and Y axes. Each "pixel" in this grid represents a specific discrete 2D spatial velocity vector $v = [v_x, v_y]$.

2. Calculating the Expected Doppler Shift (The VDM Matrix) For every possible velocity vector in that grid, the system calculates what the expected Doppler frequency shift ($f_D$) would be at each of the Wi-Fi receivers. It does this using a dot product: $$ f_{D_i} = \frac{a_i \cdot v}{\lambda} $$

$a_i$ is a projection vector that models the physical geometry (the angles between the Transmitter, the person's torso, and Receiver $i$).
$v$ is the 2D spatial velocity vector $[v_x, v_y]$ from the grid.
$\lambda$ is the wavelength of the Wi-Fi signal (e.g., 5.825 GHz).
This pre-computation creates a massive mapping tensor called the Velocity-to-Doppler Mapping (VDM) Matrix. It effectively says: "If a body part is moving at velocity $v_x, v_y$, it will show up in frequency bin $X$ at receiver $1$, frequency bin $Y$ at receiver $2$," etc.

3. The Optimization Solver (Finding the true velocities) Because a human body has multiple moving parts creating a messy, overlapping combined Doppler signal, the system uses an optimization algorithm (SciPy's SLSQP) to work backward.

It looks at the actual averaged Doppler spectrum recorded by all the receivers.
It uses the VDM mapping matrix to iteratively guess which combination of 2D velocity vectors $[v_x, v_y]$ in the grid would produce a Doppler signal that matches the real-world measurements.
It applies a "sparsity penalty," assuming human movement is sparse (you don't move in all directions at all speeds simultaneously).
The final output of this optimizer is the BVP matrix, which tells you the exact energy intensity for every 2D spatial velocity vector in the grid, allowing the system to "see" the motion of the gesture in 2D space regardless of where the Wi-Fi receivers were placed.