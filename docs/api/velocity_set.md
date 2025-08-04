# Velocity Sets

In the Lattice Boltzmann Method (LBM), a **velocity set** defines the discrete directions in which particle populations propagate on the lattice each time step. These sets, often denoted as $D_dQ_q$ (e.g., D2Q9 for 2 dimensions, 9 velocities), are fundamental to the simulation's accuracy and stability.

Each velocity set provides the essential components for the streaming step and equilibrium calculations:

- **Dimension ($d$)**: The spatial dimension of the simulation (2D or 3D).
- **Number of Velocities ($q$)**: The quantity of discrete velocity vectors at each lattice node.
- **Velocity Vectors ($e_i$)**: The set of vectors representing allowed directions of particle movement.
- **Weights ($w_i$)**: The scalar weights associated with each velocity vector, crucial for correctly recovering macroscopic fluid behavior.

---

## The `VelocitySet` Class

All velocity set objects in XLB, such as `D2Q9` or `D3Q19`, are instances of a class that inherits from a common `VelocitySet` base. This object is a critical part of the initial simulation setup.

An instance of a velocity set class provides access to its core properties and is configured for the chosen compute backend (e.g., JAX or WARP). It is typically created once and passed to `xlb.init()` at the beginning of a script.

### Usage Example

```python
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy

# 1. Choose and instantiate a velocity set for a 2D simulation
#    This configures it for a specific backend and precision.
velocity_set_2d = xlb.velocity_set.D2Q9(
    compute_backend=ComputeBackend.JAX,
    precision_policy=PrecisionPolicy.FP32FP32
)

# 2. Pass the instantiated object during initialization
xlb.init(
    velocity_set=velocity_set_2d,
    # ... other config options
)

# 3. You can now access its properties if needed
print(f"Dimension: {velocity_set_2d.d}")         # Output: 2
print(f"Num Velocities: {velocity_set_2d.q}")    # Output: 9
```

---

## Conceptual Model: Vectors and Weights

The velocity vectors $e_i$ represent the exact paths particles can take from one lattice node to another in a single time step. The collection of these vectors must satisfy specific mathematical symmetry (isotropy) conditions to ensure that the simulated fluid behaves like a real fluid.

For the standard `D2Q9` set, the 9 vectors correspond to:
- One "rest" particle (zero vector) that stays at the node.
- Four particles moving to the nearest neighbors along the coordinate axes.
- Four particles moving to the diagonal neighbors.

Visually, the vectors point from the center node (4) to the surrounding nodes:
```
  (8) \ (1) / (2)
       \ | /
  (7)---(0)---(3)
       / | \
  (6) / (5) \ (4)
```

The corresponding weights $w_i$ define the contribution of each particle population to the macroscopic density and momentum. They are carefully chosen values that ensure the LBM simulation correctly recovers the Navier-Stokes equations. For `D2Q9`, rest particles have the highest weight, followed by axis-aligned particles, and then diagonal particles.

---

## Predefined Velocity Sets

XLB provides several standard, pre-validated velocity sets suitable for a range of physics problems.

| Class | Dimension | Velocities (q) | Description |
| :--- | :--- | :--- | :--- |
| **`D2Q9`** | 2D | 9 | Standard for 2D flows. Includes a rest particle, 4 axis-aligned directions, and 4 diagonal directions. Provides a good balance of accuracy and efficiency. |
| **`D3Q19`**| 3D | 19 | An efficient choice for 3D flows. Includes a rest particle, 6 axis-aligned directions, and 12 diagonal directions to the faces of the two surrounding cubes. |
| **`D3Q27`**| 3D | 27 | A more comprehensive 3D model. Includes all 27 vectors in a 3x3x3 cube around the node. Offers higher isotropy and accuracy at a greater computational cost. |