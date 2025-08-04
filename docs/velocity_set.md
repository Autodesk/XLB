# Velocity Sets

In the Lattice Boltzmann Method (LBM), a **velocity set** defines the discrete directions in which particle distributions propagate on the lattice at each time step. These discrete velocities approximate the continuous fluid velocity space and are fundamental to accurately simulating fluid flow.

Each velocity set specifies:  
- The **dimension** (2D or 3D) of the lattice.  
- The **number of discrete velocities (Q)** per lattice node.  
- The **velocity vectors** representing allowed particle movement directions.  
- The **weights** associated with each velocity, ensuring correct macroscopic behavior.

---

## Base Class: `VelocitySet`

All velocity sets inherit from the `VelocitySet` base class, which manages:  
- The dimension and number of velocities.  
- The velocity vectors and their weights.  
- Backend compatibility for accelerated computation (JAX or WARP).  

Users typically instantiate predefined velocity sets rather than the base class directly.

---

## What Do These Velocities Represent?

Each velocity vector corresponds to a possible direction a fluid particle can move from a lattice node during streaming. For example, in a 2D lattice:  

- A zero vector (rest particle) means the particle stays at the node.  
- Unit vectors along coordinate axes represent movement to neighboring nodes (up, down, left, right).  
- Diagonal vectors represent movement to diagonal neighbors.  

The collection of these vectors must satisfy mathematical conditions to recover the Navier–Stokes equations correctly at the macroscopic scale, which govern fluid dynamics.

The associated weights define the relative probability or influence of particles moving in each direction, balancing isotropy and stability.

---

## Predefined Velocity Sets

| Class   | Dimension | Velocities (Q) | Description                            |
|---------|-----------|----------------|------------------------------------|
| `D2Q9`  | 2D        | 9              | Rest + 4 axis-aligned + 4 diagonal directions; standard for 2D flow simulations. |
| `D3Q19` | 3D        | 19             | Combines rest, axis-aligned, and select diagonal directions for efficient 3D flows. |
| `D3Q27` | 3D        | 27             | Full cubic lattice with all combinations of [-1,0,1] in 3D; higher isotropy and accuracy. |

---