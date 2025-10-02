# Core Concepts in XLB

In the [Introduction](./introduction.md), we covered the basic theory of the Lattice Boltzmann Method (LBM). Now, let's explore how that theory is translated into the practical software components you will use to build every XLB simulation.

Understanding these core concepts is the key to unlocking the full power and flexibility of the library. Each concept is represented by one or more Python objects that serve as the building blocks of your simulation.

---

## The Grid and Velocity Set

The foundation of any LBM simulation is the lattice itself. In XLB, this is defined by two components:

* **`Grid`**: This object represents the discrete simulation domain. It holds the shape of your computational grid (e.g., 500x500) and provides helpful methods for identifying regions, such as the boundaries of the domain using `grid.bounding_box_indices()`. It represents the collection of all spatial nodes $\vec{x}$.

* **`Velocity Set`**: This defines the mesoscopic "pathways" particles can take on the lattice. It specifies the set of discrete velocity vectors $\vec{e}_i$ and their corresponding weights $w_i$. The choice of velocity set depends on the physics you want to simulate and the dimensionality of your problem. Common choices include:
    * **`D2Q9`**: For 2D simulations (D_imension=2, Q_uantity-of-velocities=9).
    * **`D3Q19`** or **`D3Q27`**: For 3D simulations.

Together, the `Grid` and `Velocity Set` establish the space-time arena where your simulation will unfold.

*For more details, see the API Reference for [`Grid`](../api/grid.md) and [`Velocity Set`](../api/velocity_set.md).*

## The Distribution Function ($f_i$)

The central data structure in an XLB simulation is the **Distribution Function**. This is a large array that stores the value of the particle population $f_i(\vec{x}, t)$ for every discrete velocity $i$ at every node $\vec{x}$ on the grid.

While it is the most important piece of data, you will rarely modify it directly. Instead, the `Stepper` object (see below) reads from and writes to the distribution function arrays during the collision and streaming steps. All the complex fluid behavior emerges from the evolution of this single data structure.

*For more details, see the API Reference for [`Distribution`](../api/distribution.md).*

## Boundary Conditions

A fluid simulation is often defined by its interactions with the boundaries of the domain. In XLB, these interactions are defined by a list of `BoundaryCondition` objects. Each object targets a specific set of grid nodes and enforces a physical rule.

XLB provides several common boundary conditions out of the box.

Defining the correct boundary conditions is one of the most critical steps in setting up a valid simulation.

*For more details, see the API Reference for [`Boundary Condition`](../api/bc.md) and our [How-to Guides](./how-to-bc.md).*

## The Stepper

The `Stepper` is the engine of your simulation. It is the object that orchestrates the core LBM algorithm and advances the simulation forward in time. When you call the stepper in your main loop, it performs a complete time step, which involves:

1.  **Collision**: Applying the collision model (e.g., BGK) to every node on the grid. This relaxes the distribution functions $f_i$ towards their local equilibrium $f_i^{eq}$.
2.  **Boundary Conditions**: Applying the logic from all the `BoundaryCondition` objects you have defined.
3.  **Streaming**: Propagating the post-collision particle populations $f_i^*$ to their neighboring nodes along their velocity vectors $\vec{e}_i$.

The `IncompressibleNavierStokesStepper` is the primary stepper used for most standard fluid dynamics problems.

*For more details, see the API Reference for [`Stepper`](../api/stepper.md) and [`Streaming and Collision`](../api/streaming_and_collision.md).*

## Macroscopic Variables ($\rho, \vec{u}$)

While the LBM simulation operates on the mesoscopic distribution functions $f_i$, the results we are interested in are usually macroscopic quantities like fluid density $\rho$ and velocity $\vec{u}$.

The process of calculating these from $f_i$ is called "taking moments." XLB provides a `Macroscopic` computer for this exact purpose. You provide it with the current distribution function, and it returns the macroscopic fields by computing the sums over the discrete velocities:

* **Density:** $\rho = \sum_i f_i$
* **Momentum:** $\rho\vec{u} = \sum_i f_i \vec{e}_i$

This calculation is typically done in the post-processing stage of your simulation loop to analyze and visualize the results.

*For more details, see the API Reference for [`Macroscopic`](../api/macroscopic.md).*

## Compute Backends and Precision

A key feature of XLB is its ability to run the same simulation code on different high-performance backends. This is configured globally via `xlb.init()`.

* **`ComputeBackend`**: This tells XLB which engine to use for all numerical computations (e.g., `JAX` for CPU/GPU/TPU or `WARP` for high-performance NVIDIA GPU kernels).
* **`PrecisionPolicy`**: This controls the numerical precision (e.g., 32-bit `FP32` or 64-bit `FP64` floats) used for the simulation, allowing you to balance performance with accuracy.

This design allows you to write your physics code once and seamlessly switch the underlying computation engine to best suit your hardware.