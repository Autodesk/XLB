# StreamingOperator & CollisionOperator

In Lattice Boltzmann Method (LBM) simulations, the two core steps are **Streaming** and **Collision**. These operations define how distribution functions move across the grid and how they interact locally.

The following operators abstract these steps:

- `Stream`: handles propagation (pull-based streaming)
- `Collision`: handles local distribution interactions (e.g., BGK)

---

## Stream (Base Class)

The `Stream` operator performs the **streaming step** by pulling values from neighboring grid points, depending on the velocity set.

### Purpose

- Implements the **pull scheme** of LBM streaming.
- Ensures support for both 2D and 3D simulations.


## Collision (Base Class)

The `Collision` operator defines how particles interact locally after streaming, typically by relaxing towards equilibrium.

### Purpose

- Base class for implementing collision models.
- Uses distribution function `f`, equilibrium `feq`, and local properties (`rho`, `u`, etc.).
- Meant to be subclassed (e.g., for `BGK`).

## BGK (Bhatnagar–Gross–Krook)

- **Concept**: BGK is a common collision model where the post-collision distribution is calculated by relaxing toward equilibrium at a single relaxation rate.
- **When to use**:  
    - Standard fluid simulations.  
    - Good balance of performance and accuracy.  

---

### ForcedCollision

- **Concept**: Extends BGK by including external forces (such as gravity, pressure gradients, or body accelerations).  
- **When to use**:  
    - Flows influenced by external fields.  
    - Problems where force-driven effects are important.  

---

### KBC (Karlin–Bösch–Chikatamarla)

- **Concept**: A more advanced model that improves numerical stability and accuracy, especially for high Reynolds number flows.  
- **When to use**:  
    - Simulations at high Reynolds numbers.  
    - Turbulent or under-resolved flows where BGK may become unstable.  

---

## Summary of Support

| Operator        | JAX Support | Warp Support | Typical Use Case                          |
|-----------------|-------------|--------------|------------------------------------------|
| Stream          | Yes         | Yes          | Particle propagation                     |
| BGK             | Yes         | Yes          | Standard fluid simulations               |
| ForcedCollision | Yes         | Yes          | Flows with external forces               |
| KBC             | Yes         | Yes          | High Reynolds number / turbulent flows   |

---

## Choosing the Right Operator

- Start with **BGK** for most general-purpose LBM simulations.  
- Use **ForcedCollision** if external forces significantly affect your system.  
- Switch to **KBC** if you need more stability at high Reynolds numbers or for turbulent flows.  
- The **Stream** operator is always required to handle propagation.
