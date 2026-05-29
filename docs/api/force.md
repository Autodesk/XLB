# Force

The **Force** module provides operators for handling forces in Lattice Boltzmann Method (LBM) simulations.  
These operators cover both **adding external forces to the fluid** and **computing the forces exerted by the fluid on solid boundaries**.

---

## Overview

Forces in LBM simulations can act in two distinct ways:

- **Body forces**: External effects applied to the fluid domain (e.g., gravity, acceleration, electromagnetic forces).
- **Boundary forces**: Hydrodynamic forces exerted by the fluid on immersed solid objects (e.g., drag or lift).

XLB provides two operators to handle these cases:

| Operator            | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| **ExactDifference** | Adds body forces to the fluid using the Exact Difference Method (EDM).  |
| **MomentumTransfer** | Computes the force exerted by the fluid on boundaries via momentum exchange. |

Both operators support **JAX** and **Warp** compute backends.

---

## ExactDifference

The **Exact Difference** operator incorporates external body forces into the fluid dynamics without breaking stability or conservation.  
It uses the *Exact Difference Method (EDM)* introduced by Kupershtokh (2004), which is a stable and widely used approach in LBM.

- **Purpose**: Apply external forces uniformly to the fluid domain.
- **Use cases**: Gravity-driven flows, accelerated channel flows, magnetohydrodynamics.
- **Method**:  
    - Computes the difference between equilibrium distributions with and without a velocity shift caused by the force.  
    - Corrects the post-collision distribution functions by adding this difference.  
- **Notes**:  
    - Currently limited to constant force vectors (not spatially varying fields).  
    - Works seamlessly with the chosen velocity set (e.g., D2Q9, D3Q19, D3Q27).

---

## MomentumTransfer

The **Momentum Transfer** operator measures the hydrodynamic force exerted by the fluid on solid boundaries.  
It implements the *momentum exchange method*, introduced by Ladd (1994) and extended by Mei et al. (2002) for curved boundaries.

- **Purpose**: Compute forces such as drag, lift, or pressure exerted on immersed boundaries.
- **Use cases**: Flow around cylinders, aerodynamic lift, sedimentation of particles, boundary stress analysis.
- **Method**:  
    - Uses the post-collision distributions and applies boundary conditions.  
    - Identifies boundary nodes and their missing directions.  
    - Computes the exchanged momentum between the fluid and the solid at each node.  
- **Notes**:  
    - Should be applied **after boundary conditions** are imposed.  
    - Compatible with advanced no-slip schemes (e.g., Bouzidi) for curved geometries.  
    - Returns either a field of forces (JAX) or the net force (Warp).  

---

## Summary

- **ExactDifference** → Drives the fluid using external body forces (e.g., gravity, acceleration).  
- **MomentumTransfer** → Measures hydrodynamic forces acting on immersed solid geometries.  

Together, they enable both **forcing the fluid** and **analyzing boundary interactions**, making them essential for advanced LBM simulations.

---
