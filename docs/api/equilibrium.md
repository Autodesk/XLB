# Equilibrium

Equilibrium operators define the **target distribution functions** in the Lattice Boltzmann Method (LBM).  
They represent the state towards which the fluid relaxes during the **collision step** and are essential for computing correct fluid behavior.  

---

## Overview

- In LBM, particles are represented by discrete distributions along predefined velocity directions.  
- After collisions, these distributions relax toward an **equilibrium distribution**.  
- The chosen equilibrium model determines how density and velocity fields map to these distributions.  

The equilibrium distribution ensures that:

- Mass and momentum are conserved.  
- Macroscopic fluid quantities (like density and velocity) are correctly reproduced.  

---

## QuadraticEquilibrium

The **QuadraticEquilibrium** is the default and most widely used equilibrium model in LBM.  

- **Mathematical Basis**:  
  Approximates the Maxwell–Boltzmann distribution using a second-order Hermite polynomial expansion.  
- **Inputs**:  
    - **Density (ρ):** scalar field  
    - **Velocity (u):** vector field (2D or 3D depending on lattice)  
- **Outputs**:  
    - **Equilibrium distribution functions (feq):** one per velocity direction in the velocity set  

This model ensures correct recovery of the **Navier–Stokes equations** for incompressible flows.  

---