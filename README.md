<p align="center">
  <img src="assets/logo-transparent.png" alt="" width="700">
</p>

# XLB: Hardware-Accelerated, Scalable, and Differentiable Lattice Boltzmann Simulation Framework based on JAX

XLB (Accelerated LB) is a hardware-accelerated fully differentiable 2D/3D Lattice Boltzmann Method solver based on the JAX library. It is designed to solve fluid dynamics problems in a computationally efficient and differentiable manner.

## Documentation
Coming soon
## Showcase

The following examples showcase the capabilities of XLB:

<p align="center">
  <img src="assets/cavity.gif" alt="" width="500">
</p>
<p align="center">
  Lid-driven Cavity flow at Re=100,000 (~25 million voxels)
</p>

<p align="center">
  <img src="assets/car.png" alt="" width="500">
</p>
<p align="center">
  Q-criterion over a DrivAer model
</p>

<p align="center">
  <img src="assets/airfoil.png" width="500">
</p>
<p align="center">
  Q-criterion over a NACA airfoil
</p>

## Capabilities 

### LBM
- BGK collision model (Standard LBM collision model)
- KBC collision model (unconditionally stable for flows with high Reynolds number)

### Lattice Models
- D2Q9
- D3Q19
- D3Q27 (Must be used for KBC simulation runs)

### Output
- Binary VTK output
- ASCII VTK output
- Image Output (2D and 3D slice)
- 3D mesh voxelizer using trimesh

### Boundary conditions
- Equilibrium BC: In this boundary condition, the fluid populations are assumed to be in at equilibrium. Can be used to set prescribed velocity or pressure.

- Full-Way Bounceback BC: In this boundary condition, the velocity of the fluid populations is reflected back to the fluid side of the boundary, resulting in zero fluid velocity at the boundary.

- Half-Way Bounceback BC: Similar to the Full-Way Bounceback BC, in this boundary condition, the velocity of the fluid populations is partially reflected back to the fluid side of the boundary, resulting in a non-zero fluid velocity at the boundary.

- Do Nothing BC: In this boundary condition, the fluid populations are allowed to pass through the boundary without any reflection or modification.

- Zouhe BC: This boundary condition is used to impose a prescribed velocity or pressure profile at the boundary.
- Regularized BC: This boundary condition is used to impose a prescribed velocity or pressure profile at the boundary. This BC is more stable than Zouhe BC, but computationally more expensive.
- Extrapolation Outflow BC: A type of outflow boundary condition that uses extrapolation to avoid strong wave reflections.

### Compute Capabilities
- Distributed Multi-GPU support
- JAX shard-map and JAX Array support
- Mixed-Precision support (store vs compute)

## Installation Guide

To install XLB, you can run the following commands:

```bash
pip install --upgrade pip

# For CPU run
pip install --upgrade "jax[cpu]"

# For GPU run

# CUDA 12 and cuDNN 8.8 or newer.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 and cuDNN 8.6 or newer.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Please refer to https://github.com/google/jax for the latest installation documentation

# Run dependencies
pip install jmp pyvista numpy matplotlib Rtree trimesh jmp
```
## Citing XLB
Accompanying publication coming soon:

**M. Ataei, H. Salehipour**. XLB: Hardware-Accelerated, Scalable, and Differentiable Lattice Boltzmann Simulation Framework based on JAX. TBA
