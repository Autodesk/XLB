[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub star chart](https://img.shields.io/github/stars/Autodesk/XLB?style=social)](https://star-history.com/#Autodesk/XLB)
<p align="center">
  <img src="https://raw.githubusercontent.com/autodesk/xlb/main/assets/logo-transparent.png" alt="" width="300">
</p>

# XLB: A Differentiable Massively Parallel Lattice Boltzmann Library in Python for Physics-Based Machine Learning

🎉 **Exciting News!** 🎉 XLB version 0.2.0 has been released, featuring a complete rewrite of the library and introducing support for the NVIDIA Warp backend! 
XLB can now be installed via pip: `pip install xlb`.

XLB is a fully differentiable 2D/3D Lattice Boltzmann Method (LBM) library that leverages hardware acceleration. It supports [JAX](https://github.com/google/jax) and [NVIDIA Warp](https://github.com/NVIDIA/warp) backends, and is specifically designed to solve fluid dynamics problems in a computationally efficient and differentiable manner. Its unique combination of features positions it as an exceptionally suitable tool for applications in physics-based machine learning. With the new Warp backend, XLB now offers state-of-the-art performance for even faster simulations.

## Getting Started
To get started with XLB, you can install it using pip. There are different installation options depending on your hardware and needs:

### Basic Installation (CPU-only)
```bash
pip install xlb
```

### Installation with CUDA support (for NVIDIA GPUs)
This installation is for the JAX backend with CUDA support:
```bash
pip install "xlb[cuda]"
```

### Installation with TPU support
This installation is for the JAX backend with TPU support:
```bash
pip install "xlb[tpu]"
```

### Notes:
- For Mac users: Use the basic CPU installation command as JAX's GPU support is not available on MacOS
- The NVIDIA Warp backend is included in all installation options and supports CUDA automatically when available
- The installation options for CUDA and TPU only affect the JAX backend

To install the latest development version from source:

```bash
pip install git+https://github.com/Autodesk/XLB.git
```

The changelog for the releases can be found [here](https://github.com/Autodesk/XLB/blob/main/CHANGELOG.md).

For examples to get you started please refer to the [examples](https://github.com/Autodesk/XLB/tree/main/examples) folder.

## Accompanying Paper

Please refer to the [accompanying paper](https://doi.org/10.1016/j.cpc.2024.109187) for benchmarks, validation, and more details about the library.

## Citing XLB

If you use XLB in your research, please cite the following paper:

```
@article{ataei2024xlb,
  title={{XLB}: A differentiable massively parallel lattice {Boltzmann} library in {Python}},
  author={Ataei, Mohammadmehdi and Salehipour, Hesam},
  journal={Computer Physics Communications},
  volume={300},
  pages={109187},
  year={2024},
  publisher={Elsevier}
}
```

## Key Features
- **Multiple Backend Support:** XLB now includes support for multiple backends including JAX and NVIDIA Warp, providing *state-of-the-art* performance for lattice Boltzmann simulations. Currently, only single GPU is supported for the Warp backend.
- **Integration with JAX Ecosystem:** The library can be easily integrated with JAX's robust ecosystem of machine learning libraries such as [Flax](https://github.com/google/flax), [Haiku](https://github.com/deepmind/dm-haiku), [Optax](https://github.com/deepmind/optax), and many more.
- **Differentiable LBM Kernels:** XLB provides differentiable LBM kernels that can be used in differentiable physics and deep learning applications.
- **Scalability:** XLB is capable of scaling on distributed multi-GPU systems using the JAX backend, enabling the execution of large-scale simulations on hundreds of GPUs with billions of cells.
- **Support for Various LBM Boundary Conditions and Kernels:** XLB supports several LBM boundary conditions and collision kernels.
- **User-Friendly Interface:** Written entirely in Python, XLB emphasizes a highly accessible interface that allows users to extend the library with ease and quickly set up and run new simulations.
- **Leverages JAX Array and Shardmap:** The library incorporates the new JAX array unified array type and JAX shardmap, providing users with a numpy-like interface. This allows users to focus solely on the semantics, leaving performance optimizations to the compiler.
- **Platform Versatility:** The same XLB code can be executed on a variety of platforms including multi-core CPUs, single or multi-GPU systems, TPUs, and it also supports distributed runs on multi-GPU systems or TPU Pod slices.
- **Visualization:** XLB provides a variety of visualization options including in-situ on GPU rendering using [PhantomGaze](https://github.com/loliverhennigh/PhantomGaze).

## Showcase

<p align="center">
  <img src="https://raw.githubusercontent.com/autodesk/xlb/main/assets/wind_turbine.gif" alt="Wind Turbine Simulation" width="800">
</p>
<p align="center">
  Simulation of a wind turbine based on the immersed boundary method.
</p>


<p align="center">
  <img src="https://raw.githubusercontent.com/autodesk/xlb/main/assets/airfoil.gif" width="800">
</p>
<p align="center">
  On GPU in-situ rendering using <a href="https://github.com/loliverhennigh/PhantomGaze">PhantomGaze</a> library (no I/O). Flow over a NACA airfoil using KBC Lattice Boltzmann Simulation with ~10 million cells.
</p>


<p align="center">
  <img src="https://raw.githubusercontent.com/autodesk/xlb/main/assets/car.png" alt="" width="500">
</p>
<p align="center">
<a href=https://www.epc.ed.tum.de/en/aer/research-groups/automotive/drivaer > DrivAer model </a> in a wind-tunnel using KBC Lattice Boltzmann Simulation with approx. 317 million cells
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/autodesk/xlb/main/assets/building.png" alt="" width="700">
</p>
<p align="center">
  Airflow in to, out of, and within a building (~400 million cells)
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/autodesk/xlb/main/assets/XLB_diff.png" alt="" width="900">
</p>
<p align="center">
The stages of a fluid density field from an initial state to the emergence of the "XLB" pattern through deep learning optimization at timestep 200 (see paper for details)
</p>

<br>

<p align="center">
  <img src="https://raw.githubusercontent.com/autodesk/xlb/main/assets/cavity.gif" alt="" width="500">
</p>
<p align="center">
  Lid-driven Cavity flow at Re=100,000 (~25 million cells)
</p>

## Capabilities 

### LBM

- BGK collision model (Standard LBM collision model)
- KBC collision model (unconditionally stable for flows with high Reynolds number)

### Machine Learning

- Easy integration with JAX's ecosystem of machine learning libraries
- Differentiable LBM kernels
- Differentiable boundary conditions

### Lattice Models

- D2Q9
- D3Q19
- D3Q27 (Must be used for KBC simulation runs)

### Compute Capabilities
- Single GPU support for the Warp backend with state-of-the-art performance
- Distributed Multi-GPU support using the JAX backend
- Mixed-Precision support (store vs compute)
- Out-of-core support (coming soon)

### Output

- Binary and ASCII VTK output (based on PyVista library)
- In-situ rendering using [PhantomGaze](https://github.com/loliverhennigh/PhantomGaze) library
- [Orbax](https://github.com/google/orbax)-based distributed asynchronous checkpointing
- Image Output
- 3D mesh voxelizer using trimesh

### Boundary conditions

- **Equilibrium BC:** In this boundary condition, the fluid populations are assumed to be in at equilibrium. Can be used to set prescribed velocity or pressure.

- **Full-Way Bounceback BC:** In this boundary condition, the velocity of the fluid populations is reflected back to the fluid side of the boundary, resulting in zero fluid velocity at the boundary.

- **Half-Way Bounceback BC:** Similar to the Full-Way Bounceback BC, in this boundary condition, the velocity of the fluid populations is partially reflected back to the fluid side of the boundary, resulting in a non-zero fluid velocity at the boundary.

- **Do Nothing BC:** In this boundary condition, the fluid populations are allowed to pass through the boundary without any reflection or modification.

- **Zouhe BC:** This boundary condition is used to impose a prescribed velocity or pressure profile at the boundary.
- **Regularized BC:** This boundary condition is used to impose a prescribed velocity or pressure profile at the boundary. This BC is more stable than Zouhe BC, but computationally more expensive.
- **Extrapolation Outflow BC:** A type of outflow boundary condition that uses extrapolation to avoid strong wave reflections.

- **Interpolated Bounceback BC:** Interpolated bounce-back boundary condition for representing curved boundaries.

## Roadmap

### Work in Progress (WIP)
*Note: Some of the work-in-progress features can be found in the branches of the XLB repository. For contributions to these features, please reach out.*

 - 🌐 **Grid Refinement:** Implementing adaptive mesh refinement techniques for enhanced simulation accuracy.

 - 💾 **Out-of-Core Computations:** Enabling simulations that exceed available GPU memory, suitable for CPU+GPU coherent memory models such as NVIDIA's Grace Superchips (coming soon).


- ⚡ **Multi-GPU Acceleration using [Neon](https://github.com/Autodesk/Neon) + Warp:** Using Neon's data structure for improved scaling.

- 🗜️ **GPU Accelerated Lossless Compression and Decompression**: Implementing high-performance lossless compression and decompression techniques for larger-scale simulations and improved performance.

- 🌡️ **Fluid-Thermal Simulation Capabilities:** Incorporating heat transfer and thermal effects into fluid simulations.

- 🎯 **Adjoint-based Shape and Topology Optimization:** Implementing gradient-based optimization techniques for design optimization.

- 🧠 **Machine Learning Accelerated Simulations:** Leveraging machine learning to speed up simulations and improve accuracy.

- 📉 **Reduced Order Modeling using Machine Learning:** Developing data-driven reduced-order models for efficient and accurate simulations.


### Wishlist
*Contributions to these features are welcome. Please submit PRs for the Wishlist items.*

- 🌊 **Free Surface Flows:** Simulating flows with free surfaces, such as water waves and droplets.

- 📡 **Electromagnetic Wave Propagation:** Simulating the propagation of electromagnetic waves.

- 🛩️ **Supersonic Flows:** Simulating supersonic flows.

- 🌊🧱 **Fluid-Solid Interaction:** Modeling the interaction between fluids and solid objects.

- 🧩 **Multiphase Flow Simulation:** Simulating flows with multiple immiscible fluids.

- 🔥 **Combustion:** Simulating combustion processes and reactive flows.

- 🪨 **Particle Flows and Discrete Element Method:** Incorporating particle-based methods for granular and particulate flows.

- 🔧 **Better Geometry Processing Pipelines:** Improving the handling and preprocessing of complex geometries for simulations.

