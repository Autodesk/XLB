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
To get started with XLB, you can install it using pip:
```bash
pip install xlb
```

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
