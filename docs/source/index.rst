.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :align: left

.. image:: https://img.shields.io/github/stars/Autodesk/XLB?style=social
   :target: https://star-history.com/#Autodesk/XLB
   :align: left

.. raw:: html

   <p align="center">
     <img src="https://raw.githubusercontent.com/autodesk/xlb/main/assets/logo-transparent.png" alt="" width="300">
   </p>

XLB: A Differentiable Massively Parallel Lattice Boltzmann Library in Python for Physics-Based Machine Learning
===============================================================================================================

.. raw:: html

   <script>
     /* Hide right sidebar (page-local contents) */
     document.addEventListener("DOMContentLoaded", function () {
         const rightSidebar = document.querySelector("div.bd-sidebar-secondary");
         if (rightSidebar) {
             rightSidebar.style.display = "none";
         }
     });
   </script>





🎉 **Exciting News!** 🎉 
XLB version 0.2.0 has been released, featuring a complete rewrite of the library and introducing support for the NVIDIA Warp backend! 
XLB can now be installed via pip: `pip install xlb`.

XLB is a fully differentiable 2D/3D Lattice Boltzmann Method (LBM) library that leverages hardware acceleration. 
XLB supports `JAX <https://github.com/google/jax>`_ and `NVIDIA Warp <https://github.com/NVIDIA/warp>`_ backends,
and is specifically designed to solve fluid dynamics problems in a computationally efficient and differentiable manner. 
Its unique combination of features positions it as an exceptionally suitable tool for applications in physics-based machine learning. 
With the new Warp backend, XLB now offers state-of-the-art performance for even faster simulations.

Accompanying Paper
------------------

Please refer to the `accompanying paper <https://doi.org/10.1016/j.cpc.2024.109187>`_ for benchmarks, validation, and more details.

Citing XLB
----------

If you use XLB in your research, please cite:

.. code-block:: bibtex

   @article{ataei2024xlb,
     title={{XLB}: A differentiable massively parallel lattice {Boltzmann} library in {Python}},
     author={Ataei, Mohammadmehdi and Salehipour, Hesam},
     journal={Computer Physics Communications},
     volume={300},
     pages={109187},
     year={2024},
     publisher={Elsevier}
   }

Key Features
============

- **Multiple Backend Support:** XLB now includes support for multiple backends including JAX and NVIDIA Warp, providing *state-of-the-art* performance for lattice Boltzmann simulations. Currently, only single GPU is supported for the Warp backend.
- **Integration with JAX Ecosystem:** The library can be easily integrated with JAX's robust ecosystem of machine learning libraries such as `Flax <https://github.com/google/flax>`_, `Haiku <https://github.com/deepmind/dm-haiku>`_, `Optax <https://github.com/deepmind/optax>`_, and many more.
- **Differentiable LBM Kernels:** XLB provides differentiable LBM kernels that can be used in differentiable physics and deep learning applications.
- **Scalability:** XLB is capable of scaling on distributed multi-GPU systems using the JAX backend, enabling the execution of large-scale simulations on hundreds of GPUs with billions of cells.
- **Support for Various LBM Boundary Conditions and Kernels:** XLB supports several LBM boundary conditions and collision kernels.
- **User-Friendly Interface:** Written entirely in Python, XLB emphasizes a highly accessible interface that allows users to extend the library with ease and quickly set up and run new simulations.
- **Leverages JAX Array and Shardmap:** The library incorporates the new JAX array unified array type and JAX shardmap, providing users with a numpy-like interface. This allows users to focus solely on the semantics, leaving performance optimizations to the compiler.
- **Platform Versatility:** The same XLB code can be executed on a variety of platforms including multi-core CPUs, single or multi-GPU systems, TPUs, and it also supports distributed runs on multi-GPU systems or TPU Pod slices.
- **Visualization:** XLB provides a variety of visualization options including in-situ on GPU rendering using `PhantomGaze <https://github.com/loliverhennigh/PhantomGaze>`_.

Capabilities
============

LBM
---

- BGK collision model (Standard LBM collision model)
- KBC collision model (unconditionally stable for flows with high Reynolds number)

Machine Learning
----------------

- Easy integration with JAX's ecosystem of machine learning libraries
- Differentiable LBM kernels
- Differentiable boundary conditions

Lattice Models
--------------

- D2Q9
- D3Q19
- D3Q27 (Must be used for KBC simulation runs)

Compute Capabilities
--------------------

- Single GPU support for the Warp backend with state-of-the-art performance
- Distributed Multi-GPU support using the JAX backend
- Mixed-Precision support (store vs compute)
- Out-of-core support (coming soon)

Output
------

- Binary and ASCII VTK output (based on PyVista library)
- In-situ rendering using `PhantomGaze <https://github.com/loliverhennigh/PhantomGaze>`_ library
- `Orbax <https://github.com/google/orbax>`_-based distributed asynchronous checkpointing
- Image Output
- 3D mesh voxelizer using trimesh

Boundary Conditions
-------------------

- **Equilibrium BC:** In this boundary condition, the fluid populations are assumed to be at equilibrium. Can be used to set prescribed velocity or pressure.
- **Full-Way Bounceback BC:** The velocity of the fluid populations is reflected back to the fluid side of the boundary, resulting in zero fluid velocity at the boundary.
- **Half-Way Bounceback BC:** The velocity of the fluid populations is partially reflected back, resulting in a non-zero fluid velocity at the boundary.
- **Do Nothing BC:** The fluid populations are allowed to pass through the boundary without any reflection or modification.
- **Zouhe BC:** Used to impose a prescribed velocity or pressure profile at the boundary.
- **Regularized BC:** More stable than Zouhe BC, but computationally more expensive.
- **Extrapolation Outflow BC:** A type of outflow boundary condition that uses extrapolation to avoid strong wave reflections.
- **Interpolated Bounceback BC:** Interpolated bounce-back boundary condition for representing curved boundaries.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   installation.md
   tutorials.md
   api_reference.md
   contributing.md
   examples.md
