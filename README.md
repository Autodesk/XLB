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

## Running a Basic Example: Lid-Driven Cavity Simulation

```python
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, check_bc_overlaps
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.velocity_set import D2Q9
import numpy as np

class LidDrivenCavity2D:
    def __init__(self, omega, grid_shape, velocity_set, backend, precision_policy):
        # Initialize the backend for the XLB library with specified settings
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

        # Store the grid shape and other configurations
        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.backend = backend
        self.precision_policy = precision_policy

        # Create fields for the simulation (e.g., grid, distribution functions, masks)
        self.grid, self.f_0, self.f_1, self.missing_mask, self.bc_mask = create_nse_fields(grid_shape)
        self.stepper = None
        self.boundary_conditions = []

        # Set up the simulation by initializing boundary conditions, maskers, fields, and the stepper
        self._setup(omega)

    def _setup(self, omega):
        # Set up the boundary conditions, boundary masker, initialize fields, and create the stepper
        self.setup_boundary_conditions()
        self.setup_boundary_masker()
        self.initialize_fields()
        self.setup_stepper(omega)

    def define_boundary_indices(self):
        # Define the indices of the boundary regions of the grid
        box = self.grid.bounding_box_indices()  # Get the bounding box indices of the grid
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)  # Get bounding box indices without the edges

        # Define lid and walls for boundary conditions
        lid = box_no_edge["top"]  # Top boundary represents the moving lid
        walls = [box["bottom"][i] + box["left"][i] + box["right"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()  # Combine and remove duplicate indices for walls
        return lid, walls

    def setup_boundary_conditions(self):
        # Define the boundary indices for the lid and the walls
        lid, walls = self.define_boundary_indices()

        # Set up boundary conditions for the lid and the walls
        bc_top = EquilibriumBC(rho=1.0, u=(0.02, 0.0), indices=lid)  # Lid moves with a velocity of (0.02, 0.0)
        bc_walls = HalfwayBounceBackBC(indices=walls)  # Walls use a halfway bounce-back boundary condition

        # Store the boundary conditions in a list
        self.boundary_conditions = [bc_walls, bc_top]

    def setup_boundary_masker(self):
        # Check the boundary condition list for duplicate indices before creating the boundary mask
        check_bc_overlaps(self.boundary_conditions, self.velocity_set.d, self.backend)

        # Create a boundary masker to generate masks for the boundary and missing populations
        indices_boundary_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )

        # Apply the boundary masker to create the boundary condition mask and the missing mask
        self.bc_mask, self.missing_mask = indices_boundary_masker(self.boundary_conditions, self.bc_mask, self.missing_mask)

    def initialize_fields(self):
        # Initialize the equilibrium distribution function for the fluid based on initial conditions
        self.f_0 = initialize_eq(self.f_0, self.grid, self.velocity_set, self.precision_policy, self.backend)

    def setup_stepper(self, omega):
        # Create the time-stepping object for solving the incompressible Navier-Stokes equations
        self.stepper = IncompressibleNavierStokesStepper(omega, boundary_conditions=self.boundary_conditions)

    def run(self, num_steps, post_process_interval=100):
        # Run the simulation for a given number of steps
        for i in range(num_steps):
            # Perform one step of the simulation: swap distribution functions between f_0 and f_1
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0  # Swap references for next step

            # Periodically perform post-processing or at the final step
            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def post_process(self, i):
        # Placeholder for post-processing logic (e.g., saving output, visualizations)
        print(f"Post-processing at timestep {i}")

# Define simulation parameters
# The grid size, backend, precision, velocity set, and relaxation factor (omega) are defined here
grid_size = 500
grid_shape = (grid_size, grid_size)
# Select the compute backend between Warp or JAX
backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = D2Q9(precision_policy=precision_policy, backend=backend)
omega = 1.6

# Create an instance of the LidDrivenCavity2D class and run the simulation
simulation = LidDrivenCavity2D(omega, grid_shape, velocity_set, backend, precision_policy)
simulation.run(num_steps=5000, post_process_interval=1000)
```

For more examples please refer to the [examples](https://github.com/Autodesk/XLB/tree/main/examples) folder.

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

