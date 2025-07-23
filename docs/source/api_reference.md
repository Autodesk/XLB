# XLB API Reference

XLB is a flexible and performant lattice Boltzmann fluid solver built for multi-backend (JAX and Warp) simulation. This reference provides a structured overview of how to use the API, based on working code examples.

---


## Overview

XLB is structured around a modular design:

- **Velocity Sets:** e.g., `D2Q9`, `D3Q19`, `D3Q27`  
- **Compute Backends:** `JAX`, `WARP`  
- **Precision Policies:** Controls floating point behavior (`FP32FP32`, `FP64FP64`, etc.)  
- **Grid Factory:** Creates simulation domains  
- **Operators:** Include steppers, boundary conditions, macroscopic quantity extractors  
- **Distribute:** JAX multi-GPU or distributed computing support  

---

## Simulation Pipeline

### 1. Initialization

```python
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy

backend = ComputeBackend.JAX  # or ComputeBackend.WARP
precision = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q19(
    precision_policy=precision,
    compute_backend=backend,
)

xlb.init(
    velocity_set=velocity_set,
    default_backend=backend,
    default_precision_policy=precision,
)
```
---

### 2. Grid Creation
```python
from xlb.grid import grid_factory

grid_shape = (64, 64, 64)
grid = grid_factory(grid_shape, compute_backend=backend)
```

### 3. Boundary Indexing
```python
box = grid.bounding_box_indices(remove_edges=True)

inlet = box["left"]
outlet = box["right"]
walls = [box["top"][i] + box["bottom"][i] for i in range(velocity_set.d)]
```
Use numpy.unique() or list comprehensions to merge sets.

### 4. Boundary Conditions

From xlb.operator.boundary_condition, supported boundary conditions include:
FullwayBounceBackBC
HalfwayBounceBackBC
RegularizedBC
ExtrapolationOutflowBC

**Defining a Dynamic Velocity Profile**
```python
@wp.func
def profile(index: wp.vec3i):
    return wp.vec(0.05, 0.0, 0.0, length=1)

bc_inlet = RegularizedBC("velocity", profile=profile, indices=inlet)
```
**Defining a Static Velocity**
```python
bc_wall = RegularizedBC("velocity", prescribed_value=(0.0, 0.0, 0.0), indices=walls)
```
Combine:
```python
boundary_conditions = [bc_wall, bc_inlet, bc_outlet]
```


### 5. Stepper Setup

from xlb.operator.stepper import IncompressibleNavierStokesStepper
```python
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="BGK",  # or "KBC"
    force_vector=force_vector,  # Optional
)
```

### 6. Field Preparation
```python
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

from xlb.helper import initialize_eq

f_0 = initialize_eq(
    f_0, grid, velocity_set, precision, backend, u=initial_velocity
)
```

### 7. Running the Simulation
```python
for step in range(num_steps):
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap buffers

    if step % post_process_interval == 0:
        post_process(step, f_0)
```

### 8. Post-Processing
```python
from xlb.operator.macroscopic import Macroscopic

macro = Macroscopic(
    compute_backend=backend,
    precision_policy=precision,
    velocity_set=velocity_set
)
rho, u = macro(f_current)
```
Saving fields:
```python
from xlb.utils import save_image, save_fields_vtk

save_image(u[0][:, mid_y, :], timestep=step)
save_fields_vtk({"u_x": u[0], "rho": rho[0]}, timestep=step)
```

## Distributing Computation
```python
from xlb.distribute import distribute

stepper = IncompressibleNavierStokesStepper(...)
stepper = distribute(stepper, grid, velocity_set)
```
⚠️ Note: Distributed mode requires ComputeBackend.JAX. Warp is not supported.

## Supported Velocity Sets

| Velocity Set | Dimensions | Use Case        |
|--------------|------------|-----------------|
| D2Q9         | 2D         | Benchmark cases |
| D3Q19        | 3D         | General use     |
| D3Q27        | 3D         | High accuracy   |

**Create a velocity set:**

```python
velocity_set = xlb.velocity_set.D3Q27(precision_policy, compute_backend)
```

## Backends and Precision Policies

### Compute Backends

- `ComputeBackend.JAX`: JAX-based backend (CPU/GPU)
- `ComputeBackend.WARP`: CUDA-accelerated backend via NVIDIA Warp

### Precision Policies

| Policy    | Compute | Storage |
|-----------|---------|---------|
| FP32FP32  | float32 | float32 |
| FP64FP64  | float64 | float64 |

> Use these based on hardware support and performance needs.


## Utilities

- `save_image(field_slice, timestep)`  
  Saves 2D PNG slices of a field.

- `save_fields_vtk(fields, timestep)`  
  Outputs full 3D VTK data.

- `initialize_eq(...)`  
  Initializes distributions using macroscopic profiles.

- `wp.to_jax(warp_array)`  
  Converts a Warp array to a JAX ndarray.

- `vonKarman_loglaw_wall(yplus)`  
  Returns the analytical log-law velocity profile for wall-bounded flows.

---

## Appendix: Example Presets

### Lid-Driven Cavity

- Uses `D2Q9`
- Regularized boundary condition for inlet
- Supports distributed setup

### Channel Flow

- Uses external forcing via `force_vector`
- Includes log-law analysis and DNS reference data

### Obstacle Flow

- Demonstrates bounce-back boundary condition on an internal obstacle
- Shows custom mask creation for complex geometries
