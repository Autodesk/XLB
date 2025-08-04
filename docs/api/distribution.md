# Distribution

The `distribution` module provides tools for distributing **lattice Boltzmann operators** across multiple devices (e.g., GPUs or TPUs) using [JAX sharding](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).  
This enables simulations to run in parallel while ensuring correct **halo communication** between device partitions.

---

## Overview

In lattice Boltzmann methods (LBM), each lattice site’s distribution function depends on its neighbors.  
When running on multiple devices, the domain is split (sharded) across them, requiring **data exchange at the boundaries** after each step.

The `distribution` module handles:

- **Sharding operators** across devices.  
- **Exchanging boundary (halo) data** between devices.  
- Supporting stepper operators (like `IncompressibleNavierStokesStepper`) with or without boundary conditions.  

---

## Functions

### `distribute_operator`

```python
distribute_operator(operator, grid, velocity_set, num_results=1, ops="permute")
```
Wraps an operator to run in distributed fashion.

## Parameters

- **operator** (`Operator`)  
  The LBM operator (e.g., collision, streaming).

- **grid**  
  Grid definition with device mesh info (`grid.global_mesh`, `grid.shape`, `grid.nDevices`).

- **velocity_set**  
  Velocity set defining the LBM stencil (e.g., D2Q9, D3Q19).

- **num_results** (`int`, default=`1`)  
  Number of results returned by the operator.

- **ops** (`str`, default=`"permute"`)  
  Communication scheme. Currently supports `"permute"` for halo exchange.

---

## Details

- Uses **`shard_map`** to parallelize across devices.  
- Applies **halo communication** via `jax.lax.ppermute`:
  - Sends right-edge values to the left neighbor.  
  - Sends left-edge values to the right neighbor.  
- Returns a **JIT-compiled distributed operator**.

---

### `distribute`

```python
distribute(operator, grid, velocity_set, num_results=1, ops="permute")

```

## Description

Decides how to distribute an operator or stepper.

---

## Parameters

Same as **`distribute_operator`**.

---

## Special Case: `IncompressibleNavierStokesStepper`

- Checks if boundary conditions require **post-streaming updates**:
    - If **yes** → only the `.stream` operator is distributed.  
    - If **no** → the entire stepper is distributed.  

---

## Example

```python
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.distribution import distribute

# Create stepper
stepper = IncompressibleNavierStokesStepper(...)

# Distribute across devices
distributed_stepper = distribute(stepper, grid, velocity_set)

# Run simulation
state = distributed_stepper(state)
```
