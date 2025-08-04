# BoundaryCondition (Base Class)

The `BoundaryCondition` class is the base class for implementing all boundary conditions in Lattice Boltzmann Method (LBM) simulations.  
It extends the generic `Operator` class and provides the foundational structure for applying boundary logic in different simulation stages.

## 📌 Purpose

In LBM simulations, boundary conditions (BCs) define how the simulation behaves at domain edges — walls, inlets, outlets, etc.  
`BoundaryCondition` provides:

- A uniform interface for implementing BCs  
- GPU/TPU-compatible kernels using **JAX** or **Warp**  
- Support for auxiliary data (e.g., prescribed velocities)  
- Integration with velocity sets, precision policy, and compute backends  



## 🧩 Key Parameters

| Argument           | Description                                                         |
|--------------------|---------------------------------------------------------------------|
| `implementation_step` | When the BC is applied: `COLLISION` or `STREAMING`                 |
| `velocity_set`         | Type of LBM velocity set (optional, uses default if not provided) |
| `precision_policy`     | Controls numerical precision (optional)                           |
| `compute_backend`      | Either `JAX` or `WARP` (optional)                                 |
| `indices`              | Grid indices where the BC applies                                 |
| `mesh_vertices`        | Optional mesh information for mesh-aware BCs                      |


<!-- ## ⚙️ Features and Flags

| Flag                  | Description                                                        |
|-----------------------|--------------------------------------------------------------------|
| `needs_padding`       | True if the BC requires boundary padding in all directions         |
| `needs_mesh_distance` | True if the BC needs geometric distance to a mesh                  |
| `needs_aux_init`      | Indicates if the BC uses auxiliary data (e.g., prescribed values)  |
| `num_of_aux_data`     | How many auxiliary values are needed (if any)                      |
| `needs_aux_recovery`  | If auxiliary data must be recovered post-streaming                 | -->

<!-- ## ⚡ Backend Implementations

Subclasses are expected to register their backend-specific logic for:

- **JAX** (via `@jit`)
- **Warp** (via `@wp.kernel`)

These implementations are used to apply the boundary logic at simulation runtime.


## 🔄 Auxiliary Data Support

Some BCs (e.g., prescribed velocity or pressure) require initializing extra data at the boundary. The base class includes:

- `update_bc_auxilary_data(...)` – placeholder, can be overridden  
- `aux_data_init(...)` – initializes BC-specific auxiliary values (e.g., pre-fill velocity)

These support seamless integration of BCs requiring pre-simulation setup.

## 🔧 Custom Warp Kernels

To define Warp-compatible BCs, use:

```python
def _construct_kernel(self, functional):
```

Where functional(...) implements the per-thread boundary logic, returning updated distribution functions.

## 🧪 Example: DoNothingBC

The `DoNothingBC` subclass demonstrates a minimal example:

```python
class DoNothingBC(BoundaryCondition):
    def jax_implementation(...):
        return jnp.where(boundary_mask, f_pre, f_post)
```
This BC effectively does nothing to the boundary values — useful for debugging or placeholders. -->


---

## 🚧 **Boundary Condition Subclasses**

1. **DoNothingBC**:
In this boundary condition, the fluid populations are allowed to pass through the boundary without any reflection or modification. Useful for test cases or special boundary handling.
2. **EquilibriumBC**:
In this boundary condition, the fluid populations are assumed to be in at equilibrium. Constructor has optional macroscopic density (`rho`) and velocity (`u`) values
3. **FullwayBounceBackBC**:
In this boundary condition, the velocity of the fluid populations is reflected back to the fluid side of the boundary, resulting in zero fluid velocity at the boundary. Enforces no-slip wall conditions by reversing particle distributions at the boundary during the collision step.
4. **HalfwayBounceBackBC**:
Similar to the `FullwayBounceBackBC`, in this boundary condition, the velocity of the fluid populations is partially reflected back to the fluid side of the boundary, resulting in a non-zero fluid velocity at the boundary. Enforces no-slip conditions by reflecting distribution functions halfway between fluid and boundary nodes, improving accuracy over fullway bounce-back.
5. **ZouHeBC**:
This boundary condition is used to impose a prescribed velocity or pressure profile at the boundary. Supports only normal velocity components (only one non-zero velocity element allowed)
6. **RegularizedBC**:
This boundary condition is used to impose a prescribed velocity or pressure profile at the boundary. This BC is more stable than `ZouHeBC`, but computationally more expensive.
7. **ExtrapolationOutflowBC**:
A type of outflow boundary condition that uses extrapolation to avoid strong wave reflections.
8. **GradsApproximationBC**:
Interpolated bounce-back boundary condition for representing curved boundaries. Requires 3D velocity sets (not implemented in 2D)




## Summary Table of Boundary Conditions

| BC Class               | Purpose                                              | Implementation Step | Supports Auxiliary Data | Backend Support       |
|------------------------|------------------------------------------------------|---------------------|------------------------|-----------------------|
| `DoNothingBC`           |  Leaves boundary distributions unchanged (no-op)    | STREAMING            | No                     |JAX, Warp              |
| `EquilibriumBC`         | Prescribe equilibrium populations                    | STREAMING           | No                     | JAX, Warp             |
| `FullwayBounceBackBC`   | Classic bounce-back (no-slip)                         | COLLISION           | No                     | JAX, Warp             |
| `HalfwayBounceBackBC`   | Halfway bounce-back for no-slip walls                | STREAMING           | No                     | JAX, Warp             |
| `ZouHeBC`               |  Classical Zou-He velocity/pressure BC with non-equilibrium bounce-back | STREAMING | Yes            |JAX, Warp              |
| `RegularizedBC`         | Non-equilibrium bounce-back with second moment regularization | STREAMING           | No             | JAX, Warp             |
| `ExtrapolationOutflowBC`| Smooth outflow via extrapolation                     | STREAMING           | Yes                    | JAX, Warp             |
| `GradsApproximationBC`  | Approximate missing populations via Grad's method   | STREAMING           | No                     | Warp only             |
