# Stepper

The `Stepper` orchestrates the time-stepping of Lattice Boltzmann simulations.
It ties together streaming, collision, macroscopic updates, and boundary conditions.

A stepper advances the simulation forward by one time step. It orchestrates the sequence of operations:

1. **Streaming** → particles move along lattice directions.
2. **Boundary conditions** → enforce walls, inlets, outlets, etc.
3. **Macroscopic update** → compute density (ρ) and velocity (u).
4. **Equilibrium calculation** → build the equilibrium distribution (f_eq).
5. **Collision** → relax distributions toward equilibrium.


## IncompressibleNavierStokesStepper
A ready-to-use stepper for solving the incompressible Navier–Stokes equations with LBM.

**Functions**

`prepare_fields(initializer=None) -> (f0, f1, bc_mask, missing_mask)`

- Allocates and initializes the distribution fields and boundary condition masks.
- `initializer`: optional operator for custom initialization (otherwise uses default equilibrium with ρ=1, u=0).

Returns:

- `f0`: distribution field at the start of the step
- `f1`: buffer for the next step (double-buffering)
- `bc_mask`: IDs indicating which boundary condition applies at each node
- `missing_mask`: marks which populations are missing at boundary nodes

**Constructor**
```python
IncompressibleNavierStokesStepper(
    grid,
    boundary_conditions=[],
    collision_type="BGK",
    forcing_scheme="exact_difference",
    force_vector=None,
)
```

**Example**
```python
# Create a 3D grid
grid = grid_factory((64, 64, 64), compute_backend=ComputeBackend.JAX)

# Create stepper with BGK collision
stepper = IncompressibleNavierStokesStepper(grid, boundary_conditions=my_bcs)

# Prepare fields
f0, f1, bc_mask, missing_mask = stepper.prepare_fields()

# Advance one step
f0, f1 = stepper(f0, f1, bc_mask, missing_mask, omega=1.0, timestep=0)
```