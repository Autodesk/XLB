# Moments in LBM

In the Lattice Boltzmann Method (LBM), the fluid is represented not directly by macroscopic variables (density, velocity, pressure) but by a set of distribution functions f.

These distribution functions describe the probability of particles moving in certain discrete directions on the lattice.

Computes fluid density (`ρ`) and velocity (`u`) from the distribution field `f`.

## Methods

`macroscopic(f) -> (rho, u)`

- overview: compute macroscopic fields from the distribution.

*Parameters*

- `f`: distribution field created with grid.create_field(velocity_set.d)

*Returns*

- rho: scalar density field, shape (1, *grid.shape)
- u: velocity vector field, shape (dim, *grid.shape)

Example:

```python
from xlb.operator.macroscopic.macroscopic import Macroscopic

# Assume f is a distribution field on the grid
macroscopic = Macroscopic(velocity_set=my_velocity_set)

rho, u = macroscopic(f)
```