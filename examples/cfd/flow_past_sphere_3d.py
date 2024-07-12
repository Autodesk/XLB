import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    EquilibriumBC,
    DoNothingBC,
)
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp

# Initial setup and backend configuration
backend = ComputeBackend.JAX
velocity_set = xlb.velocity_set.D3Q19()
precision_policy = PrecisionPolicy.FP32FP32

xlb.init(
    velocity_set=velocity_set,
    default_backend=backend,
    default_precision_policy=precision_policy,
)

#TODO HS: check inconsistency between grid_shape and velocity_set
#TODO HS: why is boundary_mask and missing_mask in the same function?! they should be separated
#TODO HS: missing_mask needs to be created based on ALL boundary indices and a SINGLE streaming operation not one streaming call per bc!
#TODO HS: why bc operatores need to be stated twice: once in making boundary_mask and missing_mask and one in making bc list.
#TODO HS: proposal: we should include indices as part of the construction of the bc operators and then have a single call to construcut boundary_mask and missing_mask fields based on bc_list.

# Define grid
grid_size_x, grid_size_y, grid_size_z = 512, 128, 128
grid_shape = (grid_size_x, grid_size_y, grid_size_z)

# Define fields on the grid
grid, f_0, f_1, missing_mask, boundary_mask = create_nse_fields(grid_shape)

# Specify BC indices
inlet = grid.boundingBoxIndices['left']
outlet = grid.boundingBoxIndices['right']
walls = [grid.boundingBoxIndices['bottom'][i] + grid.boundingBoxIndices['top'][i] + 
         grid.boundingBoxIndices['front'][i] + grid.boundingBoxIndices['back'][i] for i in range(velocity_set.d)]

# indices for sphere
sphere_radius = grid_size_y // 12
x = np.arange(grid_size_x)
y = np.arange(grid_size_y)
z = np.arange(grid_size_z)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
indices = np.where(
    (X - grid_size_x // 6) ** 2
    + (Y - grid_size_y // 2) ** 2
    + (Z - grid_size_z // 2) ** 2
    < sphere_radius**2
)
sphere = [tuple(indices[i]) for i in range(velocity_set.d)]

# Instantiate BC objects
bc_left = EquilibriumBC(inlet, rho=1.0, u=(0.02, 0.0, 0.0), equilibrium_operator=QuadraticEquilibrium())
bc_walls = FullwayBounceBackBC(walls)
bc_do_nothing = DoNothingBC(outlet)
bc_sphere = FullwayBounceBackBC(sphere)

# Set boundary_id and missing_mask for all BCs in boundary_conditions list
boundary_condition_list = [bc_left, bc_walls, bc_do_nothing, bc_sphere]
indices_boundary_masker = IndicesBoundaryMasker(
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=backend,
)

boundary_mask, missing_mask = indices_boundary_masker(
    boundary_condition_list, boundary_mask, missing_mask, (0, 0, 0)
)
# Note: In case we want to remove indices from BC objects
# for bc in boundary_condition_list:
#    bc.__dict__.pop('indices', None)


# Initialize fields to start the run
f_0 = initialize_eq(f_0, grid, velocity_set, backend)
omega = 1.8

stepper = IncompressibleNavierStokesStepper(
    omega, boundary_conditions=boundary_condition_list
)

for i in range(10000):
    f_1 = stepper(f_0, f_1, boundary_mask, missing_mask, i)
    f_0, f_1 = f_1, f_0


# Write the results. We'll use JAX backend for the post-processing
if not isinstance(f_0, jnp.ndarray):
    f_0 = wp.to_jax(f_0)

macro = Macroscopic(compute_backend=ComputeBackend.JAX)

rho, u = macro(f_0)

# remove boundary cells
u = u[:, 1:-1, 1:-1, 1:-1]
u_magnitude = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5

fields = {"u_magnitude": u_magnitude}

save_fields_vtk(fields, timestep=i)
save_image(fields["u_magnitude"][:, grid_size_y // 2, :], timestep=i)
