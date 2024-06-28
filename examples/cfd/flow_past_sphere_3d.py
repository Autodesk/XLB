import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, assign_bc_id_box_faces
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    EquilibriumBC,
    DoNothingBC,
)
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp

backend = ComputeBackend.WARP
velocity_set = xlb.velocity_set.D3Q19()
precision_policy = PrecisionPolicy.FP32FP32

xlb.init(
    velocity_set=velocity_set,
    default_backend=backend,
    default_precision_policy=precision_policy,
)

grid_size_x, grid_size_y, grid_size_z = 512, 128, 128
grid_shape = (grid_size_x, grid_size_y, grid_size_z)

grid, f_0, f_1, missing_mask, boundary_mask = create_nse_fields(grid_shape)

# Velocity on left face (3D)
boundary_mask, missing_mask = assign_bc_id_box_faces(
    boundary_mask, missing_mask, grid_shape, EquilibriumBC.id, ["left"]
)


# Wall on all other faces (3D) except right
boundary_mask, missing_mask = assign_bc_id_box_faces(
    boundary_mask,
    missing_mask,
    grid_shape,
    FullwayBounceBackBC.id,
    ["bottom", "right", "front", "back"],
)

# Do nothing on right face
boundary_mask, missing_mask = assign_bc_id_box_faces(
    boundary_mask, missing_mask, grid_shape, DoNothingBC.id, ["right"]
)

bc_eq = QuadraticEquilibrium()
bc_left = EquilibriumBC(rho=1.0, u=(0.02, 0.0, 0.0), equilibrium_operator=bc_eq)
bc_walls = FullwayBounceBackBC()
bc_do_nothing = DoNothingBC()


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
indices = np.array(indices)

# Set boundary conditions on the indices
indices_boundary_masker = xlb.operator.boundary_masker.IndicesBoundaryMasker(
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=backend,
)

boundary_mask, missing_mask = indices_boundary_masker(
    indices, FullwayBounceBackBC.id, boundary_mask, missing_mask, (0, 0, 0)
)

f_0 = initialize_eq(f_0, grid, velocity_set, backend)
boundary_conditions = [bc_left, bc_walls, bc_do_nothing]
omega = 1.8

stepper = IncompressibleNavierStokesStepper(
    omega, boundary_conditions=boundary_conditions
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
