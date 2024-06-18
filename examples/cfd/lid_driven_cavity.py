import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, assign_bc_id_box_faces
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import jax.numpy as jnp

backend = ComputeBackend.WARP
velocity_set = xlb.velocity_set.D2Q9()
precision_policy = PrecisionPolicy.FP32FP32

xlb.init(
    velocity_set=velocity_set,
    default_backend=backend,
    default_precision_policy=precision_policy,
)

grid_size = 512
grid_shape = (grid_size, grid_size)

grid, f_0, f_1, missing_mask, boundary_mask = create_nse_fields(grid_shape)

# Velocity on top face (2D)
boundary_mask, missing_mask = assign_bc_id_box_faces(
    boundary_mask, missing_mask, grid_shape, EquilibriumBC.id, ["top"]
)

bc_eq = QuadraticEquilibrium()

bc_top = EquilibriumBC(rho=1.0, u=(0.02, 0.0), equilibrium_operator=bc_eq)


# Wall on all other faces (2D)
boundary_mask, missing_mask = assign_bc_id_box_faces(
    boundary_mask,
    missing_mask,
    grid_shape,
    FullwayBounceBackBC.id,
    ["bottom", "left", "right"],
)

bc_walls = FullwayBounceBackBC()


f_0 = initialize_eq(f_0, grid, velocity_set, backend)
boundary_conditions = [bc_top, bc_walls]
omega = 1.6

stepper = IncompressibleNavierStokesStepper(
    omega, boundary_conditions=boundary_conditions
)

for i in range(50000):
    f_1 = stepper(f_0, f_1, boundary_mask, missing_mask, i)
    f_0, f_1 = f_1, f_0


# Write the results. We'll use JAX backend for the post-processing
if not isinstance(f_0, jnp.ndarray):
    f_0 = wp.to_jax(f_0)

macro = Macroscopic(compute_backend=ComputeBackend.JAX)

rho, u = macro(f_0)

# remove boundary cells
rho = rho[:, 1:-1, 1:-1]
u = u[:, 1:-1, 1:-1]
u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5

fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude}

save_fields_vtk(fields, timestep=i, prefix="lid_driven_cavity")
save_image(fields["u_magnitude"], timestep=i, prefix="lid_driven_cavity")
