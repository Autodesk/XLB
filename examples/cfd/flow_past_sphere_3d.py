import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import time

# -------------------------- Simulation Setup --------------------------

omega = 1.6
grid_shape = (512 // 2, 128 // 2, 128 // 2)
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend)
u_max = 0.04
num_steps = 10000
post_process_interval = 1000

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create Grid
grid = grid_factory(grid_shape, compute_backend=compute_backend)

# Define Boundary Indices
box = grid.bounding_box_indices()
box_no_edge = grid.bounding_box_indices(remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()

sphere_radius = grid_shape[1] // 12
x = np.arange(grid_shape[0])
y = np.arange(grid_shape[1])
z = np.arange(grid_shape[2])
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
indices = np.where((X - grid_shape[0] // 6) ** 2 + (Y - grid_shape[1] // 2) ** 2 + (Z - grid_shape[2] // 2) ** 2 < sphere_radius**2)
sphere = [tuple(indices[i]) for i in range(velocity_set.d)]


# Define Boundary Conditions
def bc_profile():
    H_y = float(grid_shape[1] - 1)  # Height in y direction
    H_z = float(grid_shape[2] - 1)  # Height in z direction

    if compute_backend == ComputeBackend.JAX:

        def bc_profile_jax():
            y = jnp.arange(grid_shape[1])
            z = jnp.arange(grid_shape[2])
            Y, Z = jnp.meshgrid(y, z, indexing="ij")

            # Calculate normalized distance from center
            y_center = Y - (H_y / 2.0)
            z_center = Z - (H_z / 2.0)
            r_squared = (2.0 * y_center / H_y) ** 2.0 + (2.0 * z_center / H_z) ** 2.0

            # Parabolic profile for x velocity, zero for y and z
            u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
            u_y = jnp.zeros_like(u_x)
            u_z = jnp.zeros_like(u_x)

            return jnp.stack([u_x, u_y, u_z])

        return bc_profile_jax

    elif compute_backend == ComputeBackend.WARP:

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            # Poiseuille flow profile: parabolic velocity distribution
            y = wp.float32(index[1])
            z = wp.float32(index[2])

            # Calculate normalized distance from center
            y_center = y - (H_y / 2.0)
            z_center = z - (H_z / 2.0)
            r_squared = (2.0 * y_center / H_y) ** 2.0 + (2.0 * z_center / H_z) ** 2.0

            # Parabolic profile: u = u_max * (1 - r²)
            return wp.vec(u_max * wp.max(0.0, 1.0 - r_squared), length=1)

        return bc_profile_warp


# Initialize Boundary Conditions
bc_left = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
# Alternatively, use a prescribed velocity profile
# bc_left = RegularizedBC("velocity", prescribed_value=(u_max, 0.0, 0.0), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)
bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_sphere = HalfwayBounceBackBC(indices=sphere)
boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_sphere]

# Setup Stepper
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="BGK",
)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
)


# Post-Processing Function
def post_process(step, f_current):
    # Convert to JAX array if necessary
    if not isinstance(f_current, jnp.ndarray):
        f_current = wp.to_jax(f_current)

    rho, u = macro(f_current)

    # Remove boundary cells
    u = u[:, 1:-1, 1:-1, 1:-1]
    rho = rho[:, 1:-1, 1:-1, 1:-1]
    u_magnitude = jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

    fields = {
        "u_magnitude": u_magnitude,
        "u_x": u[0],
        "u_y": u[1],
        "u_z": u[2],
        "rho": rho[0],
    }

    # Save the u_magnitude slice at the mid y-plane
    save_image(fields["u_magnitude"][:, grid_shape[1] // 2, :], timestep=step)
    print(f"Post-processed step {step}: Saved u_magnitude slice at y={grid_shape[1] // 2}")


# -------------------------- Simulation Loop --------------------------

start_time = time.time()
for step in range(num_steps):
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap the buffers

    if step % post_process_interval == 0 or step == num_steps - 1:
        if compute_backend == ComputeBackend.WARP:
            wp.synchronize()
        post_process(step, f_0)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Completed step {step}. Time elapsed for {post_process_interval} steps: {elapsed:.6f} seconds.")
        start_time = time.time()
