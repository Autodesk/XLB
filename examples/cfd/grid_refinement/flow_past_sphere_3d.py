import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, HalfwayBounceBackBC, RegularizedBC, ExtrapolationOutflowBC, DoNothingBC, ZouHeBC
import neon
import warp as wp
import numpy as np
import jax.numpy as jnp
import time

# -------------------------- Simulation Setup --------------------------

omega = 1.6
grid_shape = (256 // 2, 256 // 2, 256 // 2)
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend)
u_max = 0.04
num_steps = 2000
post_process_interval = 100

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create the multires grid
# TODO: with rectangular cuboid for the inner box, there are some issues with the
#       multires_grid_factory. The inner box should be a cube for now!
nx, ny, nz = grid_shape
sphere_origin = (nx // 2, ny // 2, nz // 2)
sphere_radius = ny // 12
inner_box_shape = (6 * sphere_radius, 6 * sphere_radius, 6 * sphere_radius)
num_levels = 2
level_1 = np.ones((nx // 2, ny // 2, nz // 2), dtype=int)
level_0 = np.ones(inner_box_shape, dtype=int)
level_0 = np.ascontiguousarray(level_0, dtype=np.int32)
levels = [level_0, level_1]
level_origins = [((nx - inner_box_shape[0]) // 2, (ny - inner_box_shape[1]) // 2, (nz - inner_box_shape[2]) // 2), (0, 0, 0)]

grid = multires_grid_factory(
    grid_shape,
    velocity_set=velocity_set,
    sparsity_pattern_list=[level_0, level_1],
    sparsity_pattern_origins=[neon.Index_3d(*level_origins[lvl]) for lvl in range(num_levels)],
)

# Define Boundary Indices
coarsest_level = grid.count_levels - 1
box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
box_no_edge = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level), remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()

# sphere at the finest level
x = np.arange(nx)
y = np.arange(ny)
z = np.arange(nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
indices = np.where((X - sphere_origin[0]) ** 2 + (Y - sphere_origin[1]) ** 2 + (Z - sphere_origin[2]) ** 2 < sphere_radius**2)
sphere = [tuple(indices[i]) for i in range(velocity_set.d)]

# Convert bc indices to a list of list (first entry corresponds to the finest level)
inlet = [[], inlet]
outlet = [[], outlet]
walls = [[], walls]
sphere = [sphere, []]


# Define Boundary Conditions
def bc_profile():
    assert compute_backend == ComputeBackend.NEON

    # Note nx, ny, nz are the dimensions of the grid at the finest level
    H_y = float(ny // 2 - 1)  # Height in y direction
    H_z = float(nz // 2 - 1)  # Height in z direction

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
bc_walls = FullwayBounceBackBC(indices=walls)  # TODO: issues with halfway bounce back only here!
# bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_outlet = DoNothingBC(indices=outlet)
bc_sphere = HalfwayBounceBackBC(indices=sphere)
boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_sphere]

# Define a multi-resolution simulation manager
sim = xlb.helper.MultiresSimulationManager(
    omega=omega,
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="BGK",
)

# -------------------------- Simulation Loop --------------------------

wp.synchronize()
start_time = time.time()
for step in range(num_steps):
    sim.step()

    if step % post_process_interval == 0 or step == num_steps - 1:
        # TODO: Issues in the vtk output for rectangular cuboids (as if a duboid grid with the largest side is assumed)
        sim.export_macroscopic("multires_flow_over_sphere_3d_")
        wp.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Completed step {step}. Time elapsed for {post_process_interval} steps: {elapsed:.6f} seconds.")
        start_time = time.time()
