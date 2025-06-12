import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import FullwayBounceBackBC, HalfwayBounceBackBC, RegularizedBC, ExtrapolationOutflowBC, DoNothingBC, ZouHeBC
import neon
import warp as wp
import numpy as np
import time

# -------------------------- Simulation Setup --------------------------

Re = 500.0
grid_shape = (512 // 2, 128 // 2, 128 // 2)
compute_backend = ComputeBackend.NEON
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

# Create the multires grid
nx, ny, nz = grid_shape
sphere_origin = (nx // 6, ny // 2, nz // 2)
sphere_radius = min(nx, ny, nz) // 12  # Radius of the sphere
num_levels = 3
level_origins = []
level_list = []
for lvl in range(num_levels):
    divider = 2**lvl
    growth = 1.5**lvl
    shape = grid_shape[0] // divider, grid_shape[1] // divider, grid_shape[2] // divider
    if lvl == num_levels - 1:
        level = np.ascontiguousarray(np.ones(shape, dtype=int), dtype=np.int32)
        box_origin = (0, 0, 0)  # The coarsest level has no origin offset
    else:
        box_size = tuple([int(shape[i] // 4 * growth) for i in range(3)])
        box_origin = tuple([sphere_origin[0] // divider - 4 * sphere_radius // divider] + [shape[i] // 2 - box_size[i] // 2 for i in range(1, 3)])
        level = np.ascontiguousarray(np.ones(box_size, dtype=int), dtype=np.int32)
    level_list.append(level)
    level_origins.append(neon.Index_3d(*box_origin))

grid = multires_grid_factory(
    grid_shape,
    velocity_set=velocity_set,
    sparsity_pattern_list=level_list,
    sparsity_pattern_origins=level_origins,
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
inlet = [[] for _ in range(num_levels - 1)] + [inlet]
outlet = [[] for _ in range(num_levels - 1)] + [outlet]
walls = [[] for _ in range(num_levels - 1)] + [walls]
sphere = [sphere] + [[] for _ in range(num_levels - 1)]


# Define Boundary Conditions
def bc_profile():
    assert compute_backend == ComputeBackend.NEON

    # Note nx, ny, nz are the dimensions of the grid at the finest level
    H_y = float(ny // 2 ** (num_levels - 1) - 1)  # Height in y direction
    H_z = float(nz // 2 ** (num_levels - 1) - 1)  # Height in z direction

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


# Configure the simulation relaxation time
visc = 2.0 * u_max * sphere_radius / Re
omega = 1.0 / (3.0 * visc + 0.5)

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
