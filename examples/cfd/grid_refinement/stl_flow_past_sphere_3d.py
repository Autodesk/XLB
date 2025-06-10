import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, HalfwayBounceBackBC, RegularizedBC, ExtrapolationOutflowBC, DoNothingBC, ZouHeBC
from xlb.utils import make_cuboid_mesh
import neon
import warp as wp
import numpy as np
import time


def generate_cuboid_mesh(stl_filename, num_finest_voxels_across_part, grid_shape):
    """
    Generate a cuboid mesh based on the provided voxel size and domain multipliers.
    """
    import open3d as o3d
    import os

    # STL position
    nx, ny, nz = grid_shape
    sphere_origin = (nx // 6, ny // 2, nz // 2)

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    if mesh.is_empty():
        raise ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    partSize = max_bound - min_bound
    sphere_diameter_phys_units = float(min(partSize))

    # smallest voxel size
    voxel_size = sphere_diameter_phys_units / num_finest_voxels_across_part
    sphere_radius = sphere_diameter_phys_units / voxel_size / 2.0

    # Compute translation to put mesh into first octant of that domain—
    shift = np.array(sphere_origin) * voxel_size - sphere_diameter_phys_units / 2.0 - min_bound

    # Apply translation and save out temp stl
    mesh.translate(shift)
    mesh.compute_vertex_normals()
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    o3d.io.write_triangle_mesh("temp.stl", mesh)
    os.remove("temp.stl")

    # Mesh base don temp stl
    # Create the multires grid
    num_levels = 3
    level_origins = []
    level_data = []
    for lvl in range(num_levels):
        divider = 2**lvl
        growth = 1.25**lvl
        shape = nx // divider, ny // divider, nz // divider
        if lvl == num_levels - 1:
            level = np.ascontiguousarray(np.ones(shape, dtype=int), dtype=np.int32)
            box_origin = (0, 0, 0)  # The coarsest level has no origin offset
        else:
            box_size = tuple([int(shape[i] // 4 * growth) for i in range(3)])
            if lvl == 0:
                box_origin = tuple(
                    [sphere_origin[0] // divider - int(2 * growth * sphere_radius // divider)]
                    + [shape[i] // 2 - box_size[i] // 2 for i in range(1, 3)]
                )
            else:
                finer_box_size = level_data[-1].shape
                finer_box_origin = np.array(level_origins[-1])
                shift = np.array(box_size) - np.array(finer_box_size) // 2
                box_origin = finer_box_origin // 2 - shift // 2
            level = np.ascontiguousarray(np.ones(box_size, dtype=int), dtype=np.int32)
        level_data.append(level)
        level_origins.append(box_origin)

    return level_data, level_origins, mesh_vertices


# -------------------------- Simulation Setup --------------------------

# The following parameters define the resolution of the voxelized grid
num_finest_voxels_across_part = 10

# Other setup parameters
Re = 500.0
grid_shape = (512 // 2, 128 // 2, 128 // 2)
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
u_max = 0.04
num_steps = 1000
post_process_interval = 100

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate the cuboid mesh and sphere vertices
stl_filename = "examples/cfd/stl-files/sphere.stl"
level_data, level_origins, sphere = generate_cuboid_mesh(stl_filename, num_finest_voxels_across_part, grid_shape)

# get the number of levels
num_levels = len(level_data)

# Create the multires grid
grid = multires_grid_factory(
    grid_shape,
    velocity_set=velocity_set,
    sparsity_pattern_list=level_data,
    sparsity_pattern_origins=[neon.Index_3d(*box_origin) for box_origin in level_origins],
)

# Define Boundary Indices
coarsest_level = grid.count_levels - 1
box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
box_no_edge = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level), remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()


# Define Boundary Conditions
def bc_profile():
    assert compute_backend == ComputeBackend.NEON

    # Note nx, ny, nz are the dimensions of the grid at the finest level while the inlet is defined at the coarsest level
    nx, ny, nz = grid_shape
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


# Convert bc indices to a list of list (first entry corresponds to the finest level)
inlet = [[] for _ in range(num_levels - 1)] + [inlet]
outlet = [[] for _ in range(num_levels - 1)] + [outlet]
walls = [[] for _ in range(num_levels - 1)] + [walls]

# Initialize Boundary Conditions
bc_left = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
# Alternatively, use a prescribed velocity profile
# bc_left = RegularizedBC("velocity", prescribed_value=(u_max, 0.0, 0.0), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)  # TODO: issues with halfway bounce back only here!
# bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_outlet = DoNothingBC(indices=outlet)
bc_sphere = HalfwayBounceBackBC(mesh_vertices=sphere)
boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_sphere]

# Configure the simulation relaxation time
visc = u_max * num_finest_voxels_across_part / Re
omega = 1.0 / (3.0 * visc + 0.5)

# Define a multi-resolution simulation manager
sim = xlb.helper.MultiresSimulationManager(
    omega=omega,
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
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
