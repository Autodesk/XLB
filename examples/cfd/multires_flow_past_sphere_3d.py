"""
3D flow past a sphere with multi-resolution LBM.

Demonstrates the multi-resolution Neon backend for a 3-D Poiseuille-
inlet flow past a sphere inside a nested cuboid multi-resolution domain.
Uses AABB-Close voxelization with halfway bounce-back on the sphere surface and
computes lift/drag via momentum transfer.
"""

try:
    import neon
except ModuleNotFoundError:
    import sys

    raise ModuleNotFoundError(
        "The 'neon' module is required for this example (Neon backend). "
        "Install with: pip install 'xlb[neon]'. "
        f"You are on Python {sys.version_info.major}.{sys.version_info.minor}; "
        "the current Neon wheel requires Python 3.11 or 3.12. Use an appropriate environment (e.g. pyenv, conda, or venv)."
    ) from None
import warp as wp
import numpy as np
import time

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    DoNothingBC,
    ZouHeBC,
    HybridBC,
)
from xlb.operator.boundary_masker import MeshVoxelizationMethod
from xlb.utils.mesher import make_cuboid_mesh, prepare_sparsity_pattern
from xlb.operator.force import MultiresMomentumTransfer


def generate_cuboid_mesh(stl_filename, num_finest_voxels_across_part):
    """
    Generate a cuboid mesh based on the provided voxel size and domain multipliers.
    """
    import trimesh
    import os

    # Domain multipliers for each refinement level
    # First entry should be full domain size
    # Domain multipliers
    domainMultiplier = [
        [7, 22, 7, 7, 7, 7],  # -x, x, -y, y, -z, z  (sphere at 1/4 domain from inlet)
        [3, 12, 5, 5, 5, 5],  # -x, x, -y, y, -z, z  (wake-biased)
        [2, 8, 4, 4, 4, 4],
        [1, 5, 2, 2, 2, 2],
        # [1, 2, 1, 1, 1, 1],
        # [0.4, 1, 0.4, 0.4, 0.4, 0.4],
        # [0.2, 0.4, 0.2, 0.2, 0.2, 0.2],
    ]

    # Load the mesh
    mesh = trimesh.load_mesh(stl_filename, process=False)
    assert not mesh.is_empty, ValueError("Loaded mesh is empty or invalid.")

    # Compute original bounds
    # Find voxel size and sphere radius
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)
    partSize = max_bound - min_bound

    # smallest voxel size
    voxel_size = min(partSize) / num_finest_voxels_across_part

    # Compute translation to put mesh into first octant of that domain—
    shift = np.array(
        [
            domainMultiplier[0][0] * partSize[0] - min_bound[0],
            domainMultiplier[0][2] * partSize[1] - min_bound[1],
            domainMultiplier[0][4] * partSize[2] - min_bound[2],
        ],
        dtype=float,
    )

    # Apply translation and save out temp stl
    mesh.apply_translation(shift)
    _ = mesh.vertex_normals
    mesh_vertices = np.asarray(mesh.vertices) / voxel_size
    mesh.export("temp.stl")

    # Mesh based on temp stl
    level_data = make_cuboid_mesh(
        voxel_size,
        domainMultiplier,
        "temp.stl",
    )
    grid_shape_finest = tuple([i * 2 ** (len(level_data) - 1) for i in level_data[-1][0].shape])
    print(f"Full shape based on finest voxels size is {grid_shape_finest}")
    os.remove("temp.stl")
    return level_data, mesh_vertices, tuple([int(a) for a in grid_shape_finest])


# -------------------------- Simulation Setup --------------------------

# The following parameters define the resolution of the voxelized grid
sphere_radius = 5
num_finest_voxels_across_part = 2 * sphere_radius

# Other setup parameters
Re = 5000.0
compute_backend = ComputeBackend.NEON
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
u_max = 0.04
num_steps = 10000
post_process_interval = 1000

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Generate the cuboid mesh and sphere vertices
stl_filename = "../stl-files/sphere.stl"
level_data, sphere, grid_shape_finest = generate_cuboid_mesh(stl_filename, num_finest_voxels_across_part)


# Define exporter object for hdf5 output
from xlb.utils import MultiresIO

# Define an exporter for the multiresolution data
exporter = MultiresIO({"velocity": 3, "density": 1}, level_data)

# Prepare the sparsity pattern and origins from the level data
sparsity_pattern, level_origins = prepare_sparsity_pattern(level_data)

# get the number of levels
num_levels = len(level_data)

# Create the multires grid
grid = multires_grid_factory(
    grid_shape_finest,
    velocity_set=velocity_set,
    sparsity_pattern_list=sparsity_pattern,
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
    """Build a Warp function for a Poiseuille parabolic inlet velocity profile."""
    assert compute_backend == ComputeBackend.NEON
    # IMPORTANT NOTE: the user defined functional must be defined in terms of the indices at the finest level
    _, ny, nz = grid_shape_finest
    dtype = precision_policy.compute_precision.wp_dtype
    H_y = dtype(ny)  # Length in y direction (finest level)
    H_z = dtype(nz)  # Length in z direction (finest level)
    two = dtype(2.0)
    one = dtype(1.0)
    zero = dtype(0.0)
    u_max_wp = dtype(u_max)
    _u_vec = wp.vec(velocity_set.d, dtype=dtype)

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        # Poiseuille flow profile: parabolic velocity distribution
        y = dtype(index[1])
        z = dtype(index[2])

        # Calculate normalized distance from center
        y_center = y - (H_y / two)
        z_center = z - (H_z / two)
        r_squared = (two * y_center / H_y) ** two + (two * z_center / H_z) ** two

        # Parabolic profile: u = u_max * (1 - r²)
        # Note that unlike RegularizedBC and ZouHeBC which only accept normal velocity, hybridBC accepts the full velocity vector

        # For hybridBC
        # return _u_vec(u_max_wp * wp.max(zero, one - r_squared), zero, zero)

        # For Regularized and ZouHe
        return wp.vec(u_max_wp * wp.max(zero, one - r_squared), length=1)

    return bc_profile_warp


# Convert bc indices to a list of list (first entry corresponds to the finest level)
inlet = [[] for _ in range(num_levels - 1)] + [inlet]
outlet = [[] for _ in range(num_levels - 1)] + [outlet]
walls = [[] for _ in range(num_levels - 1)] + [walls]

# Initialize Boundary Conditions
bc_left = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
# Alternatives:
# bc_left = HybridBC(bc_method="bounceback_regularized", profile=bc_profile(), indices=inlet)
# bc_left = RegularizedBC("velocity", prescribed_value=(u_max, 0.0, 0.0), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)
bc_outlet = DoNothingBC(indices=outlet)
# bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_sphere = HybridBC(
    bc_method="nonequilibrium_regularized", mesh_vertices=sphere, voxelization_method=MeshVoxelizationMethod("AABB"), use_mesh_distance=True
)
# bc_sphere = HalfwayBounceBackBC(mesh_vertices=sphere, voxelization_method=MeshVoxelizationMethod('AABB'))

boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_sphere]

# Configure the simulation relaxation time
visc = u_max * num_finest_voxels_across_part / Re
omega_finest = 1.0 / (3.0 * visc + 0.5)

# Make initializer operator
from xlb.helper.initializers import CustomMultiresInitializer

initializer = CustomMultiresInitializer(
    bc_id=bc_outlet.id,
    constant_velocity_vector=(u_max, 0.0, 0.0),
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

# Define a multi-resolution simulation manager
sim = xlb.helper.MultiresSimulationManager(
    omega_finest=omega_finest,
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
    initializer=initializer,
    mres_perf_opt=xlb.mres_perf_optimization_type.MresPerfOptimizationType.FUSION_AT_FINEST,
)

# Setup Momentum Transfer for Force Calculation
bc_sphere = boundary_conditions[-1]
momentum_transfer = MultiresMomentumTransfer(bc_sphere, mres_perf_opt=sim.mres_perf_opt, compute_backend=compute_backend)


def print_lift_drag(sim):
    """Compute and print drag and lift coefficients from the simulation state."""
    boundary_force = momentum_transfer(sim.f_0, sim.f_1, sim.bc_mask, sim.missing_mask)
    drag = boundary_force[0]  # x-direction
    lift = boundary_force[2]
    sphere_cross_section = np.pi * sphere_radius**2
    u_avg = 0.5 * u_max
    cd = 2.0 * drag / (u_avg**2 * sphere_cross_section)
    cl = 2.0 * lift / (u_avg**2 * sphere_cross_section)
    print(f"\tCD={cd}, CL={cl}")


# -------------------------- Simulation Loop --------------------------

wp.synchronize()
start_time = time.time()
for step in range(num_steps):
    sim.step()

    if step % post_process_interval == 0 or step == num_steps - 1:
        # # Export VTK for comparison
        # tic_write = time.perf_counter()
        # sim.export_macroscopic("multires_flow_over_sphere_3d_")
        # toc_write = time.perf_counter()
        # print(f"\tVTK file written in {toc_write - tic_write:0.1f} seconds")

        # Call the Macroscopic operator to compute macroscopic fields
        wp.synchronize()
        sim.macro(sim.f_0, sim.bc_mask, sim.rho, sim.u, streamId=0)

        # Call the exporter to save the current state
        nx, ny, nz = grid_shape_finest
        filename = f"multires_flow_past_sphere_3d_{step:04d}"
        wp.synchronize()
        exporter.to_hdf5(filename, {"velocity": sim.u, "density": sim.rho}, compression="gzip", compression_opts=2)
        exporter.to_slice_image(
            filename,
            {"velocity": sim.u},
            plane_point=(nx // 2, ny // 2, nz // 2),
            plane_normal=(0, 0, 1),
            grid_res=256,
            slice_thickness=2 ** (num_levels - 1),
            bounds=(0.1, 0.6, 0.3, 0.7),
        )

        # Print lift and drag coefficients
        print_lift_drag(sim)
        wp.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\tCompleted step {step}. Time elapsed for {post_process_interval} steps: {elapsed:.6f} seconds.")
        start_time = time.time()
