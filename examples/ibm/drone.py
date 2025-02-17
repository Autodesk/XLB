import os
import xlb
import trimesh
import time
from tqdm import tqdm
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper import IBMStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from xlb.helper.ibm_helper import (
    prepare_immersed_boundary,
    transform_mesh,
    reconstruct_mesh_from_vertices_and_faces,
)
from xlb.grid import grid_factory

from pxr import Usd, UsdGeom, Vt, Gf

from xlb.operator.postprocess import QCriterion, Vorticity, GridToPoint


@wp.kernel
def get_color(
    low: float,
    high: float,
    values: wp.array(dtype=float),
    out_color: wp.array(dtype=wp.vec3),
) -> wp.vec3:
    tid = wp.tid()

    v = values[tid]

    r = 1.0
    g = 1.0
    b = 1.0

    if v < low:
        v = low
    if v > high:
        v = high

    dv = high - low

    if v < (low + 0.25 * dv):
        r = 0.0
        g = 4.0 * (v - low) / dv
    elif v < (low + 0.5 * dv):
        r = 0.0
        b = 1.0 + 4.0 * (low + 0.25 * dv - v) / dv
    elif v < (low + 0.75 * dv):
        r = 4.0 * (v - low - 0.5 * dv) / dv
        b = 0.0
    else:
        g = 1.0 + 4.0 * (low + 0.75 * dv - v) / dv
        b = 0.0

    out_color[tid] = wp.vec3(r, g, b)


# -------------------------- Simple grid_to_point Helper --------------------------
def grid_to_point(field, verts, out):
    """
    A simple nearest-neighbor sampler.
    field: NumPy array of shape (nx, ny, nz)
    verts: Warp array of shape (n, 3) containing vertex positions (assumed to be in grid indices)
    out: Warp array of shape (n,) to store the sampled scalar values.
    """
    verts_np = verts.numpy()  # Convert to NumPy array
    nx, ny, nz = field.shape
    out_np = out.numpy()  # Make a NumPy copy to allow assignment
    for i, v in enumerate(verts_np):
        x = int(round(v[0]))
        y = int(round(v[1]))
        z = int(round(v[2]))
        x = min(max(x, 0), nx - 1)
        y = min(max(y, 0), ny - 1)
        z = min(max(z, 0), nz - 1)
        out_np[i] = field[x, y, z]
    return wp.array(out_np, dtype=wp.float32, device=out.device)


# -------------------------- USD Output Helper using Q Criterion --------------------------
def save_usd_vorticity(
    timestep, post_process_interval, bc_mask, f_current, grid_shape, usd_mesh, vorticity_operator, precision_policy, vorticity_threshold, usd_stage
):
    # Compute macroscopic quantities using WARP backend for vorticity
    device = "cuda:0"  # Use the same device as the simulation
    macro_wp = Macroscopic(
        compute_backend=ComputeBackend.WARP,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP),
    )
    rho = wp.zeros((1, *grid_shape), dtype=wp.float32, device=device)
    u = wp.zeros((3, *grid_shape), dtype=wp.float32, device=device)
    rho, u = macro_wp(f_current, rho, u)
    u = u[:, 20:-20, 20:-20, 20:-20]  # Remove bc

    # Allocate arrays for vorticity and its magnitude
    vorticity = wp.zeros((3, *u.shape[1:]), dtype=wp.float32, device=device)
    vorticity_magnitude = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)

    # Compute vorticity
    vorticity, vorticity_magnitude = vorticity_operator(u, bc_mask, vorticity, vorticity_magnitude)

    # Setup marching cubes on vorticity_magnitude[0]
    max_verts = grid_shape[0] * grid_shape[1] * grid_shape[2] * 5
    max_tris = grid_shape[0] * grid_shape[1] * grid_shape[2] * 3
    mc = wp.MarchingCubes(
        nx=u.shape[1],
        ny=u.shape[2],
        nz=u.shape[3],
        max_verts=max_verts,
        max_tris=max_tris,
        device=device,
    )

    # Extract isosurface at vorticity_threshold
    mc.surface(vorticity_magnitude[0], vorticity_threshold)

    if mc.verts.shape[0] == 0:
        print(f"Warning: No vertices found for vorticity at timestep {timestep}. Try adjusting vorticity_threshold.")
        return

    grid_to_point_op = GridToPoint(
        precision_policy=precision_policy,
        compute_backend=ComputeBackend.WARP,
    )

    scalars = wp.zeros(mc.verts.shape[0], dtype=wp.float32, device=device)
    scalars = grid_to_point_op(vorticity_magnitude, mc.verts, scalars)

    colors = wp.empty(mc.verts.shape[0], dtype=wp.vec3, device=device)
    # Compute dynamic range based on actual vorticity values
    scalars_np = scalars.numpy()
    vorticity_min = float(np.percentile(scalars_np, 5))  # 5th percentile to exclude outliers
    vorticity_max = float(np.percentile(scalars_np, 95))  # 95th percentile to exclude outliers

    if abs(vorticity_max - vorticity_min) < 1e-6:
        vorticity_max = vorticity_min + 1e-6

    wp.launch(
        get_color,
        dim=mc.verts.shape[0],
        inputs=(vorticity_min, vorticity_max, scalars),
        outputs=(colors,),
        device=device,
    )

    vertices = mc.verts.numpy()
    indices = mc.indices.numpy()
    tri_count = len(indices) // 3

    usd_mesh.GetPointsAttr().Set(vertices.tolist(), time=timestep // post_process_interval)
    usd_mesh.GetFaceVertexCountsAttr().Set([3] * tri_count, time=timestep // post_process_interval)
    usd_mesh.GetFaceVertexIndicesAttr().Set(indices.tolist(), time=timestep // post_process_interval)
    usd_mesh.GetDisplayColorAttr().Set(colors.numpy().tolist(), time=timestep // post_process_interval)
    UsdGeom.Primvar(usd_mesh.GetDisplayColorAttr()).SetInterpolation("vertex")

    # usd_stage.SetStartTimeCode(0)
    # usd_stage.SetEndTimeCode(timestep)
    # usd_stage.SetTimeCodesPerSecond(24)

    # usd_stage.Save()

    print(f"Vorticity visualization at timestep {timestep}:")
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Number of triangles: {tri_count}")
    print(f"  Vorticity range: [{vorticity_min:.6f}, {vorticity_max:.6f}]")


def save_usd_q_criterion(timestep, bc_mask, f_current, grid_shape, usd_mesh, q_criterion_operator, precision_policy, q_threshold, usd_stage):
    # Compute macroscopic quantities using WARP backend for Q-criterion
    device = "cuda:0"  # Use the same device as the simulation
    macro_wp = Macroscopic(
        compute_backend=ComputeBackend.WARP,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP),
    )
    rho = wp.zeros((1, *grid_shape), dtype=wp.float32, device=device)
    u = wp.zeros((3, *grid_shape), dtype=wp.float32, device=device)
    rho, u = macro_wp(f_current, rho, u)
    u = u[:, 1:-1, 1:-1, 1:-1]  # Remove bc

    # Allocate arrays for norm_mu and q_field on the same device as input
    norm_mu = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)
    q_field = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)

    # Compute Q criterion
    norm_mu, q_field = q_criterion_operator(u, bc_mask, norm_mu, q_field)

    # Setup marching cubes on q_field[0] (use grid_shape minus ghost layers)
    max_verts = grid_shape[0] * grid_shape[1] * grid_shape[2] * 5
    max_tris = grid_shape[0] * grid_shape[1] * grid_shape[2] * 3
    mc = wp.MarchingCubes(
        nx=u.shape[1],
        ny=u.shape[2],
        nz=u.shape[3],
        max_verts=max_verts,
        max_tris=max_tris,
        device=device,
    )

    # Extract isosurface at q_threshold
    mc.surface(q_field[0], q_threshold)

    if mc.verts.shape[0] == 0:
        print(f"Warning: No vertices found for Q-criterion at timestep {timestep}. Try adjusting q_threshold.")
        return

    grid_to_point_op = GridToPoint(
        precision_policy=precision_policy,
        compute_backend=ComputeBackend.WARP,
    )

    scalars = wp.zeros(mc.verts.shape[0], dtype=wp.float32, device=device)
    scalars = grid_to_point_op(norm_mu, mc.verts, scalars)

    colors = wp.empty(mc.verts.shape[0], dtype=wp.vec3, device=device)
    vorticity_min = 0.0
    vorticity_max = 0.1
    wp.launch(
        get_color,
        dim=mc.verts.shape[0],
        inputs=(vorticity_min, vorticity_max, scalars),
        outputs=(colors,),
        device=device,
    )

    vertices = mc.verts.numpy()
    indices = mc.indices.numpy()
    tri_count = len(indices) // 3

    usd_mesh.GetPointsAttr().Set(vertices.tolist(), time=timestep)
    usd_mesh.GetFaceVertexCountsAttr().Set([3] * tri_count, time=timestep)
    usd_mesh.GetFaceVertexIndicesAttr().Set(indices.tolist(), time=timestep)
    usd_mesh.GetDisplayColorAttr().Set(colors.numpy().tolist(), time=timestep)
    UsdGeom.Primvar(usd_mesh.GetDisplayColorAttr()).SetInterpolation("vertex")

    print(f"Q-criterion visualization at timestep {timestep}:")
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Number of triangles: {tri_count}")
    print(f"  Vorticity range: [{vorticity_min:.6f}, {vorticity_max:.6f}]")


# -------------------------- Function to Save Drone Body and Blades to USD --------------------------
def save_usd_drone_and_blades(
    timestep, post_process_interval, vertices_wp, num_body_vertices, body_faces_np, blades_faces_np, usd_drone_body, usd_drone_blades, usd_stage
):
    # Get vertices and adjust for boundary condition offset
    body_vertices = vertices_wp.numpy()[:num_body_vertices]
    blades_vertices = vertices_wp.numpy()[num_body_vertices:]

    # Adjust vertices by subtracting the boundary condition offset (20 cells)
    body_vertices = body_vertices - 20
    blades_vertices = blades_vertices - 20

    # Process body mesh
    drone_body_tri_count = len(body_faces_np)
    usd_drone_body.GetPointsAttr().Set(body_vertices.tolist(), time=timestep // post_process_interval)
    usd_drone_body.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * drone_body_tri_count), time=timestep // post_process_interval)
    usd_drone_body.GetFaceVertexIndicesAttr().Set(Vt.IntArray(body_faces_np.flatten().tolist()), time=timestep // post_process_interval)

    # Process blades mesh
    # Rebase blade face indices to the local blade vertex array
    blades_faces_np_corrected = blades_faces_np - num_body_vertices
    drone_blades_tri_count = len(blades_faces_np_corrected)
    usd_drone_blades.GetPointsAttr().Set(blades_vertices.tolist(), time=timestep // post_process_interval)
    usd_drone_blades.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * drone_blades_tri_count), time=timestep // post_process_interval)
    usd_drone_blades.GetFaceVertexIndicesAttr().Set(Vt.IntArray(blades_faces_np_corrected.flatten().tolist()), time=timestep // post_process_interval)

    print(f"Drone body and blades USD updated at timestep {timestep}")


# -------------------------- Mesh Loading and IBM Setup --------------------------
def load_and_prepare_meshes(grid_shape, stl_dir="../stl-files/X8_new"):
    if not os.path.exists(stl_dir):
        raise FileNotFoundError(f"STL directory {stl_dir} does not exist.")
    main_body_stl = os.path.join(stl_dir, "X8_Main_Body.stl")
    if not os.path.isfile(main_body_stl):
        raise FileNotFoundError(f"Main body STL file {main_body_stl} not found.")
    main_body_mesh = trimesh.load_mesh(main_body_stl, process=False)
    num_blades = 8
    blade_ranges = []
    cumulative_vertex_count = 0
    target_size = 50.0
    body_scale = target_size / max(main_body_mesh.extents)
    domain_center = np.array([(s - 1) / 2 for s in grid_shape])
    main_body_mesh = transform_mesh(main_body_mesh, translation=domain_center, scale=body_scale)
    body_vertices_wp, body_vertex_areas_wp, body_faces = prepare_immersed_boundary(
        main_body_mesh, max_lbm_length=target_size, translation=None, scale=None
    )
    all_vertices = body_vertices_wp.numpy()
    all_vertex_areas = body_vertex_areas_wp.numpy()
    all_faces = body_faces.copy()
    transformed_centers = []
    for blade_id in range(1, num_blades + 1):
        blade_stl = os.path.join(stl_dir, f"Blade{blade_id}.stl")
        if not os.path.isfile(blade_stl):
            raise FileNotFoundError(f"Blade STL file {blade_stl} not found.")
        blade_mesh = trimesh.load_mesh(blade_stl, process=False)
        blade_com = blade_mesh.center_mass
        blade_mesh = transform_mesh(blade_mesh, translation=domain_center, scale=body_scale)
        blade_vertices_wp, blade_vertex_areas_wp, blade_faces = prepare_immersed_boundary(
            blade_mesh, max_lbm_length=target_size, translation=None, scale=None
        )
        blade_vertices_np = blade_vertices_wp.numpy()
        blade_vertex_areas_np = blade_vertex_areas_wp.numpy()
        blade_faces_offset = blade_faces + cumulative_vertex_count + len(body_vertices_wp)
        all_faces = np.vstack([all_faces, blade_faces_offset])
        all_vertices = np.vstack([all_vertices, blade_vertices_np])
        all_vertex_areas = np.hstack([all_vertex_areas, blade_vertex_areas_np])
        start_idx = cumulative_vertex_count + len(body_vertices_wp)
        end_idx = start_idx + len(blade_vertices_np)
        blade_ranges.append((start_idx, end_idx))
        cumulative_vertex_count += len(blade_vertices_np)
        transformed_com = (blade_com * body_scale) + domain_center
        transformed_centers.append(transformed_com)
        print(f"Blade {blade_id} processed:")
        print(f"  Number of vertices: {len(blade_vertices_np)}")
        print(f"  Center of mass (transformed): {transformed_com}")
    vertices_wp = wp.array(all_vertices, dtype=wp.vec3)
    vertex_areas_wp = wp.array(all_vertex_areas, dtype=wp.float32)
    faces_np = all_faces
    num_body_vertices = len(body_vertices_wp)
    num_blade_vertices = len(all_vertices) - num_body_vertices

    # Separate body and blade faces based on the original ordering
    body_faces_np = body_faces
    blades_faces_np = all_faces[len(body_faces_np) :]

    print("\nBlade Centers (Transformed COM):")
    for i, center in enumerate(transformed_centers):
        print(f"Blade {i + 1}: {center}")
    print(f"\nTotal number of vertices: {len(all_vertices)}")
    print(f"Number of body vertices: {num_body_vertices}")
    print(f"Number of blade vertices: {num_blade_vertices}")
    return {
        "vertices_wp": vertices_wp,
        "vertex_areas_wp": vertex_areas_wp,
        "faces_np": faces_np,
        "blade_ranges": blade_ranges,
        "blade_centers": np.array(transformed_centers),
        "num_body_vertices": num_body_vertices,
        "num_blade_vertices": num_blade_vertices,
        "body_faces_np": body_faces_np,
        "blades_faces_np": blades_faces_np,
    }


# -------------------------- Boundary Conditions --------------------------
def define_boundary_indices(grid, velocity_set):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["top"]
    outlet = box_no_edge["bottom"]
    walls = [box["front"][i] + box["back"][i] + box["left"][i] + box["right"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    return inlet, outlet, walls


def bc_profile(precision_policy, grid_shape, u_max):
    _dtype = precision_policy.store_precision.wp_dtype
    u_max_d = _dtype(u_max)
    L_x = _dtype(grid_shape[0] - 1)
    L_y = _dtype(grid_shape[1] - 1)

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        x = _dtype(index[0])
        y = _dtype(index[1])
        x_center = x - (L_x / _dtype(2.0))
        y_center = y - (L_y / _dtype(2.0))
        r_squared = (_dtype(2.0) * x_center / L_x) ** _dtype(2.0) + (_dtype(2.0) * y_center / L_y) ** _dtype(2.0)
        velocity_z = u_max_d * wp.max(_dtype(0.0), _dtype(1.0) - r_squared)
        return wp.vec(-velocity_z, length=1)

    return bc_profile_warp


def setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max):
    inlet, outlet, walls = define_boundary_indices(grid, velocity_set)
    bc_inlet = RegularizedBC("velocity", indices=inlet, profile=bc_profile(precision_policy, grid_shape, -u_max))
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    return [bc_walls, bc_inlet, bc_outlet]


# -------------------------- Stepper Setup --------------------------
def setup_stepper(grid, boundary_conditions, lbm_omega):
    return IBMStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    )


# -------------------------- Post-Processing (Extended with USD Output using Q Criterion) --------------------------
def post_process(
    i,
    post_process_interval,
    f_current,
    bc_mask,
    grid,
    faces_np,
    vertices_wp,
    precision_policy,
    grid_shape,
    blade_ranges,
    num_blades,
    usd_mesh,
    vorticity_operator,
    usd_stage,
):
    # # Compute macroscopic quantities using the JAX backend for visualization
    # if not isinstance(f_current, jnp.ndarray):
    #     f_jax = wp.to_jax(f_current)
    # macro_jax = Macroscopic(
    #     compute_backend=ComputeBackend.JAX,
    #     precision_policy=precision_policy,
    #     velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
    # )
    # rho, u = macro_jax(f_jax)
    # u = u[:, 1:-1, 1:-1, 1:-1]  # Remove ghost layers if present

    # # Save standard visualization outputs
    # fields = {
    #     "u_magnitude": (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5,
    #     "u_x": u[0],
    #     "u_y": u[1],
    #     "u_z": u[2],
    # }
    # save_fields_vtk(fields, timestep=i)
    # reconstruct_mesh_from_vertices_and_faces(vertices_wp=vertices_wp, faces_np=faces_np, save_path=f"mesh_{i:06d}.stl")
    # save_image(fields["u_magnitude"][grid_shape[0] // 2, :, :], timestep=i)

    # # Plot blade positions
    # plt.figure(figsize=(10, 10))
    # padding = 20
    # x_min = -padding
    # x_max = grid_shape[0] + padding
    # y_min = -padding
    # y_max = grid_shape[1] + padding
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.autoscale(False)
    # for blade_id in range(num_blades):
    #     start, end = blade_ranges[blade_id]
    #     blade_vertices = vertices_wp[start:end].numpy()
    #     plt.scatter(blade_vertices[:, 0], blade_vertices[:, 1], s=20, label=f"Blade {blade_id + 1}")
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", markerscale=0.8)
    # plt.title(f"Blade Positions at Step {i}")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.savefig(f"blade_positions_{i:06d}.png", bbox_inches="tight", dpi=150, pad_inches=0.2)
    # plt.close()

    save_usd_vorticity(
        i,
        post_process_interval,
        bc_mask,
        f_current,
        grid_shape,
        usd_mesh,
        vorticity_operator,
        precision_policy=precision_policy,
        vorticity_threshold=9e-3,
        usd_stage=usd_stage,
    )


# -------------------------- Simulation Parameters --------------------------
grid_shape = (100, 200, 200)
u_max = 0.01
num_steps = 100000
post_process_interval = 100
print_interval = 1000
num_blades = 8
angular_velocity = 0.001

Re = 50000.0
clength = grid_shape[2] - 1
visc = u_max * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)

# -------------------------- Initialize Simulation --------------------------
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

grid = grid_factory(grid_shape, compute_backend=compute_backend)

print("\n" + "=" * 50 + "\n")
print("Simulation Configuration:")
print(f"Grid size: {grid_shape[0]} x {grid_shape[1]} x {grid_shape[2]}")
print(f"Omega: {omega}")
print(f"Backend: {compute_backend}")
print(f"Velocity set: {velocity_set}")
print(f"Precision policy: {precision_policy}")
print(f"Prescribed velocity: {-u_max}")
print(f"Reynolds number: {Re}")
print(f"Max iterations: {num_steps}")
print("\n" + "=" * 50 + "\n")

# -------------------------- Create USD stage and meshes for the outputs --------------------------
usd_output_directory = "usd_output"
os.makedirs(usd_output_directory, exist_ok=True)
usd_file = os.path.join(usd_output_directory, "output.usd")
usd_stage = Usd.Stage.CreateNew(usd_file)
usd_mesh = UsdGeom.Mesh.Define(usd_stage, "/World/FluidField")
usd_drone_body = UsdGeom.Mesh.Define(usd_stage, "/World/DroneBody")
usd_drone_blades = UsdGeom.Mesh.Define(usd_stage, "/World/DroneBlades")

mesh_data = load_and_prepare_meshes(grid_shape)
vertices_wp = mesh_data["vertices_wp"]
vertex_areas_wp = mesh_data["vertex_areas_wp"]
faces_np = mesh_data["faces_np"]
blade_ranges = mesh_data["blade_ranges"]
blade_centers = mesh_data["blade_centers"]
num_body_vertices = mesh_data["num_body_vertices"]
num_blade_vertices = mesh_data["num_blade_vertices"]
body_faces_np = mesh_data["body_faces_np"]
blades_faces_np = mesh_data["blades_faces_np"]

bc_list = setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max)
stepper = setup_stepper(grid, bc_list, omega)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

q_criterion_operator = QCriterion(
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

vorticity_operator = Vorticity(
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)
_num_body_vertices = wp.constant(num_body_vertices)
_num_blades = wp.constant(num_blades)
velocities_wp = wp.zeros(shape=vertices_wp.shape[0], dtype=wp.vec3)
blade_ranges_start = [start for (start, end) in blade_ranges]
blade_ranges_end = [end for (start, end) in blade_ranges]


blade_ranges_start_wp = wp.constant(wp.vec(len(blade_ranges_start), dtype=int)(blade_ranges_start))
blade_ranges_end_wp = wp.constant(wp.vec(len(blade_ranges_end), dtype=int)(blade_ranges_end))

# Flatten blade centers into separate x, y, z components
blade_centers_x = [center[0] for center in blade_centers]
blade_centers_y = [center[1] for center in blade_centers]
blade_centers_z = [center[2] for center in blade_centers]

blade_centers_x_wp = wp.constant(wp.vec(len(blade_centers_x), dtype=float)(blade_centers_x))
blade_centers_y_wp = wp.constant(wp.vec(len(blade_centers_y), dtype=float)(blade_centers_y))
blade_centers_z_wp = wp.constant(wp.vec(len(blade_centers_z), dtype=float)(blade_centers_z))


# -------------------------- Blade Rotation Kernel --------------------------
@wp.kernel
def rotate_blades(
    timestep: int,
    forces: wp.array(dtype=wp.vec3),
    vertices: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    idx = wp.tid()
    if idx < _num_body_vertices:
        velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        return
    blade_id = wp.int32(0)
    for b in range(_num_blades):
        start = blade_ranges_start_wp[b]
        end = blade_ranges_end_wp[b]
        if start <= idx < end:
            blade_id = b
            break
    else:
        velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    # Reconstruct center from flattened components
    center = wp.vec3(blade_centers_x_wp[blade_id], blade_centers_y_wp[blade_id], blade_centers_z_wp[blade_id])

    rel_pos = vertices[idx] - center
    direction = wp.float32(1.0)
    if blade_id == 1 or blade_id == 5 or blade_id == 3 or blade_id == 7:
        direction = wp.float32(-1.0)
    theta = direction * angular_velocity
    cos_theta = wp.cos(theta)
    sin_theta = wp.sin(theta)
    new_rel_pos = wp.vec3(cos_theta * rel_pos[0] - sin_theta * rel_pos[1], sin_theta * rel_pos[0] + cos_theta * rel_pos[1], rel_pos[2])
    vertices[idx] = new_rel_pos + center
    velocities[idx] = wp.vec3(-direction * angular_velocity * rel_pos[1], direction * angular_velocity * rel_pos[0], 0.0)


start_time = time.time()
progress_bar = tqdm(range(num_steps), desc="Simulation Progress", unit="steps")

for i in progress_bar:
    f_0, f_1 = stepper(
        f_0,
        f_1,
        vertices_wp,
        vertex_areas_wp,
        velocities_wp,
        rotate_blades,
        bc_mask,
        missing_mask,
        omega,
        i,
    )
    f_0, f_1 = f_1, f_0

    if (i + 1) % print_interval == 0:
        elapsed_time = time.time() - start_time
        progress_bar.set_postfix({"Time elapsed": f"{elapsed_time:.2f}s"})

    if i % post_process_interval == 0 or i == num_steps - 1:
        post_process(
            i,
            post_process_interval,
            f_0,
            bc_mask,
            grid,
            faces_np,
            vertices_wp,
            precision_policy,
            grid_shape,
            blade_ranges,
            num_blades,
            usd_mesh,
            vorticity_operator,
            usd_stage,
        )
        save_usd_drone_and_blades(
            i,
            post_process_interval,
            vertices_wp,
            num_body_vertices,
            body_faces_np,
            blades_faces_np,
            usd_drone_body,
            usd_drone_blades,
            usd_stage,
        )

usd_stage.SetStartTimeCode(0)
usd_stage.SetEndTimeCode(num_steps // post_process_interval)
usd_stage.SetTimeCodesPerSecond(30)

usd_stage.Save()
