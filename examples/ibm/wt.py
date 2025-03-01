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
from mpl_toolkits.mplot3d import Axes3D
from xlb.helper.ibm_helper import (
    prepare_immersed_boundary,
    transform_mesh,
    reconstruct_mesh_from_vertices_and_faces,
)
from xlb.grid import grid_factory
from pxr import Usd, UsdGeom, Vt
from xlb.operator.postprocess import QCriterion, Vorticity, GridToPoint


@wp.kernel
def get_color(
    low: float,
    high: float,
    values: wp.array(dtype=float),
    out_color: wp.array(dtype=wp.vec3),
):
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


def grid_to_point(field, verts, out):
    verts_np = verts.numpy()
    nx, ny, nz = field.shape
    out_np = out.numpy()
    for i, v in enumerate(verts_np):
        x = int(round(v[0]))
        y = int(round(v[1]))
        z = int(round(v[2]))
        x = min(max(x, 0), nx - 1)
        y = min(max(y, 0), ny - 1)
        z = min(max(z, 0), nz - 1)
        out_np[i] = field[x, y, z]
    return wp.array(out_np, dtype=wp.float32, device=out.device)


def save_usd_vorticity(
    timestep,
    post_process_interval,
    bc_mask,
    f_current,
    grid_shape,
    usd_mesh,
    vorticity_operator,
    precision_policy,
    vorticity_threshold,
    usd_stage,
):
    device = "cuda:1"
    f_current_new = wp.clone(f_current, device=device)
    bc_mask_new = wp.clone(bc_mask, device=device)
    with wp.ScopedDevice(device):
        macro_wp = Macroscopic(
            compute_backend=ComputeBackend.WARP,
            precision_policy=precision_policy,
            velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP),
        )
        rho = wp.zeros((1, *grid_shape), dtype=wp.float32, device=device)
        u = wp.zeros((3, *grid_shape), dtype=wp.float32, device=device)
        rho, u = macro_wp(f_current_new, rho, u)
        u = u[:, 20:-20, 20:-20, 0:-20]

        vorticity = wp.zeros((3, *u.shape[1:]), dtype=wp.float32, device=device)
        vorticity_magnitude = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)
        vorticity, vorticity_magnitude = vorticity_operator(u, bc_mask_new, vorticity, vorticity_magnitude)

        max_verts = grid_shape[0] * grid_shape[1] * grid_shape[2] * 5
        max_tris = grid_shape[0] * grid_shape[1] * grid_shape[2] * 3
        mc = wp.MarchingCubes(nx=u.shape[1], ny=u.shape[2], nz=u.shape[3], max_verts=max_verts, max_tris=max_tris, device=device)
        mc.surface(vorticity_magnitude[0], vorticity_threshold)
        if mc.verts.shape[0] == 0:
            print(f"Warning: No vertices found for vorticity at timestep {timestep}.")
            return

        grid_to_point_op = GridToPoint(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP)
        scalars = wp.zeros(mc.verts.shape[0], dtype=wp.float32, device=device)
        scalars = grid_to_point_op(vorticity_magnitude, mc.verts, scalars)

        colors = wp.empty(mc.verts.shape[0], dtype=wp.vec3, device=device)
        scalars_np = scalars.numpy()
        vorticity_min = float(np.percentile(scalars_np, 5))
        vorticity_max = float(np.percentile(scalars_np, 95))
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

    print(f"Vorticity visualization at timestep {timestep}:")
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Number of triangles: {tri_count}")
    print(f"  Vorticity range: [{vorticity_min:.6f}, {vorticity_max:.6f}]")


def save_usd_q_criterion(
    timestep,
    post_process_interval,
    bc_mask,
    f_current,
    grid_shape,
    usd_mesh,
    q_criterion_operator,
    precision_policy,
    q_threshold,
    usd_stage,
):
    device = "cuda:1"
    f_current_new = wp.clone(f_current, device=device)
    bc_mask_new = wp.clone(bc_mask, device=device)
    with wp.ScopedDevice(device):
        macro_wp = Macroscopic(
            compute_backend=ComputeBackend.WARP,
            precision_policy=precision_policy,
            velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP),
        )
        rho = wp.zeros((1, *grid_shape), dtype=wp.float32, device=device)
        u = wp.zeros((3, *grid_shape), dtype=wp.float32, device=device)
        rho, u = macro_wp(f_current_new, rho, u)
        u = u[:, 20:-20, 20:-20, 0:-20]

        norm_mu = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)
        q_field = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)
        norm_mu, q_field = q_criterion_operator(u, bc_mask_new, norm_mu, q_field)

        max_verts = grid_shape[0] * grid_shape[1] * grid_shape[2] * 5
        max_tris = grid_shape[0] * grid_shape[1] * grid_shape[2] * 3
        mc = wp.MarchingCubes(nx=u.shape[1], ny=u.shape[2], nz=u.shape[3], max_verts=max_verts, max_tris=max_tris, device=device)
        mc.surface(q_field[0], q_threshold)
        if mc.verts.shape[0] == 0:
            print(f"Warning: No vertices found for Q-criterion at timestep {timestep}.")
            return

        grid_to_point_op = GridToPoint(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP)
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

    usd_mesh.GetPointsAttr().Set(vertices.tolist(), time=timestep // post_process_interval)
    usd_mesh.GetFaceVertexCountsAttr().Set([3] * tri_count, time=timestep // post_process_interval)
    usd_mesh.GetFaceVertexIndicesAttr().Set(indices.tolist(), time=timestep // post_process_interval)
    usd_mesh.GetDisplayColorAttr().Set(colors.numpy().tolist(), time=timestep // post_process_interval)
    UsdGeom.Primvar(usd_mesh.GetDisplayColorAttr()).SetInterpolation("vertex")

    print(f"Q-criterion visualization at timestep {timestep}:")
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Number of triangles: {tri_count}")
    print(f"  Vorticity range: [{vorticity_min:.6f}, {vorticity_max:.6f}]")


def save_usd_car_parts(
    timestep,
    post_process_interval,
    vertices_wp,
    num_body_vertices,
    body_faces_np,
    wheels_faces_np,
    usd_car_body,
    usd_car_wheels,
    usd_stage,
):
    # Apply offset to match velocity field slicing:
    # X/Y: 20:-20 slice requires -20 offset
    # Z: 0:-20 slice requires no offset
    offset = np.array([20.0, 20.0, 0.0])
    
    body_vertices = vertices_wp.numpy()[:num_body_vertices] - offset
    wheels_vertices = vertices_wp.numpy()[num_body_vertices:] - offset
    car_body_tri_count = len(body_faces_np)

    usd_car_body.GetPointsAttr().Set(body_vertices.tolist(), time=timestep // post_process_interval)
    usd_car_body.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * car_body_tri_count), time=timestep // post_process_interval)
    usd_car_body.GetFaceVertexIndicesAttr().Set(Vt.IntArray(body_faces_np.flatten().tolist()), time=timestep // post_process_interval)

    wheels_faces_np_corrected = wheels_faces_np - num_body_vertices
    car_wheels_tri_count = len(wheels_faces_np_corrected)

    usd_car_wheels.GetPointsAttr().Set(wheels_vertices.tolist(), time=timestep // post_process_interval)
    usd_car_wheels.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * car_wheels_tri_count), time=timestep // post_process_interval)
    usd_car_wheels.GetFaceVertexIndicesAttr().Set(Vt.IntArray(wheels_faces_np_corrected.flatten().tolist()), time=timestep // post_process_interval)

    print(f"Car body and wheels USD updated at timestep {timestep}")


def visualize_car_placement(vertices_wp, grid_shape):
    verts = vertices_wp.numpy()
    car_min = verts.min(axis=0)
    car_max = verts.max(axis=0)

    plt.figure(figsize=(10, 5))
    domain_x = [0, grid_shape[0], grid_shape[0], 0, 0]
    domain_y = [0, 0, grid_shape[1], grid_shape[1], 0]
    plt.plot(domain_x, domain_y, "k-", label="Domain", linewidth=1)

    car_x = [car_min[0], car_max[0], car_max[0], car_min[0], car_min[0]]
    car_y = [car_min[1], car_min[1], car_max[1], car_max[1], car_min[1]]
    plt.plot(car_x, car_y, "r-", linewidth=2, label="Car")

    # target_x = grid_shape[0] / 3.0
    # plt.axvline(x=target_x, color='b', linestyle='--', label='Target Front')

    plt.title("Car Placement (Top View)")
    plt.xlabel("X (Flow Direction →)")
    plt.ylabel("Y (Width)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.axis("equal")
    plt.legend()

    plt.savefig("car_placement.png", dpi=150, bbox_inches="tight")
    plt.close()


def load_and_prepare_meshes_car(grid_shape, stl_dir="/home/mehdi/Repos/stl-files/ford"):
    if not os.path.exists(stl_dir):
        raise FileNotFoundError(f"STL directory {stl_dir} does not exist.")

    body_stl = os.path.join(stl_dir, "S550_GT500_BS_5p.stl")
    wheel_fr_stl = os.path.join(stl_dir, "S550_GT500_BS_FR_5p.stl")
    wheel_fl_stl = os.path.join(stl_dir, "S550_GT500_BS_FL_5p.stl")
    wheel_rr_stl = os.path.join(stl_dir, "S550_GT500_BS_RR_5p.stl")
    wheel_rl_stl = os.path.join(stl_dir, "S550_GT500_BS_RL_5p.stl")
    for stl_file in [body_stl, wheel_fr_stl, wheel_fl_stl, wheel_rr_stl, wheel_rl_stl]:
        if not os.path.isfile(stl_file):
            raise FileNotFoundError(f"STL file {stl_file} not found.")

    # Load body FIRST to get reference dimensions
    body_mesh = trimesh.load_mesh(body_stl, process=False)
    body_bounds = body_mesh.bounds
    orig_body_width = body_bounds[1][1] - body_bounds[0][1]  # Use body width for scaling

    # Now load other components
    fr_mesh = trimesh.load_mesh(wheel_fr_stl, process=False)
    fl_mesh = trimesh.load_mesh(wheel_fl_stl, process=False)
    rr_mesh = trimesh.load_mesh(wheel_rr_stl, process=False)
    rl_mesh = trimesh.load_mesh(wheel_rl_stl, process=False)

    # Calculate scale based on BODY width only
    Nx, Ny, Nz = grid_shape
    target_body_width = Ny / 3.0
    scale_factor = target_body_width / orig_body_width  # Use body's original width

    # Apply scale to ALL components
    body_mesh.apply_scale(scale_factor)
    fr_mesh.apply_scale(scale_factor)
    fl_mesh.apply_scale(scale_factor)
    rr_mesh.apply_scale(scale_factor)
    rl_mesh.apply_scale(scale_factor)

    # Now combine scaled meshes
    combined = trimesh.util.concatenate([body_mesh, fr_mesh, fl_mesh, rr_mesh, rl_mesh])

    R_y = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
    R_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    combined.apply_transform(R_y)
    combined.apply_transform(R_x)

    bnds = combined.bounds
    min_x, min_y, min_z = bnds[0]
    max_x, max_y, max_z = bnds[1]
    car_length = max_x - min_x
    car_height = max_z - min_z

    target_front_x = Nx / 3.0
    shift_x = target_front_x - min_x

    back_x_after_shift = max_x + shift_x
    if back_x_after_shift > Nx:
        raise ValueError(
            f"Car too long to fit in domain with front at 1/3!\n"
            f"Car length: {car_length:.1f}\n"
            f"Available space: {Nx - target_front_x:.1f}\n"
            f"Would need additional {back_x_after_shift - Nx:.1f} units"
        )

    shift_y = (Ny - (max_y - min_y)) / 2.0 - min_y

    ground_clearance = 2.0
    shift_z = ground_clearance - min_z

    combined.apply_translation([shift_x, shift_y, shift_z])

    bnds = combined.bounds
    min_x, min_y, min_z = bnds[0]
    max_x, max_y, max_z = bnds[1]

    tolerance = 0.1
    front_tolerance = target_front_x * tolerance
    width_tolerance = target_body_width * tolerance

    if not (target_front_x - front_tolerance < min_x < target_front_x + front_tolerance):
        raise ValueError(f"Car front not at 1/3 domain length! Front at {min_x:.1f}, should be {target_front_x:.1f}")

    scaled_body_width = body_mesh.bounds[1][1] - body_mesh.bounds[0][1]
    if not (target_body_width - width_tolerance < scaled_body_width < target_body_width + width_tolerance):
        raise ValueError(f"Car BODY width mismatch! Current: {scaled_body_width:.1f}, Target: {target_body_width:.1f}")

    print("\nScaled dimensions verification:")
    print(f"Body width: {scaled_body_width:.1f} (target: {target_body_width:.1f})")
    print(f"Total car width (body + wheels): {max_y - min_y:.1f}")

    def apply_transform(mesh):
        mesh.apply_scale(scale_factor)
        mesh.apply_transform(R_y)
        mesh.apply_transform(R_x)
        mesh.apply_translation([shift_x, shift_y, shift_z])

    body_mesh = trimesh.load_mesh(body_stl, process=False)
    fr_mesh = trimesh.load_mesh(wheel_fr_stl, process=False)
    fl_mesh = trimesh.load_mesh(wheel_fl_stl, process=False)
    rr_mesh = trimesh.load_mesh(wheel_rr_stl, process=False)
    rl_mesh = trimesh.load_mesh(wheel_rl_stl, process=False)

    apply_transform(body_mesh)
    apply_transform(fr_mesh)
    apply_transform(fl_mesh)
    apply_transform(rr_mesh)
    apply_transform(rl_mesh)

    body_v_wp, body_a_wp, body_faces = prepare_immersed_boundary(body_mesh, max_lbm_length=Ny / 2.0)
    fr_v_wp, fr_a_wp, fr_faces = prepare_immersed_boundary(fr_mesh, max_lbm_length=Ny / 2.0)
    fl_v_wp, fl_a_wp, fl_faces = prepare_immersed_boundary(fl_mesh, max_lbm_length=Ny / 2.0)
    rr_v_wp, rr_a_wp, rr_faces = prepare_immersed_boundary(rr_mesh, max_lbm_length=Ny / 2.0)
    rl_v_wp, rl_a_wp, rl_faces = prepare_immersed_boundary(rl_mesh, max_lbm_length=Ny / 2.0)

    all_vertices = body_v_wp.numpy()
    all_areas = body_a_wp.numpy()
    all_faces = body_faces.copy()
    current_offset = len(body_v_wp)

    wheels = [
        (fr_v_wp, fr_a_wp, fr_faces),
        (fl_v_wp, fl_a_wp, fl_faces),
        (rr_v_wp, rr_a_wp, rr_faces),
        (rl_v_wp, rl_a_wp, rl_faces),
    ]
    wheel_ranges = []
    for wv_wp, wa_wp, wf in wheels:
        wv_np = wv_wp.numpy()
        wa_np = wa_wp.numpy()
        wf_offset = wf + current_offset
        all_vertices = np.vstack([all_vertices, wv_np])
        all_areas = np.hstack([all_areas, wa_np])
        all_faces = np.vstack([all_faces, wf_offset])
        start_idx = current_offset
        end_idx = start_idx + len(wv_np)
        wheel_ranges.append((start_idx, end_idx))
        current_offset += len(wv_np)

    vertices_wp = wp.array(all_vertices, dtype=wp.vec3)
    areas_wp = wp.array(all_areas, dtype=wp.float32)
    faces_np = all_faces
    num_body_vertices = len(body_v_wp)
    num_wheel_vertices = len(all_vertices) - num_body_vertices
    body_faces_np = body_faces
    wheels_faces_np = faces_np[len(body_faces_np) :]

    wheel_centers = []
    for start_idx, end_idx in wheel_ranges:
        center = np.mean(all_vertices[start_idx:end_idx], axis=0)
        wheel_centers.append(center)

    print("\nMesh preparation summary:")
    print(f"Target size (car width): {target_body_width:.2f}")
    print(f"Scale factor applied: {scale_factor:.4f}")
    print(f"Total vertices: {len(all_vertices)}")
    print(f"Body vertices: {num_body_vertices}")
    print(f"Wheel vertices: {num_wheel_vertices}")
    print(f"Number of wheels: {len(wheel_ranges)}")

    return {
        "vertices_wp": vertices_wp,
        "areas_wp": areas_wp,
        "faces_np": faces_np,
        "wheel_ranges": wheel_ranges,
        "wheel_centers": wheel_centers,
        "num_body_vertices": num_body_vertices,
        "num_wheel_vertices": num_wheel_vertices,
        "body_faces_np": body_faces_np,
        "wheels_faces_np": wheels_faces_np,
    }


def define_boundary_indices(grid, velocity_set):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["right"]
    outlet = box_no_edge["left"]
    walls = [box["front"][i] + box["back"][i] + box["top"][i] + box["bottom"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    return inlet, outlet, walls


def bc_profile(precision_policy, grid_shape, u_max):
    _dtype = precision_policy.store_precision.wp_dtype
    u_max_d = _dtype(u_max)
    L_y = _dtype(grid_shape[1] - 1)
    L_z = _dtype(grid_shape[2] - 1)

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        y = _dtype(index[1])
        z = _dtype(index[2])
        y_center = y - (L_y / _dtype(2.0))
        z_center = z - (L_z / _dtype(2.0))
        r_sq = ((_dtype(2.0) * y_center) / L_y) ** _dtype(2.0) + ((_dtype(2.0) * z_center) / L_z) ** _dtype(2.0)
        velocity_x = u_max_d * wp.max(_dtype(0.0), _dtype(1.0) - r_sq)
        return wp.vec(velocity_x, length=1)

    return bc_profile_warp


def setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max):
    inlet, outlet, walls = define_boundary_indices(grid, velocity_set)
    bc_inlet = RegularizedBC("velocity", indices=inlet, profile=bc_profile(precision_policy, grid_shape, u_max))
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    return [bc_walls, bc_inlet, bc_outlet]


def setup_stepper(grid, boundary_conditions, lbm_omega):
    return IBMStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    )


@wp.kernel
def rotate_wheels(
    timestep: int,
    forces: wp.array(dtype=wp.vec3),
    vertices: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    idx = wp.tid()

    if idx < _num_body_vertices:
        velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    wheel_id = wp.int32(-1)
    for b in range(_num_wheels):
        start = _wheel_starts[b]
        end = _wheel_ends[b]
        if (idx >= start) and (idx < end):
            wheel_id = b
            break

    if wheel_id == -1:
        velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    center = wp.vec3(
        _wheel_centers_x[wheel_id],
        _wheel_centers_y[wheel_id],
        _wheel_centers_z[wheel_id],
    )

    rel_pos = vertices[idx] - center
    theta = _wheel_speed
    c = wp.cos(theta)
    s = wp.sin(theta)

    x_new = c * rel_pos[0] - s * rel_pos[2]
    z_new = s * rel_pos[0] + c * rel_pos[2]
    new_rel_pos = wp.vec3(x_new, rel_pos[1], z_new)
    vertices[idx] = new_rel_pos + center
    velocities[idx] = wp.vec3(-_wheel_speed * rel_pos[2], 0.0, _wheel_speed * rel_pos[0])


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
    wheel_ranges,
    num_wheels,
    usd_mesh_vorticity,
    usd_mesh_q_criterion,
    vorticity_operator,
    q_criterion_operator,
    usd_stage,
):
    if not isinstance(f_current, jnp.ndarray):
        f_jax = wp.to_jax(f_current)
    else:
        f_jax = f_current

    macro_jax = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
    )
    rho, u = macro_jax(f_jax)
    u = u[:, 1:-1, 1:-1, 1:-1]

    fields = {
        "u_magnitude": (u[0] ** 2.0 + u[1] ** 2.0 + u[2] ** 2.0) ** 0.5,
        "u_x": u[0],
        "u_y": u[1],
        "u_z": u[2],
    }
    slice_idy = grid_shape[1] // 2
    save_image(fields["u_magnitude"][:, slice_idy, :], timestep=i)
    # save_fields_vtk(fields, i)

    save_usd_vorticity(
        i,
        post_process_interval,
        bc_mask,
        f_current,
        grid_shape,
        usd_mesh_vorticity,
        vorticity_operator,
        precision_policy=precision_policy,
        vorticity_threshold=1e-2,
        usd_stage=usd_stage,
    )

    save_usd_q_criterion(
        i,
        post_process_interval,
        bc_mask,
        f_current,
        grid_shape,
        usd_mesh_q_criterion,
        q_criterion_operator,
        precision_policy=precision_policy,
        q_threshold=5e-6,
        usd_stage=usd_stage,
    )

grid_shape = (1024, 500, 200)
u_max = 0.02
iter_per_flow_passes = grid_shape[0] / u_max
num_steps = 2
post_process_interval = 1000
print_interval = 1000
num_wheels = 4
wheel_rotation_speed = -0.002

Re = 2e6
clength = grid_shape[0] - 1
visc = u_max * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)
grid = grid_factory(grid_shape, compute_backend=compute_backend)

print("Car Simulation Configuration:")
print(f"  Grid size: {grid_shape}")
print(f"  Omega: {omega}")
print(f"  Backend: {compute_backend}")
print(f"  Velocity set: {velocity_set}")
print(f"  Precision policy: {precision_policy}")
print(f"  Inlet velocity: {u_max}")
print(f"  Reynolds number: {Re}")
print(f"  Max steps: {num_steps}")

usd_output_directory = "usd_output_car"
os.makedirs(usd_output_directory, exist_ok=True)
usd_file = os.path.join(usd_output_directory, "car_output.usd")
usd_stage = Usd.Stage.CreateNew(usd_file)
usd_mesh_vorticity = UsdGeom.Mesh.Define(usd_stage, "/World/Vorticity")
usd_mesh_q_criterion = UsdGeom.Mesh.Define(usd_stage, "/World/QCriterion")
usd_car_body = UsdGeom.Mesh.Define(usd_stage, "/World/CarBody")
usd_car_wheels = UsdGeom.Mesh.Define(usd_stage, "/World/CarWheels")

mesh_data = load_and_prepare_meshes_car(grid_shape)
vertices_wp = mesh_data["vertices_wp"]
areas_wp = mesh_data["areas_wp"]
faces_np = mesh_data["faces_np"]
wheel_ranges = mesh_data["wheel_ranges"]
wheel_centers = mesh_data["wheel_centers"]
num_body_vertices = mesh_data["num_body_vertices"]
num_wheel_vertices = mesh_data["num_wheel_vertices"]
body_faces_np = mesh_data["body_faces_np"]
wheels_faces_np = mesh_data["wheels_faces_np"]

visualize_car_placement(vertices_wp, grid_shape)

_num_body_vertices = wp.constant(num_body_vertices)
_num_wheels = wp.constant(num_wheels)
_wheel_speed = wp.constant(wheel_rotation_speed)

starts_list = [rng[0] for rng in wheel_ranges]
ends_list = [rng[1] for rng in wheel_ranges]
cx_list = [c[0] for c in wheel_centers]
cy_list = [c[1] for c in wheel_centers]
cz_list = [c[2] for c in wheel_centers]

_wheel_starts = wp.constant(wp.vec(len(starts_list), dtype=int)(starts_list))
_wheel_ends = wp.constant(wp.vec(len(ends_list), dtype=int)(ends_list))
_wheel_centers_x = wp.constant(wp.vec(len(cx_list), dtype=float)(cx_list))
_wheel_centers_y = wp.constant(wp.vec(len(cy_list), dtype=float)(cy_list))
_wheel_centers_z = wp.constant(wp.vec(len(cz_list), dtype=float)(cz_list))

bc_list = setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max)
stepper = setup_stepper(grid, bc_list, omega)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

device = "cuda:1"
with wp.ScopedDevice(device):
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

velocities_wp = wp.zeros(shape=vertices_wp.shape[0], dtype=wp.vec3)

start_time = time.time()
progress_bar = tqdm(range(num_steps), desc="Simulation Progress", unit="steps")

for i in progress_bar:
    f_0, f_1 = stepper(
        f_0,
        f_1,
        vertices_wp,
        areas_wp,
        velocities_wp,
        rotate_wheels,
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
            wheel_ranges,
            num_wheels,
            usd_mesh_vorticity,
            usd_mesh_q_criterion,
            vorticity_operator,
            q_criterion_operator,
            usd_stage,
        )
        save_usd_car_parts(
            i,
            post_process_interval,
            vertices_wp,
            num_body_vertices,
            body_faces_np,
            wheels_faces_np,
            usd_car_body,
            usd_car_wheels,
            usd_stage,
        )

usd_stage.SetStartTimeCode(0)
usd_stage.SetEndTimeCode(num_steps // post_process_interval)
usd_stage.SetTimeCodesPerSecond(30)
usd_stage.Save()
print("Simulation finished. USD file saved.")
