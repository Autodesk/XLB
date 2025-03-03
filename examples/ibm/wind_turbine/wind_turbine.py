import os
import xlb
import trimesh
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
import matplotlib.pyplot as plt
from xlb.helper.ibm_helper import prepare_immersed_boundary
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
        # Clip a boundary slice just for clarity in visualization
        u = u[:, 20:-20, 20:-20, 5:-20]

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
        # Clip a boundary slice just for clarity
        u = u[:, 20:-20, 20:-20, 5:-20]

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


def save_usd_turbine_parts(
    timestep,
    post_process_interval,
    vertices_wp,
    num_body_vertices,
    body_faces_np,
    rotor_faces_np,
    usd_turbine_body,
    usd_turbine_rotor,
    usd_stage,
    lag_forces=None,
):
    offset = np.array([20, 20, 5])
    body_vertices = vertices_wp.numpy()[:num_body_vertices] - offset
    rotor_vertices = vertices_wp.numpy()[num_body_vertices:] - offset
    body_tri_count = len(body_faces_np)

    usd_turbine_body.GetPointsAttr().Set(body_vertices.tolist(), time=timestep // post_process_interval)
    usd_turbine_body.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * body_tri_count), time=timestep // post_process_interval)
    usd_turbine_body.GetFaceVertexIndicesAttr().Set(Vt.IntArray(body_faces_np.flatten().tolist()), time=timestep // post_process_interval)

    rotor_faces_np_corrected = rotor_faces_np - num_body_vertices
    rotor_tri_count = len(rotor_faces_np_corrected)

    usd_turbine_rotor.GetPointsAttr().Set(rotor_vertices.tolist(), time=timestep // post_process_interval)
    usd_turbine_rotor.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * rotor_tri_count), time=timestep // post_process_interval)
    usd_turbine_rotor.GetFaceVertexIndicesAttr().Set(Vt.IntArray(rotor_faces_np_corrected.flatten().tolist()), time=timestep // post_process_interval)

    if lag_forces is not None:
        body_force_np = lag_forces.numpy()[:num_body_vertices, 0]
        rotor_force_np = lag_forces.numpy()[num_body_vertices:, 0]

        min_body_force = np.min(body_force_np)
        max_body_force = np.max(body_force_np)
        min_rotor_force = np.min(rotor_force_np)
        max_rotor_force = np.max(rotor_force_np)

        body_colors = wp.zeros(num_body_vertices, dtype=wp.vec3)
        rotor_colors = wp.zeros(vertices_wp.shape[0] - num_body_vertices, dtype=wp.vec3)

        wp.launch(
            kernel=get_color,
            dim=num_body_vertices,
            inputs=[min_body_force, max_body_force, wp.from_numpy(body_force_np), body_colors],
        )

        wp.launch(
            kernel=get_color,
            dim=vertices_wp.shape[0] - num_body_vertices,
            inputs=[min_rotor_force, max_rotor_force, wp.from_numpy(rotor_force_np), rotor_colors],
        )

        body_colors_np = body_colors.numpy()
        rotor_colors_np = rotor_colors.numpy()

        usd_turbine_body.GetDisplayColorAttr().Set(body_colors_np.tolist(), time=timestep // post_process_interval)
        UsdGeom.Primvar(usd_turbine_body.GetDisplayColorAttr()).SetInterpolation("vertex")

        usd_turbine_rotor.GetDisplayColorAttr().Set(rotor_colors_np.tolist(), time=timestep // post_process_interval)
        UsdGeom.Primvar(usd_turbine_rotor.GetDisplayColorAttr()).SetInterpolation("vertex")

    print(f"Turbine mesh updated at timestep {timestep}")


def define_boundary_indices(grid, velocity_set):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["front"]
    outlet = box_no_edge["back"]
    walls = [box["right"][i] + box["left"][i] + box["top"][i] + box["bottom"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    return inlet, outlet, walls


def bc_profile(precision_policy, grid_shape, u_max):
    _dtype = precision_policy.store_precision.wp_dtype
    u_max_d = _dtype(u_max)
    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        return wp.vec(u_max_d, length=1)

    return bc_profile_warp

def setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, inlet_speed):
    inlet, outlet, walls = define_boundary_indices(grid, velocity_set)
    bc_inlet = RegularizedBC("velocity", indices=inlet, profile=bc_profile(precision_policy, grid_shape, inlet_speed))
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    return [bc_inlet, bc_outlet, bc_walls]


def setup_stepper(grid, boundary_conditions, lbm_omega):
    return IBMStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    )


def visualize_turbine_placement(vertices_wp, grid_shape):
    verts = vertices_wp.numpy()
    turbine_min = verts.min(axis=0)
    turbine_max = verts.max(axis=0)

    plt.figure(figsize=(10, 5))
    domain_x = [0, grid_shape[0], grid_shape[0], 0, 0]
    domain_y = [0, 0, grid_shape[1], grid_shape[1], 0]
    plt.plot(domain_x, domain_y, "k-", label="Domain", linewidth=1)

    # Show the bounding box of the turbine in X-Y plane (ignore Z)
    tx = [turbine_min[0], turbine_max[0], turbine_max[0], turbine_min[0], turbine_min[0]]
    ty = [turbine_min[1], turbine_min[1], turbine_max[1], turbine_max[1], turbine_min[1]]
    plt.plot(tx, ty, "r-", linewidth=2, label="Turbine bounding box")

    plt.title("Turbine Placement (X-Y Top View)")
    plt.xlabel("X (left→right)")
    plt.ylabel("Y (front→back)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.axis("equal")
    plt.legend()

    plt.savefig("turbine_placement.png", dpi=150, bbox_inches="tight")
    plt.close()


def load_and_prepare_meshes_turbine(grid_shape, stl_dir="./examples/ibm/wind_turbine/"):
    rotor_stl = os.path.join(stl_dir, "turbine_wind_turbine.stl")
    body_stl = os.path.join(stl_dir, "body_wind_turbine.stl")
    if not os.path.isfile(rotor_stl):
        raise FileNotFoundError(f"Cannot find {rotor_stl}")
    if not os.path.isfile(body_stl):
        raise FileNotFoundError(f"Cannot find {body_stl}")

    rotor_mesh = trimesh.load_mesh(rotor_stl, process=False)
    body_mesh = trimesh.load_mesh(body_stl, process=False)

    # Identify rotor bounding box dimension that dictates scaling
    rotor_bounds = rotor_mesh.bounds
    rotor_size = rotor_bounds[1] - rotor_bounds[0]
    rotor_diameter = max(rotor_size)  # largest dimension in the rotor
    desired_diameter = 150.0

    scale_factor = desired_diameter / rotor_diameter
    print(f"Scale factor: {scale_factor:.4f}")

    # We apply the scale to both rotor and body
    rotor_mesh.apply_scale(scale_factor)
    body_mesh.apply_scale(scale_factor)

    R_dummy = trimesh.transformations.rotation_matrix(np.radians(0), [1, 0, 0])
    rotor_mesh.apply_transform(R_dummy)
    body_mesh.apply_transform(R_dummy)

    combined = trimesh.util.concatenate([rotor_mesh, body_mesh])
    bnds = combined.bounds
    min_x, min_y, min_z = bnds[0]
    max_x, max_y, max_z = bnds[1]
    center_x = 0.5 * grid_shape[0]
    center_y = 0.5 * grid_shape[1]

    shift_x = center_x - 0.5 * (min_x + max_x)
    shift_y = center_y - 0.5 * (min_y + max_y)
    shift_z = -min_z

    combined.apply_translation([shift_x, shift_y, shift_z])
    rotor_mesh = trimesh.load_mesh(rotor_stl, process=False)
    body_mesh = trimesh.load_mesh(body_stl, process=False)
    rotor_mesh.apply_scale(scale_factor)
    body_mesh.apply_scale(scale_factor)
    rotor_mesh.apply_transform(R_dummy)
    body_mesh.apply_transform(R_dummy)
    rotor_mesh.apply_translation([shift_x, shift_y, shift_z])
    body_mesh.apply_translation([shift_x, shift_y, shift_z])

    Nx, Ny, Nz = grid_shape
    rotor_v_wp, rotor_a_wp, rotor_faces = prepare_immersed_boundary(rotor_mesh, max_lbm_length=max(Nx, Ny, Nz))
    body_v_wp, body_a_wp, body_faces = prepare_immersed_boundary(body_mesh, max_lbm_length=max(Nx, Ny, Nz))

    all_vertices = body_v_wp.numpy()
    all_areas = body_a_wp.numpy()
    all_faces = body_faces.copy()
    current_offset = len(body_v_wp)

    rotor_v_np = rotor_v_wp.numpy()
    rotor_a_np = rotor_a_wp.numpy()
    rotor_f_offset = rotor_faces + current_offset
    all_vertices = np.vstack([all_vertices, rotor_v_np])
    all_areas = np.hstack([all_areas, rotor_a_np])
    all_faces = np.vstack([all_faces, rotor_f_offset])

    vertices_wp = wp.array(all_vertices, dtype=wp.vec3)
    areas_wp = wp.array(all_areas, dtype=wp.float32)
    faces_np = all_faces
    num_body_vertices = len(body_v_wp)
    num_rotor_vertices = len(rotor_v_wp)
    body_faces_np = body_faces
    rotor_faces_np = faces_np[len(body_faces_np):]

    print("\nTurbine mesh preparation summary:")
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  Rotor diameter (desired ~200): {desired_diameter}")
    print(f"  Total vertices: {len(all_vertices)}")
    print(f"  Body vertices: {num_body_vertices}")
    print(f"  Rotor vertices: {num_rotor_vertices}")

    return {
        "vertices_wp": vertices_wp,
        "areas_wp": areas_wp,
        "faces_np": faces_np,
        "num_body_vertices": num_body_vertices,
        "num_rotor_vertices": num_rotor_vertices,
        "body_faces_np": body_faces_np,
        "rotor_faces_np": rotor_faces_np,
    }


@wp.kernel
def rotate_rotor(
    timestep: int,
    forces: wp.array(dtype=wp.vec3),
    vertices: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    idx = wp.tid()

    if idx < _num_body_vertices:
        velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    # For rotor vertices, rotate about the negative Y axis (axis = -Y).
    center = wp.vec3(_rotor_center_x, _rotor_center_y, _rotor_center_z)
    pos = vertices[idx]
    rel_pos = pos - center

    radius_x = rel_pos[0]
    radius_z = rel_pos[2]
    r = wp.sqrt(radius_x * radius_x + radius_z * radius_z)
    if r < 1e-6:
        velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    nx = radius_x / r
    nz = radius_z / r
    theta = _rotor_speed
    c = wp.cos(theta)
    s = wp.sin(theta)

    x_new = r * (c * nx - s * nz)
    z_new = r * (s * nx + c * nz)

    # Update position
    new_rel = wp.vec3(x_new, rel_pos[1], z_new)
    vertices[idx] = new_rel + center

    # Tangential velocity from rotation
    velocities[idx] = wp.vec3(z_new * _rotor_speed, 0.0, -x_new * _rotor_speed)


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
    usd_mesh_vorticity,
    usd_mesh_q_criterion,
    vorticity_operator,
    q_criterion_operator,
    usd_stage,
    turbine_body_mesh,
    turbine_rotor_mesh,
    num_body_vertices,
    body_faces_np,
    rotor_faces_np,
    lag_forces=None,
):
    # if not isinstance(f_current, jnp.ndarray):
    #     f_jax = wp.to_jax(f_current)
    # else:
    #     f_jax = f_current

    # macro_jax = Macroscopic(
    #     compute_backend=ComputeBackend.JAX,
    #     precision_policy=precision_policy,
    #     velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
    # )
    # rho, u = macro_jax(f_jax)
    # u = u[:, 20:-20, 20:-20, 5:-20]

    # fields = {
    #     "u_magnitude": (u[0] ** 2.0 + u[1] ** 2.0 + u[2] ** 2.0) ** 0.5,
    #     "u_x": u[0],
    #     "u_y": u[1],
    #     "u_z": u[2],
    # }

    # slice_idx = grid_shape[0] // 2
    # slice_idy = grid_shape[1] // 2
    # save_image(fields["u_magnitude"][slice_idx, :, :], timestep=i, prefix="slice_idx")
    # save_image(fields["u_magnitude"][:, slice_idy, :], timestep=i, prefix="slice_idy")

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

    save_usd_turbine_parts(
        i,
        post_process_interval,
        vertices_wp,
        num_body_vertices,
        body_faces_np,
        rotor_faces_np,
        turbine_body_mesh,
        turbine_rotor_mesh,
        usd_stage,
        lag_forces=lag_forces,
    )


#
# Main simulation
#

grid_shape = (256, 450, 256)            # example domain size (Nx, Ny, Nz)
u_inlet = 0.05                          # inlet flow speed
num_steps = 25000
post_process_interval = 100
print_interval = 100
turbine_rotation_speed = -0.0005         # user-controlled rotor speed (radians per timestep)
Re = 5e5

clength = grid_shape[0] - 1
visc = u_inlet * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)
grid = grid_factory(grid_shape, compute_backend=compute_backend)

print("Wind Turbine Simulation Configuration:")
print(f"  Grid size: {grid_shape}")
print(f"  Omega: {omega:.6f}")
print(f"  Backend: {compute_backend}")
print(f"  Velocity set: {velocity_set}")
print(f"  Precision policy: {precision_policy}")
print(f"  Inlet velocity: {u_inlet}")
print(f"  Reynolds number: {Re}")
print(f"  Turbine rotation speed (rad/step): {turbine_rotation_speed}")
print(f"  Max steps: {num_steps}")

usd_output_directory = "usd_output_turbine"
os.makedirs(usd_output_directory, exist_ok=True)
usd_file = os.path.join(usd_output_directory, "turbine_output.usd")
usd_stage = Usd.Stage.CreateNew(usd_file)
usd_mesh_vorticity = UsdGeom.Mesh.Define(usd_stage, "/World/Vorticity")
usd_mesh_q_criterion = UsdGeom.Mesh.Define(usd_stage, "/World/QCriterion")
usd_turbine_body = UsdGeom.Mesh.Define(usd_stage, "/World/TurbineBody")
usd_turbine_rotor = UsdGeom.Mesh.Define(usd_stage, "/World/TurbineRotor")

mesh_data = load_and_prepare_meshes_turbine(grid_shape)
vertices_wp = mesh_data["vertices_wp"]
areas_wp = mesh_data["areas_wp"]
faces_np = mesh_data["faces_np"]
num_body_vertices = mesh_data["num_body_vertices"]
num_rotor_vertices = mesh_data["num_rotor_vertices"]
body_faces_np = mesh_data["body_faces_np"]
rotor_faces_np = mesh_data["rotor_faces_np"]

visualize_turbine_placement(vertices_wp, grid_shape)

# Calculate rotor center (simple approach: average rotor vertex positions)
rotor_center_np = vertices_wp.numpy()[num_body_vertices:].mean(axis=0)
_num_body_vertices = wp.constant(int(num_body_vertices))
_rotor_speed = wp.constant(float(turbine_rotation_speed))
_rotor_center_x = wp.constant(float(rotor_center_np[0]))
_rotor_center_y = wp.constant(float(rotor_center_np[1]))
_rotor_center_z = wp.constant(float(rotor_center_np[2]))

bc_list = setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_inlet)
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

try:
    for i in range(num_steps):
        f_0, f_1, lag_forces = stepper(
            f_0,
            f_1,
            vertices_wp,
            areas_wp,
            velocities_wp,
            bc_mask,
            missing_mask,
            omega,
            i,
        )
        f_0, f_1 = f_1, f_0

        wp.launch(
            kernel=rotate_rotor,
            dim=vertices_wp.shape[0],
            inputs=[
                i,
                lag_forces,
                vertices_wp,
                velocities_wp,
            ],
        )

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
                usd_mesh_vorticity,
                usd_mesh_q_criterion,
                vorticity_operator,
                q_criterion_operator,
                usd_stage,
                usd_turbine_body,
                usd_turbine_rotor,
                num_body_vertices,
                body_faces_np,
                rotor_faces_np,
                lag_forces,
            )
except KeyboardInterrupt:
    print("\nSimulation interrupted by user. Saving current USD state...")
    current_time_code = i // post_process_interval
    usd_stage.SetStartTimeCode(0)
    usd_stage.SetEndTimeCode(current_time_code)
    usd_stage.SetTimeCodesPerSecond(30)
    usd_stage.Save()
    print(f"USD file saved with {current_time_code+1} frames. Exiting.")
    import sys
    sys.exit(0)

usd_stage.SetStartTimeCode(0)
usd_stage.SetEndTimeCode(num_steps // post_process_interval)
usd_stage.SetTimeCodesPerSecond(30)
usd_stage.Save()
print("Simulation finished. USD file saved.")
