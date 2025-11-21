import numpy as np
import trimesh
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warp as wp
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper import IBMStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, RegularizedBC, ExtrapolationOutflowBC
from xlb.operator.macroscopic import Macroscopic
from xlb.helper.ibm_helper import prepare_immersed_boundary
from xlb.grid import grid_factory
from xlb.utils import save_image


def generate_naca_profile(chord_length, thickness_ratio, n_points=400):
    x = np.linspace(0.0, chord_length, n_points)
    x_c = x / chord_length
    coeffs = np.array([0.2969, -0.1260, -0.3516, 0.2843, -0.1015], dtype=np.float64)
    powers = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    terms = np.stack([x_c**p for p in powers], axis=0)
    thickness = 5.0 * thickness_ratio * chord_length * np.tensordot(coeffs, terms, axes=1)
    upper = np.stack([x, thickness], axis=1)
    lower = np.stack([x[::-1], -thickness[::-1]], axis=1)
    profile = np.vstack([upper, lower[1:-1]])
    profile[:, 0] -= chord_length * 0.5
    return profile


def extrude_profile_to_mesh(profile, span_length):
    lower_z = -0.5 * span_length
    upper_z = 0.5 * span_length
    lower = np.concatenate([profile, np.full((profile.shape[0], 1), lower_z)], axis=1)
    upper = np.concatenate([profile, np.full((profile.shape[0], 1), upper_z)], axis=1)
    vertices = np.vstack([lower, upper])
    faces = []
    n = profile.shape[0]
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
    top_offset = n
    for i in range(1, n - 1):
        faces.append([top_offset, top_offset + i, top_offset + i + 1])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, top_offset + j])
        faces.append([i, top_offset + j, top_offset + i])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=False)


def create_airfoil_mesh(chord_length, thickness_ratio, span_length, n_points=400):
    profile = generate_naca_profile(chord_length, thickness_ratio, n_points)
    mesh = extrude_profile_to_mesh(profile, span_length)
    return mesh


def define_boundary_indices(grid, velocity_set):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    walls = [box["front"][i] + box["back"][i] + box["top"][i] + box["bottom"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    return inlet, outlet, walls


def bc_profile(precision_policy, grid_shape, u_max):
    dtype = precision_policy.store_precision.wp_dtype
    u_max_d = dtype(u_max)

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        return wp.vec(dtype(u_max_d), length=1)

    return bc_profile_warp


def setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max):
    inlet, outlet, walls = define_boundary_indices(grid, velocity_set)
    bc_inlet = RegularizedBC("velocity", indices=inlet, profile=bc_profile(precision_policy, grid_shape, u_max))
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    return [bc_walls, bc_inlet, bc_outlet]


def setup_stepper(grid, boundary_conditions, ibm_max_iterations=2, ibm_tolerance=1e-5, ibm_relaxation=1.0):
    return IBMStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
        ibm_max_iterations=ibm_max_iterations,
        ibm_tolerance=ibm_tolerance,
        ibm_relaxation=ibm_relaxation,
    )


def calculate_force_coefficients(lag_forces, areas_np, reference_velocity, reference_area):
    forces_np = lag_forces.numpy()
    weighted = forces_np * areas_np[:, None]
    total_force = -np.sum(weighted, axis=0)
    dynamic_pressure = 0.5 * reference_velocity**2
    denom = dynamic_pressure * reference_area if dynamic_pressure * reference_area != 0.0 else 1.0
    cd = total_force[0] / denom
    cl = total_force[1] / denom
    return cd, cl, total_force


def post_process(
    step,
    post_process_interval,
    f_current,
    precision_policy,
    grid_shape,
    lag_forces,
    cd_values,
    cl_values,
    reference_velocity,
    reference_area,
    areas_np,
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
        "u_magnitude": np.sqrt(u[0] ** 2.0 + u[1] ** 2.0 + u[2] ** 2.0),
        "u_x": u[0],
        "u_y": u[1],
        "u_z": u[2],
    }
    slice_idz = grid_shape[2] // 2
    save_image(fields["u_magnitude"][:, :, slice_idz], timestep=step)
    cd, cl, total_force = calculate_force_coefficients(lag_forces, areas_np, reference_velocity, reference_area)
    cd_values.append((step, float(cd)))
    cl_values.append((step, float(cl)))
    if step % post_process_interval == 0:
        window = 10
        if len(cd_values) >= window:
            avg_cd = float(np.mean([v for _, v in cd_values[-window:]]))
            avg_cl = float(np.mean([v for _, v in cl_values[-window:]]))
        else:
            avg_cd = float(np.mean([v for _, v in cd_values]))
            avg_cl = float(np.mean([v for _, v in cl_values]))
        print(
            f"Step {step}: Cd = {cd:.6f}, Cl = {cl:.6f}, Cd(avg{window}) = {avg_cd:.6f}, Cl(avg{window}) = {avg_cl:.6f}, "
            f"Fx = {total_force[0]:.6f}, Fy = {total_force[1]:.6f}"
        )


def save_force_coefficients(cd_values, cl_values, filename):
    with open(filename, "w") as f:
        f.write("timestep,cd,cl\n")
        for (timestep_cd, cd), (_, cl) in zip(cd_values, cl_values):
            f.write(f"{timestep_cd},{cd},{cl}\n")
    timesteps = [t for t, _ in cd_values]
    cds = [cd for _, cd in cd_values]
    cls = [cl for _, cl in cl_values]
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, cds, "r-", label="Cd")
    plt.plot(timesteps, cls, "b-", label="Cl")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Coefficient")
    plt.title("Airfoil Force Coefficients")
    plt.legend()
    plt.tight_layout()
    plt.savefig("airfoil_force_coefficients.png", dpi=150)
    plt.close()


@wp.kernel
def update_airfoil_pose(
    step: int,
    total_steps: int,
    start_angle: float,
    total_rotation: float,
    origin: wp.vec3,
    base_vertices: wp.array(dtype=wp.vec3),
    vertices: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    idx = wp.tid()
    total_span = wp.float32(total_steps - 1)
    progress = wp.float32(0.0)
    if total_span > 0.0:
        progress = wp.float32(step) / total_span
        if progress > 1.0:
            progress = wp.float32(1.0)
    start_angle_f = wp.float32(start_angle)
    total_rotation_f = wp.float32(total_rotation)
    angle = start_angle_f + total_rotation_f * progress
    c = wp.cos(angle)
    s = wp.sin(angle)
    base = base_vertices[idx] - origin
    rotated = wp.vec3(
        c * base[0] - s * base[1],
        s * base[0] + c * base[1],
        base[2],
    )
    vertices[idx] = rotated + origin
    angular_rate = wp.float32(0.0)
    if total_span > 0.0:
        angular_rate = total_rotation_f / total_span
    velocities[idx] = wp.vec3(
        -angular_rate * rotated[1],
        angular_rate * rotated[0],
        0.0,
    )


if __name__ == "__main__":
    chord_length = 60.0 * 1.3
    span_length = 50.0 * 1.3
    thickness_ratio = 0.12
    upstream = int(2 * chord_length)
    downstream = int(4 * chord_length)
    ly = int(3.0 * chord_length)
    lz = int(2.0 * span_length)
    lx = upstream + downstream + int(chord_length)
    grid_shape = (lx, ly, lz)
    u_max = 0.05
    Re = 20000
    start_angle_deg = 0.0
    total_rotation_deg = -45.0
    start_angle_rad = np.deg2rad(start_angle_deg)
    total_rotation_rad = np.deg2rad(total_rotation_deg)
    num_steps = 30000
    post_process_interval = 100
    print_interval = 100
    ibm_max_iterations = 1
    ibm_tolerance = 1e-5
    ibm_relaxation = 0.5
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
    xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)
    grid = grid_factory(grid_shape, compute_backend=compute_backend)
    print("Airfoil IBM Simulation Configuration:")
    print(f"  Grid size: {grid_shape}")
    print(f"  Chord length: {chord_length}")
    print(f"  Span length: {span_length}")
    print(f"  Thickness ratio: {thickness_ratio}")
    print(f"  Inlet velocity: {u_max}")
    print(f"  Reynolds number: {Re}")
    print(f"  Start angle: {start_angle_deg}")
    print(f"  Total rotation: {total_rotation_deg}")
    print(f"  Max steps: {num_steps}")
    print(f"  IBM max iterations: {ibm_max_iterations}")
    print(f"  IBM tolerance: {ibm_tolerance}")
    print(f"  IBM relaxation: {ibm_relaxation}")
    airfoil_mesh = create_airfoil_mesh(chord_length, thickness_ratio, span_length)
    airfoil_center = np.array([float(upstream + 0.6 * chord_length), grid_shape[1] * 0.5, grid_shape[2] * 0.5], dtype=np.float64)
    translation = airfoil_center - airfoil_mesh.centroid
    airfoil_mesh.apply_translation(translation)
    vertices_wp, areas_wp, faces_np = prepare_immersed_boundary(airfoil_mesh, max_lbm_length=max(chord_length, span_length))
    vertices_np = vertices_wp.numpy()
    base_vertices_wp = wp.array(vertices_np, dtype=wp.vec3)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3)
    areas_np = areas_wp.numpy()
    leading_edge_x = float(np.min(vertices_np[:, 0]))
    rotation_center_y = float(np.mean(vertices_np[:, 1]))
    rotation_center_z = float(np.mean(vertices_np[:, 2]))
    rotation_origin = np.array(
        [
            leading_edge_x + 0.1 * chord_length,
            rotation_center_y,
            rotation_center_z,
        ],
        dtype=np.float64,
    )
    origin_wp = wp.vec3(float(rotation_origin[0]), float(rotation_origin[1]), float(rotation_origin[2]))
    reference_area = chord_length * span_length
    bc_list = setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max)
    stepper = setup_stepper(grid, bc_list, ibm_max_iterations, ibm_tolerance, ibm_relaxation)
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
    velocities_wp = wp.zeros(shape=vertices_wp.shape[0], dtype=wp.vec3)
    device = vertices_wp.device
    wp.launch(
        kernel=update_airfoil_pose,
        dim=vertices_wp.shape[0],
        inputs=[
            0,
            num_steps,
            start_angle_rad,
            total_rotation_rad,
            origin_wp,
            base_vertices_wp,
            vertices_wp,
            velocities_wp,
        ],
        device=device,
    )
    cd_values = []
    cl_values = []
    visc = u_max * chord_length / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    print(f"  Omega: {omega}")
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
            if print_interval > 0 and i % print_interval == 0:
                print(f"Step {i}/{num_steps} completed")
            if i % post_process_interval == 0 or i == num_steps - 1:
                post_process(
                    i,
                    post_process_interval,
                    f_0,
                    precision_policy,
                    grid_shape,
                    lag_forces,
                    cd_values,
                    cl_values,
                    u_max,
                    reference_area,
                    areas_np,
                )
            next_step = i + 1
            if next_step < num_steps:
                wp.launch(
                    kernel=update_airfoil_pose,
                    dim=vertices_wp.shape[0],
                    inputs=[
                        next_step,
                        num_steps,
                        start_angle_rad,
                        total_rotation_rad,
                        origin_wp,
                        base_vertices_wp,
                        vertices_wp,
                        velocities_wp,
                    ],
                    device=device,
                )
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        if cd_values and cl_values:
            save_force_coefficients(cd_values, cl_values, "airfoil_force_coefficients.csv")
            print("Force coefficient data saved to airfoil_force_coefficients.csv")
        raise
    if cd_values and cl_values:
        save_force_coefficients(cd_values, cl_values, "airfoil_force_coefficients.csv")
        print("Force coefficient data saved to airfoil_force_coefficients.csv")
        print(f"Final Cd (avg last 10): {np.mean([cd for _, cd in cd_values[-10:]]):.6f}")
        print(f"Final Cl (avg last 10): {np.mean([cl for _, cl in cl_values[-10:]]):.6f}")
    print("Simulation finished.")
