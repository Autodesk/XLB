"""
Flow past a sphere (IBM) — Drag coefficient validation

References
- Johnson, T. A., & Patel, V. C. (1999). Flow past a sphere up to Re = 300.
  Journal of Fluid Mechanics, 378, 19–70. (domain sizing, Cd at Re ≈ 100)
- Uhlmann, M. (2005). An immersed boundary method with direct forcing for
  particulate flows. Journal of Computational Physics, 209(2), 448–476.
  (IBM forcing and hydrodynamic force evaluation)
- Clift, R., Grace, J. R., & Weber, M. E. (1978). Bubbles, Drops, and
  Particles. Academic Press. (Cd correlations vs Reynolds number)
- Achenbach, E. (1972). Experiments on the flow past spheres at very high
  Reynolds numbers. Journal of Fluid Mechanics, 54(3), 565–575.
  (experimental Cd curve)
"""

import os
import xlb
import trimesh
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warp as wp
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
from xlb.helper.ibm_helper import prepare_immersed_boundary
from xlb.grid import grid_factory


def create_sphere_mesh(center, radius, subdivisions=3):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(center)
    return sphere


def define_boundary_indices(grid, velocity_set):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    walls = [box["front"][i] + box["back"][i] + box["top"][i] + box["bottom"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    return inlet, outlet, walls


def bc_profile(precision_policy, grid_shape, u_max):
    _dtype = precision_policy.store_precision.wp_dtype
    u_max_d = _dtype(u_max)

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        return wp.vec(_dtype(u_max_d), length=1)

    return bc_profile_warp


def calculate_drag_coefficient(lag_forces, reference_velocity, frontal_area, areas_wp):
    forces_np = lag_forces.numpy()
    drag_forces = forces_np[:, 0]

    total_drag = -np.sum(drag_forces * areas_wp.numpy())

    dynamic_pressure = 0.5 * reference_velocity**2
    cd = total_drag / (dynamic_pressure * frontal_area)

    return cd, total_drag


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


def post_process(
    i,
    post_process_interval,
    f_current,
    precision_policy,
    grid_shape,
    lag_forces,
    cd_values,
    reference_velocity,
    frontal_area,
    areas_wp,
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

    cd, total_drag = calculate_drag_coefficient(lag_forces, reference_velocity, frontal_area, areas_wp)
    cd_values.append((i, cd))
    if i % post_process_interval == 0:
        window = 10
        if len(cd_values) >= window:
            avg_cd = float(np.mean([v for _, v in cd_values[-window:]]))
        else:
            avg_cd = float(np.mean([v for _, v in cd_values]))
        print(f"Step {i}: Cd = {cd:.6f}, Cd(avg{window}) = {avg_cd:.6f}, Total Drag = {total_drag:.6f}")


def save_drag_coefficient(cd_values, filename):
    with open(filename, "w") as f:
        f.write("timestep,cd\n")
        for timestep, cd in cd_values:
            f.write(f"{timestep},{cd}\n")

    timesteps = [t for t, _ in cd_values]
    cds = [cd for _, cd in cd_values]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, cds, "b-")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Drag Coefficient (Cd)")
    plt.title("Drag Coefficient vs Time")
    plt.tight_layout()
    plt.savefig("drag_coefficient_sphere.png", dpi=150)
    plt.close()


sphere_radius = 25.0

diameter = 2.0 * sphere_radius
upstream = int(1.5 * diameter)  # 1.5D upstream
downstream = int(7 * diameter)  # 2.5D downstream
ly = int(3.0 * diameter)  # 3D lateral
lz = int(3.0 * diameter)  # 3D vertical
lx = upstream + downstream
grid_shape = (lx, ly, lz)


# Uniform inlet velocity
u_max = 0.02

# Place sphere at 1/3 from entrance
sphere_center = [float(lx / 3), grid_shape[1] / 2.0, grid_shape[2] / 2.0]

Re = 300
visc = u_max * (2.0 * sphere_radius) / Re
omega = 1.0 / (3.0 * visc + 0.5)

num_steps = 20000
post_process_interval = 1000
print_interval = 1000

ibm_max_iterations = 4
ibm_tolerance = 1e-5
ibm_relaxation = 0.5

compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)
grid = grid_factory(grid_shape, compute_backend=compute_backend)

print("Sphere IBM Simulation Configuration:")
print(f"  Grid size: {grid_shape}")
print(f"  Sphere radius: {sphere_radius}")
print(f"  Sphere center: {sphere_center}")
print(f"  Omega: {omega}")
print(f"  Inlet velocity: {u_max}")
print(f"  Reynolds number: {Re}")
print(f"  Max steps: {num_steps}")
print(f"  IBM max iterations: {ibm_max_iterations}")
print(f"  IBM tolerance: {ibm_tolerance}")
print(f"  IBM relaxation: {ibm_relaxation}")

sphere_mesh = create_sphere_mesh(sphere_center, sphere_radius, subdivisions=4)
vertices_wp, areas_wp, faces_np = prepare_immersed_boundary(sphere_mesh, max_lbm_length=sphere_radius * 2)

frontal_area = np.pi * sphere_radius**2
print(f"Frontal area (theoretical): {frontal_area:.2f}")

bc_list = setup_boundary_conditions(grid, velocity_set, precision_policy, grid_shape, u_max)
stepper = setup_stepper(grid, bc_list, ibm_max_iterations, ibm_tolerance, ibm_relaxation)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

velocities_wp = wp.zeros(shape=vertices_wp.shape[0], dtype=wp.vec3)
cd_values = []

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
                u_max,
                frontal_area,
                areas_wp,
            )

except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")
    if cd_values:
        save_drag_coefficient(cd_values, "drag_coefficient_sphere.csv")
        print("Drag coefficient data saved to drag_coefficient_sphere.csv")
    import sys

    sys.exit(0)

if cd_values:
    save_drag_coefficient(cd_values, "drag_coefficient_sphere.csv")
    print("Drag coefficient data saved to drag_coefficient_sphere.csv")
    print(f"Final Cd (average of last 10 values): {np.mean([cd for _, cd in cd_values[-10:]]):.6f}")

print("Simulation finished.")
