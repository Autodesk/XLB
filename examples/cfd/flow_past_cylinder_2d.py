"""
Flow past a cylinder in a 2D channel (D2Q9, Re=100).

Classic bluff-body channel flow: Poiseuille inflow passes a circular cylinder and
exits through an open outlet. Used for wake structure, vortex shedding, and drag/lift
coefficients at moderate Reynolds number.

Physical parameters
-------------------
* Reynolds number ``Re = 100`` (based on cylinder diameter and ``prescribed_vel``).
* Relaxation rate ``omega = 1 / (3 * nu + 0.5)`` with
  ``nu = prescribed_vel * diam / Re``.
* Reference inlet speed ``prescribed_vel = 0.02 (For more accurate results reduce this value)

Domain and geometry
-------------------
* Channel size: ``(22 * diam, 4.1 * diam)`` lattice nodes in x and y.
* Cylinder: center ``(2 * diam, 2 * diam)``, radius ``diam / 2``.
* Default ``diam = 80`` → grid ``(1760, 328)`` and ~2.7M time steps to
  ``100 * diam / prescribed_vel``. Use a smaller ``diam`` (e.g. 20) for quick tests.

Boundary conditions
-------------------
* **Inlet (left):** ``RegularizedBC`` with parabolic Poiseuille profile;
  peak speed ``u_peak = 1.5 * prescribed_vel``.
* **Outlet (right):** ``ExtrapolationOutflowBC``.
* **Top / bottom:** ``FullwayBounceBackBC`` (no-slip channel walls).
* **Cylinder:** ``HalfwayBounceBackBC`` (no-slip body).

Compute backends
----------------
* **WARP (default):** Recommended for large 2D GPU runs.
* **NEON:** Recommended for extremely large multi-GPU runs.
* **JAX:** Set ``compute_backend = ComputeBackend.JAX``. On NVIDIA Ampere and
  newer GPUs, keep the TF32 overrides at the top of this file (or export
  ``NVIDIA_TF32_OVERRIDE=0`` before launch) so ``jnp.tensordot`` in equilibrium
  and macroscopic operators use full FP32 accuracy.

Post-processing uses ``Macroscopic`` on JAX (populations are converted from Warp
when needed). PNG snapshots are written with prefix ``flow_past_cylinder_2d``.

Forces
------
After step ``> 0.5 * num_steps``, ``MomentumTransfer`` reports ``CD`` and ``CL``
(drag along x, lift along y), normalized by ``prescribed_vel**2 * diam``. Values
are stored at each post-process step.

Usage
-----
From the repository root (with ``PYTHONPATH`` pointing at this repo)::

    python3 examples/cfd/flow_past_cylinder_2d.py

For JAX on GPU with full FP32 matmul precision::

    NVIDIA_TF32_OVERRIDE=0 python3 examples/cfd/flow_past_cylinder_2d.py
"""

import os
import time

# Full FP32 matmul on NVIDIA GPUs (Macroscopic and other JAX operators use tensordot).
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np
import warp as wp
from tqdm import tqdm
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import (
    ExtrapolationOutflowBC,
    HalfwayBounceBackBC,
    RegularizedBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import PrecisionPolicy
from xlb.utils import save_image, warp_array_to_jax

# -------------------------- Simulation Setup --------------------------

# Geometry and grid parameters
diam = 20  # Cylinder diameter in lattice units; increase (e.g. 80) for faster runs and note to scale down prescribed_vel accordingly
grid_shape = (int(22 * diam), int(4.1 * diam))
cylinder_center = (2.0 * diam, 2.0 * diam)
cylinder_radius = diam / 2.0

# Physical parameters
Re = 100.0
prescribed_vel = 0.02
visc = prescribed_vel * diam / Re
omega = 1.0 / (3.0 * visc + 0.5)
u_peak = 1.5 * prescribed_vel

# Compute backend and precision policy
compute_backend = ComputeBackend.JAX
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)
flow_pass = int(grid_shape[0] / u_peak)
num_steps = 5 * flow_pass
post_process_interval = int(0.1 * flow_pass)


def bc_profile():
    """Parabolic inlet profile: u_x = 4 u_peak / d² (y d - y²), u_y = 0."""
    channel_height = float(grid_shape[1] - 1)

    if compute_backend == ComputeBackend.JAX:

        def bc_profile_jax():
            y = jnp.arange(grid_shape[1], dtype=jnp.float32)
            u_x = jnp.maximum(
                0.0,
                4.0 * u_peak / channel_height**2 * (y * channel_height - y**2),
            )
            u_y = jnp.zeros_like(u_x)
            return jnp.stack([u_x, u_y])

        return bc_profile_jax

    wp_dtype = precision_policy.compute_precision.wp_dtype
    ch = wp_dtype(channel_height)
    four = wp_dtype(4.0)
    u_peak_wp = wp_dtype(u_peak)

    @wp.func
    def bc_profile_warp(index: wp.vec3i):
        y = wp_dtype(index[1])
        u_x = four * u_peak_wp / (ch * ch) * (y * ch - y * y)
        return wp.vec(wp.max(wp_dtype(0.0), u_x), length=1)

    return bc_profile_warp


def main() -> None:
    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    grid = grid_factory(grid_shape, compute_backend=compute_backend)

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    walls = [box["bottom"][i] + box["top"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    x = np.arange(grid_shape[0])
    y = np.arange(grid_shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    cx, cy = cylinder_center
    cyl_idx = np.where((X - cx) ** 2 + (Y - cy) ** 2 <= cylinder_radius**2)
    cylinder = [tuple(cyl_idx[i].tolist()) for i in range(velocity_set.d)]

    bc_inlet = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
    bc_walls = HalfwayBounceBackBC(indices=walls)
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    bc_cylinder = HalfwayBounceBackBC(indices=cylinder)
    boundary_conditions = [bc_walls, bc_inlet, bc_outlet, bc_cylinder]

    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="BGK",
    )
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    macro = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D2Q9(
            precision_policy=precision_policy,
            compute_backend=ComputeBackend.JAX,
        ),
    )
    momentum_transfer = MomentumTransfer(bc_cylinder, compute_backend=compute_backend)

    force_steps: list[int] = []
    cl_history: list[float] = []
    cd_history: list[float] = []

    def post_process(step: int, f_0, f_1) -> None:
        wp.synchronize()

        if step > 0.5 * num_steps:
            boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
            drag = boundary_force[0]
            lift = boundary_force[1]
            cd = 2.0 * drag / (prescribed_vel**2 * diam)
            cl = 2.0 * lift / (prescribed_vel**2 * diam)
            force_steps.append(step)
            cd_history.append(cd)
            cl_history.append(cl)
            tqdm.write(f"step={step:7d}, CL={cl: .6f}, CD={cd: .6f}")

        if not isinstance(f_0, jnp.ndarray):
            # Warp pads 2D domains with a singleton z dimension
            f_0 = warp_array_to_jax(f_0)[..., 0]
            wp.synchronize()

        _, u = macro(f_0)

        # Vorticity ω_z = ∂u_y/∂x − ∂u_x/∂y (central differences on the full grid)
        u_y_dx = jnp.gradient(u[1], axis=0)
        u_x_dy = jnp.gradient(u[0], axis=1)
        vort_z = u_y_dx - u_x_dy

        save_image(vort_z, timestep=step, prefix="flow_past_cylinder_2d", cmap="seismic", vmin=-1e-2, vmax=1e-2)
        tqdm.write(f"Post-processed step {step}: saved vorticity (prefix=flow_past_cylinder_2d)")

    print(
        f"grid_shape={grid_shape}, Re={Re}, omega={omega:.6f}, prescribed_vel={prescribed_vel}, num_steps={num_steps}, backend={compute_backend.name}"
    )

    start_time = time.time()
    pbar = tqdm(range(num_steps), unit="step")
    for step in pbar:
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
        f_0, f_1 = f_1, f_0

        if step % post_process_interval == 0 or step == num_steps - 1:
            post_process(step, f_0, f_1)
            elapsed = time.time() - start_time
            pbar.set_postfix(chunk_s=f"{elapsed:.1f}")
            start_time = time.time()

    print(f"Stored {len(cl_history)} force samples")
    if cl_history:
        print(f"Final CL={cl_history[-1]:.6f}, CD={cd_history[-1]:.6f}")


if __name__ == "__main__":
    main()
