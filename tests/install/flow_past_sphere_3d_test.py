"""
Flow past a sphere (3D) — smoke test for JAX / WARP backends.

NEON is **not** run: ``HalfwayBounceBackBC`` (sphere) has no NEON implementation in
XLB yet; NEON is listed under "Skipped (unsupported)" when the package is
installed.

Run from the repository root::

    python tests/install/flow_past_sphere_3d_test.py

Domain and step counts are kept small for fast CI / install verification.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import jax.numpy as jnp
import warp as wp
import xlb
import xlb.velocity_set
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import PrecisionPolicy
from xlb.utils import save_image


def _run_flow_past_sphere_for_backend(compute_backend: ComputeBackend) -> None:
    # Small domain for install / smoke tests (original example uses 256×64×64).
    grid_shape = (32, 16, 16)
    omega = 1.6
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend)
    u_max = 0.04
    num_steps = 20
    post_process_interval = 10

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
    walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    sphere_radius = max(grid_shape[1] // 12, 1)
    x = np.arange(grid_shape[0])
    y = np.arange(grid_shape[1])
    z = np.arange(grid_shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    indices = np.where((X - grid_shape[0] // 6) ** 2 + (Y - grid_shape[1] // 2) ** 2 + (Z - grid_shape[2] // 2) ** 2 < sphere_radius**2)
    sphere = [tuple(indices[i].tolist()) for i in range(velocity_set.d)]

    def bc_profile():
        H_y = float(grid_shape[1] - 1)
        H_z = float(grid_shape[2] - 1)

        if compute_backend == ComputeBackend.JAX:

            def bc_profile_jax():
                yy = jnp.arange(grid_shape[1])
                zz = jnp.arange(grid_shape[2])
                Y, Z = jnp.meshgrid(yy, zz, indexing="ij")
                y_center = Y - (H_y / 2.0)
                z_center = Z - (H_z / 2.0)
                r_squared = (2.0 * y_center / H_y) ** 2.0 + (2.0 * z_center / H_z) ** 2.0
                u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
                u_y = jnp.zeros_like(u_x)
                u_z = jnp.zeros_like(u_x)
                return jnp.stack([u_x, u_y, u_z])

            return bc_profile_jax

        wp_dtype = precision_policy.compute_precision.wp_dtype
        H_y_w = wp_dtype(grid_shape[1] - 1)
        H_z_w = wp_dtype(grid_shape[2] - 1)
        two = wp_dtype(2.0)

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            y = wp_dtype(index[1])
            z = wp_dtype(index[2])
            y_center = y - (H_y_w / two)
            z_center = z - (H_z_w / two)
            r_squared = (two * y_center / H_y_w) ** two + (two * z_center / H_z_w) ** two
            return wp.vec(wp_dtype(u_max) * wp.max(wp_dtype(0.0), wp_dtype(1.0) - r_squared), length=1)

        return bc_profile_warp

    bc_left = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    bc_outlet = ExtrapolationOutflowBC(indices=outlet)
    bc_sphere = HalfwayBounceBackBC(indices=sphere)
    boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_sphere]

    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="BGK",
    )
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    macro = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
    )
    to_jax = xlb.utils.ToJAX("populations", velocity_set.q, grid_shape)

    momentum_transfer = MomentumTransfer(bc_sphere, compute_backend=compute_backend)
    sphere_cross_section = float(np.pi * sphere_radius**2)

    prefix = f"flow_past_sphere_{compute_backend.name.lower()}"

    def post_process(step: int, f_0, f_1) -> None:
        if compute_backend in (ComputeBackend.WARP, ComputeBackend.NEON):
            wp.synchronize()

        boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
        drag = boundary_force[0]
        lift = boundary_force[2]
        cd = 2.0 * drag / (u_max**2 * sphere_cross_section)
        cl = 2.0 * lift / (u_max**2 * sphere_cross_section)
        print(f"CD={cd}, CL={cl}")

        if not isinstance(f_0, jnp.ndarray):
            f_0 = to_jax(f_0)
            if compute_backend in (ComputeBackend.WARP, ComputeBackend.NEON):
                wp.synchronize()

        rho, u = macro(f_0)

        u = u[:, 1:-1, 1:-1, 1:-1]
        rho = rho[:, 1:-1, 1:-1, 1:-1]
        u_magnitude = jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

        fields = {
            "u_magnitude": u_magnitude,
            "u_x": u[0],
            "u_y": u[1],
            "u_z": u[2],
            "rho": rho[0],
        }

        save_image(fields["u_magnitude"][:, grid_shape[1] // 2, :], timestep=step, prefix=prefix)
        print(f"Post-processed step {step}: saved u_magnitude slice (prefix={prefix})")

    start_time = time.time()
    for step in range(num_steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
        f_0, f_1 = f_1, f_0

        if step % post_process_interval == 0 or step == num_steps - 1:
            post_process(step, f_0, f_1)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Completed step {step}. Elapsed for last chunk: {elapsed:.6f} s.")
            start_time = time.time()


def run_flow_past_sphere_smoke() -> dict[str, Any]:
    """Run JAX and WARP; skip NEON (unsupported BC). Missing NEON package -> ImportError path unused."""
    backends_order: tuple[ComputeBackend, ...] = (
        ComputeBackend.WARP,
        ComputeBackend.JAX,
        ComputeBackend.NEON,
    )

    executed: list[str] = []
    skipped_not_installed: list[str] = []
    skipped_unsupported: list[str] = []
    failed: list[tuple[str, str]] = []

    for backend in backends_order:
        print(f"\n--- Backend: {backend.name} ---")
        if backend == ComputeBackend.NEON:
            reason = "HalfwayBounceBackBC on the sphere has no NEON implementation in XLB (see bc_halfway_bounce_back.neon_implementation)."
            print(f"SKIP (unsupported): NEON — {reason}")
            skipped_unsupported.append(f"NEON ({reason})")
            continue

        try:
            _run_flow_past_sphere_for_backend(backend)
            executed.append(backend.name)
            print(f"OK: {backend.name} finished.")
        except ImportError:
            skipped_not_installed.append(backend.name)
        except Exception as exc:
            failed.append((backend.name, str(exc)))
            print(f"FAIL {backend.name}:")
            traceback.print_exc()

    print("\n=== Summary ===")
    print(f"Executed: {', '.join(executed) if executed else '(none)'}")
    if skipped_not_installed:
        print("Skipped (not installed): " + ", ".join(skipped_not_installed) + " — required package not available.")
    else:
        print("Skipped (not installed): (none)")
    if skipped_unsupported:
        print("Skipped (unsupported configuration):")
        for s in skipped_unsupported:
            print(f"  - {s}")
    else:
        print("Skipped (unsupported configuration): (none)")
    if failed:
        print("Failed:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
    else:
        print("Failed: (none)")

    return {
        "executed": executed,
        "skipped_not_installed": skipped_not_installed,
        "skipped_unsupported": skipped_unsupported,
        "failed": failed,
    }


def main() -> int:
    result = run_flow_past_sphere_smoke()
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
