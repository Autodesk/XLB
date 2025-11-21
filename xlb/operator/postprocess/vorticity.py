import warp as wp
from typing import Any
from functools import partial
from jax import jit

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class Vorticity(Operator):
    """
    Compute vorticity vector and magnitude for flow visualization and analysis.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Call the parent constructor
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

    def _construct_warp(self):
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            u: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            vorticity: wp.array4d(dtype=Any),
            vorticity_magnitude: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Add ghost cells to index
            i += 1
            j += 1
            k += 1

            # Check if anything on edges
            b_id_2_1_1 = bc_mask[0, i + 1, j, k]
            b_id_1_2_1 = bc_mask[0, i, j + 1, k]
            b_id_1_1_2 = bc_mask[0, i, j, k + 1]
            b_id_0_1_1 = bc_mask[0, i - 1, j, k]
            b_id_1_0_1 = bc_mask[0, i, j - 1, k]
            b_id_1_1_0 = bc_mask[0, i, j, k - 1]
            if (
                b_id_2_1_1 != wp.uint8(0)
                or b_id_1_2_1 != wp.uint8(0)
                or b_id_1_1_2 != wp.uint8(0)
                or b_id_0_1_1 != wp.uint8(0)
                or b_id_1_0_1 != wp.uint8(0)
                or b_id_1_1_0 != wp.uint8(0)
            ):
                return

            # Get derivatives using central differences
            u_x_dy = (u[0, i, j + 1, k] - u[0, i, j - 1, k]) / 2.0
            u_x_dz = (u[0, i, j, k + 1] - u[0, i, j, k - 1]) / 2.0
            u_y_dx = (u[1, i + 1, j, k] - u[1, i - 1, j, k]) / 2.0
            u_y_dz = (u[1, i, j, k + 1] - u[1, i, j, k - 1]) / 2.0
            u_z_dx = (u[2, i + 1, j, k] - u[2, i - 1, j, k]) / 2.0
            u_z_dy = (u[2, i, j + 1, k] - u[2, i, j - 1, k]) / 2.0

            # Compute vorticity components (curl of velocity)
            vort_x = u_z_dy - u_y_dz
            vort_y = u_x_dz - u_z_dx
            vort_z = u_y_dx - u_x_dy

            # Store vorticity vector components
            vorticity[0, i, j, k] = vort_x
            vorticity[1, i, j, k] = vort_y
            vorticity[2, i, j, k] = vort_z

            # Compute and store vorticity magnitude
            vorticity_magnitude[0, i, j, k] = wp.sqrt(vort_x * vort_x + vort_y * vort_y + vort_z * vort_z)

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, u, bc_mask, vorticity, vorticity_magnitude):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[u, bc_mask, vorticity, vorticity_magnitude],
            dim=[i - 2 for i in u.shape[1:]],
        )
        return vorticity, vorticity_magnitude

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, u, bc_mask, vorticity, vorticity_magnitude):
        # TODO: Implement JAX version
        raise NotImplementedError("JAX implementation not yet available")
