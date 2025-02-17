import warp as wp
from typing import Any
from functools import partial
from jax import jit

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class QCriterion(Operator):
    """
    Compute Q-criterion and vorticity magnitude for flow visualization and analysis.

    The Q-criterion is the second invariant of the velocity gradient tensor,
    defined as Q = 1/2(|Ω|^2 - |S|^2) where Ω is the vorticity tensor and
    S is the rate-of-strain tensor.
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
            norm_mu: wp.array4d(dtype=Any),
            q: wp.array4d(dtype=Any),
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

            # Get derivatives
            u_x_dx = (u[0, i + 1, j, k] - u[0, i - 1, j, k]) / 2.0
            u_x_dy = (u[0, i, j + 1, k] - u[0, i, j - 1, k]) / 2.0
            u_x_dz = (u[0, i, j, k + 1] - u[0, i, j, k - 1]) / 2.0
            u_y_dx = (u[1, i + 1, j, k] - u[1, i - 1, j, k]) / 2.0
            u_y_dy = (u[1, i, j + 1, k] - u[1, i, j - 1, k]) / 2.0
            u_y_dz = (u[1, i, j, k + 1] - u[1, i, j, k - 1]) / 2.0
            u_z_dx = (u[2, i + 1, j, k] - u[2, i - 1, j, k]) / 2.0
            u_z_dy = (u[2, i, j + 1, k] - u[2, i, j - 1, k]) / 2.0
            u_z_dz = (u[2, i, j, k + 1] - u[2, i, j, k - 1]) / 2.0

            # Compute vorticity
            mu_x = u_z_dy - u_y_dz
            mu_y = u_x_dz - u_z_dx
            mu_z = u_y_dx - u_x_dy
            mu = wp.sqrt(mu_x**2.0 + mu_y**2.0 + mu_z**2.0)

            # Compute strain rate
            s_0_0 = u_x_dx
            s_0_1 = 0.5 * (u_x_dy + u_y_dx)
            s_0_2 = 0.5 * (u_x_dz + u_z_dx)
            s_1_0 = s_0_1
            s_1_1 = u_y_dy
            s_1_2 = 0.5 * (u_y_dz + u_z_dy)
            s_2_0 = s_0_2
            s_2_1 = s_1_2
            s_2_2 = u_z_dz
            s_dot_s = s_0_0**2.0 + s_0_1**2.0 + s_0_2**2.0 + s_1_0**2.0 + s_1_1**2.0 + s_1_2**2.0 + s_2_0**2.0 + s_2_1**2.0 + s_2_2**2.0

            # Compute omega
            omega_0_0 = 0.0
            omega_0_1 = 0.5 * (u_x_dy - u_y_dx)
            omega_0_2 = 0.5 * (u_x_dz - u_z_dx)
            omega_1_0 = -omega_0_1
            omega_1_1 = 0.0
            omega_1_2 = 0.5 * (u_y_dz - u_z_dy)
            omega_2_0 = -omega_0_2
            omega_2_1 = -omega_1_2
            omega_2_2 = 0.0
            omega_dot_omega = (
                omega_0_0**2.0
                + omega_0_1**2.0
                + omega_0_2**2.0
                + omega_1_0**2.0
                + omega_1_1**2.0
                + omega_1_2**2.0
                + omega_2_0**2.0
                + omega_2_1**2.0
                + omega_2_2**2.0
            )

            # Compute q-criterion
            q_value = 0.5 * (omega_dot_omega - s_dot_s)

            # Set the output
            norm_mu[0, i, j, k] = mu
            q[0, i, j, k] = q_value

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, u, bc_mask, norm_mu, q):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[u, bc_mask, norm_mu, q],
            dim=[i - 2 for i in u.shape[1:]],
        )
        return norm_mu, q

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, u, bc_mask, norm_mu, q):
        # TODO: Implement JAX version
        raise NotImplementedError("JAX implementation not yet available")
