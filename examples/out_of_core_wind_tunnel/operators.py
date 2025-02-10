# Custom XLB operators for out of core simulations

from typing import Any
import os
from time import time
import numpy as np
import warp as wp

wp.init()

import xlb
from xlb.operator import Operator
from xlb.operator.stepper import Stepper

class UniformInitializer(Operator):

    def _construct_warp(self):
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            boundary_id: wp.array4d(dtype=wp.uint8),
            vel: float,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Set the velocity
            u[0, i, j, k] = vel
            u[1, i, j, k] = 0.0
            u[2, i, j, k] = 0.0

            # Set the density
            rho[0, i, j, k] = 1.0

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, rho, u, boundary_id, vel):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                boundary_id,
                vel,
            ],
            dim=rho.shape[1:],
        )
        return rho, u

class MomentumTransfer(Operator):

    def __init__(self, halfway_bounce_back, velocity_set, precision_policy, compute_backend):
        self.halfway_bounce_back = halfway_bounce_back
        super().__init__(velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.wp_c
        _opp_indices = self.velocity_set.wp_opp_indices
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(
            self.velocity_set.q, dtype=wp.uint8
        )  # TODO fix vec bool

        # Find velocity index for 0, 0, 0
        for l in range(self.velocity_set.q):
            if _c[0, l] == 0 and _c[1, l] == 0 and _c[2, l] == 0:
                zero_index = l
        _zero_index = wp.int32(zero_index)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            boundary_id: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            momentum: wp.array(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the boundary id
            _boundary_id = boundary_id[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Determin if boundary is an edge by checking if center is missing
            is_edge = wp.bool(False)
            if _boundary_id == wp.uint8(xlb.operator.boundary_condition.HalfwayBounceBackBC.id):
                if _missing_mask[_zero_index] == wp.uint8(0):
                    is_edge = wp.bool(True)

            # If the boundary is an edge then add the momentum transfer
            m = wp.vec3()
            if is_edge:

                # Get the distribution function
                f_post_collision = _f_vec()
                for l in range(self.velocity_set.q):
                    f_post_collision[l] = f[l, index[0], index[1], index[2]]

                # Apply streaming
                f_post_stream = self.halfway_bounce_back.warp_functional(
                    f, _missing_mask, index
                )
 
                # Compute the momentum transfer
                for l in range(self.velocity_set.q):
                    if _missing_mask[l] == wp.uint8(1):
                        phi = f_post_collision[_opp_indices[l]] + f_post_stream[l]
                        for d in range(self.velocity_set.d):
                            m[d] += phi * wp.float32(_c[d, _opp_indices[l]])

            wp.atomic_add(momentum, 0, m)

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, f, boundary_id, missing_mask):

        # Allocate the momentum field
        momentum = wp.zeros((1), dtype=wp.vec3)

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f, boundary_id, missing_mask, momentum],
            dim=f.shape[1:],
        )
        return momentum.numpy()


class QCriterion(Operator):

    @wp.kernel
    def q_kernel(
        u: wp.array4d(dtype=Any),
        boundary_id: wp.array4d(dtype=wp.uint8),
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
        b_id_2_1_1 = boundary_id[0, i + 1, j, k]
        b_id_1_2_1 = boundary_id[0, i, j + 1, k]
        b_id_1_1_2 = boundary_id[0, i, j, k + 1]
        b_id_0_1_1 = boundary_id[0, i - 1, j, k]
        b_id_1_0_1 = boundary_id[0, i, j - 1, k]
        b_id_1_1_0 = boundary_id[0, i, j, k - 1]
        if b_id_2_1_1 != wp.uint8(0) or b_id_1_2_1 != wp.uint8(0) or b_id_1_1_2 != wp.uint8(0) or b_id_0_1_1 != wp.uint8(0) or b_id_1_0_1 != wp.uint8(0) or b_id_1_1_0 != wp.uint8(0):
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
        mu = wp.sqrt(mu_x ** 2.0 + mu_y ** 2.0 + mu_z ** 2.0)

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
        s_dot_s = (
            s_0_0 ** 2.0 + s_0_1 ** 2.0 + s_0_2 ** 2.0 +
            s_1_0 ** 2.0 + s_1_1 ** 2.0 + s_1_2 ** 2.0 +
            s_2_0 ** 2.0 + s_2_1 ** 2.0 + s_2_2 ** 2.0
        )

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
            omega_0_0 ** 2.0 + omega_0_1 ** 2.0 + omega_0_2 ** 2.0 +
            omega_1_0 ** 2.0 + omega_1_1 ** 2.0 + omega_1_2 ** 2.0 +
            omega_2_0 ** 2.0 + omega_2_1 ** 2.0 + omega_2_2 ** 2.0
        )

        # Compute q-criterion
        q_value = 0.5 * (omega_dot_omega - s_dot_s)

        # Set the output
        norm_mu[0, i, j, k] = mu
        q[0, i, j, k] = q_value

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, u, boundary_id, norm_mu, q):

        # Launch the warp kernel
        wp.launch(
            self.q_kernel,
            inputs=[
                u,
                boundary_id,
                norm_mu,
                q,
            ],
            dim=[i - 2 for i in u.shape[1:]],
        )


        return norm_mu, q

class GridToPoint(Operator):

    @wp.kernel
    def grid_to_point(
        grid: wp.array4d(dtype=Any),
        points: wp.array(dtype=wp.vec3),
        point_values: wp.array(dtype=Any),
    ):
        # Get the global index
        i = wp.tid()

        # Get the point
        point = points[i]

        # Get lower and upper bounds
        lower_0_0_0 = wp.vec3i(wp.int32(point[0]), wp.int32(point[1]), wp.int32(point[2]))
        lower_0_0_1 = lower_0_0_0 + wp.vec3i(0, 0, 1)
        lower_0_1_0 = lower_0_0_0 + wp.vec3i(0, 1, 0)
        lower_0_1_1 = lower_0_0_0 + wp.vec3i(0, 1, 1)
        lower_1_0_0 = lower_0_0_0 + wp.vec3i(1, 0, 0)
        lower_1_0_1 = lower_0_0_0 + wp.vec3i(1, 0, 1)
        lower_1_1_0 = lower_0_0_0 + wp.vec3i(1, 1, 0)
        lower_1_1_1 = lower_0_0_0 + wp.vec3i(1, 1, 1)

        # Get grid values
        grid_0_0_0 = grid[0, lower_0_0_0[0], lower_0_0_0[1], lower_0_0_0[2]]
        grid_0_0_1 = grid[0, lower_0_0_1[0], lower_0_0_1[1], lower_0_0_1[2]]
        grid_0_1_0 = grid[0, lower_0_1_0[0], lower_0_1_0[1], lower_0_1_0[2]]
        grid_0_1_1 = grid[0, lower_0_1_1[0], lower_0_1_1[1], lower_0_1_1[2]]
        grid_1_0_0 = grid[0, lower_1_0_0[0], lower_1_0_0[1], lower_1_0_0[2]]
        grid_1_0_1 = grid[0, lower_1_0_1[0], lower_1_0_1[1], lower_1_0_1[2]]
        grid_1_1_0 = grid[0, lower_1_1_0[0], lower_1_1_0[1], lower_1_1_0[2]]
        grid_1_1_1 = grid[0, lower_1_1_1[0], lower_1_1_1[1], lower_1_1_1[2]]

        # Compute the interpolation weights
        dx = point[0] - wp.float32(lower_0_0_0[0])
        dy = point[1] - wp.float32(lower_0_0_0[1])
        dz = point[2] - wp.float32(lower_0_0_0[2])
        w_000 = (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        w_001 = (1.0 - dx) * (1.0 - dy) * dz
        w_010 = (1.0 - dx) * dy * (1.0 - dz)
        w_011 = (1.0 - dx) * dy * dz
        w_100 = dx * (1.0 - dy) * (1.0 - dz)
        w_101 = dx * (1.0 - dy) * dz
        w_110 = dx * dy * (1.0 - dz)
        w_111 = dx * dy * dz

        # Compute the interpolated value
        point_value = (
            w_000 * grid_0_0_0 + w_001 * grid_0_0_1 + w_010 * grid_0_1_0 + w_011 * grid_0_1_1 +
            w_100 * grid_1_0_0 + w_101 * grid_1_0_1 + w_110 * grid_1_1_0 + w_111 * grid_1_1_1
        )

        # Set the output
        point_values[i] = point_value


    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, grid, points, point_values):

        # Launch the warp kernel
        wp.launch(
            self.grid_to_point,
            inputs=[grid, points, point_values],
            dim=[points.shape[0]],
        )

        return point_values


class MyCopy(Operator):

    @wp.kernel
    def simple_copy(
        dest: wp.array4d(dtype=Any),
        src: wp.array4d(dtype=Any),
        q: wp.int32,
    ):
        # Get the global index
        i, j, k = wp.tid()

        # Copy the data
        for ii in range(q):
            dest[ii, i, j, k] = src[ii, i, j, k]

    @wp.kernel
    def upsample_copy(
        dest: wp.array4d(dtype=Any),
        src: wp.array4d(dtype=Any),
        q: wp.int32,
    ):
        # Get the global index
        i, j, k = wp.tid()

        # Copy the data
        for ii in range(q):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        dest[ii, 2 * i + l, 2 * j + m, 2 * k + n] = src[ii, i, j, k]

    @wp.kernel
    def downsample_copy_float(
        dest: wp.array4d(dtype=wp.float32),
        src: wp.array4d(dtype=wp.float32),
        q: wp.int32,
    ):
        # Get the global index
        i, j, k = wp.tid()

        # Copy the data
        for ii in range(q):
            val = wp.float32(0.0)
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        val += src[ii, 2 * i + l, 2 * j + m, 2 * k + n]
            dest[ii, i, j, k] = val / 8.0

    @wp.kernel
    def downsample_copy_int(
        dest: wp.array4d(dtype=Any),
        src: wp.array4d(dtype=Any),
        q: wp.int32,
    ):
        # Get the global index
        i, j, k = wp.tid()

        # Copy the data
        for ii in range(q):
            dest[ii, i, j, k] = src[ii, 2 * i, 2 * j, 2 * k]


    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, dest, src):

        # Check if we need to upsample or downsample
        if 2 * src.shape[1] == dest.shape[1]:
            # Launch the warp kernel
            wp.launch(
                self.upsample_copy,
                inputs=[
                    dest,
                    src,
                    dest.shape[0]
                ],
                dim=src.shape[1:],
            )
        elif src.shape[1] == dest.shape[1]:
            # Launch the warp kernel
            wp.launch(
                self.simple_copy,
                inputs=[
                    dest,
                    src,
                    dest.shape[0]
                ],
                dim=dest.shape[1:],
            )
        elif src.shape[1] == 2 * dest.shape[1]:
            if dest.dtype == wp.float32:
                # Launch the warp kernel
                wp.launch(
                    self.downsample_copy_float,
                    inputs=[
                        dest,
                        src,
                        dest.shape[0]
                    ],
                    dim=dest.shape[1:],
                )
            else:
                # Launch the warp kernel
                wp.launch(
                    self.downsample_copy_int,
                    inputs=[
                        dest,
                        src,
                        dest.shape[0]
                    ],
                    dim=dest.shape[1:],
                )
        else:
            print(src.shape, dest.shape)
            raise ValueError("Invalid source and destination shapes")
        return dest

class IncompressibleNavierStokesStepper(Stepper):
    """
    Fast NS stepper with only equilibrium and full way BC
    """

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(
            self.velocity_set.q, dtype=wp.uint8
        )  # TODO fix vec bool

        # Get the boundary condition ids
        _equilibrium_bc = wp.uint8(self.equilibrium_bc.id)
        _do_nothing_bc = wp.uint8(self.do_nothing_bc.id)
        _halfway_bounce_back_bc = wp.uint8(self.halfway_bounce_back_bc.id)
        _fullway_bounce_back_bc = wp.uint8(self.fullway_bounce_back_bc.id)

        # Construct the kernel
        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            boundary_id: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            timestep: int,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO warp should fix this

            # Get the boundary id and missing mask
            _boundary_id = boundary_id[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply streaming boundary conditions
            if (_boundary_id == wp.uint8(0)) or _boundary_id == _fullway_bounce_back_bc:
                # Regular streaming
                f_post_stream = self.stream.warp_functional(f_0, index)
            elif _boundary_id == _equilibrium_bc:
                # Equilibrium boundary condition
                f_post_stream = self.equilibrium_bc.warp_functional(
                    f_0, _missing_mask, index
                )

            # Compute rho and u
            rho, u = self.macroscopic.warp_functional(f_post_stream)

            # Compute equilibrium
            feq = self.equilibrium.warp_functional(rho, u)

            # Apply collision
            f_post_collision = self.collision.warp_functional(
                f_post_stream,
                feq,
                rho,
                u,
            )

            # Apply collision type boundary conditions
            if _boundary_id == _fullway_bounce_back_bc:
                # Full way boundary condition
                f_post_collision = self.fullway_bounce_back_bc.warp_functional(
                    f_post_stream,
                    f_post_collision,
                    _missing_mask,
                )

            # Set the output
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1], index[2]] = f_post_collision[l]

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, boundary_id, missing_mask, timestep, omega):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f_0,
                f_1,
                boundary_id,
                missing_mask,
                timestep,
            ],
            dim=f_0.shape[1:],
        )
        return f_1
