import warp as wp
from typing import Any
from functools import partial
from jax import jit

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class GridToPoint(Operator):
    """
    Interpolate values from a grid to arbitrary points using trilinear interpolation.
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
            # Trilinear interpolation: sum contributions from each corner of the cube
            point_value = (
                (w_000 * grid_0_0_0) +  # (0,0,0) corner
                (w_001 * grid_0_0_1) +  # (0,0,1) corner
                (w_010 * grid_0_1_0) +  # (0,1,0) corner
                (w_011 * grid_0_1_1) +  # (0,1,1) corner
                (w_100 * grid_1_0_0) +  # (1,0,0) corner
                (w_101 * grid_1_0_1) +  # (1,0,1) corner
                (w_110 * grid_1_1_0) +  # (1,1,0) corner
                (w_111 * grid_1_1_1)    # (1,1,1) corner
            )

            # Set the output
            point_values[i] = point_value

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, grid, points, point_values):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[grid, points, point_values],
            dim=[points.shape[0]],
        )
        return point_values

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, grid, points, point_values):
        # TODO: Implement JAX version
        raise NotImplementedError("JAX implementation not yet available")
