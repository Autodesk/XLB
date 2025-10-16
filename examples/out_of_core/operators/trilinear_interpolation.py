import warp as wp


class TrilinearInterpolation:
    """
    Operator for trilinear interpolation from a grid to points in space.

    The grid is assumed to be a 4D array with shape (q, nx, ny, nz) where:
    - q: number of quantities to interpolate
    - nx, ny, nz: grid dimensions in each direction

    Values are assumed to be cell-centered.
    """

    @wp.kernel
    def _trilinear_interpolation(
        grid: wp.array4d(dtype=float),
        points: wp.array(dtype=wp.vec3),
        point_values: wp.array2d(dtype=float),
        origin: wp.vec3,
        spacing: wp.vec3,
    ):
        # Get the global index
        i = wp.tid()

        # Get the point
        point = points[i]

        # Convert point to grid coordinates (cell-centered)
        x = (point[0] - origin[0]) / spacing[0] - 0.5
        y = (point[1] - origin[1]) / spacing[1] - 0.5
        z = (point[2] - origin[2]) / spacing[2] - 0.5

        # Clamp to valid range
        nx = grid.shape[1] - 1
        ny = grid.shape[2] - 1
        nz = grid.shape[3] - 1

        x = wp.clamp(x, 0.0, float(nx))
        y = wp.clamp(y, 0.0, float(ny))
        z = wp.clamp(z, 0.0, float(nz))

        # Get lower and upper bounds
        lower_0_0_0 = wp.vec3i(wp.int32(x), wp.int32(y), wp.int32(z))

        # Ensure we don't exceed grid bounds
        lower_0_0_0[0] = wp.min(lower_0_0_0[0], nx - 1)
        lower_0_0_0[1] = wp.min(lower_0_0_0[1], ny - 1)
        lower_0_0_0[2] = wp.min(lower_0_0_0[2], nz - 1)

        lower_0_0_1 = lower_0_0_0 + wp.vec3i(0, 0, 1)
        lower_0_1_0 = lower_0_0_0 + wp.vec3i(0, 1, 0)
        lower_0_1_1 = lower_0_0_0 + wp.vec3i(0, 1, 1)
        lower_1_0_0 = lower_0_0_0 + wp.vec3i(1, 0, 0)
        lower_1_0_1 = lower_0_0_0 + wp.vec3i(1, 0, 1)
        lower_1_1_0 = lower_0_0_0 + wp.vec3i(1, 1, 0)
        lower_1_1_1 = lower_0_0_0 + wp.vec3i(1, 1, 1)

        # Compute the interpolation weights
        dx = x - wp.float32(lower_0_0_0[0])
        dy = y - wp.float32(lower_0_0_0[1])
        dz = z - wp.float32(lower_0_0_0[2])
        w_000 = (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        w_001 = (1.0 - dx) * (1.0 - dy) * dz
        w_010 = (1.0 - dx) * dy * (1.0 - dz)
        w_011 = (1.0 - dx) * dy * dz
        w_100 = dx * (1.0 - dy) * (1.0 - dz)
        w_101 = dx * (1.0 - dy) * dz
        w_110 = dx * dy * (1.0 - dz)
        w_111 = dx * dy * dz

        # Loop over values to interpolate
        for n in range(grid.shape[0]):
            # Get grid values
            grid_0_0_0 = grid[n, lower_0_0_0[0], lower_0_0_0[1], lower_0_0_0[2]]
            grid_0_0_1 = grid[n, lower_0_0_1[0], lower_0_0_1[1], lower_0_0_1[2]]
            grid_0_1_0 = grid[n, lower_0_1_0[0], lower_0_1_0[1], lower_0_1_0[2]]
            grid_0_1_1 = grid[n, lower_0_1_1[0], lower_0_1_1[1], lower_0_1_1[2]]
            grid_1_0_0 = grid[n, lower_1_0_0[0], lower_1_0_0[1], lower_1_0_0[2]]
            grid_1_0_1 = grid[n, lower_1_0_1[0], lower_1_0_1[1], lower_1_0_1[2]]
            grid_1_1_0 = grid[n, lower_1_1_0[0], lower_1_1_0[1], lower_1_1_0[2]]
            grid_1_1_1 = grid[n, lower_1_1_1[0], lower_1_1_1[1], lower_1_1_1[2]]

            # Compute the interpolated value
            point_value = (
                w_000 * grid_0_0_0
                + w_001 * grid_0_0_1
                + w_010 * grid_0_1_0
                + w_011 * grid_0_1_1
                + w_100 * grid_1_0_0
                + w_101 * grid_1_0_1
                + w_110 * grid_1_1_0
                + w_111 * grid_1_1_1
            )

            # Set the output
            point_values[n, i] = point_value

    def __call__(
        self,
        grid: wp.array4d,
        points: wp.array,
        origin: wp.vec3,
        spacing: wp.vec3,
        point_values: wp.array2d,
    ) -> wp.array2d:
        """
        Interpolate values from a grid to points in space.

        Parameters
        ----------
        grid : wp.array4d(dtype=float)
            Input grid with shape (q, nx, ny, nz)
        points : wp.array(dtype=wp.vec3)
            Points to interpolate to
        origin : wp.vec3
            Origin of the grid (lower corner)
        spacing : wp.vec3
            Grid spacing in each direction
        point_values : wp.array2d(dtype=float)
            Output array with shape (q, num_points).

        Returns
        -------
        wp.array2d
            Interpolated values at each point
        """

        # Launch the kernel
        wp.launch(
            self._trilinear_interpolation,
            dim=points.shape[0],
            inputs=[grid, points, point_values, origin, spacing],
        )

        return point_values
