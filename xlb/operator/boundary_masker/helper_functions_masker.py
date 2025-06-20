import warp as wp
from typing import Any
from xlb import DefaultConfig, ComputeBackend


class HelperFunctionsMasker(object):
    """
    A collection of helper functions used for the boundary masker operators.
    """

    def __init__(self, velocity_set=None, precision_policy=None, compute_backend=None):
        if compute_backend == ComputeBackend.JAX:
            raise ValueError("This helper class contains helper functions only for the WARP implementation of some BCs not JAX!")

        # Set the default values from the global config
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.default_precision_policy
        self.compute_backend = compute_backend or DefaultConfig.default_backend

        # Set local constants
        _d = self.velocity_set.d
        _c = self.velocity_set.c

        @wp.func
        def neon_index_to_warp(neon_field_hdl: Any, index: Any):
            # Unpack the global index in Neon
            cIdx = wp.neon_global_idx(neon_field_hdl, index)
            gx = wp.neon_get_x(cIdx)
            gy = wp.neon_get_y(cIdx)
            gz = wp.neon_get_z(cIdx)

            # TODO@Max - XLB is flattening the z dimension in 3D, while neon uses the y dimension
            if _d == 2:
                gy, gz = gz, gy

            # Get warp indices
            index_wp = wp.vec3i(gx, gy, gz)
            return index_wp

        @wp.func
        def index_to_position_warp(field: Any, index: wp.vec3i):
            # position of the point
            ijk = wp.vec3(wp.float32(index[0]), wp.float32(index[1]), wp.float32(index[2]))
            pos = ijk + wp.vec3(0.5, 0.5, 0.5)  # cell center
            return pos

        @wp.func
        def index_to_position_neon(field: Any, index: Any):
            # position of the point
            index_wp = neon_index_to_warp(field, index)
            return index_to_position_warp(field, index_wp)

        @wp.func
        def is_in_bounds(index: wp.vec3i, field: wp.array4d(dtype=wp.uint8)):
            grid_shape = wp.vec3i(field.shape[1], field.shape[2], field.shape[3])
            return (
                index[0] >= 0
                and index[0] < grid_shape[0]
                and index[1] >= 0
                and index[1] < grid_shape[1]
                and index[2] >= 0
                and index[2] < grid_shape[2]
            )

        @wp.func
        def get_grid_shape_neon(bc_mask: wp.array4d(dtype=wp.uint8)):
            return bc_mask.get_grid().dim.x, bc_mask.get_grid().dim.y, bc_mask.get_grid().dim.z

        @wp.func
        def get_pull_index_warp(
            field: wp.array4d(dtype=wp.uint8),
            lattice_dir: wp.int32,
            index: wp.vec3i,
        ):
            pull_index = wp.vec3i()
            for d in range(self.velocity_set.d):
                pull_index[d] = index[d] - _c[d, lattice_dir]

            return pull_index, pull_index

        @wp.func
        def get_pull_index_neon(
            field: wp.array4d(dtype=wp.uint8),
            lattice_dir: wp.int32,
            index: wp.vec3i,
        ):
            # Convert the index to warp
            index_wp = neon_index_to_warp(field, index)
            pull_index_wp = get_pull_index_warp(field, lattice_dir, index_wp)
            pull_index_neon = wp.neon_ngh_idx(wp.int8(-_c[0, lattice_dir]), wp.int8(-_c[1, lattice_dir]), wp.int8(-_c[2, lattice_dir]))
            return pull_index_wp, pull_index_neon

        @wp.func
        def is_in_bc_indices_warp(
            field: wp.array4d(dtype=wp.uint8),
            index: wp.vec3i,
            bc_indices: wp.array2d(dtype=wp.int32),
            ii: wp.int32,
        ):
            return bc_indices[0, ii] == index[0] and bc_indices[1, ii] == index[1] and bc_indices[2, ii] == index[2]

        @wp.func
        def is_in_bc_indices_neon(
            field: Any,
            index: Any,
            bc_indices: wp.array2d(dtype=wp.int32),
            ii: wp.int32,
        ):
            index_wp = neon_index_to_warp(field, index)
            return is_in_bc_indices_warp(field, index_wp, bc_indices, ii)

        # Construct some helper warp functions
        self.is_in_bounds = is_in_bounds
        self.index_to_position = index_to_position_warp if self.compute_backend == ComputeBackend.WARP else index_to_position_neon
        self.get_pull_index = get_pull_index_warp if self.compute_backend == ComputeBackend.WARP else get_pull_index_neon
        self.is_in_bc_indices = is_in_bc_indices_warp if self.compute_backend == ComputeBackend.WARP else is_in_bc_indices_neon

    def get_grid_shape(self, bc_mask):
        """
        Get the grid shape from the boundary mask.
        """
        if self.compute_backend == ComputeBackend.WARP:
            return bc_mask.shape[1:]
        elif self.compute_backend == ComputeBackend.NEON:
            return self.get_grid_shape_neon(bc_mask)
        else:
            raise ValueError(f"Unsupported compute backend: {self.compute_backend}")
