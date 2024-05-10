# Base class for all equilibriums

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Tuple
from jax.numpy import where, einsum, full_like

from xlb.default_config import DefaultConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.stream.stream import Stream


class PlanarBoundaryMasker(Operator):
    """
    Operator for creating a boundary mask on a plane of the domain
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

    @Operator.register_backend(ComputeBackend.JAX)
    # @partial(jit, static_argnums=(0, 1, 2, 3, 4, 7))
    def jax_implementation(
        self,
        lower_bound,
        upper_bound,
        direction,
        id_number,
        boundary_id_field,
        mask,
        start_index=None,
    ):
        if start_index is None:
            start_index = (0,) * self.velocity_set.d

        _, *dimensions = boundary_id_field.shape

        indices = [
            (max(0, lb + start), min(dim, ub + start))
            for lb, ub, start, dim in zip(
                lower_bound, upper_bound, start_index, dimensions
            )
        ]

        slices = [slice(None)]
        slices.extend(slice(lb, ub) for lb, ub in indices)
        boundary_id_field = boundary_id_field.at[tuple(slices)].set(id_number)

        return boundary_id_field, None

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.wp_c
        _q = wp.constant(self.velocity_set.q)

        @wp.kernel
        def kernel2d(
            lower_bound: wp.vec3i,
            upper_bound: wp.vec3i,
            direction: wp.vec2i,
            id_number: wp.int32,
            boundary_id_field: wp.array3d(dtype=wp.uint8),
            mask: wp.array3d(dtype=wp.bool),
            start_index: wp.vec2i,
        ):
            # Get the indices of the plane to mask
            plane_i, plane_j = wp.tid()

            # Get local indices
            if direction[0] != 0:
                i = lower_bound[0] - start_index[0]
                j = plane_i + lower_bound[1] - start_index[1]
            elif direction[1] != 0:
                i = plane_i + lower_bound[0] - start_index[0]
                j = lower_bound[1] - start_index[1]

            # Check if in bounds
            if i >= 0 and i < mask.shape[1] and j >= 0 and j < mask.shape[2]:
                # Set the boundary id
                boundary_id_field[0, i, j] = wp.uint8(id_number)

                # Set mask for just directions coming from the boundary
                for l in range(_q):
                    d_dot_c = (
                        direction[0] * _c[0, l]
                        + direction[1] * _c[1, l]
                        + direction[2] * _c[2, l]
                    )
                    if d_dot_c >= 0:
                        mask[l, i, j] = wp.bool(True)

        @wp.kernel
        def kernel3d(
            lower_bound: wp.vec3i,
            upper_bound: wp.vec3i,
            direction: wp.vec3i,
            id_number: wp.int32,
            boundary_id_field: wp.array4d(dtype=wp.uint8),
            mask: wp.array4d(dtype=wp.bool),
            start_index: wp.vec3i,
        ):
            # Get the indices of the plane to mask
            plane_i, plane_j = wp.tid()

            # Get local indices
            if direction[0] != 0:
                i = lower_bound[0] - start_index[0]
                j = plane_i + lower_bound[1] - start_index[1]
                k = plane_j + lower_bound[2] - start_index[2]
            elif direction[1] != 0:
                i = plane_i + lower_bound[0] - start_index[0]
                j = lower_bound[1] - start_index[1]
                k = plane_j + lower_bound[2] - start_index[2]
            elif direction[2] != 0:
                i = plane_i + lower_bound[0] - start_index[0]
                j = plane_j + lower_bound[1] - start_index[1]
                k = lower_bound[2] - start_index[2]

            # Check if in bounds
            if (
                i >= 0
                and i < mask.shape[1]
                and j >= 0
                and j < mask.shape[2]
                and k >= 0
                and k < mask.shape[3]
            ):
                # Set the boundary id
                boundary_id_field[0, i, j, k] = wp.uint8(id_number)

                # Set mask for just directions coming from the boundary
                for l in range(_q):
                    d_dot_c = (
                        direction[0] * _c[0, l]
                        + direction[1] * _c[1, l]
                        + direction[2] * _c[2, l]
                    )
                    if d_dot_c >= 0:
                        mask[l, i, j, k] = wp.bool(True)

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        lower_bound,
        upper_bound,
        direction,
        id_number,
        boundary_id_field,
        mask,
        start_index=(0, 0, 0),
    ):
        # Get plane dimensions
        if direction[0] != 0:
            dim = (
                upper_bound[1] - lower_bound[1],
                upper_bound[2] - lower_bound[2],
            )
        elif direction[1] != 0:
            dim = (
                upper_bound[0] - lower_bound[0],
                upper_bound[2] - lower_bound[2],
            )
        elif direction[2] != 0:
            dim = (
                upper_bound[0] - lower_bound[0],
                upper_bound[1] - lower_bound[1],
            )

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                lower_bound,
                upper_bound,
                direction,
                id_number,
                boundary_id_field,
                mask,
                start_index,
            ],
            dim=dim,
        )

        return boundary_id_field, mask
