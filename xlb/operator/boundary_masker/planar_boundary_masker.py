# Base class for all equilibriums

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Tuple

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
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.JAX,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

    @Operator.register_backend(ComputeBackend.JAX)
    # @partial(jit, static_argnums=(0), inline=True) TODO: Fix this
    def jax_implementation(
        self,
        lower_bound,
        upper_bound,
        direction,
        id_number,
        boundary_id,
        mask,
        start_index=(0, 0, 0),
    ):
        # TODO: Optimize this
        
        # x plane
        if direction[0] != 0:

            # Set boundary id
            boundary_id = boundary_id.at[0, lower_bound[0], lower_bound[1] : upper_bound[1], lower_bound[2] : upper_bound[2]].set(id_number)

            # Set mask
            for l in range(self.velocity_set.q):
                d_dot_c = (
                    direction[0] * self.velocity_set.c[0, l]
                    + direction[1] * self.velocity_set.c[1, l]
                    + direction[2] * self.velocity_set.c[2, l]
                )
                if d_dot_c >= 0:
                    mask = mask.at[l, lower_bound[0], lower_bound[1] : upper_bound[1], lower_bound[2] : upper_bound[2]].set(True)

        # y plane
        elif direction[1] != 0:

            # Set boundary id
            boundary_id = boundary_id.at[0, lower_bound[0] : upper_bound[0], lower_bound[1], lower_bound[2] : upper_bound[2]].set(id_number)

            # Set mask
            for l in range(self.velocity_set.q):
                d_dot_c = (
                    direction[0] * self.velocity_set.c[0, l]
                    + direction[1] * self.velocity_set.c[1, l]
                    + direction[2] * self.velocity_set.c[2, l]
                )
                if d_dot_c >= 0:
                    mask = mask.at[l, lower_bound[0] : upper_bound[0], lower_bound[1], lower_bound[2] : upper_bound[2]].set(True)

        # z plane
        elif direction[2] != 0:

            # Set boundary id
            boundary_id = boundary_id.at[0, lower_bound[0] : upper_bound[0], lower_bound[1] : upper_bound[1], lower_bound[2]].set(id_number)

            # Set mask
            for l in range(self.velocity_set.q):
                d_dot_c = (
                    direction[0] * self.velocity_set.c[0, l]
                    + direction[1] * self.velocity_set.c[1, l]
                    + direction[2] * self.velocity_set.c[2, l]
                )
                if d_dot_c >= 0:
                    mask = mask.at[l, lower_bound[0] : upper_bound[0], lower_bound[1] : upper_bound[1], lower_bound[2]].set(True)

        return boundary_id, mask


    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.wp_c
        _q = wp.constant(self.velocity_set.q)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            lower_bound: wp.vec3i,
            upper_bound: wp.vec3i,
            direction: wp.vec3i,
            id_number: wp.int32,
            boundary_id: wp.array4d(dtype=wp.uint8),
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
                boundary_id[0, i, j, k] = wp.uint8(id_number)

                # Set mask for just directions coming from the boundary
                for l in range(_q):
                    d_dot_c = (
                        direction[0] * _c[0, l]
                        + direction[1] * _c[1, l]
                        + direction[2] * _c[2, l]
                    )
                    if d_dot_c >= 0:
                        mask[l, i, j, k] = wp.bool(True)

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        lower_bound,
        upper_bound,
        direction,
        id_number,
        boundary_id,
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
                boundary_id,
                mask,
                start_index,
            ],
            dim=dim,
        )

        return boundary_id, mask
