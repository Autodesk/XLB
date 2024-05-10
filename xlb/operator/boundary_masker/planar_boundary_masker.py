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
    @partial(jit, static_argnums=(0, 1, 2, 3, 4, 7))
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
            lower_bound: wp.vec2i,
            upper_bound: wp.vec2i,
            direction: wp.vec2i,
            id_number: wp.uint8,
            boundary_id_field: wp.array3d(dtype=wp.uint8),
            mask: wp.array3d(dtype=wp.bool),
            start_index: wp.vec2i,
        ):
            i, j = wp.tid()
            lb_x, lb_y = lower_bound.x + start_index.x, lower_bound.y + start_index.y
            ub_x, ub_y = upper_bound.x + start_index.x, upper_bound.y + start_index.y

            if lb_x <= i < ub_x and lb_y <= j < ub_y:
                boundary_id_field[0, i, j] = id_number

        @wp.kernel
        def kernel3d(
            lower_bound: wp.vec3i,
            upper_bound: wp.vec3i,
            direction: wp.vec3i,
            id_number: wp.uint8,
            boundary_id_field: wp.array4d(dtype=wp.uint8),
            mask: wp.array4d(dtype=wp.bool),
            start_index: wp.vec3i,
        ):
            i, j, k = wp.tid()
            lb_x, lb_y, lb_z = (
                lower_bound.x + start_index.x,
                lower_bound.y + start_index.y,
                lower_bound.z + start_index.z,
            )
            ub_x, ub_y, ub_z = (
                upper_bound.x + start_index.x,
                upper_bound.y + start_index.y,
                upper_bound.z + start_index.z,
            )

            if lb_x <= i < ub_x and lb_y <= j < ub_y and lb_z <= k < ub_z:
                boundary_id_field[0, i, j, k] = id_number

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
        start_index=None,
    ):
        if start_index is None:
            start_index = (0,) * self.velocity_set.d
    
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
            dim=mask.shape[1:],
        )

        return boundary_id_field, mask
