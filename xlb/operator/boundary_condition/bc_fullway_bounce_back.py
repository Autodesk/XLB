"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)
from xlb.operator.boundary_condition.boundary_condition_registry import (
    boundary_condition_registry,
)


class FullwayBounceBackBC(BoundaryCondition):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """

    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
    ):
        super().__init__(
            ImplementationStep.COLLISION,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def apply_jax(self, f_pre, f_post, boundary_mask, missing_mask):
        boundary = boundary_mask == self.id
        boundary = jnp.repeat(boundary, self.velocity_set.q, axis=0)
        return jnp.where(boundary, f_pre[self.velocity_set.opp_indices, ...], f_post)

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _opp_indices = self.velocity_set.wp_opp_indices
        _q = wp.constant(self.velocity_set.q)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)  # TODO fix vec bool

        # Construct the funcional to get streamed indices
        @wp.func
        def functional(
            f_pre: Any,
            missing_mask: Any,
            index: Any,
        ):
            fliped_f = _f_vec()
            for l in range(_q):
                fliped_f[l] = f_pre[_opp_indices[l]]
            return fliped_f

        @wp.kernel
        def kernel2d(
            f_pre: wp.array3d(dtype=Any),
            f_post: wp.array3d(dtype=Any),
            boundary_mask: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.bool),
        ):  # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # Get the boundary id and missing mask
            _boundary_id = boundary_mask[0, index[0], index[1]]

            # Make vectors for the lattice
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                _f_pre[l] = f_pre[l, index[0], index[1]]
                _f_post[l] = f_post[l, index[0], index[1]]

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1]]:
                    _mask[l] = wp.uint8(1)
                else:
                    _mask[l] = wp.uint8(0)

            # Check if the boundary is active
            if _boundary_id == wp.uint8(FullwayBounceBackBC.id):
                _f = functional(_f_pre, _mask, index)
            else:
                _f = _f_post

            # Write the result to the output
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1]] = _f[l]

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            boundary_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the boundary id and missing mask
            _boundary_id = boundary_mask[0, index[0], index[1], index[2]]

            # Make vectors for the lattice
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                _f_pre[l] = f_pre[l, index[0], index[1], index[2]]
                _f_post[l] = f_post[l, index[0], index[1], index[2]]

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _mask[l] = wp.uint8(1)
                else:
                    _mask[l] = wp.uint8(0)

            # Check if the boundary is active
            if _boundary_id == wp.uint8(FullwayBounceBackBC.id):
                _f = functional(_f_pre, _mask, index)
            else:
                _f = _f_post

            # Write the result to the output
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1], index[2]] = _f[l]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, boundary_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, boundary_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post
