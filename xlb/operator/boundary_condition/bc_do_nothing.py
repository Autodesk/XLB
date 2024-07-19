"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
import warp as wp
from typing import Any, List

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)
from xlb.operator.boundary_condition.boundary_condition_registry import (
    boundary_condition_registry,
)


class DoNothingBC(BoundaryCondition):
    """
    Do nothing boundary condition. This boundary condition skips the streaming step for the
    boundary nodes.
    """

    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices = None,
    ):
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, boundary_mask, missing_mask):
        boundary = boundary_mask == self.id
        return jnp.where(boundary, f_pre, f_post)

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(
            self.velocity_set.q, dtype=wp.uint8
        )  # TODO fix vec bool

        # Construct the funcional to get streamed indices

        @wp.func
        def functional2d(
            f: wp.array3d(dtype=Any),
            missing_mask: Any,
            index: Any,
        ):
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1]]
            return _f

        @wp.func
        def functional3d(
            f: wp.array4d(dtype=Any),
            missing_mask: Any,
            index: Any,
        ):
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            return _f

        @wp.kernel
        def kernel2d(
            f_pre: wp.array3d(dtype=Any),
            f_post: wp.array3d(dtype=Any),
            boundary_mask: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.uint8),
            f: wp.array3d(dtype=Any),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # Get the boundary id and missing mask
            _boundary_id = boundary_mask[0, index[0], index[1]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply the boundary condition
            if _boundary_id == wp.uint8(DoNothingBC.id):
                _f = functional3d(f_pre, _missing_mask, index)
            else:
                _f = _f_vec()
                for l in range(self.velocity_set.q):
                    _f[l] = f_post[l, index[0], index[1]]

            # Write the result
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1]] = _f[l]

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            boundary_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            f: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the boundary id and missing mask
            _boundary_id = boundary_mask[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply the boundary condition
            if _boundary_id == wp.uint8(DoNothingBC.id):
                _f = functional3d(f_pre, _missing_mask, index)
            else:
                _f = _f_vec()
                for l in range(self.velocity_set.q):
                    _f[l] = f_post[l, index[0], index[1], index[2]]

            # Write the result
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1], index[2]] = _f[l]

        functional = functional3d if self.velocity_set.d == 3 else functional2d
        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, boundary_mask, missing_mask, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, boundary_mask, missing_mask, f],
            dim=f_pre.shape[1:],
        )
        return f
