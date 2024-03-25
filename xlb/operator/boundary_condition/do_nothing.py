"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np
import warp as wp
from typing import Tuple, Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep, BoundaryCondition
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry

class DoNothingBC(BoundaryCondition):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """
    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend,
    ):
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3, 4))
    def apply_jax(self, f_pre, f_post, boundary_id, missing_mask):
        boundary = boundary_id == self.id
        flip = jnp.repeat(boundary, self.velocity_set.q, axis=0)
        skipped_f = lax.select(flip, f_pre, f_post)
        return skipped_f

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        # Construct the funcional to get streamed indices
        @wp.func
        def functional(
            f: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=wp.bool),
            index: Any,
        ):
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            return _f

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            boundary_id: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get boundary id
            if boundary_id[0, index[0], index[1], index[2]] == wp.uint8(DoNothing.id):
                _f = functional(f_pre, index)
                for l in range(_q):
                    f_post[l, index[0], index[1], index[2]] = _f[l]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, f, boundary, mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel, inputs=[f_pre, f_post, f, boundary, mask], dim=f_pre.shape[1:]
        )
        return f
