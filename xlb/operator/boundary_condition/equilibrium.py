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

class EquilibriumBC(BoundaryCondition):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """
    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        rho: float,
        u: Tuple[float, float, float],
        equilibrium_operator: Operator,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend,
    ):
        # Store the equilibrium information
        self.rho = rho
        self.u = u 
        self.equilibrium_operator = equilibrium_operator

        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3, 4))
    def apply_jax(self, f_pre, f_post, boundary_id, missing_mask):
        raise NotImplementedError

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _rho = wp.float32(self.rho)
        _u = _u_vec(self.u[0], self.u[1], self.u[2])

        # Construct the funcional to get streamed indices
        @wp.func
        def functional(
            f: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=wp.bool),
            index: Any,
        ):
            _f = self.equilibrium_operator.warp_functional(_rho, _u)
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