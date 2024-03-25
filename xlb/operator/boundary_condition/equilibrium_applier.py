"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np
import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.boundary_condition.boundary_applier import (
    BoundaryApplier,
    ImplementationStep,
)
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry


class EquilibriumApplier(BoundaryApplier):
    """
    Apply Equilibrium boundary condition to the distribution function.
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
    def apply_jax(self, f_pre, f_post, boundary, mask):
        equilibrium_mask = jnp.repeat(boundary[..., None], self.velocity_set.q, axis=-1)
        equilibrium_f = jnp.repeat(self.f[None, ...], boundary.shape[0], axis=0)
        equilibrium_f = jnp.repeat(equilibrium_f[:, None], boundary.shape[1], axis=1)
        equilibrium_f = jnp.repeat(equilibrium_f[:, :, None], boundary.shape[2], axis=2)
        f = lax.select(equilibrium_mask, equilibrium_f, f_post)
        return f
