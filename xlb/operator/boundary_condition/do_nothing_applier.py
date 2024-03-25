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


class DoNothingApplier(BoundaryApplier):
    """
    Do nothing boundary condition. Basically skips the streaming step
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
        do_nothing = jnp.repeat(boundary[..., None], self.velocity_set.q, axis=-1)
        f = lax.select(do_nothing, f_pre, f_post)
        return f
