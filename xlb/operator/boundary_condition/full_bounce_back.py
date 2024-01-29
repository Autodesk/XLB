"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)

class FullBounceBack(BoundaryCondition):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """

    def __init__(
            self,
            set_boundary,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackends = ComputeBackends.JAX,
        ):
        super().__init__(
            set_boundary=set_boundary,
            implementation_step=ImplementationStep.COLLISION,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )

    @classmethod
    def from_indices(
            cls,
            indices,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackends = ComputeBackends.JAX,
        ):
        """
        Creates a boundary condition from a list of indices.
        """
        
        return cls(
            set_boundary=cls._set_boundary_from_indices(indices),
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )

    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3, 4))
    def apply_jax(self, f_pre, f_post, boundary, mask):
        flip = jnp.repeat(boundary[..., jnp.newaxis], self.velocity_set.q, axis=-1)
        flipped_f = lax.select(flip, f_pre[..., self.velocity_set.opp_indices], f_post)
        return flipped_f
