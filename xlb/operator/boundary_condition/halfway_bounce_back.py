import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.stream.stream import Stream
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry


class HalfwayBounceBack(BoundaryCondition):
    """
    Halfway Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """
    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        set_boundary,
        velocity_set: VelocitySet,
        compute_backend: ComputeBackend = ComputeBackend.JAX,
    ):
        super().__init__(
            set_boundary=set_boundary,
            implementation_step=ImplementationStep.STREAMING,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )

    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3, 4))
    def apply_jax(self, f_pre, f_post, boundary, mask):
        flip_mask = boundary[..., jnp.newaxis] & mask
        flipped_f = lax.select(
            flip_mask, f_pre[..., self.velocity_set.opp_indices], f_post
        )
        return flipped_f
