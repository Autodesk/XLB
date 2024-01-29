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

class DoNothing(BoundaryCondition):
    """
    A boundary condition that skips the streaming step.
    """

    def __init__(
            self,
            set_boundary,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackends = ComputeBackends.JAX,
        ):
        super().__init__(
            set_boundary=set_boundary,
            implementation_step=ImplementationStep.STREAMING,
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
        do_nothing = jnp.repeat(boundary[..., None], self.velocity_set.q, axis=-1)
        f = lax.select(do_nothing, f_pre, f_post)
        return f
