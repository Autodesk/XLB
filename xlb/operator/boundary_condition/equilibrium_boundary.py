import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.stream.stream import Stream
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)

class EquilibriumBoundary(BoundaryCondition):
    """
    A boundary condition that skips the streaming step.
    """

    def __init__(
            self,
            set_boundary,
            rho: float,
            u: tuple[float, float],
            equilibrium: Equilibrium,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackend = ComputeBackend.JAX,
        ):
        super().__init__(
            set_boundary=set_boundary,
            implementation_step=ImplementationStep.STREAMING,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )
        self.f = equilibrium(rho, u)

    @classmethod
    def from_indices(
            cls,
            indices,
            rho: float,
            u: tuple[float, float],
            equilibrium: Equilibrium,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackend = ComputeBackend.JAX,
        ):
        """
        Creates a boundary condition from a list of indices.
        """
       
        return cls(
            set_boundary=cls._set_boundary_from_indices(indices),
            rho=rho,
            u=u,
            equilibrium=equilibrium,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )


    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3, 4))
    def apply_jax(self, f_pre, f_post, boundary, mask):
        equilibrium_mask = jnp.repeat(boundary[..., None], self.velocity_set.q, axis=-1)
        equilibrium_f = jnp.repeat(self.f[None, ...], boundary.shape[0], axis=0)
        equilibrium_f = jnp.repeat(equilibrium_f[:, None], boundary.shape[1], axis=1)
        equilibrium_f = jnp.repeat(equilibrium_f[:, :, None], boundary.shape[2], axis=2)
        f = lax.select(equilibrium_mask, equilibrium_f, f_post)
        return f
