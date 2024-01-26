# Base class for all equilibriums

from functools import partial
import jax.numpy as jnp
from jax import jit

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator.operator import Operator


class Macroscopic(Operator):
    """
    Base class for all macroscopic operators

    TODO: Currently this is only used for the standard rho and u moments.
    In the future, this should be extended to include higher order moments
    and other physic types (e.g. temperature, electromagnetism, etc...)
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        compute_backend=ComputeBackends.JAX,
    ):
        super().__init__(velocity_set, compute_backend)

    @Operator.register_backend(ComputeBackends.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f):
        """
        Apply the macroscopic operator to the lattice distribution function
        """
        rho = jnp.sum(f, axis=0, keepdims=True)
        u = jnp.tensordot(self.velocity_set.c, f, axes=(-1, 0)) / rho

        return rho, u
