# Base class for all equilibriums
from xlb.global_config import GlobalConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


from functools import partial
import jax.numpy as jnp
from jax import jit


class Macroscopic(Operator):
    """
    Base class for all macroscopic operators

    TODO: Currently this is only used for the standard rho and u moments.
    In the future, this should be extended to include higher order moments
    and other physic types (e.g. temperature, electromagnetism, etc...)
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        compute_backend=None,
    ):
        self.velocity_set = velocity_set or GlobalConfig.velocity_set
        self.compute_backend = compute_backend or GlobalConfig.compute_backend

        super().__init__(velocity_set, compute_backend)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f):
        """
        Apply the macroscopic operator to the lattice distribution function
        TODO: Check if the following implementation is more efficient (
        as the compiler may be able to remove operations resulting in zero)
        c_x = tuple(self.velocity_set.c[0])
        c_y = tuple(self.velocity_set.c[1])

        u_x = 0.0
        u_y = 0.0

        rho = jnp.sum(f, axis=0, keepdims=True)

        for i in range(self.velocity_set.q):
            u_x += c_x[i] * f[i, ...]
            u_y += c_y[i] * f[i, ...]
        return rho, jnp.stack((u_x, u_y))
        """
        rho = jnp.sum(f, axis=0, keepdims=True)
        u = jnp.tensordot(self.velocity_set.c, f, axes=(-1, 0)) / rho

        return rho, u
