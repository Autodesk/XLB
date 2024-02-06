# Base class for all equilibriums
from xlb.global_config import GlobalConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
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

    @Operator.register_backend(ComputeBackends.JAX)
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

    @Operator.register_backend(ComputeBackends.PALLAS)
    def pallas_implementation(self, f):
        # TODO: Maybe this can be done with jnp.sum
        rho = jnp.sum(f, axis=0, keepdims=True)
        
        u = jnp.zeros((3, *rho.shape[1:]))
        u.at[0].set(
            -f[9]
            - f[10]
            - f[11]
            - f[12]
            - f[13]
            + f[14]
            + f[15]
            + f[16]
            + f[17]
            + f[18]
        ) / rho
        u.at[1].set(
            -f[3] - f[4] - f[5] + f[6] + f[7] + f[8] - f[12] + f[13] - f[17] + f[18]
        ) / rho
        u.at[2].set(
            -f[1] + f[2] - f[4] + f[5] - f[7] + f[8] - f[10] + f[11] - f[15] + f[16]
        ) / rho

        return rho, jnp.array(u)
