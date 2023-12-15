# Base class for all equilibriums

from functools import partial
import jax.numpy as jnp
from jax import jit
import numba
from numba import cuda

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class Equilibrium(Operator):
    """
    Base class for all equilibriums
    """

    def __init__(
            self,
            velocity_set: VelocitySet,
            compute_backend=ComputeBackend.JAX,
        ):
        super().__init__(velocity_set, compute_backend)


class QuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.

    TODO: move this to a separate file and lower and higher order equilibriums
    """

    def __init__(
            self,
            velocity_set: VelocitySet,
            compute_backend=ComputeBackend.JAX,
        ):
        super().__init__(velocity_set, compute_backend)

    @partial(jit, static_argnums=(0), donate_argnums=(1, 2))
    def apply_jax(self, rho, u):
        """
        JAX implementation of the equilibrium distribution function.

        # TODO: This might be optimized using a for loop for because
        # the compiler will remove 0 c terms.
        """
        cu = 3.0 * jnp.dot(u, jnp.array(self.velocity_set.c, dtype=rho.dtype))
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        feq = (
            rho
            * jnp.array(self.velocity_set.w, dtype=rho.dtype)
            * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        )
        return feq

    def construct_numba(self, velocity_set: VelocitySet, dtype=numba.float32):
        """
        Numba implementation of the equilibrium distribution function.
        """
        # Get needed values for numba functions
        q = velocity_set.q
        c = velocity_set.c.T
        w = velocity_set.w

        # Make numba functions
        @cuda.jit(device=True)
        def _equilibrium(rho, u, feq):
            # Compute the equilibrium distribution function
            usqr = dtype(1.5) * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
            for i in range(q):
                cu = dtype(3.0) * (
                    u[0] * dtype(c[i, 0])
                    + u[1] * dtype(c[i, 1])
                    + u[2] * dtype(c[i, 2])
                )
                feq[i] = (
                    rho[0]
                    * dtype(w[i])
                    * (dtype(1.0) + cu * (dtype(1.0) + dtype(0.5) * cu) - usqr)
                )

            # Return the equilibrium distribution function
            return feq  # comma is needed for numba to return a tuple, seems like a bug in numba

        return _equilibrium
