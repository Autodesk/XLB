"""
BGK collision operator for LBM.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from numba import cuda, float32

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision


class BGK(Collision):
    """
    BGK collision operator for LBM.

    The BGK collision operator is the simplest collision operator for LBM.
    It is based on the Bhatnagar-Gross-Krook approximation to the Boltzmann equation.
    Reference: https://en.wikipedia.org/wiki/Bhatnagar%E2%80%93Gross%E2%80%93Krook_operator
    """

    def __init__(
            self,
            omega: float,
            velocity_set: VelocitySet,
            compute_backend=ComputeBackend.JAX,
        ):
        super().__init__(
            omega=omega,
            velocity_set=velocity_set,
            compute_backend=compute_backend
        )

    @partial(jit, static_argnums=(0), donate_argnums=(1,2,3,4))
    def apply_jax(
        self,
        f: jnp.ndarray,
        feq: jnp.ndarray,
        rho: jnp.ndarray,
        u : jnp.ndarray,
    ):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.

        Parameters
        ----------
        f : jax.numpy.ndarray
            The distribution function
        feq : jax.numpy.ndarray
            The equilibrium distribution function
        rho : jax.numpy.ndarray
            The macroscopic density
        u : jax.numpy.ndarray
            The macroscopic velocity

        """
        fneq = f - feq
        fout = f - self.omega * fneq
        return fout

    def construct_numba(self):
        """
        Numba implementation of the collision operator.

        Returns
        -------
        _collision : numba.cuda.jit
            The compiled numba function for the collision operator.
        """

        # Get needed parameters for numba function
        omega = self.omega
        omega = float32(omega)

        # Make numba function
        @cuda.jit(device=True)
        def _collision(f, feq, rho, u, fout):
            """
            Numba BGK collision step for lattice.

            The collision step is where the main physics of the LBM is applied. In the BGK approximation,
            the distribution function is relaxed towards the equilibrium distribution function.

            Parameters
            ----------
            f : cuda.local.array
                The distribution function
            feq : cuda.local.array
                The equilibrium distribution function
            rho : cuda.local.array
                The macroscopic density
            u : cuda.local.array
                The macroscopic velocity
            fout : cuda.local.array
                The output distribution function
            """

            # Relaxation
            for i in range(f.shape[0]):
                fout[i] = f[i] - omega * (f[i] - feq[i])

            return fout

        return _collision
