"""
KBC collision operator for LBM.
"""

import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any
from functools import partial

from xlb.velocity_set import VelocitySet, D2Q9, D3Q27
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from xlb.operator.macroscopic import SecondMoment as MomentumFlux


class KBC(Collision):
    """
    KBC collision operator for LBM.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        self.momentum_flux = MomentumFlux()
        self.epsilon = 1e-32

        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,), donate_argnums=(1, 2, 3))
    def jax_implementation(
        self,
        f: jnp.ndarray,
        feq: jnp.ndarray,
        rho: jnp.ndarray,
        u: jnp.ndarray,
        omega,
    ):
        """
        KBC collision step for lattice.

        Parameters
        ----------
        f : jax.numpy.array
            Distribution function.
        feq : jax.numpy.array
            Equilibrium distribution function.
        rho : jax.numpy.array
            Density.
        u : jax.numpy.array
            Velocity.
        """
        fneq = f - feq
        if isinstance(self.velocity_set, D2Q9):
            shear = self.decompose_shear_d2q9_jax(fneq)
            delta_s = shear * rho / 4.0
        elif isinstance(self.velocity_set, D3Q27):
            shear = self.decompose_shear_d3q27_jax(fneq)
            delta_s = shear * rho
        else:
            raise NotImplementedError("Velocity set not supported: {}".format(type(self.velocity_set)))

        # Compute required constants based on the input omega (omega is the inverse relaxation time)
        beta = self.compute_dtype(0.5) * self.compute_dtype(omega)
        inv_beta = 1.0 / beta

        # Perform collision
        delta_h = fneq - delta_s
        gamma = inv_beta - (2.0 - inv_beta) * self.entropic_scalar_product(delta_s, delta_h, feq) / (
            self.epsilon + self.entropic_scalar_product(delta_h, delta_h, feq)
        )

        fout = f - beta * (2.0 * delta_s + gamma[None, ...] * delta_h)

        return fout

    @partial(jit, static_argnums=(0,), inline=True)
    def entropic_scalar_product(self, x: jnp.ndarray, y: jnp.ndarray, feq: jnp.ndarray):
        """
        Compute the entropic scalar product of x and y to approximate gamma in KBC.

        Returns
        -------
        jax.numpy.array
            Entropic scalar product of x, y, and feq.
        """
        return jnp.sum(x * y / feq, axis=0)

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d3q27_jax(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components of fneq.
        """

        # Calculate the momentum flux
        Pi = self.momentum_flux(fneq)
        # Calculating Nxz and Nyz with indices moved to the first dimension
        Nxz = Pi[0, ...] - Pi[5, ...]
        Nyz = Pi[3, ...] - Pi[5, ...]

        # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
        s = jnp.zeros_like(fneq)
        s = s.at[9, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[18, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[3, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[6, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[1, ...].set((-Nxz - Nyz) / 6.0)
        s = s.at[2, ...].set((-Nxz - Nyz) / 6.0)

        # For c = (i, j, 0)
        s = s.at[12, ...].set(Pi[1, ...] / 4.0)
        s = s.at[24, ...].set(Pi[1, ...] / 4.0)
        s = s.at[21, ...].set(-Pi[1, ...] / 4.0)
        s = s.at[15, ...].set(-Pi[1, ...] / 4.0)

        # For c = (i, 0, k)
        s = s.at[10, ...].set(Pi[2, ...] / 4.0)
        s = s.at[20, ...].set(Pi[2, ...] / 4.0)
        s = s.at[19, ...].set(-Pi[2, ...] / 4.0)
        s = s.at[11, ...].set(-Pi[2, ...] / 4.0)

        # For c = (0, j, k)
        s = s.at[8, ...].set(Pi[4, ...] / 4.0)
        s = s.at[4, ...].set(Pi[4, ...] / 4.0)
        s = s.at[7, ...].set(-Pi[4, ...] / 4.0)
        s = s.at[5, ...].set(-Pi[4, ...] / 4.0)

        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d2q9_jax(self, fneq):
        """
        Decompose fneq into shear components for D2Q9 lattice.

        Parameters
        ----------
        fneq : jax.numpy.array
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.array
            Shear components of fneq.
        """
        Pi = self.momentum_flux(fneq)
        N = Pi[0, ...] - Pi[2, ...]
        s = jnp.zeros_like(fneq)
        s = s.at[3, ...].set(N)
        s = s.at[6, ...].set(N)
        s = s.at[2, ...].set(-N)
        s = s.at[1, ...].set(-N)
        s = s.at[8, ...].set(Pi[1, ...])
        s = s.at[4, ...].set(-Pi[1, ...])
        s = s.at[5, ...].set(-Pi[1, ...])
        s = s.at[7, ...].set(Pi[1, ...])

        return s

    def _construct_warp(self):
        # Raise error if velocity set is not supported
        if not (isinstance(self.velocity_set, D3Q27) or isinstance(self.velocity_set, D2Q9)):
            raise NotImplementedError("Velocity set not supported for warp backend: {}".format(type(self.velocity_set)))

        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _epsilon = wp.constant(self.compute_dtype(self.epsilon))

        @wp.func
        def decompose_shear_d2q9(fneq: Any):
            pi = self.momentum_flux.warp_functional(fneq)
            N = pi[0] - pi[2]
            s = _f_vec()
            s[3] = N
            s[6] = N
            s[2] = -N
            s[1] = -N
            s[8] = pi[1]
            s[4] = -pi[1]
            s[5] = -pi[1]
            s[7] = pi[1]
            return s

        # Construct functional for decomposing shear
        @wp.func
        def decompose_shear_d3q27(
            fneq: Any,
        ):
            # Get momentum flux
            pi = self.momentum_flux.warp_functional(fneq)
            nxz = pi[0] - pi[5]
            nyz = pi[3] - pi[5]

            # set shear components
            s = _f_vec()

            # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
            two = self.compute_dtype(2.0)
            four = self.compute_dtype(4.0)
            six = self.compute_dtype(6.0)

            s[9] = (two * nxz - nyz) / six
            s[18] = (two * nxz - nyz) / six
            s[3] = (-nxz + two * nyz) / six
            s[6] = (-nxz + two * nyz) / six
            s[1] = (-nxz - nyz) / six
            s[2] = (-nxz - nyz) / six

            # For c = (i, j, 0)
            s[12] = pi[1] / four
            s[24] = pi[1] / four
            s[21] = -pi[1] / four
            s[15] = -pi[1] / four

            # For c = (i, 0, k)
            s[10] = pi[2] / four
            s[20] = pi[2] / four
            s[19] = -pi[2] / four
            s[11] = -pi[2] / four

            # For c = (0, j, k)
            s[8] = pi[4] / four
            s[4] = pi[4] / four
            s[7] = -pi[4] / four
            s[5] = -pi[4] / four

            return s

        # Construct functional for computing entropic scalar product
        @wp.func
        def entropic_scalar_product(
            x: Any,
            y: Any,
            feq: Any,
        ):
            e = wp.cw_div(wp.cw_mul(x, y), feq)
            e_sum = self.compute_dtype(0.0)
            for i in range(self.velocity_set.q):
                e_sum += e[i]
            return e_sum

        # Construct the functional
        @wp.func
        def functional(
            f: Any,
            feq: Any,
            rho: Any,
            u: Any,
            omega: Any,
        ):
            # Compute shear and delta_s
            fneq = f - feq
            if wp.static(self.velocity_set.d == 3):
                shear = decompose_shear_d3q27(fneq)
                delta_s = shear * rho
            else:
                shear = decompose_shear_d2q9(fneq)
                delta_s = shear * rho / self.compute_dtype(4.0)

            # Compute required constants based on the input omega (omega is the inverse relaxation time)
            _beta = self.compute_dtype(0.5) * self.compute_dtype(omega)
            _inv_beta = self.compute_dtype(1.0) / _beta

            # Perform collision
            delta_h = fneq - delta_s
            two = self.compute_dtype(2.0)
            gamma = _inv_beta - (two - _inv_beta) * entropic_scalar_product(delta_s, delta_h, feq) / (
                _epsilon + entropic_scalar_product(delta_h, delta_h, feq)
            )
            fout = f - _beta * (two * delta_s + gamma * delta_h)

            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            omega: Any,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            _d = self.velocity_set.d
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]
            _u = _u_vec()
            for l in range(_d):
                _u[l] = u[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]

            # Compute the collision
            _fout = functional(_f, _feq, _rho, _u, omega)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u, omega):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                feq,
                fout,
                rho,
                u,
                omega,
            ],
            dim=f.shape[1:],
        )
        return fout
