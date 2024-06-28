"""
KBC collision operator for LBM.
"""

import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.velocity_set import VelocitySet, D2Q9, D3Q27
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from functools import partial


class KBC(Collision):
    """
    KBC collision operator for LBM.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(
        self,
        omega: float,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        self.epsilon = 1e-32
        self.beta = omega * 0.5
        self.inv_beta = 1.0 / self.beta

        super().__init__(
            omega=omega,
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
            delta_s = shear * rho / 4.0  # TODO: Check this
        elif isinstance(self.velocity_set, D3Q27):
            shear = self.decompose_shear_d3q27_jax(fneq)
            delta_s = shear * rho
        else:
            raise NotImplementedError(
                "Velocity set not supported: {}".format(type(self.velocity_set))
            )

        # Perform collision
        delta_h = fneq - delta_s
        gamma = self.inv_beta - (2.0 - self.inv_beta) * self.entropic_scalar_product(
            delta_s, delta_h, feq
        ) / (self.epsilon + self.entropic_scalar_product(delta_h, delta_h, feq))

        fout = f - self.beta * (2.0 * delta_s + gamma[None, ...] * delta_h)

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

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def momentum_flux_jax(
        self,
        fneq: jnp.ndarray,
    ):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann
        Method (LBM).

        # TODO: probably move this to equilibrium calculation

        Parameters
        ----------
        fneq: jax.numpy.ndarray
            The non-equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The computed momentum flux.
        """

        return jnp.tensordot(self.velocity_set.cc, fneq, axes=(0, 0))

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
        Pi = self.momentum_flux_jax(fneq)
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
        Pi = self.momentum_flux_jax(fneq)
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
        if not isinstance(self.velocity_set, D3Q27):
            raise NotImplementedError(
                "Velocity set not supported for warp backend: {}".format(
                    type(self.velocity_set)
                )
            )

        # Set local constants TODO: This is a hack and should be fixed with warp update
        _w = self.velocity_set.wp_w
        _cc = self.velocity_set.wp_cc
        _omega = wp.constant(self.compute_dtype(self.omega))
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _pi_dim = self.velocity_set.d * (self.velocity_set.d + 1) // 2
        _pi_vec = wp.vec(
            _pi_dim,
            dtype=self.compute_dtype,
        )
        _epsilon = wp.constant(self.compute_dtype(self.epsilon))
        _beta = wp.constant(self.compute_dtype(self.beta))
        _inv_beta = wp.constant(self.compute_dtype(1.0 / self.beta))

        # Construct functional for computing momentum flux
        @wp.func
        def momentum_flux_warp(
            fneq: Any,
        ):
            # Get momentum flux
            pi = _pi_vec()
            for d in range(_pi_dim):
                pi[d] = 0.0
                for q in range(self.velocity_set.q):
                    pi[d] += _cc[q, d] * fneq[q]
            return pi

        @wp.func
        def decompose_shear_d2q9(fneq: Any):
            pi = momentum_flux_warp(fneq)
            N = pi[0] - pi[1]
            s = wp.vec9(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            s[3] = N
            s[6] = N
            s[2] = -N
            s[1] = -N
            s[8] = pi[2]
            s[4] = -pi[2]
            s[5] = -pi[2]
            s[7] = pi[2]
            return s

        # Construct functional for decomposing shear
        @wp.func
        def decompose_shear_d3q27(
            fneq: Any,
        ):
            # Get momentum flux
            pi = momentum_flux_warp(fneq)
            nxz = pi[0] - pi[5]
            nyz = pi[3] - pi[5]

            # set shear components
            s = _f_vec()

            # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
            s[9] = (2.0 * nxz - nyz) / 6.0
            s[18] = (2.0 * nxz - nyz) / 6.0
            s[3] = (-nxz + 2.0 * nyz) / 6.0
            s[6] = (-nxz + 2.0 * nyz) / 6.0
            s[1] = (-nxz - nyz) / 6.0
            s[2] = (-nxz - nyz) / 6.0

            # For c = (i, j, 0)
            s[12] = pi[1] / 4.0
            s[24] = pi[1] / 4.0
            s[21] = -pi[1] / 4.0
            s[15] = -pi[1] / 4.0

            # For c = (i, 0, k)
            s[10] = pi[2] / 4.0
            s[20] = pi[2] / 4.0
            s[19] = -pi[2] / 4.0
            s[11] = -pi[2] / 4.0

            # For c = (0, j, k)
            s[8] = pi[4] / 4.0
            s[4] = pi[4] / 4.0
            s[7] = -pi[4] / 4.0
            s[5] = -pi[4] / 4.0

            return s

        # Construct functional for computing entropic scalar product
        @wp.func
        def entropic_scalar_product(
            x: Any,
            y: Any,
            feq: Any,
        ):
            e = wp.cw_div(wp.cw_mul(x, y), feq)
            e_sum = wp.float32(0.0)
            for i in range(self.velocity_set.q):
                e_sum += e[i]
            return e_sum

        # Construct the functional
        @wp.func
        def functional2d(
            f: Any,
            feq: Any,
            rho: Any,
            u: Any,
        ):
            # Compute shear and delta_s
            fneq = f - feq
            shear = decompose_shear_d2q9(fneq)
            delta_s = shear * rho  # TODO: Check this

            # Perform collision
            delta_h = fneq - delta_s
            gamma = _inv_beta - (2.0 - _inv_beta) * entropic_scalar_product(
                delta_s, delta_h, feq
            ) / (_epsilon + entropic_scalar_product(delta_h, delta_h, feq))
            fout = f - _beta * (2.0 * delta_s + gamma * delta_h)

            return fout

        # Construct the functional
        @wp.func
        def functional3d(
            f: Any,
            feq: Any,
            rho: Any,
            u: Any,
        ):
            # Compute shear and delta_s
            fneq = f - feq
            shear = decompose_shear_d3q27(fneq)
            delta_s = shear * rho  # TODO: Check this

            # Perform collision
            delta_h = fneq - delta_s
            gamma = _inv_beta - (2.0 - _inv_beta) * entropic_scalar_product(
                delta_s, delta_h, feq
            ) / (_epsilon + entropic_scalar_product(delta_h, delta_h, feq))
            fout = f - _beta * (2.0 * delta_s + gamma * delta_h)

            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel2d(
            f: wp.array3d(dtype=Any),
            feq: wp.array3d(dtype=Any),
            rho: wp.array3d(dtype=Any),
            u: wp.array3d(dtype=Any),
            fout: wp.array3d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j)  # TODO: Warp needs to fix this

            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1]]
                _feq[l] = feq[l, index[0], index[1]]
            _u = self._warp_u_vec()
            for l in range(_d):
                _u[l] = u[l, index[0], index[1]]
            _rho = rho[0, index[0], index[1]]

            # Compute the collision
            _fout = functional(_f, _feq, _rho, _u)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1]] = _fout[l]

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]
            _u = self._warp_u_vec()
            for l in range(_d):
                _u[l] = u[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]

            # Compute the collision
            _fout = functional(_f, _feq, _rho, _u)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = _fout[l]

        functional = functional3d if self.velocity_set.d == 3 else functional2d
        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                feq,
                fout,
                rho,
                u,
            ],
            dim=f.shape[1:],
        )
        return fout
