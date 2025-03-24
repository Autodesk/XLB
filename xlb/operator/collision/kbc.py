"""
KBC collision operator for LBM with fused scalar products optimization.
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

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method,
    optimized with fused scalar products to reduce redundant division operations.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        """Initialize the KBC collision operator."""
        self.momentum_flux = MomentumFlux()
        self.epsilon = 1e-32  # Small constant to prevent division by zero

        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

    ### JAX Backend Implementation ###

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
        JAX implementation of the KBC collision step with fused scalar products.

        Parameters
        ----------
        f : jax.numpy.ndarray
            Distribution function.
        feq : jax.numpy.ndarray
            Equilibrium distribution function.
        rho : jax.numpy.ndarray
            Density.
        u : jax.numpy.ndarray
            Velocity.
        omega : float
            Relaxation parameter (inverse relaxation time).

        Returns
        -------
        jax.numpy.ndarray
            Post-collision distribution function.
        """
        fneq = f - feq
        if isinstance(self.velocity_set, D2Q9):
            shear = self.decompose_shear_d2q9_jax(fneq)
            delta_s = shear * rho / 4.0
        elif isinstance(self.velocity_set, D3Q27):
            shear = self.decompose_shear_d3q27_jax(fneq)
            delta_s = shear * rho
        else:
            raise NotImplementedError(f"Velocity set not supported: {type(self.velocity_set)}")

        # Compute constants
        beta = self.compute_dtype(0.5) * self.compute_dtype(omega)
        inv_beta = 1.0 / beta

        # Compute fused scalar products using symmetry pairs
        delta_h = fneq - delta_s
        sp1, sp2 = self.compute_scalar_products_jax(delta_s, delta_h, feq)
        gamma = inv_beta - (2.0 - inv_beta) * sp1 / (self.epsilon + sp2)
        fout = f - beta * (2.0 * delta_s + gamma[None, ...] * delta_h)

        return fout

    @partial(jit, static_argnums=(0,), inline=True)
    def compute_scalar_products_jax(self, delta_s, delta_h, feq):
        """
        Compute fused entropic scalar products for JAX backend using symmetry pairs.

        The full sum over q is computed by summing the contributions of the center velocity (handled separately)
        and contributions from each symmetry pair.

        Parameters
        ----------
        delta_s : jax.numpy.ndarray
            Shear component of the non-equilibrium distribution.
        delta_h : jax.numpy.ndarray
            Higher-order component of the non-equilibrium distribution.
        feq : jax.numpy.ndarray
            Equilibrium distribution function.

        Returns
        -------
        tuple
            (sp1, sp2) where:
            - sp1 = sum(delta_s * delta_h / feq)
            - sp2 = sum(delta_h * delta_h / feq)
        """
        temp = delta_h / feq
        # Use the symmetry pairs stored in the velocity set
        pairs = self.velocity_set.symmetry_pairs  # shape (n, 2)
        # Sum contributions from each pair (processing two directions at once)
        sp1_pairs = jnp.sum(delta_s[pairs[:, 0]] * temp[pairs[:, 0]] +
                            delta_s[pairs[:, 1]] * temp[pairs[:, 1]], axis=0)
        sp2_pairs = jnp.sum(delta_h[pairs[:, 0]] * temp[pairs[:, 0]] +
                            delta_h[pairs[:, 1]] * temp[pairs[:, 1]], axis=0)
        # Process the center velocity separately
        center = jnp.argmin(jnp.sum(jnp.abs(self.velocity_set.c), axis=1))
        sp1_center = delta_s[0] * temp[0]
        sp2_center = delta_h[0] * temp[0]
        sp1 = sp1_pairs + sp1_center
        sp2 = sp2_pairs + sp2_center
        return sp1, sp2

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d3q27_jax(self, fneq):
        """
        Decompose the non-equilibrium distribution into shear components for D3Q27.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components.
        """
        Pi = self.momentum_flux(fneq)
        Nxz = Pi[0, ...] - Pi[5, ...]
        Nyz = Pi[3, ...] - Pi[5, ...]
        s = jnp.zeros_like(fneq)
        s = s.at[9, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[18, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[3, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[6, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[1, ...].set((-Nxz - Nyz) / 6.0)
        s = s.at[2, ...].set((-Nxz - Nyz) / 6.0)
        s = s.at[12, ...].set(Pi[1, ...] / 4.0)
        s = s.at[24, ...].set(Pi[1, ...] / 4.0)
        s = s.at[21, ...].set(-Pi[1, ...] / 4.0)
        s = s.at[15, ...].set(-Pi[1, ...] / 4.0)
        s = s.at[10, ...].set(Pi[2, ...] / 4.0)
        s = s.at[20, ...].set(Pi[2, ...] / 4.0)
        s = s.at[19, ...].set(-Pi[2, ...] / 4.0)
        s = s.at[11, ...].set(-Pi[2, ...] / 4.0)
        s = s.at[8, ...].set(Pi[4, ...] / 4.0)
        s = s.at[4, ...].set(Pi[4, ...] / 4.0)
        s = s.at[7, ...].set(-Pi[4, ...] / 4.0)
        s = s.at[5, ...].set(-Pi[4, ...] / 4.0)
        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d2q9_jax(self, fneq):
        """
        Decompose the non-equilibrium distribution into shear components for D2Q9.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components.
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

    ### Warp Backend Implementation ###

    def _construct_warp(self):
        """Construct Warp functionals and kernel for the KBC collision step."""
        if not (isinstance(self.velocity_set, D3Q27) or isinstance(self.velocity_set, D2Q9)):
            raise NotImplementedError(f"Velocity set not supported for Warp backend: {type(self.velocity_set)}")

        # Define Warp types and constants
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _epsilon = wp.constant(self.compute_dtype(self.epsilon))
        _zero = wp.constant(self.compute_dtype(0.0))
        _two = wp.constant(self.compute_dtype(2.0))
        _inv_four = wp.constant(self.compute_dtype(1.0)/self.compute_dtype(4.0))
        _inv_six = wp.constant(self.compute_dtype(1.0)/self.compute_dtype(6.0))
        # Capture symmetry pairs and center index as local constants.
        symmetry_pairs = self.velocity_set.symmetry_pairs  
        n_pairs = self.velocity_set.n_symmetry_pairs


        @wp.func
        def decompose_shear_d2q9(fneq: Any):
            """Decompose shear components for D2Q9 in Warp."""
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

        @wp.func
        def decompose_shear_d3q27(fneq: Any):
            """Decompose shear components for D3Q27 in Warp."""
            pi = self.momentum_flux.warp_functional(fneq)
            nxz = pi[0] - pi[5]
            nyz = pi[3] - pi[5]
            shear1_inv6 = (_two * nxz - nyz) * _inv_six
            shear2_inv6 = (-nxz + _two * nyz) * _inv_six
            shear3_inv6 = (-nxz - nyz) * _inv_six
            pi_invFour1 = pi[1] * _inv_four
            pi_invFour2 = pi[2] * _inv_four
            pi_invFour4 = pi[2] * _inv_four
            s = _f_vec()
            s[9]  = shear1_inv6
            s[18] = shear1_inv6
            s[3]  = shear2_inv6
            s[6]  = shear2_inv6
            s[1]  = shear3_inv6
            s[2]  = shear3_inv6
            s[12] = pi_invFour1
            s[24] = pi_invFour1
            s[21] = -pi_invFour1
            s[15] = -pi_invFour1
            s[10]  = pi_invFour2
            s[20] = pi_invFour2
            s[19] = -pi_invFour2
            s[11] = -pi_invFour2
            s[8]  = pi_invFour4
            s[4]  = pi_invFour4
            s[7]  = -pi_invFour4
            s[5]  = -pi_invFour4
            return s

        @wp.func
        def compute_scalar_products(delta_s: Any, delta_h: Any, feq: Any):
            """
            Compute fused entropic scalar products for Warp backend using symmetry pairs.

            Instead of iterating over all q, we first process the center (zero) velocity and then loop
            over the symmetry pairs, thereby reducing the number of iterations by nearly half.

            Returns
            -------
            tuple
                (sp1, sp2)
            """
            s1 = _zero  # Sum for sp1
            c1 = _zero  # Compensation for sp1
            s2 = _zero  # Sum for sp2
            c2 = _zero  # Compensation for sp2

            # Process center velocity separately
            #center = self.velocity_set.center_index
            temp = delta_h[0] / feq[0]
            x1 = delta_s[0] * temp
            t1 = s1 + x1
            c1 += wp.select(wp.abs(s1) >= wp.abs(x1), (x1 - t1) + s1, (s1 - t1) + x1)
            s1 = t1

            x2 = delta_h[0] * temp
            t2 = s2 + x2
            c2 += wp.select(wp.abs(s2) >= wp.abs(x2), (x2 - t2) + s2, (s2 - t2) + x2)
            s2 = t2

            # Loop over symmetry pairs
            for p in range(n_pairs):
                i_idx = symmetry_pairs[p, 0]
                opp   = symmetry_pairs[p, 1]
                temp   = delta_h[i_idx] / feq[i_idx]
                temp_opp = delta_h[opp]   / feq[opp]
                x1 = delta_s[i_idx] * temp + delta_s[opp] * temp_opp
                t1 = s1 + x1
                c1 += wp.select(wp.abs(s1) >= wp.abs(x1), (x1 - t1) + s1, (s1 - t1) + x1)
                s1 = t1

                x2 = delta_h[i_idx] * temp + delta_h[opp] * temp_opp
                t2 = s2 + x2
                c2 += wp.select(wp.abs(s2) >= wp.abs(x2), (x2 - t2) + s2, (s2 - t2) + x2)
                s2 = t2

            sp1 = s1 + c1
            sp2 = s2 + c2
            return sp1, sp2

        @wp.func
        def functional(
            f: Any,
            feq: Any,
            rho: Any,
            u: Any,
            omega: Any,
        ):
            """Warp functional for KBC collision with fused scalar products."""
            fneq = f - feq
            # Use the dimension to decide which shear decomposition to use
            if wp.static(self.velocity_set.d == 3):
                shear = decompose_shear_d3q27(fneq)
                delta_s = shear * rho
            else:
                shear = decompose_shear_d2q9(fneq)
                delta_s = shear * rho / 4.0

            _beta = self.compute_dtype(0.5) * self.compute_dtype(omega)
            _inv_beta = self.compute_dtype(1.0) / _beta
            delta_h = fneq - delta_s
            sp1, sp2 = compute_scalar_products(delta_s, delta_h, feq)
            gamma = _inv_beta - (_two - _inv_beta) * sp1 / (_epsilon + sp2)
            fout = f - _beta * (_two * delta_s + gamma * delta_h)
            return fout

        @wp.func
        def load_vector(data: Any, idx: wp.vec3i):
            vec = _f_vec()  # allocate local vector of length q
            for q in range(self.velocity_set.q):
                vec[q] = data[q, idx[0], idx[1], idx[2]]
            return vec
        
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            omega: Any,
        ):
            """Warp kernel to launch the KBC collision step using symmetry pairs."""
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            _f = _f_vec()
            _feq = _f_vec()
            '''
            # --- Copy the center velocity first ---
            #center = self.velocity_set.center_index
            _f[0] = f[0, index[0], index[1], index[2]]
            _feq[0] = feq[0, index[0], index[1], index[2]]
            # --- Copy all symmetry pairs (processing two directions per loop iteration) ---
            n_pairs = self.velocity_set.symmetry_pairs.shape[0]
            for p in range(n_pairs):
                i_idx = self.velocity_set.symmetry_pairs[p, 0]
                opp   = self.velocity_set.symmetry_pairs[p, 1]
                _f[i_idx] = f[i_idx, index[0], index[1], index[2]]
                _f[opp]   = f[opp,   index[0], index[1], index[2]]
                _feq[i_idx] = feq[i_idx, index[0], index[1], index[2]]
                _feq[opp]   = feq[opp,   index[0], index[1], index[2]]
            '''    
            _f = load_vector(f, index)
            _feq = load_vector(feq, index)
            _u = _u_vec()
            for l in range(self.velocity_set.d):
                _u[l] = u[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]
            _fout = functional(_f, _feq, _rho, _u, omega)
            # --- Write updated distribution back using symmetry pairs ---
            fout[0, index[0], index[1], index[2]] = self.store_dtype(_fout[0])
            for p in range(n_pairs):
                i_idx = self.velocity_set.symmetry_pairs[p, 0]
                opp   = self.velocity_set.symmetry_pairs[p, 1]
                fout[i_idx, index[0], index[1], index[2]] = self.store_dtype(_fout[i_idx])
                fout[opp,   index[0], index[1], index[2]] = self.store_dtype(_fout[opp])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u, omega):
        """
        Warp implementation of the KBC collision step.

        Parameters
        ----------
        f, feq, fout, rho, u, omega : Warp arrays and scalar
            Inputs and output for the collision step.

        Returns
        -------
        Warp array
            Post-collision distribution function.
        """
        wp.launch(
            self.warp_kernel,
            inputs=[f, feq, fout, rho, u, omega],
            dim=f.shape[1:],
        )
        return fout
