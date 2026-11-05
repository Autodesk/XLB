"""
BGK collision operator with Smagorinsky large-eddy-simulation sub-grid model.
"""

import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any
import numpy as np

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from functools import partial


class SmagorinskyLESBGK(Collision):
    """BGK collision with Smagorinsky LES turbulence modelling.

    Adjusts the effective relaxation time based on the local strain rate
    estimated from the non-equilibrium stress tensor, using the
    Smagorinsky model constant *C_s*.

    Parameters
    ----------
    velocity_set : VelocitySet, optional
    precision_policy : PrecisionPolicy, optional
    compute_backend : ComputeBackend, optional
    smagorinsky_coef : float
        Smagorinsky model constant (default 0.17).
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
        smagorinsky_coef: float = 0.17,
    ):
        self.smagorinsky_coef = smagorinsky_coef
        super().__init__(velocity_set, precision_policy, compute_backend)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f: jnp.ndarray, feq: jnp.ndarray, omega):
        fneq = f - feq

        pi_neq = jnp.tensordot(self.velocity_set.cc, fneq, axes=(0, 0))

        if self.velocity_set.d == 3:
            diag = pi_neq[(0, 3, 5), ...]
            offdiag = pi_neq[(1, 2, 4), ...]
        else:
            diag = pi_neq[(0, 2), ...]
            offdiag = pi_neq[(1,), ...]

        strain = jnp.sum(diag * diag, axis=0) + self.compute_dtype(2.0) * jnp.sum(offdiag * offdiag, axis=0)

        tau0 = self.compute_dtype(1.0) / self.compute_dtype(omega)
        cs = self.compute_dtype(self.smagorinsky_coef)
        tau = self.compute_dtype(0.5) * (tau0 + jnp.sqrt(tau0 * tau0 + self.compute_dtype(36.0) * (cs * cs) * jnp.sqrt(strain)))

        omega_eff = self.compute_dtype(1.0) / tau
        fout = f - omega_eff[None, ...] * fneq
        return fout

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _d = self.velocity_set.d
        _cc = self.velocity_set.cc
        _smagorinsky_coef = wp.constant(self.compute_dtype(self.smagorinsky_coef))
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _pi_dim = self.velocity_set.d * (self.velocity_set.d + 1) // 2
        _pi_vec = wp.vec(_pi_dim, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Construct the functional
        @wp.func
        def functional(
            f: Any,
            feq: Any,
            omega: Any,
        ):
            # Compute the non-equilibrium distribution
            fneq = f - feq

            # Compute strain
            pi_neq = _pi_vec()
            for a in range(_pi_dim):
                pi_neq[a] = self.compute_dtype(0.0)
                for l in range(self.velocity_set.q):
                    pi_neq[a] += _cc[l, a] * fneq[l]

            strain = self.compute_dtype(0.0)
            if wp.static(_d == 3):
                strain += pi_neq[0] * pi_neq[0] + pi_neq[3] * pi_neq[3] + pi_neq[5] * pi_neq[5]
                strain += self.compute_dtype(2.0) * (pi_neq[1] * pi_neq[1] + pi_neq[2] * pi_neq[2] + pi_neq[4] * pi_neq[4])
            else:
                strain += pi_neq[0] * pi_neq[0] + pi_neq[2] * pi_neq[2]
                strain += self.compute_dtype(2.0) * (pi_neq[1] * pi_neq[1])

            # Compute the Smagorinsky model
            _tau = self.compute_dtype(1.0) / self.compute_dtype(omega)
            tau = _tau + (
                self.compute_dtype(0.5) * (wp.sqrt(_tau * _tau + self.compute_dtype(36.0) * (_smagorinsky_coef**2.0) * wp.sqrt(strain)) - _tau)
            )

            # Compute the collision
            fout = f - (self.compute_dtype(1.0) / tau) * fneq
            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            omega: wp.float32,
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

            # Compute the collision
            _fout = functional(_f, _feq, omega)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, omega):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                feq,
                fout,
                omega,
            ],
            dim=f.shape[1:],
        )
        return fout

    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None
