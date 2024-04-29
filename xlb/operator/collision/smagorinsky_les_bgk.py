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
    """
    BGK collision operator for LBM with Smagorinsky LES model.
    """

    def __init__(
        self,
        omega: float,
        smagorinsky_coef: float = 0.17,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        self.smagorinsky_coef = smagorinsky_coef
        super().__init__(omega, velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _w = self.velocity_set.wp_w
        _c = self.velocity_set.wp_c
        _omega = wp.constant(self.compute_dtype(self.omega))
        _tau = wp.constant(self.compute_dtype(1.0 / self.omega))
        _smagorinsky_coef = wp.constant(self.compute_dtype(self.smagorinsky_coef))
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        # Construct the functional
        @wp.func
        def functional(
            f: Any,
            feq: Any,
            rho: Any,
            u: Any,
        ):
            # Compute the non-equilibrium distribution
            fneq = f - feq

            # Sailfish implementation
            #{
            #  float tmp, strain;

            #  strain = 0.0f;

            #  // Off-diagonal components, count twice for symmetry reasons.
            #  %for a in range(0, dim):
            #    %for b in range(a + 1, dim):
            #       tmp = ${cex(sym.ex_flux(grid, 'd0', a, b, config), pointers=True)} -
            #           ${cex(sym.ex_eq_flux(grid, a, b))};
            #       strain += 2.0f * tmp * tmp;
            #    %endfor
            #  %endfor

            #  // Diagonal components.
            #  %for a in range(0, dim):
            #    tmp = ${cex(sym.ex_flux(grid, 'd0', a, a, config), pointers=True)} -
            #        ${cex(sym.ex_eq_flux(grid, a, a))};
            #    strain += tmp * tmp;
            #  %endfor

            #  tau0 += 0.5f * (sqrtf(tau0 * tau0 + 36.0f * ${cex(smagorinsky_const**2)} * sqrtf(strain)) - tau0);
            #}

            # Compute strain
            strain = wp.float32(0.0)
            for l in range(self.velocity_set.q):
                # diagonal terms
                if (_c[0, l] + _c[1, l] + _c[2, l]) == 1: 
                    strain += fneq[l] * fneq[l]

                # Off-diagonal terms
                if (_c[0, l] + _c[1, l] + _c[2, l]) >= 2:
                    strain += 2.0 * fneq[l] * fneq[l]

            # Compute the Smagorinsky model
            tau = _tau + (
                0.5 * (wp.sqrt(_tau * _tau + 36.0 * (_smagorinsky_coef ** 2.0) * wp.sqrt(strain)) - _tau)
            )

            # Compute the collision
            fout = f - (1.0 / tau) * fneq
            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
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

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, rho, u, fout):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                feq,
                rho,
                u,
                fout,
            ],
            dim=f.shape[1:],
        )
        return fout
