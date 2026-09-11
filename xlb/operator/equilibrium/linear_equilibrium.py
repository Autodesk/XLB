"""
First-order (linear) equilibrium for a passive scalar.

Used by the advection-diffusion / thermal populations, where only the
zeroth and first moments need to be recovered.
"""

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp

from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import Equilibrium
from xlb.operator import Operator


class LinearEquilibrium(Equilibrium):
    """
    Linear equilibrium of the advection-diffusion equation, truncated at first order in the
    advecting velocity::

        g_eq[l] = w[l] * phi * (1 + (c[l] . u) / cs^2)

    The quadratic terms of :class:`QuadraticEquilibrium` are dropped because the
    advection-diffusion equation only requires the zeroth and first moments of the scalar
    populations. Keeping the expansion linear also makes the anti-bounce-back Dirichlet
    condition exact, since ``g_eq[l] + g_eq[opp[l]] = 2 * w[l] * phi`` for any velocity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, phi, u):
        cu = self.velocity_set.inv_cs2 * jnp.tensordot(self.velocity_set.c, u, axes=(0, 0))
        w = self.velocity_set.w.reshape((-1,) + (1,) * (len(u.shape) - 1))
        geq = phi * w * (1.0 + cu)
        return geq

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.c
        _w = self.velocity_set.w
        _inv_cs2 = self.compute_dtype(self.velocity_set.inv_cs2)
        _g_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Construct the equilibrium functional
        @wp.func
        def functional(
            phi: Any,
            u: Any,
        ):
            # Allocate the equilibrium
            geq = _g_vec()

            # Compute the equilibrium
            for l in range(self.velocity_set.q):
                # Compute cu
                cu = self.compute_dtype(0.0)
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        cu += u[d]
                    elif _c[d, l] == -1:
                        cu -= u[d]
                cu *= _inv_cs2

                # Compute geq
                geq[l] = phi * _w[l] * (self.compute_dtype(1.0) + cu)

            return geq

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            phi: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            g: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _u = _u_vec()
            for d in range(self.velocity_set.d):
                _u[d] = self.compute_dtype(u[d, index[0], index[1], index[2]])
            _phi = self.compute_dtype(phi[0, index[0], index[1], index[2]])
            geq = functional(_phi, _u)

            # Set the output
            for l in range(self.velocity_set.q):
                g[l, index[0], index[1], index[2]] = self.store_dtype(geq[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, phi, u, g):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                phi,
                u,
                g,
            ],
            dim=phi.shape[1:],
        )
        return g

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, phi, u, g):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
