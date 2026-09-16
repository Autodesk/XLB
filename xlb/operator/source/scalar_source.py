"""
Volumetric source term for the advection-diffusion populations.
"""

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.source.source import Source
from xlb.operator import Operator


class ScalarSource(Source):
    """Volumetric source term for a scalar transport equation.

    The source is projected onto the populations with the lattice weights and added after
    collision::

        g_out[l] = g[l] + w[l] * source

    Since the weights sum to one and the projection carries no momentum, the scalar
    ``phi = sum_l g[l]`` gains exactly *source* per time step. Chapman-Enskog analysis
    shows the recovered equation is

        d(phi)/dt + div(phi * u) = div(diffusivity * grad(phi)) + source - dt / 2 * d(source)/dt

    so the discretization is exact for a source that is constant in time (however it varies
    in space) and first order in ``dt`` for a time-dependent one. Because ``dt`` is the
    lattice time step, that defect is negligible for any resolved source transient.

    For the heat equation ``rho * cp * dT/dt = k * laplacian(T) + q'''`` the lattice source
    is ``source = q''' * dt / (rho * cp)`` in lattice units.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, g: jnp.ndarray, source: jnp.ndarray):
        w = self.velocity_set.w.reshape((-1,) + (1,) * (len(g.shape) - 1))
        gout = g + w * source
        return gout

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _w = self.velocity_set.w
        _g_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        # Construct the functional
        @wp.func
        def functional(g: Any, source: Any):
            gout = _g_vec()
            for l in range(self.velocity_set.q):
                gout[l] = g[l] + _w[l] * source
            return gout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            g: wp.array4d(dtype=Any),
            source: wp.array4d(dtype=Any),
            gout: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Load needed values
            _g = _g_vec()
            for l in range(self.velocity_set.q):
                _g[l] = self.compute_dtype(g[l, index[0], index[1], index[2]])
            _source = self.compute_dtype(source[0, index[0], index[1], index[2]])

            # Apply the source term
            _gout = functional(_g, _source)

            # Write the result
            for l in range(self.velocity_set.q):
                gout[l, index[0], index[1], index[2]] = self.store_dtype(_gout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, g, source, gout):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                g,
                source,
                gout,
            ],
            dim=g.shape[1:],
        )
        return gout

    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, g, source, gout):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
