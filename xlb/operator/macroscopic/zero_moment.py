from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class ZeroMoment(Operator):
    """A class to compute the zeroth moment (density) of distribution functions."""

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f):
        return jnp.sum(f, axis=0, keepdims=True)

    def _construct_warp(self):
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        @wp.func
        def functional(f: _f_vec):
            # Simple sum for Warp autodiff compatibility
            # Neumaier sum with conditionals breaks gradient flow
            total = self.compute_dtype(0.0)
            for l in range(self.velocity_set.q):
                total = total + f[l]
            return total

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            _rho = functional(_f)

            rho[0, index[0], index[1], index[2]] = _rho

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, rho):
        wp.launch(self.warp_kernel, inputs=[f], outputs=[rho], dim=rho.shape[1:])
        return rho

    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f, rho):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
