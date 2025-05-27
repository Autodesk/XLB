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
        def neumaier_sum(f: _f_vec):
            total = self.compute_dtype(0.0)
            compensation = self.compute_dtype(0.0)
            for l in range(self.velocity_set.q):
                x = f[l]
                t = total + x
                # Using wp.abs to compute absolute value
                if wp.abs(total) >= wp.abs(x):
                    compensation = compensation + ((total - t) + x)
                else:
                    compensation = compensation + ((x - t) + total)
                total = t
            return total + compensation

        @wp.func
        def functional(f: _f_vec):
            return neumaier_sum(f)

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
        wp.launch(self.warp_kernel, inputs=[f, rho], dim=rho.shape[1:])
        return rho
