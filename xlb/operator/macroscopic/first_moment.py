from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class FirstMoment(Operator):
    """A class to compute the first moment (velocity) of distribution functions."""

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f, rho):
        u = jnp.tensordot(self.velocity_set.c, f, axes=(-1, 0)) / rho
        return u

    def _construct_warp(self):
        _c = self.velocity_set.c
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        @wp.func
        def functional(
            f: _f_vec,
            rho: Any,
        ):
            u = _u_vec()
            for l in range(self.velocity_set.q):
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        u[d] += f[l]
                    elif _c[d, l] == -1:
                        u[d] -= f[l]
            u /= rho
            return u

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]
            _u = functional(_f, _rho)

            for d in range(self.velocity_set.d):
                u[d, index[0], index[1], index[2]] = self.store_dtype(_u[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, rho, u):
        wp.launch(
            self.warp_kernel,
            inputs=[f, rho, u],
            dim=u.shape[1:],
        )
        return u
