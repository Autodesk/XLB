from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic.zero_moment import ZeroMoment
from xlb.operator.macroscopic.first_moment import FirstMoment

class Macroscopic(Operator):
    """A class to compute both zero and first moments of distribution functions (rho, u)."""

    def __init__(self, *args, **kwargs):
        self.zero_moment = ZeroMoment(*args, **kwargs)
        self.first_moment = FirstMoment(*args, **kwargs)
        super().__init__(*args, **kwargs)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f):
        rho = self.zero_moment(f)
        u = self.first_moment(f, rho)
        return rho, u

    def _construct_warp(self):
        zero_moment_func = self.zero_moment.warp_functional
        first_moment_func = self.first_moment.warp_functional
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        @wp.func
        def functional(f: _f_vec):
            rho = zero_moment_func(f)
            u = first_moment_func(f, rho)
            return rho, u

        @wp.kernel
        def kernel3d(
            f: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            _rho, _u = functional(_f)

            rho[0, index[0], index[1], index[2]] = _rho
            for d in range(self.velocity_set.d):
                u[d, index[0], index[1], index[2]] = _u[d]

        @wp.kernel
        def kernel2d(
            f: wp.array3d(dtype=Any),
            rho: wp.array3d(dtype=Any),
            u: wp.array3d(dtype=Any),
        ):
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1]]
            _rho, _u = functional(_f)

            rho[0, index[0], index[1]] = _rho
            for d in range(self.velocity_set.d):
                u[d, index[0], index[1]] = _u[d]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, rho, u):
        wp.launch(
            self.warp_kernel,
            inputs=[f, rho, u],
            dim=rho.shape[1:],
        )
        return rho, u