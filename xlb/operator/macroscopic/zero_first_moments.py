# Base class for all equilibriums

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class ZeroAndFirstMoments(Operator):
    """
    A class to compute first and zeroth moments of distribution functions.

    TODO: Currently this is only used for the standard rho and u moments.
    In the future, this should be extended to include higher order moments
    and other physic types (e.g. temperature, electromagnetism, etc...)
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), inline=True)
    def jax_implementation(self, f):
        """
        Apply the macroscopic operator to the lattice distribution function
        TODO: Check if the following implementation is more efficient (
        as the compiler may be able to remove operations resulting in zero)
        c_x = tuple(self.velocity_set.c[0])
        c_y = tuple(self.velocity_set.c[1])

        u_x = 0.0
        u_y = 0.0

        rho = jnp.sum(f, axis=0, keepdims=True)

        for i in range(self.velocity_set.q):
            u_x += c_x[i] * f[i, ...]
            u_y += c_y[i] * f[i, ...]
        return rho, jnp.stack((u_x, u_y))
        """
        rho = jnp.sum(f, axis=0, keepdims=True)
        u = jnp.tensordot(self.velocity_set.c, f, axes=(-1, 0)) / rho

        return rho, u

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Construct the functional
        @wp.func
        def functional(f: _f_vec):
            # Compute rho and u
            rho = self.compute_dtype(0.0)
            u = _u_vec()
            for l in range(self.velocity_set.q):
                rho += f[l]
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        u[d] += f[l]
                    elif _c[d, l] == -1:
                        u[d] -= f[l]
            u /= rho

            return rho, u

        # Construct the kernel
        @wp.kernel
        def kernel3d(
            f: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            (_rho, _u) = functional(_f)

            # Set the output
            rho[0, index[0], index[1], index[2]] = _rho
            for d in range(self.velocity_set.d):
                u[d, index[0], index[1], index[2]] = _u[d]

        @wp.kernel
        def kernel2d(
            f: wp.array3d(dtype=Any),
            rho: wp.array3d(dtype=Any),
            u: wp.array3d(dtype=Any),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # Get the equilibrium
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1]]
            (_rho, _u) = functional(_f)

            # Set the output
            rho[0, index[0], index[1]] = _rho
            for d in range(self.velocity_set.d):
                u[d, index[0], index[1]] = _u[d]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, rho, u):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                rho,
                u,
            ],
            dim=rho.shape[1:],
        )
        return rho, u
