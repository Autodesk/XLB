# Base class for all equilibriums

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Tuple

from xlb.global_config import GlobalConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class Macroscopic(Operator):
    """
    Base class for all macroscopic operators

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
        _c = wp.constant(self._warp_stream_mat(self.velocity_set.c))
        _q = wp.constant(self.velocity_set.q)
        _d = wp.constant(self.velocity_set.d)

        # Construct the functional
        @wp.func
        def functional(f: self._warp_lattice_vec):
            # Compute rho and u
            rho = self.compute_dtype(0.0)
            u = self._warp_u_vec()
            for l in range(_q):
                rho += f[l]
                for d in range(_d):
                    if _c[l, d] == 1:
                        u[d] += f[l]
                    elif _c[l, d] == -1:
                        u[d] -= f[l]
            u /= rho

            return rho, u
            # return u, rho

        # Construct the kernel
        @wp.kernel
        def kernel(
            f: self._warp_array_type,
            rho: self._warp_array_type,
            u: self._warp_array_type,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Get the equilibrium
            _f = self._warp_lattice_vec()
            for l in range(_q):
                _f[l] = f[l, i, j, k]
            (_rho, _u) = functional(_f)

            # Set the output
            rho[0, i, j, k] = _rho
            for d in range(_d):
                u[d, i, j, k] = _u[d]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, rho, u):
        # Launch the warp kernel
        wp.launch(
            self._kernel,
            inputs=[
                f,
                rho,
                u,
            ],
            dim=rho.shape[1:],
        )
        return rho, u
