from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator import Operator


class QuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, rho, u):
        cu = 3.0 * jnp.tensordot(self.velocity_set.c, u, axes=(0, 0))
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=0, keepdims=True)
        w = self.velocity_set.w.reshape((-1,) + (1,) * self.velocity_set.d)
        feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        return feq

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.c
        _w = self.velocity_set.w
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Construct the equilibrium functional
        @wp.func
        def functional(
            rho: Any,
            u: Any,
        ):
            # Allocate the equilibrium
            feq = _f_vec()

            # Compute the equilibrium
            for l in range(self.velocity_set.q):
                # Compute cu
                cu = self.compute_dtype(0.0)
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        cu += u[d]
                    elif _c[d, l] == -1:
                        cu -= u[d]
                cu *= self.compute_dtype(3.0)

                # Compute usqr
                usqr = self.compute_dtype(1.5) * wp.dot(u, u)

                # Compute feq
                feq[l] = rho * _w[l] * (self.compute_dtype(1.0) + cu * (self.compute_dtype(1.0) + self.compute_dtype(0.5) * cu) - usqr)

            return feq

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            f: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _u = _u_vec()
            for d in range(self.velocity_set.d):
                _u[d] = u[d, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]
            feq = functional(_rho, _u)

            # Set the output
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1], index[2]] = self.store_dtype(feq[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, rho, u, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                f,
            ],
            dim=rho.shape[1:],
        )
        return f
