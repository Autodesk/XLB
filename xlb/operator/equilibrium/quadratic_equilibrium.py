from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator import Operator
from xlb.default_config import DefaultConfig


class QuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), donate_argnums=(1, 2))
    def jax_implementation(self, rho, u):
        cu = 3.0 * jnp.tensordot(self.velocity_set.c, u, axes=(0, 0))
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=0, keepdims=True)
        w = self.velocity_set.w.reshape((-1,) + (1,) * (len(rho.shape) - 1))
        feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        return feq

    @Operator.register_backend(ComputeBackend.PALLAS)
    def pallas_implementation(self, rho, u):
        u0, u1, u2 = u[0], u[1], u[2]
        usqr = 1.5 * (u0**2 + u1**2 + u2**2)

        eq = [
            rho[0] * (1.0 / 18.0) * (1.0 - 3.0 * u0 + 4.5 * u0 * u0 - usqr),
            rho[0] * (1.0 / 18.0) * (1.0 - 3.0 * u1 + 4.5 * u1 * u1 - usqr),
            rho[0] * (1.0 / 18.0) * (1.0 - 3.0 * u2 + 4.5 * u2 * u2 - usqr),
        ]

        combined_velocities = [u0 + u1, u0 - u1, u0 + u2, u0 - u2, u1 + u2, u1 - u2]

        for vel in combined_velocities:
            eq.append(
                rho[0] * (1.0 / 36.0) * (1.0 - 3.0 * vel + 4.5 * vel * vel - usqr)
            )

        eq.append(rho[0] * (1.0 / 3.0) * (1.0 - usqr))

        for i in range(3):
            eq.append(eq[i] + rho[0] * (1.0 / 18.0) * 6.0 * u[i])

        for i, vel in enumerate(combined_velocities, 3):
            eq.append(eq[i] + rho[0] * (1.0 / 36.0) * 6.0 * vel)

        return jnp.array(eq)

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.wp_c
        _w = self.velocity_set.wp_w
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
                usqr = 1.5 * wp.dot(u, u)

                # Compute feq
                feq[l] = rho * _w[l] * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

            return feq

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
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
                f[l, index[0], index[1], index[2]] = feq[l]

        @wp.kernel
        def kernel2d(
            rho: wp.array3d(dtype=Any),
            u: wp.array3d(dtype=Any),
            f: wp.array3d(dtype=Any),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # Get the equilibrium
            _u = _u_vec()
            for d in range(self.velocity_set.d):
                _u[d] = u[d, index[0], index[1]]
            _rho = rho[0, index[0], index[1]]
            feq = functional(_rho, _u)

            # Set the output
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1]] = feq[l]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

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
