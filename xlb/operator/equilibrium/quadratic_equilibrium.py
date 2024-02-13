from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator import Operator
from xlb.global_config import GlobalConfig


class QuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.

    TODO: move this to a separate file and lower and higher order equilibriums
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), donate_argnums=(1, 2))
    def jax_implementation(self, rho, u):
        cu = 3.0 * jnp.tensordot(self.velocity_set.c, u, axes=(0, 0))
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=0, keepdims=True)
        w = self.velocity_set.w.reshape((-1,) + (1,) * (len(rho.shape) - 1))
        feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        return feq

    def _construct_warp(self):
        # Make constants for warp
        _c = wp.constant(self._warp_stream_mat(self.velocity_set.c))
        _q = wp.constant(self.velocity_set.q)
        _w = wp.constant(self._warp_lattice_vec(self.velocity_set.w))
        _d = wp.constant(self.velocity_set.d)

        # Construct the equilibrium functional
        @wp.func
        def functional(
            rho: self.compute_dtype, u: self._warp_u_vec
        ) -> self._warp_lattice_vec:
            feq = self._warp_lattice_vec()  # empty lattice vector
            for l in range(_q):
                # Compute cu
                cu = self.compute_dtype(0.0)
                for d in range(_d):
                    if _c[l, d] == 1:
                        cu += u[d]
                    elif _c[l, d] == -1:
                        cu -= u[d]
                cu *= self.compute_dtype(3.0)

                # Compute usqr
                usqr = 1.5 * wp.dot(u, u)

                # Compute feq
                feq[l] = rho * _w[l] * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

            return feq

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: self._warp_array_type,
            u: self._warp_array_type,
            f: self._warp_array_type,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Get the equilibrium
            _u = self._warp_u_vec()
            for d in range(_d):
                _u[i] = u[d, i, j, k]
            _rho = rho[0, i, j, k]
            feq = functional(_rho, _u)

            # Set the output
            for l in range(_q):
                f[l, i, j, k] = feq[l]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, rho, u, f):
        # Launch the warp kernel
        wp.launch(
            self._kernel,
            inputs=[
                rho,
                u,
                f,
            ],
            dim=rho.shape[1:],
        )
        return f
