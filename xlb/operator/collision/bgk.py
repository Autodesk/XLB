import jax.numpy as jnp
from jax import jit
import warp as wp

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from functools import partial


class BGK(Collision):
    """
    BGK collision operator for LBM.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(
        self, f: jnp.ndarray, feq: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray
    ):
        fneq = f - feq
        fout = f - self.compute_dtype(self.omega) * fneq
        return fout

    @Operator.register_backend(ComputeBackend.PALLAS)
    def pallas_implementation(
        self, f: jnp.ndarray, feq: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray
    ):
        fneq = f - feq
        fout = f - self.omega * fneq
        return fout

    def _construct_warp(self):
        # Make constants for warp
        _c = wp.constant(self._warp_stream_mat(self.velocity_set.c))
        _q = wp.constant(self.velocity_set.q)
        _w = wp.constant(self._warp_lattice_vec(self.velocity_set.w))
        _d = wp.constant(self.velocity_set.d)
        _omega = wp.constant(self.compute_dtype(self.omega))

        # Construct the functional
        @wp.func
        def functional(
            f: self._warp_lattice_vec,
            feq: self._warp_lattice_vec,
            rho: self.compute_dtype,
            u: self._warp_u_vec,
        ) -> self._warp_lattice_vec:
            fneq = f - feq
            fout = f - _omega * fneq
            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: self._warp_array_type,
            feq: self._warp_array_type,
            rho: self._warp_array_type,
            u: self._warp_array_type,
            fout: self._warp_array_type,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Get the equilibrium
            _f = self._warp_lattice_vec()
            _feq = self._warp_lattice_vec()
            for l in range(_q):
                _f[l] = f[l, i, j, k]
                _feq[l] = feq[l, i, j, k]
            _u = self._warp_u_vec()
            for l in range(_d):
                _u[l] = u[l, i, j, k]
            _rho = rho[0, i, j, k]
            _fout = functional(_f, _feq, _rho, _u)

            # Write the result
            for l in range(_q):
                fout[l, i, j, k] = _fout[l]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, rho, u, fout):
        # Launch the warp kernel
        wp.launch(
            self._kernel,
            inputs=[
                f,
                feq,
                rho,
                u,
                fout,
            ],
            dim=f.shape[1:],
        )
        return fout
