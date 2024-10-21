# Base class for all equilibriums

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class SecondMoment(Operator):
    """
    Operator to calculate the second moment of distribution functions.

    The second moment may be used to compute the momentum flux in the computation of
    the stress tensor in the Lattice Boltzmann Method (LBM).

    Important Note:
    Note that this rank 2 symmetric tensor (dim*dim) has been converted into a rank one
    vector where the diagonal and off-diagonal components correspond to the following elements of
    the vector:
    if self.grid.dim == 3:
        diagonal    = (0, 3, 5)
        offdiagonal = (1, 2, 4)
    elif self.grid.dim == 2:
        diagonal    = (0, 2)
        offdiagonal = (1,)

    ** For any reduction operation on the full tensor it is crucial to account for the full tensor by
    considering all diagonal and off-diagonal components.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def jax_implementation(
        self,
        fneq: jnp.ndarray,
    ):
        """
        This function computes the second order moment, which is the product of the
        distribution functions (f) and the lattice moments (cc).

        Parameters
        ----------
        fneq: jax.numpy.ndarray
            The distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The computed second moment.
        """
        return jnp.tensordot(self.velocity_set.cc, fneq, axes=(0, 0))

    def _construct_warp(self):
        # Make constants for warp
        _cc = self.velocity_set.cc
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _pi_dim = self.velocity_set.d * (self.velocity_set.d + 1) // 2
        _pi_vec = wp.vec(
            _pi_dim,
            dtype=self.compute_dtype,
        )

        # Construct functional for computing second moment
        @wp.func
        def functional(
            fneq: Any,
        ):
            # Get second order moment (a symmetric tensore shaped into a vector)
            pi = _pi_vec()
            for d in range(_pi_dim):
                pi[d] = self.compute_dtype(0.0)
                for q in range(self.velocity_set.q):
                    pi[d] += _cc[q, d] * fneq[q]
            return pi

        # Construct the kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            pi: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            _pi = functional(_f)

            # Set the output
            for d in range(_pi_dim):
                pi[d, index[0], index[1], index[2]] = self.store_dtype(_pi[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, pi):
        # Launch the warp kernel
        wp.launch(self.warp_kernel, inputs=[f, pi], dim=pi.shape[1:])
        return pi
