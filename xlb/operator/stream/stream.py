# Base class for all streaming operators

from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class Stream(Operator):
    """
    Base class for all streaming operators.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f):
        """
        JAX implementation of the streaming step.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution function.
        """

        def _streaming_jax_i(f, c):
            """
            Perform individual streaming operation in a direction.

            Parameters
            ----------
                f: The distribution function.
                c: The streaming direction vector.

            Returns
            -------
                jax.numpy.ndarray
                The updated distribution function after streaming.
            """
            if self.velocity_set.d == 2:
                return jnp.roll(f, (c[0], c[1]), axis=(0, 1))
            elif self.velocity_set.d == 3:
                return jnp.roll(f, (c[0], c[1], c[2]), axis=(0, 1, 2))

        return vmap(_streaming_jax_i, in_axes=(0, 0), out_axes=0)(
            f, jnp.array(self.velocity_set.c).T
        )

    def _construct_warp(self):
        # Make constants for warp
        _c = wp.constant(self._warp_int_stream_mat(self.velocity_set.c))
        _q = wp.constant(self.velocity_set.q)
        _d = wp.constant(self.velocity_set.d)

        # Construct the funcional to get streamed indices
        @wp.func
        def functional(
            l: int,
            i: int,
            j: int,
            k: int,
            max_i: int,
            max_j: int,
            max_k: int,
        ):
            streamed_i = i + _c[0, l]
            streamed_j = j + _c[1, l]
            streamed_k = k + _c[2, l]
            if streamed_i < 0:
                streamed_i = max_i - 1
            elif streamed_i >= max_i:
                streamed_i = 0
            if streamed_j < 0:
                streamed_j = max_j - 1
            elif streamed_j >= max_j:
                streamed_j = 0
            if streamed_k < 0:
                streamed_k = max_k - 1
            elif streamed_k >= max_k:
                streamed_k = 0
            return streamed_i, streamed_j, streamed_k

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_0: self._warp_array_type,
            f_1: self._warp_array_type,
            max_i: int,
            max_j: int,
            max_k: int,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Set the output
            for l in range(_q):
                streamed_i, streamed_j, streamed_k = functional(
                    l, i, j, k, max_i, max_j, max_k
                )
                f_1[l, streamed_i, streamed_j, streamed_k] = f_0[l, i, j, k]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1):
        # Launch the warp kernel
        wp.launch(
            self._kernel,
            inputs=[
                f_0,
                f_1,
                f_0.shape[1],
                f_0.shape[2],
                f_0.shape[3],
            ],
            dim=f_0.shape[1:],
        )
        return f_1
