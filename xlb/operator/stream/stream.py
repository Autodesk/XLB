# Base class for all streaming operators

from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class Stream(Operator):
    """
    Base class for all streaming operators. This is used for pulling the distribution
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f):
        """
        JAX implementation of the streaming step.

        TODO: Make sure this works with pull scheme.

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
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.wp_c
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        # Construct the warp functional
        @wp.func
        def functional2d(
            f: wp.array3d(dtype=Any),
            index: Any,
        ):
            # Pull the distribution function
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                # Get pull index
                pull_index = type(index)()
                for d in range(self.velocity_set.d):
                    pull_index[d] = index[d] - _c[d, l]

                    if pull_index[d] < 0:
                        pull_index[d] = f.shape[d + 1] - 1
                    elif pull_index[d] >= f.shape[d + 1]:
                        pull_index[d] = 0

                # Read the distribution function
                _f[l] = f[l, pull_index[0], pull_index[1]]

            return _f

        @wp.kernel
        def kernel2d(
            f_0: wp.array3d(dtype=Any),
            f_1: wp.array3d(dtype=Any),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # Set the output
            _f = functional2d(f_0, index)

            # Write the output
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1]] = _f[l]

        # Construct the funcional to get streamed indices
        @wp.func
        def functional3d(
            f: wp.array4d(dtype=Any),
            index: Any,
        ):
            # Pull the distribution function
            _f = _f_vec()
            for l in range(self.velocity_set.q):
                # Get pull index
                pull_index = type(index)()
                for d in range(self.velocity_set.d):
                    pull_index[d] = index[d] - _c[d, l]

                    if pull_index[d] < 0:
                        pull_index[d] = f.shape[d + 1] - 1
                    elif pull_index[d] >= f.shape[d + 1]:
                        pull_index[d] = 0

                # Read the distribution function
                _f[l] = f[l, pull_index[0], pull_index[1], pull_index[2]]

            return _f

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Set the output
            _f = functional3d(f_0, index)

            # Write the output
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1], index[2]] = _f[l]

        functional = functional3d if self.velocity_set.d == 3 else functional2d
        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f_0,
                f_1,
            ],
            dim=f_0.shape[1:],
        )
        return f_1
