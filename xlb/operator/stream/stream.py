# Base class for all streaming operators

from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
import numba

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator.operator import Operator


class Stream(Operator):
    """
    Base class for all streaming operators.

    TODO: Currently only this one streaming operator is implemented but
    in the future we may have more streaming operators. For example,
    one might want a multi-step streaming operator.
    """

    def __init__(
            self,
            velocity_set: VelocitySet,
            compute_backend=ComputeBackends.JAX,
        ):
        super().__init__(velocity_set, compute_backend)

    @partial(jit, static_argnums=(0), donate_argnums=(1,))
    def apply_jax(self, f):
        """
        JAX implementation of the streaming step.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution function.
        """

        def _streaming(f, c):
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

        return vmap(_streaming, in_axes=(-1, 0), out_axes=-1)(
            f, jnp.array(self.velocity_set.c).T
        )

    def construct_numba(self, dtype=numba.float32):
        """
        Numba implementation of the streaming step.
        """

        # Get needed values for numba functions
        d = velocity_set.d
        q = velocity_set.q
        c = velocity_set.c.T

        # Make numba functions
        @cuda.jit(device=True)
        def _streaming(f_array, f, ijk):
            # Stream to the next node
            for _ in range(q):
                if d == 2:
                    i = (ijk[0] + int32(c[_, 0])) % f_array.shape[0]
                    j = (ijk[1] + int32(c[_, 1])) % f_array.shape[1]
                    f_array[i, j, _] = f[_]
                else:
                    i = (ijk[0] + int32(c[_, 0])) % f_array.shape[0]
                    j = (ijk[1] + int32(c[_, 1])) % f_array.shape[1]
                    k = (ijk[2] + int32(c[_, 2])) % f_array.shape[2]
                    f_array[i, j, k, _] = f[_]

        return _streaming
