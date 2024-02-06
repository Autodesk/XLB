# Base class for all streaming operators

from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator import Operator
from xlb.operator import ParallelOperator

class Stream(Operator):
    """
    Base class for all streaming operators.
    """

    def __init__(self, grid, velocity_set: VelocitySet = None, compute_backend=None):
        self.grid = grid
        self.parallel_operator = ParallelOperator(grid, self._streaming_jax_p, velocity_set)
        super().__init__(velocity_set, compute_backend)

    @Operator.register_backend(ComputeBackends.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f):
        """
        JAX implementation of the streaming step.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution function.
        """
        return self.parallel_operator(f)

    def _streaming_jax_p(self, f):
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
