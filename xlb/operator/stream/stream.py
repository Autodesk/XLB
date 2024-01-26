# Base class for all streaming operators

from functools import partial
import jax.numpy as jnp
from jax import jit, vmap, lax

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator.operator import Operator
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


class Stream(Operator):
    """
    Base class for all streaming operators.
    """

    def __init__(self, grid, velocity_set: VelocitySet = None, compute_backend=None):
        self.grid = grid
        super().__init__(velocity_set, compute_backend)

    @Operator.register_backend(ComputeBackends.JAX)
    # @partial(jit, static_argnums=(0))
    def jax_implementation(self, f):
        """
        JAX implementation of the streaming step.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution function.
        """
        in_specs = P(*((None, "x") + (self.grid.dim - 1) * (None,)))
        out_specs = in_specs
        return shard_map(
            self._streaming_jax_m,
            mesh=self.grid.global_mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )(f)

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

    def _streaming_jax_m(self, f):
        """
        This function performs the streaming step in the Lattice Boltzmann Method, which is
        the propagation of the distribution functions in the lattice.

        To enable multi-GPU/TPU functionality, it extracts the left and right boundary slices of the
        distribution functions that need to be communicated to the neighboring processes.

        The function then sends the left boundary slice to the right neighboring process and the right
        boundary slice to the left neighboring process. The received data is then set to the
        corresponding indices in the receiving domain.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The array holding the distribution functions for the simulation.

        Returns
        -------
        jax.numpy.ndarray
            The distribution functions after the streaming operation.
        """
        rightPerm = [(i, (i + 1) % self.grid.nDevices) for i in range(self.grid.nDevices)]
        leftPerm = [((i + 1) % self.grid.nDevices, i) for i in range(self.grid.nDevices)]

        f = self._streaming_jax_p(f)
        left_comm, right_comm = (
            f[self.velocity_set.right_indices, :1, ...],
            f[self.velocity_set.left_indices, -1:, ...],
        )
        left_comm, right_comm = (
            lax.ppermute(left_comm, perm=rightPerm, axis_name='x'),
            lax.ppermute(right_comm, perm=leftPerm, axis_name='x'),
        )
        f = f.at[self.velocity_set.right_indices, :1, ...].set(left_comm)
        f = f.at[self.velocity_set.left_indices, -1:, ...].set(right_comm)

        return f
