from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax import lax


class ParallelOperator:
    """
    A generic class for parallelizing operations across multiple GPUs/TPUs.
    """

    def __init__(self, grid, func, velocity_set):
        """
        Initialize the ParallelOperator.

        Parameters
        ----------
        grid : Grid object
            The computational grid.
        func : function
            The function to be parallelized.
        velocity_set : VelocitySet object
            The velocity set used in the Lattice Boltzmann Method.
        """
        self.grid = grid
        self.func = func
        self.velocity_set = velocity_set

    def __call__(self, f):
        """
        Execute the parallel operation.

        Parameters
        ----------
        f : jax.numpy.ndarray
            The input data for the operation.

        Returns
        -------
        jax.numpy.ndarray
            The result after applying the parallel operation.
        """
        in_specs = P(*((None, "x") + (self.grid.dim - 1) * (None,)))
        out_specs = in_specs

        f = shard_map(
            self._parallel_func,
            mesh=self.grid.global_mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )(f)
        return f

    def _parallel_func(self, f):
        """
        Internal function to handle data communication and apply the given function in parallel.

        Parameters
        ----------
        f : jax.numpy.ndarray
            The input data.

        Returns
        -------
        jax.numpy.ndarray
            The processed data.
        """
        rightPerm = [(i, (i + 1) % self.grid.nDevices) for i in range(self.grid.nDevices)]
        leftPerm = [((i + 1) % self.grid.nDevices, i) for i in range(self.grid.nDevices)]
        f = self.func(f)
        left_comm, right_comm = (
            f[self.velocity_set.right_indices, :1, ...],
            f[self.velocity_set.left_indices, -1:, ...],
        )
        left_comm, right_comm = (
            lax.ppermute(left_comm, perm=rightPerm, axis_name="x"),
            lax.ppermute(right_comm, perm=leftPerm, axis_name="x"),
        )
        f = f.at[self.velocity_set.right_indices, :1, ...].set(left_comm)
        f = f.at[self.velocity_set.left_indices, -1:, ...].set(right_comm)

        return f
