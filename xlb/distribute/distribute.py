from jax.sharding import PartitionSpec as P
from xlb.operator import Operator
from xlb import DefaultConfig
from jax import lax, sharding
from jax import jit
import warp as wp


def distribute(
    operator: Operator, grid, velocity_set, num_results=2, ops="permute"
) -> Operator:
    # Define the sharded operator
    def _sharded_operator(*args):
        results = operator(*args)

        if not isinstance(results, tuple):
            results = (results,)

        if DefaultConfig.default_backend == DefaultConfig.ComputeBackend.WARP:
            for i, result in enumerate(results):
                if isinstance(result, wp.array):
                    # Convert to jax array (zero copy)
                    results[i] = wp.to_jax(result)

        if ops == "permute":
            # Define permutation rules for right and left communication
            rightPerm = [(i, (i + 1) % grid.nDevices) for i in range(grid.nDevices)]
            leftPerm = [((i + 1) % grid.nDevices, i) for i in range(grid.nDevices)]

            right_comm = [
                lax.ppermute(
                    arg[velocity_set.right_indices, :1, ...],
                    perm=rightPerm,
                    axis_name="x",
                )
                for arg in results
            ]
            left_comm = [
                lax.ppermute(
                    arg[velocity_set.left_indices, -1:, ...],
                    perm=leftPerm,
                    axis_name="x",
                )
                for arg in results
            ]

            updated_results = []
            for result in results:
                result = result.at[velocity_set.right_indices, :1, ...].set(
                    right_comm.pop(0)
                )
                result = result.at[velocity_set.left_indices, -1:, ...].set(
                    left_comm.pop(0)
                )
                updated_results.append(result)

            return (
                tuple(updated_results)
                if len(updated_results) > 1
                else updated_results[0]
            )
        else:
            raise NotImplementedError(f"Operation {ops} not implemented")

    in_specs = (P(*((None, "x") + (grid.dim - 1) * (None,)))) * len(num_results)
    out_specs = in_specs

    distributed_operator = sharding.shard_map(
        _sharded_operator,
        mesh=grid.global_mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )

    if DefaultConfig.default_backend == DefaultConfig.ComputeBackend.JAX:
        distributed_operator = jit(distributed_operator)

    return distributed_operator
