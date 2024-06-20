from jax.sharding import PartitionSpec as P
from xlb.operator import Operator
from xlb import DefaultConfig
from xlb import ComputeBackend
from jax import lax
from jax.experimental.shard_map import shard_map
from jax import jit
import jax.numpy as jnp
import warp as wp
from typing import Tuple


def distribute(
    operator: Operator,
    grid,
    velocity_set,
    sharding_flags: Tuple[bool, ...],
    num_results=1,
    ops="permute",
) -> Operator:
    if DefaultConfig.default_backend == ComputeBackend.WARP:
        from warp.jax_experimental import jax_kernel
        operator = jax_kernel(operator.warp_kernel)

    # Define the sharded operator
    def _sharded_operator(*args):
        results = operator(*args[:-2])

        # if not isinstance(results, tuple):
        #     results = (results,)

        if ops == "permute":
            # Define permutation rules for right and left communication
            rightPerm = [(i, (i + 1) % grid.nDevices) for i in range(grid.nDevices)]
            leftPerm = [((i + 1) % grid.nDevices, i) for i in range(grid.nDevices)]

            right_comm = [
                lax.ppermute(
                    result[velocity_set.right_indices, :1, ...],
                    perm=rightPerm,
                    axis_name="x",
                )
                for result in results
            ]
            left_comm = [
                lax.ppermute(
                    result[velocity_set.left_indices, -1:, ...],
                    perm=leftPerm,
                    axis_name="x",
                )
                for result in results
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

    in_specs = tuple(
        P(*((None, "x") + (grid.dim - 1) * (None,))) if flag else P()
        for flag in sharding_flags
    )
    out_specs = tuple(
        P(*((None, "x") + (grid.dim - 1) * (None,))) for _ in range(num_results)
    )

    if len(out_specs) == 1:
        out_specs = out_specs[0]

    distributed_operator = shard_map(
        _sharded_operator,
        mesh=grid.global_mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )
    distributed_operator = jit(distributed_operator)

    return distributed_operator
