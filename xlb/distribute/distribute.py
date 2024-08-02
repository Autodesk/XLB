from jax.sharding import PartitionSpec as P
from xlb.operator import Operator
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from jax import lax
from jax.experimental.shard_map import shard_map
from jax import jit


def distribute_operator(
    operator: Operator,
    grid,
    velocity_set,
    num_results=1,
    ops="permute",
) -> Operator:
    # Define the sharded operator
    def _sharded_operator(*args):
        result = operator(*args)

        if ops == "permute":
            # Define permutation rules for right and left communication
            rightPerm = [(i, (i + 1) % grid.nDevices) for i in range(grid.nDevices)]
            leftPerm = [((i + 1) % grid.nDevices, i) for i in range(grid.nDevices)]

            left_comm, right_comm = (
                result[velocity_set.right_indices, :1, ...],
                result[velocity_set.left_indices, -1:, ...],
            )

            left_comm = lax.ppermute(
                left_comm,
                perm=rightPerm,
                axis_name="x",
            )

            right_comm = lax.ppermute(
                right_comm,
                perm=leftPerm,
                axis_name="x",
            )

            result = result.at[velocity_set.right_indices, :1, ...].set(left_comm)
            result = result.at[velocity_set.left_indices, -1:, ...].set(right_comm)

            return result
        else:
            raise NotImplementedError(f"Operation {ops} not implemented")

    # Build sharding_flags and in_specs based on args
    def build_specs(grid, *args):
        sharding_flags = []
        in_specs = []
        for arg in args:
            if arg.shape[1:] == grid.shape:
                sharding_flags.append(True)
            else:
                sharding_flags.append(False)

        in_specs = tuple(P(*((None, "x") + (grid.dim - 1) * (None,))) if flag else P() for flag in sharding_flags)
        out_specs = tuple(P(*((None, "x") + (grid.dim - 1) * (None,))) for _ in range(num_results))
        return tuple(sharding_flags), in_specs, out_specs

    def _wrapped_operator(*args):
        sharding_flags, in_specs, out_specs = build_specs(grid, *args)

        if len(out_specs) == 1:
            out_specs = out_specs[0]

        distributed_operator = shard_map(
            _sharded_operator,
            mesh=grid.global_mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
        return distributed_operator(*args)

    return jit(_wrapped_operator)


def distribute(operator, grid, velocity_set, num_results=1, ops="permute"):
    """
    Distribute an operator or a stepper.
    If the operator is a stepper, check for post-streaming boundary conditions
    before deciding how to distribute.
    """
    if isinstance(operator, IncompressibleNavierStokesStepper):
        # Check for post-streaming boundary conditions
        has_post_streaming_bc = any(bc.implementation_step == ImplementationStep.STREAMING for bc in operator.boundary_conditions)

        if has_post_streaming_bc:
            # If there are post-streaming BCs, only distribute the stream operator
            distributed_stream = distribute_operator(operator.stream, grid, velocity_set)
            operator.stream = distributed_stream
        else:
            # If no post-streaming BCs, distribute the whole operator
            distributed_op = distribute_operator(operator, grid, velocity_set, num_results=num_results, ops=ops)
            return distributed_op

        return operator
    else:
        # For other operators, apply the original distribution logic
        distributed_op = distribute_operator(operator, grid, velocity_set, num_results=num_results, ops=ops)
        return distributed_op
