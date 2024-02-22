from xlb.grid.grid import Grid
from xlb.compute_backends import ComputeBackends
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding, Mesh
from jax.experimental import mesh_utils
from xlb.operator.initializer import ConstInitializer
import jax


class JaxGrid(Grid):
    def __init__(self, grid_shape, velocity_set, precision_policy, grid_backend):
        super().__init__(grid_shape, velocity_set, precision_policy, grid_backend)
        self._initialize_jax_backend()

    def _initialize_jax_backend(self):
        self.nDevices = jax.device_count()
        self.backend = jax.default_backend()
        device_mesh = (
            mesh_utils.create_device_mesh((1, self.nDevices, 1))
            if self.dim == 2
            else mesh_utils.create_device_mesh((1, self.nDevices, 1, 1))
        )
        self.global_mesh = (
            Mesh(device_mesh, axis_names=("cardinality", "x", "y"))
            if self.dim == 2
            else Mesh(device_mesh, axis_names=("cardinality", "x", "y", "z"))
        )
        self.sharding = (
            NamedSharding(self.global_mesh, P("cardinality", "x", "y"))
            if self.dim == 2
            else NamedSharding(self.global_mesh, P("cardinality", "x", "y", "z"))
        )
        self.grid_shape_per_gpu = (
            self.grid_shape[0] // self.nDevices,
        ) + self.grid_shape[1:]


    def parallelize_operator(self, operator: Operator):
        # TODO: fix this

        # Make parallel function
        def _parallel_operator(f):
            rightPerm = [
                (i, (i + 1) % self.grid.nDevices) for i in range(self.grid.nDevices)
            ]
            leftPerm = [
                ((i + 1) % self.grid.nDevices, i) for i in range(self.grid.nDevices)
            ]
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




    def create_field(self, name: str, cardinality: int, callback=None):
        # Get shape of the field
        shape = (cardinality,) + (self.shape)

        # Create field
        if callback is None:
            f = jax.numpy.full(shape, 0.0, dtype=self.precision_policy)
            if self.sharding is not None:
                f = jax.make_sharded_array(self.sharding, f)
        else:
            f = jax.make_array_from_callback(shape, self.sharding, callback)

        # Add field to the field dictionary
        self.fields[name] = f
