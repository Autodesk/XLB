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
