from xlb.grid.grid import Grid
from xlb.compute_backends import ComputeBackends
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding, Mesh
from jax.experimental import mesh_utils
from xlb.operator.initializer import ConstInitializer
import jax


class JaxGrid(Grid):
    def __init__(self, grid_shape, velocity_set, compute_backend):
        super().__init__(grid_shape, velocity_set, compute_backend)
        self.initialize_jax_backend()

    def initialize_jax_backend(self):
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

    def global_to_local_shape(self, shape):
        if len(shape) < 2:
            raise ValueError("Shape must have at least two dimensions")

        new_second_index = shape[1] // self.nDevices

        return shape[:1] + (new_second_index,) + shape[2:]

    def create_field(self, cardinality, callback=None):
        if callback is None:
            callback = ConstInitializer(self, cardinality, const_value=0.0)
        shape = (cardinality,) + (self.grid_shape)
        return jax.make_array_from_callback(shape, self.sharding, callback)
