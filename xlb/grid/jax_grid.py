from typing import Any, Literal, Optional, Tuple
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding, Mesh
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from xlb.compute_backend import ComputeBackend

import jax.numpy as jnp
from jax import lax
import jax

from xlb.default_config import DefaultConfig
from .grid import Grid
from xlb.operator import Operator
from xlb.precision_policy import Precision


class JaxGrid(Grid):
    def __init__(self, shape):
        super().__init__(shape, ComputeBackend.JAX)

    def _initialize_backend(self):
        self.nDevices = jax.device_count()
        self.backend = jax.default_backend()
        device_mesh = (
            mesh_utils.create_device_mesh((1, self.nDevices, 1))
            if self.dim == 2
            else mesh_utils.create_device_mesh((1, self.nDevices, 1, 1))
        )
        global_mesh = (
            Mesh(device_mesh, axis_names=("cardinality", "x", "y"))
            if self.dim == 2
            else Mesh(device_mesh, axis_names=("cardinality", "x", "y", "z"))
        )
        self.sharding = (
            NamedSharding(global_mesh, P("cardinality", "x", "y"))
            if self.dim == 2
            else NamedSharding(global_mesh, P("cardinality", "x", "y", "z"))
        )

    def create_field(
        self,
        cardinality: int,
        dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16, Precision.BOOL] = None,
        fill_value=None,
    ):
        sharding_dim = self.shape[-1] // self.nDevices
        device_shape = (cardinality, sharding_dim, *self.shape[1:])
        full_shape = (cardinality, *self.shape)
        arrays = []

        dtype = dtype.jax_dtype if dtype else DefaultConfig.default_precision_policy.store_precision.jax_dtype

        for d, index in self.sharding.addressable_devices_indices_map(
            full_shape
        ).items():
            jax.default_device = d
            if fill_value:
                x = jnp.full(device_shape, fill_value, dtype=dtype)
            else:
                x = jnp.zeros(shape=device_shape, dtype=dtype)
            arrays += [jax.device_put(x, d)]
        jax.default_device = jax.devices()[0]
        return jax.make_array_from_single_device_arrays(
            full_shape, self.sharding, arrays
        )
