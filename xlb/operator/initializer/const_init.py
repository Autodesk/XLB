from xlb.velocity_set import VelocitySet
from xlb.global_config import GlobalConfig
from xlb.compute_backends import ComputeBackends
from xlb.operator.operator import Operator
from xlb.grid.grid import Grid
from functools import partial
import numpy as np
import jax


class ConstInitializer(Operator):
    def __init__(
        self,
        type=np.float32,
        velocity_set: VelocitySet = None,
        compute_backend: ComputeBackends = None,
    ):
        self.type = type
        self.grid = grid
        velocity_set = velocity_set or GlobalConfig.velocity_set
        compute_backend = compute_backend or GlobalConfig.compute_backend

        super().__init__(velocity_set, compute_backend)

    @Operator.register_backend(ComputeBackends.JAX)
    @partial(jax.jit, static_argnums=(0, 2))
    def jax_implementation(self, const_value, sharding=None):
        if sharding is None:
            sharding = self.grid.sharding
        x = jax.numpy.full(shape=self.shape, fill_value=const_value, dtype=self.type)
        return jax.lax.with_sharding_constraint(x, sharding)

    @Operator.register_backend(ComputeBackends.PALLAS)
    @partial(jax.jit, static_argnums=(0, 2))
    def jax_implementation(self, const_value, sharding=None):
        return self.jax_implementation(const_value, sharding)
