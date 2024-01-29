from xlb.velocity_set import VelocitySet
from xlb.global_config import GlobalConfig
from xlb.compute_backends import ComputeBackends
from xlb.operator.operator import Operator
from xlb.grid.grid import Grid
import numpy as np
import jax


class EquilibriumInitializer(Operator):
    def __init__(
        self,
        grid: Grid,
        velocity_set: VelocitySet = None,
        compute_backend: ComputeBackends = None,
    ):
        velocity_set = velocity_set or GlobalConfig.velocity_set
        compute_backend = compute_backend or GlobalConfig.compute_backend
        local_shape = (-1,) + (1,) * (len(grid.pop_shape) - 1)
        
        self.init_values = np.zeros(
            grid.global_to_local_shape(grid.pop_shape)
        ) + velocity_set.w.reshape(local_shape)

        super().__init__(velocity_set, compute_backend)

    @Operator.register_backend(ComputeBackends.JAX)
    def jax_implementation(self, index):
        return self.init_values
