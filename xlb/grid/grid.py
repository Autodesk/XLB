from abc import ABC, abstractmethod
from xlb.compute_backends import ComputeBackends
from xlb.global_config import GlobalConfig
from xlb.velocity_set import VelocitySet


class Grid(ABC):
    def __init__(self, grid_shape, velocity_set, compute_backend):
        self.velocity_set: VelocitySet = velocity_set
        self.compute_backend = compute_backend
        self.grid_shape = grid_shape
        self.pop_shape = (self.velocity_set.q, *grid_shape)
        self.u_shape = (self.velocity_set.d, *grid_shape)
        self.rho_shape = (1, *grid_shape)
        self.dim = self.velocity_set.d

    @abstractmethod
    def create_field(self, cardinality, callback=None):
        pass

    @staticmethod
    def create(grid_shape, velocity_set=None, compute_backend=None):
        compute_backend = compute_backend or GlobalConfig.compute_backend
        velocity_set = velocity_set or GlobalConfig.velocity_set

        if compute_backend == ComputeBackends.JAX or compute_backend == ComputeBackends.PALLAS:
            from xlb.grid.jax_grid import JaxGrid  # Avoids circular import

            return JaxGrid(grid_shape, velocity_set, compute_backend)
        raise ValueError(f"Compute backend {compute_backend} is not supported")

    @abstractmethod
    def field_global_to_local_shape(self, shape):
        pass
