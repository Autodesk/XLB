from abc import ABC, abstractmethod
from xlb.compute_backend import ComputeBackend
from xlb.global_config import GlobalConfig
from xlb.velocity_set import VelocitySet


class Grid(ABC):

    def __init__(self, shape, velocity_set, precision_policy, grid_backend):
        # Set parameters
        self.shape = shape
        self.velocity_set = velocity_set
        self.precision_policy = precision_policy
        self.grid_backend = grid_backend
        self.dim = self.velocity_set.d

        # Create field dict
        self.fields = {}

    def parallelize_operator(self, operator):
        raise NotImplementedError("Parallelization not implemented, child class must implement")

    @abstractmethod
    def create_field(
        self, name: str, cardinality: int, precision: Precision, callback=None
    ):
        pass

    def get_field(self, name: str):
        return self.fields[name]

    def swap_fields(self, field1, field2):
        self.fields[field1], self.fields[field2] = self.fields[field2], self.fields[field1]
