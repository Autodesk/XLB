from abc import ABC, abstractmethod

from xlb.global_config import GlobalConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy, Precision
from xlb.velocity_set import VelocitySet
from xlb.operator import Operator


class Grid(ABC):

    def __init__(
        self,
        shape : tuple,
    ):
        # Set parameters
        self.shape = shape
        self.dim = len(shape)

    def parallelize_operator(self, operator: Operator):
        raise NotImplementedError("Parallelization not implemented, child class must implement")

    @abstractmethod
    def create_field(
        self, name: str, cardinality: int, precision: Precision, callback=None
    ):
        raise NotImplementedError("create_field not implemented, child class must implement")
