from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple

from xlb.default_config import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision


def grid(
    shape: Tuple[int, ...],
    compute_backend: ComputeBackend = None,
    parallel: bool = False,
):
    compute_backend = compute_backend or DefaultConfig.default_backend
    if compute_backend == ComputeBackend.WARP:
        from xlb.grid.warp_grid import WarpGrid

        return WarpGrid(shape)
    elif compute_backend == ComputeBackend.JAX:
        from xlb.grid.jax_grid import JaxGrid

        return JaxGrid(shape)

    raise ValueError(f"Compute backend {compute_backend} is not supported")


class Grid(ABC):
    def __init__(self, shape: Tuple[int, ...], compute_backend: ComputeBackend):
        self.shape = shape
        self.dim = len(shape)
        self.parallel = False
        self.compute_backend = compute_backend
        self._initialize_backend()

    @abstractmethod
    def _initialize_backend(self):
        pass

    # @abstractmethod
    # def parallelize_operator(self, operator: Operator):
    #     pass
