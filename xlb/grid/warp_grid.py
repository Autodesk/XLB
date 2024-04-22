from dataclasses import field
import warp as wp

from xlb.grid import Grid
from xlb.operator import Operator
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Literal
from xlb.default_config import DefaultConfig
import numpy as np


class WarpGrid(Grid):
    def __init__(self, shape):
        super().__init__(shape, ComputeBackend.WARP)

    def _initialize_backend(self):
        pass

    def parallelize_operator(self, operator: Operator):
        # TODO: Implement parallelization of the operator
        raise NotImplementedError(
            "Parallelization of the operator is not implemented yet for the WarpGrid"
        )

    def create_field(
        self,
        cardinality: int,
        dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None,
        init_val=None,
    ):
        dtype = (
            dtype.wp_dtype
            if dtype
            else DefaultConfig.default_precision_policy.store_precision.wp_dtype
        )
        shape = (cardinality,) + (self.shape)

        if init_val is None:
            f = wp.zeros(shape, dtype=dtype)
        else:
            f = wp.full(shape, init_val, dtype=dtype)
        return f
