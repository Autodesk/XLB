from dataclasses import field
import warp as wp

from .grid import Grid
from xlb.operator import Operator
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Literal
from xlb import DefaultConfig
import numpy as np


class WarpGrid(Grid):
    def __init__(self, shape):
        super().__init__(shape, ComputeBackend.WARP)

    def _initialize_backend(self):
        pass

    def create_field(
        self,
        cardinality: int,
        dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None,
        fill_value=None,
    ):
        dtype = dtype.wp_dtype if dtype else DefaultConfig.default_precision_policy.store_precision.wp_dtype
        shape = (cardinality,) + (self.shape)

        if fill_value is None:
            f = wp.zeros(shape, dtype=dtype)
        else:
            f = wp.full(shape, fill_value, dtype=dtype)
        return f
