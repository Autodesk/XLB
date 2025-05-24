import numpy as np
import warp as wp
import neon
from .grid import Grid
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Literal, List
from xlb import DefaultConfig


class NeonMultiresGrid(Grid):
    def __init__(
        self,
        shape,
        velocity_set,
        sparsity_pattern_list: List[np.ndarray],
        sparsity_pattern_origins: List[neon.Index_3d],
    ):
        self.bk = None
        self.dim = None
        self.grid = None
        self.velocity_set = velocity_set
        self.sparsity_pattern_list = sparsity_pattern_list
        self.sparsity_pattern_origins = sparsity_pattern_origins
        self.count_levels = len(sparsity_pattern_list)
        self.refinement_factor = 2

        super().__init__(shape, ComputeBackend.NEON)

    def _get_velocity_set(self):
        return self.velocity_set

    def _initialize_backend(self):
        # FIXME@max: for now we hardcode the number of devices to 0
        num_devs = 1
        dev_idx_list = list(range(num_devs))

        if len(self.shape) == 2:
            import py_neon

            self.dim = py_neon.Index_3d(self.shape[0], 1, self.shape[1])
            self.neon_stencil = []
            for q in range(self.velocity_set.q):
                xval, yval = self.velocity_set._c[:, q]
                self.neon_stencil.append([xval, 0, yval])

        else:
            self.dim = neon.Index_3d(self.shape[0], self.shape[1], self.shape[2])

            self.neon_stencil = []
            for q in range(self.velocity_set.q):
                xval, yval, zval = self.velocity_set._c[:, q]
                self.neon_stencil.append([xval, yval, zval])

        self.bk = neon.Backend(runtime=neon.Backend.Runtime.stream, dev_idx_list=dev_idx_list)

        """
         backend: neon.Backend,
         dim,
         sparsity_pattern_list: List[np.ndarray],
         sparsity_pattern_origins: List[neon.Index_3d],
         stencil: List[List[int]]):"""
        self.grid = neon.multires.mGrid(
            backend=self.bk,
            dim=self.dim,
            sparsity_pattern_list=self.sparsity_pattern_list,
            sparsity_pattern_origins=self.sparsity_pattern_origins,
            stencil=self.neon_stencil,
        )
        pass

    def create_field(
        self,
        cardinality: int,
        dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None,
        fill_value=None,
    ):
        dtype = dtype.wp_dtype if dtype else DefaultConfig.default_precision_policy.store_precision.wp_dtype
        field = self.grid.new_field(
            cardinality=cardinality,
            dtype=dtype,
        )
        for l in range(self.count_levels):
            if fill_value is None:
                field.zero_run(l, stream_idx=0)
            else:
                field.fill_run(level=l, value=fill_value, stream_idx=0)
        return field

    def get_neon_backend(self):
        return self.bk

    def level_to_shape(self, level):
        # level = 0 corresponds to the finest level
        return tuple(x // self.refinement_factor**level for x in self.shape)
