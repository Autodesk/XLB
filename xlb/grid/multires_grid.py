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
        from .warp_grid import WarpGrid

        self.bk = None
        self.dim = None
        self.grid = None
        self.xlb_lattice = velocity_set
        self.warp_grid = WarpGrid(shape)
        self.sparsity_pattern_list = sparsity_pattern_list
        self.sparsity_pattern_origins = sparsity_pattern_origins
        self.count_levels = len(sparsity_pattern_list)

        super().__init__(shape, ComputeBackend.NEON)

    def _get_velocity_set(self):
        return self.xlb_lattice

    def _initialize_backend(self):
        # FIXME@max: for now we hardcode the number of devices to 0
        num_devs = 1
        dev_idx_list = list(range(num_devs))

        if len(self.shape) == 2:
            import py_neon

            self.dim = py_neon.Index_3d(self.shape[0], 1, self.shape[1])
            self.neon_stencil = []
            for c_idx in range(len(self.xlb_lattice._c[0])):
                xval = self.xlb_lattice._c[0][c_idx]
                yval = self.xlb_lattice._c[1][c_idx]
                self.neon_stencil.append([xval, 0, yval])

        else:
            self.dim = neon.Index_3d(self.shape[0], self.shape[1], self.shape[2])

            self.neon_stencil = []
            for c_idx in range(len(self.xlb_lattice._c[0])):
                xval = self.xlb_lattice._c[0][c_idx]
                yval = self.xlb_lattice._c[1][c_idx]
                zval = self.xlb_lattice._c[2][c_idx]
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

    def _create_warp_field(
        self, cardinality: int, dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None, fill_value=None, ne_field=None
    ):
        print("WARNING: allocating warp fields for mres is temporary and only a work around!")
        warp_field = self.warp_grid.create_field(cardinality, dtype, fill_value)
        if ne_field is None:
            return warp_field

        _d = self.xlb_lattice.d

        import typing

        @neon.Container.factory(mame="cloning-warp")
        def container(src_field: typing.Any, dst_field: typing.Any, cardinality: wp.int32):
            def loading_step(loader: neon.Loader):
                loader.declare_execution_scope(self.grid, level=0)
                src_pn = loader.get_read_handel(src_field)

                @wp.func
                def cloning(gridIdx: typing.Any):
                    cIdx = wp.neon_global_idx(src_pn, gridIdx)
                    gx = wp.neon_get_x(cIdx)
                    gy = wp.neon_get_y(cIdx)
                    gz = wp.neon_get_z(cIdx)

                    # TODO@Max - XLB is flattening the z dimension in 3D, while neon uses the y dimension
                    if _d == 2:
                        gy, gz = gz, gy

                    for card in range(cardinality):
                        value = wp.neon_read(src_pn, gridIdx, card)
                        dst_field[card, gx, gy, gz] = value

                loader.declare_kernel(cloning)

            return loading_step

        c = container(src_field=ne_field, dst_field=warp_field, cardinality=cardinality)
        c.run(0)
        wp.synchronize()
        return warp_field

    def get_neon_backend(self):
        return self.bk
