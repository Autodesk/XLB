from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import Macroscopic, ZeroMoment, FirstMoment


class MultiresMacroscopic(Macroscopic):
    """A class to compute both zero and first moments of distribution functions (rho, u) on a multi-resolution grid."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

    def _construct_neon(self):
        import neon, typing

        # Redefine the zero and first moment operators for the neon backend
        # This is because the neon backend relies on the warp functionals for its operations.
        self.zero_moment = ZeroMoment(compute_backend=ComputeBackend.WARP)
        self.first_moment = FirstMoment(compute_backend=ComputeBackend.WARP)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        functional, _ = self._construct_warp()

        @neon.Container.factory("macroscopic")
        def container(
            level: int,
            f_field: Any,
            bc_mask: Any,
            rho_field: Any,
            u_fild: Any,
        ):
            _d = self.velocity_set.d

            def macroscopic_ll(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)

                rho = loader.get_mres_write_handle(rho_field)
                u = loader.get_mres_write_handle(u_fild)
                f = loader.get_mres_read_handle(f_field)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)

                @wp.func
                def macroscopic_cl(gIdx: typing.Any):
                    _f = _f_vec()
                    _boundary_id = wp.neon_read(bc_mask_pn, gIdx, 0)

                    for l in range(self.velocity_set.q):
                        _f[l] = wp.neon_read(f, gIdx, l)

                    _rho, _u = functional(_f)

                    if _boundary_id != wp.uint8(0):
                        _rho = self.compute_dtype(1.0)
                        for d in range(_d):
                            _u[d] = self.compute_dtype(0.0)

                    if _boundary_id == wp.uint8(255) or wp.neon_has_child(f, gIdx):
                        _rho = self.compute_dtype(0.0)
                        for d in range(_d):
                            _u[d] = self.compute_dtype(0.0)

                    wp.neon_write(rho, gIdx, 0, _rho)
                    for d in range(_d):
                        wp.neon_write(u, gIdx, d, _u[d])

                loader.declare_kernel(macroscopic_cl)

            return macroscopic_ll

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f, bc_mask, rho, u, streamId=0):
        grid = f.get_grid()
        for level in range(grid.num_levels):
            c = self.neon_container(level, f, bc_mask, rho, u)
            c.run(streamId)
        return rho, u
