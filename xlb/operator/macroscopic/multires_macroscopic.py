from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic.zero_moment import ZeroMoment
from xlb.operator.macroscopic.first_moment import FirstMoment


class MultiresMacroscopic(Operator):
    """A class to compute both zero and first moments of distribution functions (rho, u)."""

    def __init__(self, *args, **kwargs):
        self.zero_moment = ZeroMoment(*args, **kwargs)
        self.first_moment = FirstMoment(*args, **kwargs)
        super().__init__(*args, **kwargs)


    def _construct_warp(self):
        zero_moment_func = self.zero_moment.warp_functional
        first_moment_func = self.first_moment.warp_functional
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        @wp.func
        def functional(f: _f_vec):
            rho = zero_moment_func(f)
            u = first_moment_func(f, rho)
            return rho, u

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _f = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
            _rho, _u = functional(_f)

            rho[0, index[0], index[1], index[2]] = self.store_dtype(_rho)
            for d in range(self.velocity_set.d):
                u[d, index[0], index[1], index[2]] = self.store_dtype(_u[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, rho, u):
        wp.launch(
            self.warp_kernel,
            inputs=[f, rho, u],
            dim=rho.shape[1:],
        )
        return rho, u

    def _construct_neon(self):
        zero_moment_func = self.zero_moment.neon_functional
        first_moment_func = self.first_moment.neon_functional
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        @wp.func
        def functional(f: _f_vec):
            rho = zero_moment_func(f)
            u = first_moment_func(f, rho)
            return rho, u


        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        import neon, typing
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

                rho=loader.get_mres_read_handle(rho_field)
                u =loader.get_mres_read_handle(u_fild)
                f=loader.get_mres_write_handle(f_field)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)

                @wp.func
                def macroscopic_cl(gIdx: typing.Any):
                    _f = _f_vec()
                    _boundary_id = wp.neon_read(bc_mask_pn, gIdx, 0)

                    for l in range(self.velocity_set.q):
                        _f[l] = wp.neon_read(f, gIdx,l)
                    _rho, _u = functional(_f)
                    if _boundary_id != wp.uint8(0):
                        _rho = self.compute_dtype(1.0)
                        for d in range(_d):
                            _u[d] = self.compute_dtype(0.0)
                    if _boundary_id == wp.uint8(255):
                        _rho = self.compute_dtype(0.0)
                        for d in range(_d):
                            _u[d] = self.compute_dtype(0.0)

                    wp.neon_write(rho, gIdx, 0, _rho)
                    for d in range(_d):
                        wp.neon_write(u, gIdx, d, _u[d])

                    if wp.neon_has_children(f, gIdx):
                        offVal = self.compute_dtype(-33000.0)
                        zero_val = self.compute_dtype(0.0)
                        wp.neon_write(rho, gIdx, 0, zero_val)
                        wp.neon_write(u, gIdx, 0, offVal)
                        wp.neon_write(u, gIdx, 1, zero_val)
                        wp.neon_write(u, gIdx, 2, zero_val)
                    else:
                        offVal = self.compute_dtype(+33000.0)
                        zero_val = self.compute_dtype(0.0)
                        wp.neon_write(rho, gIdx, 0, zero_val)
                        wp.neon_write(u, gIdx, 0, offVal)
                        wp.neon_write(u, gIdx, 1, zero_val)
                        wp.neon_write(u, gIdx, 2, zero_val)
                loader.declare_kernel(macroscopic_cl)
            return macroscopic_ll
        return functional, container

    def get_containers(self, target_level, f_0, f_1, bc_mask, rho, u):
        _, container = self._construct_neon()
        evenList = []
        oddList = []
        evenList.append(container(target_level, f_0, bc_mask,   rho, u))
        oddList.append( container(target_level, f_1, bc_mask,  rho, u))
        return {'even':evenList ,
                'odd':oddList }

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f, rho, u):
        c = self.neon_container(f, rho, u)
        c.run(0)
        wp.synchronize()
        return rho, u
