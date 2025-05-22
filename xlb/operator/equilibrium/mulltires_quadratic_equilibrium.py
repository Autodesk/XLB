from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
import os

import neon
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator import Operator


class MultiresQuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.
    """

    def _construct_neon(self):
        import neon

        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.c
        _w = self.velocity_set.w
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Construct the equilibrium functional
        @wp.func
        def functional(
            rho: Any,
            u: Any,
        ):
            # Allocate the equilibrium
            feq = _f_vec()

            # Compute the equilibrium
            for l in range(self.velocity_set.q):
                # Compute cu
                cu = self.compute_dtype(0.0)
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        cu += u[d]
                    elif _c[d, l] == -1:
                        cu -= u[d]
                cu *= self.compute_dtype(3.0)

                # Compute usqr
                usqr = self.compute_dtype(1.5) * wp.dot(u, u)

                # Compute feq
                feq[l] = rho * _w[l] * (self.compute_dtype(1.0) + cu * (self.compute_dtype(1.0) + self.compute_dtype(0.5) * cu) - usqr)

            return feq

        import typing

        @neon.Container.factory(name="QuadraticEquilibrium")
        def container(
            level,
            rho: Any,
            u: Any,
            f: Any,
        ):
            def quadratic_equilibrium_ll(loader: neon.Loader):
                loader.set_mres_grid(rho.get_grid(), level)

                rho_pn = loader.get_mres_read_handle(rho)
                u_pn = loader.get_mres_read_handle(u)
                f_pn = loader.get_mres_write_handle(f)

                @wp.func
                def quadratic_equilibrium_cl(index: typing.Any):
                    _u = _u_vec()
                    for d in range(self.velocity_set.d):
                        _u[d] = wp.neon_read(u_pn, index, d)
                    _rho = wp.neon_read(rho_pn, index, 0)
                    feq = functional(_rho, _u)

                    if wp.neon_has_child(f_pn, index):
                        for l in range(self.velocity_set.q):
                            feq[l] = self.compute_dtype(0.0)
                    # Set the output
                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_pn, index, l, feq[l])

                loader.declare_kernel(quadratic_equilibrium_cl)

            return quadratic_equilibrium_ll

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, level, rho, u, f, stream):
        c = self.neon_container(level, rho, u, f)
        c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)

        return f
