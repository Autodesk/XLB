"""
Multi-resolution quadratic equilibrium operator for the Neon backend.
"""

import warp as wp
from typing import Any
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator import Operator


class MultiresQuadraticEquilibrium(QuadraticEquilibrium):
    """Quadratic equilibrium operator for multi-resolution grids (Neon only).

    Computes the second-order Hermite-polynomial equilibrium distribution
    from density and velocity at every active cell on each grid level.
    Cells that have child refinement (halo cells) are zeroed out.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

    def _construct_neon(self):
        import neon

        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        @neon.Container.factory(name="QuadraticEquilibrium")
        def container(
            rho: Any,
            u: Any,
            f: Any,
            level,
        ):
            def quadratic_equilibrium_ll(loader: neon.Loader):
                loader.set_mres_grid(rho.get_grid(), level)

                rho_pn = loader.get_mres_read_handle(rho)
                u_pn = loader.get_mres_read_handle(u)
                f_pn = loader.get_mres_write_handle(f)

                @wp.func
                def quadratic_equilibrium_cl(index: Any):
                    _u = _u_vec()
                    for d in range(self.velocity_set.d):
                        _u[d] = self.compute_dtype(wp.neon_read(u_pn, index, d))
                    _rho = self.compute_dtype(wp.neon_read(rho_pn, index, 0))
                    feq = functional(_rho, _u)

                    if wp.neon_has_child(f_pn, index):
                        for l in range(self.velocity_set.q):
                            feq[l] = self.compute_dtype(0.0)
                    # Set the output
                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_pn, index, l, self.store_dtype(feq[l]))

                loader.declare_kernel(quadratic_equilibrium_cl)

            return quadratic_equilibrium_ll

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, rho, u, f, stream=0):
        import neon

        grid = f.get_grid()
        for level in range(grid.num_levels):
            c = self.neon_container(rho, u, f, level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return f
