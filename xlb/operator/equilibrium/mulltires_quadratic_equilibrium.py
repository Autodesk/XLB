import warp as wp
import neon
from typing import Any
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator import Operator


class MultiresQuadraticEquilibrium(QuadraticEquilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

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
                def quadratic_equilibrium_cl(index: Any):
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
