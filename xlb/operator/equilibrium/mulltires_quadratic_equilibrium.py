import warp as wp
import neon
from typing import Any
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator import Operator


class MultiresQuadraticEquilibrium(QuadraticEquilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials for multires grids.

    This class computes the equilibrium distribution function on multi-resolution grids,
    with proper handling of refined regions and boundary conditions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        # Define vector types for cleaner code
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        @neon.Container.factory(name="MultiresQuadraticEquilibrium")
        def container(
            rho: Any,
            u: Any,
            f: Any,
            level: int,
        ):
            def equilibrium_computation(loader: neon.Loader):
                loader.set_mres_grid(rho.get_grid(), level)

                # Get field handles
                rho_handle = loader.get_mres_read_handle(rho)
                u_handle = loader.get_mres_read_handle(u)
                f_handle = loader.get_mres_write_handle(f)

                @wp.func
                def equilibrium_kernel(index: Any):
                    # Read macroscopic properties
                    _rho = wp.neon_read(rho_handle, index, 0)
                    _u = _u_vec()
                    for d in range(self.velocity_set.d):
                        _u[d] = wp.neon_read(u_handle, index, d)

                    # Compute equilibrium distribution
                    feq = functional(_rho, _u)

                    # Handle refined cells (set to zero if this cell has children)
                    has_child = wp.neon_has_child(f_handle, index)
                    if has_child:
                        feq = _f_vec()  # Initialize to zero
                        for l in range(self.velocity_set.q):
                            feq[l] = self.compute_dtype(0.0)

                    # Write results to output field
                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_handle, index, l, feq[l])

                loader.declare_kernel(equilibrium_kernel)

            return equilibrium_computation

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, rho, u, f, stream=0):
        """
        Execute equilibrium computation for all levels in the multires grid.

        Args:
            rho: Density field
            u: Velocity field
            f: Distribution function field (output)
            stream: Stream ID for execution

        Returns:
            Updated distribution function field
        """
        grid = f.get_grid()

        for level in range(grid.num_levels):
            computation = self.neon_container(rho, u, f, level)
            computation.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)

        return f
