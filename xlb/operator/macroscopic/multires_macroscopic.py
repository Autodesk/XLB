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

        # Reuse Warp functionals for the Neon backend
        self.zero_moment = ZeroMoment(compute_backend=ComputeBackend.WARP)
        self.first_moment = FirstMoment(compute_backend=ComputeBackend.WARP)

        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        functional, _ = self._construct_warp()

        @neon.Container.factory("macroscopic")
        def container(
            level: int,
            f_field: Any,
            bc_mask: Any,
            rho_field: Any,
            u_field: Any,
        ):
            def macroscopic_computation(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)

                # Get handles for all fields
                rho_handle = loader.get_mres_write_handle(rho_field)
                u_handle = loader.get_mres_write_handle(u_field)
                f_handle = loader.get_mres_read_handle(f_field)
                bc_mask_handle = loader.get_mres_read_handle(bc_mask)

                @wp.func
                def macroscopic_kernel(grid_index: typing.Any):
                    # Read boundary condition ID
                    _boundary_id = wp.neon_read(bc_mask_handle, grid_index, 0)

                    # Read all populations
                    _f = _f_vec()
                    for l in range(self.velocity_set.q):
                        _f[l] = wp.neon_read(f_handle, grid_index, l)

                    # Compute macroscopic properties
                    _rho, _u = functional(_f)

                    # Apply boundary condition corrections
                    if _boundary_id != wp.uint8(0):
                        _rho = self.compute_dtype(1.0)
                        for d in range(self.velocity_set.d):
                            _u[d] = self.compute_dtype(0.0)

                    # Handle inactive cells (refined or boundary cells)
                    has_child = wp.neon_has_child(f_handle, grid_index)
                    if _boundary_id == wp.uint8(255) or has_child:
                        _rho = self.compute_dtype(0.0)
                        for d in range(self.velocity_set.d):
                            _u[d] = self.compute_dtype(0.0)

                    # Write results to output fields
                    wp.neon_write(rho_handle, grid_index, 0, _rho)
                    for d in range(self.velocity_set.d):
                        wp.neon_write(u_handle, grid_index, d, _u[d])

                loader.declare_kernel(macroscopic_kernel)

            return macroscopic_computation

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f, bc_mask, rho, u, streamId=0):
        """
        Execute macroscopic computation for all levels in the multires grid.

        Args:
            f: Distribution function field
            bc_mask: Boundary condition mask
            rho: Density field (output)
            u: Velocity field (output)
            streamId: Stream ID for execution

        Returns:
            Tuple of (rho, u) fields
        """
        grid = f.get_grid()

        for level in range(grid.num_levels):
            computation = self.neon_container(level, f, bc_mask, rho, u)
            computation.run(streamId)

        return rho, u
