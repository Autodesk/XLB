import warp as wp
import neon, typing, copy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.grid import grid_factory
from xlb.precision_policy import Precision
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker


class MultiresBoundaryMasker(Operator):
    """
    Operator for creating boundary masks for multi-resolution grids.

    This class handles the creation of boundary condition masks across different
    refinement levels in a multires grid, ensuring proper mapping between levels.
    """

    def __init__(
        self,
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
    ):
        if compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {compute_backend} backend.")

        super().__init__(velocity_set, precision_policy, compute_backend)

        # Create boundary maskers using the WARP backend for compatibility
        self.indices_masker = IndicesBoundaryMasker(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=ComputeBackend.WARP,
        )
        self.mesh_masker = MeshBoundaryMasker(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=ComputeBackend.WARP,
        )

    def _validate_multires_grid(self, bc_mask):
        """Validate that the operation is being performed on a multires grid."""
        if bc_mask.get_grid().get_name() != "mGrid":
            raise ValueError(f"Operation {self.__class__.__name__} is only applicable to multi-resolution cases")

    def _create_level_grid_and_fields(self, xlb_grid, level):
        """Create dense grid and fields for a specific level."""
        refinement = 2**level
        grid_shape = tuple(x // refinement for x in xlb_grid.shape)
        grid_dense = grid_factory(grid_shape, compute_backend=ComputeBackend.WARP)

        missing_mask_warp = grid_dense.create_field(cardinality=self.velocity_set.q, dtype=Precision.UINT8)
        bc_mask_warp = grid_dense.create_field(cardinality=1, dtype=Precision.UINT8)

        return grid_dense, bc_mask_warp, missing_mask_warp, refinement

    def _prepare_level_boundary_conditions(self, bclist, level):
        """Prepare boundary conditions specific to a given level."""
        bclist_level = []
        for bc in bclist:
            if bc.indices is not None and bc.indices[level]:
                bc_copy = copy.copy(bc)  # Shallow copy of the whole object
                bc_copy.indices = copy.deepcopy(bc.indices[level])  # Deep copy only the modified part
                bclist_level.append(bc_copy)
        return bclist_level

    def _create_mask_transfer_container(self, bc_mask_warp, missing_mask_warp, bc_mask, missing_mask, level, refinement):
        """Create a container for transferring masks from Warp to Neon format."""

        @neon.Container.factory(name="MultiresBoundaryMaskTransfer")
        def mask_transfer_container(
            bc_mask_warp_field: typing.Any,
            missing_mask_warp_field: typing.Any,
            bc_mask_neon_field: typing.Any,
            missing_mask_neon_field: typing.Any,
        ):
            def mask_transfer_computation(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_neon_field.get_grid(), level)

                # Get field handles
                bc_mask_handle = loader.get_mres_write_handle(bc_mask_neon_field)
                missing_mask_handle = loader.get_mres_write_handle(missing_mask_neon_field)

                @wp.func
                def mask_transfer_kernel(grid_index: typing.Any):
                    # Get global coordinates
                    global_idx = wp.neon_global_idx(bc_mask_handle, grid_index)
                    global_x = wp.neon_get_x(global_idx)
                    global_y = wp.neon_get_y(global_idx)
                    global_z = wp.neon_get_z(global_idx)

                    # Transform to local coordinates for this level
                    local_x = global_x // refinement
                    local_y = global_y // refinement
                    local_z = global_z // refinement

                    # Note: XLB flattens the y dimension in 3D, while Neon uses the z dimension
                    # So we need to swap y and z coordinates
                    boundary_mask_value = bc_mask_warp_field[0, local_x, local_z, local_y]
                    wp.neon_write(bc_mask_handle, grid_index, 0, boundary_mask_value)

                    # Transfer missing mask values for all directions
                    for q in range(self.velocity_set.q):
                        missing_value = wp.uint8(missing_mask_warp_field[q, local_x, local_z, local_y])
                        wp.neon_write(missing_mask_handle, grid_index, q, missing_value)

                loader.declare_kernel(mask_transfer_kernel)

            return mask_transfer_computation

        return mask_transfer_container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bclist, bc_mask, missing_mask, start_index=None, xlb_grid=None):
        """
        Create boundary masks for all levels in a multires grid.

        Args:
            bclist: List of boundary conditions
            bc_mask: Boundary condition mask field (output)
            missing_mask: Missing direction mask field (output)
            start_index: Starting index for boundary condition IDs
            xlb_grid: The XLB grid object

        Returns:
            Tuple of (bc_mask, missing_mask) fields
        """
        self._validate_multires_grid(bc_mask)

        num_levels = bc_mask.get_grid().get_num_levels()

        for level in range(num_levels):
            # Create dense grid and fields for this level
            grid_dense, bc_mask_warp, missing_mask_warp, refinement = self._create_level_grid_and_fields(xlb_grid, level)

            # Prepare boundary conditions for this level
            bclist_level = self._prepare_level_boundary_conditions(bclist, level)

            # Apply boundary masking using Warp backend
            if bclist_level:  # Only process if there are boundary conditions for this level
                bc_mask_warp, missing_mask_warp = self.indices_masker(bclist_level, bc_mask_warp, missing_mask_warp, start_index, xlb_grid)

            # Transfer masks from Warp to Neon format
            mask_transfer_container = self._create_mask_transfer_container(bc_mask_warp, missing_mask_warp, bc_mask, missing_mask, level, refinement)

            # Execute the transfer
            transfer_computation = mask_transfer_container(bc_mask_warp, missing_mask_warp, bc_mask, missing_mask)
            transfer_computation.run(0)
            wp.synchronize()

            # Clean up temporary fields
            del bc_mask_warp
            del missing_mask_warp
            del grid_dense

        return bc_mask, missing_mask
