import warp as wp
import neon, typing, copy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.grid import grid_factory
from xlb.precision_policy import Precision
from xlb.operator.boundary_masker import (
    IndicesBoundaryMasker,
    MeshVoxelizationMethod,
    MeshMaskerAABB,
    MeshMaskerRay,
    MeshMaskerWinding,
    MeshMaskerAABBFill,
)


class MultiresBoundaryMasker(Operator):
    """
    Operator for creating a boundary mask for multi-resolution grids
    """

    def __init__(
        self,
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
    ):
        if compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name} not supported in {compute_backend} backend.")

        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Create boundary maskers using the WARP backend
        self.indices_masker = IndicesBoundaryMasker(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=ComputeBackend.WARP,
        )

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bclist, f_1, bc_mask, missing_mask, start_index=None, xlb_grid=None):
        # Ensure that this operator is called on multires grids
        assert bc_mask.get_grid().get_name() == "mGrid", f"Operation {self.__class__.__name} is only applicable to multi-resolution cases"

        # Make constants
        _d = self.velocity_set.d

        # number of levels
        num_levels = bc_mask.get_grid().get_num_levels()
        for level in range(num_levels):
            # Use the warp backend to create dense fields to be written in multi-res NEON fields
            refinement = 2**level
            grid_shape = tuple(x // refinement for x in xlb_grid.shape)
            grid_dense = grid_factory(grid_shape, compute_backend=ComputeBackend.WARP)
            missing_mask_warp = grid_dense.create_field(cardinality=self.velocity_set.q, dtype=Precision.UINT8)
            bc_mask_warp = grid_dense.create_field(cardinality=1, dtype=Precision.UINT8)
            f_1_warp = grid_dense.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)

            # Set local constants
            lattice_central_index = self.velocity_set.center_index

            # create a new bclist for this level only
            bc_with_indices = []
            for bc in bclist:
                if bc.indices is not None and bc.indices[level]:
                    bc_copy = copy.copy(bc)  # shallow copy of the whole object
                    bc_copy.indices = copy.deepcopy(bc.indices[level])  # deep copy only the modified part
                    bc_with_indices.append(bc_copy)
                elif bc.mesh_vertices is not None:
                    bc_copy = copy.copy(bc)  # shallow copy of the whole object
                    bc_copy.mesh_vertices = copy.deepcopy(bc.mesh_vertices) / refinement

                    # call mesh masker for this bc at this level
                    if bc.voxelization_method is MeshVoxelizationMethod.AABB:
                        mesh_masker = MeshMaskerAABB(
                            velocity_set=self.velocity_set,
                            precision_policy=self.precision_policy,
                            compute_backend=ComputeBackend.WARP,
                        )
                    elif bc.voxelization_method is MeshVoxelizationMethod.RAY:
                        mesh_masker = MeshMaskerRay(
                            velocity_set=self.velocity_set,
                            precision_policy=self.precision_policy,
                            compute_backend=ComputeBackend.WARP,
                        )
                    elif bc.voxelization_method is MeshVoxelizationMethod.WINDING:
                        mesh_masker = MeshMaskerWinding(
                            velocity_set=self.velocity_set,
                            precision_policy=self.precision_policy,
                            compute_backend=ComputeBackend.WARP,
                        )
                    elif bc.voxelization_method is MeshVoxelizationMethod.AABB_FILL:
                        mesh_masker = MeshMaskerAABBFill(
                            velocity_set=self.velocity_set,
                            precision_policy=self.precision_policy,
                            compute_backend=ComputeBackend.WARP,
                        )
                    else:
                        raise ValueError(f"Unsupported voxelization method: {bc.voxelization_method}")
                    f_1_warp, bc_mask_warp, missing_mask_warp = mesh_masker(bc_copy, f_1_warp, bc_mask_warp, missing_mask_warp)

            # call indices masker for all BC's with indices at this level
            bc_mask_warp, missing_mask_warp = self.indices_masker(bc_with_indices, bc_mask_warp, missing_mask_warp, start_index)

            @neon.Container.factory(name="MultiresBoundaryMasker")
            def container(
                f_1_warp: typing.Any,
                bc_mask_warp: typing.Any,
                missing_mask_warp: typing.Any,
                f_1_field: typing.Any,
                bc_mask_field: typing.Any,
                missing_mask_field: typing.Any,
            ):
                def loading_step(loader: neon.Loader):
                    loader.set_mres_grid(bc_mask_field.get_grid(), level)
                    f_1_hdl = loader.get_mres_write_handle(f_1_field)
                    bc_mask_hdl = loader.get_mres_write_handle(bc_mask_field)
                    missing_mask_hdl = loader.get_mres_write_handle(missing_mask_field)

                    @wp.func
                    def masker(gridIdx: typing.Any):
                        cIdx = wp.neon_global_idx(bc_mask_hdl, gridIdx)
                        # get local indices by dividing the global indices (associated with the finest level) by 2^level
                        lx = wp.neon_get_x(cIdx) // refinement
                        ly = wp.neon_get_y(cIdx) // refinement
                        lz = wp.neon_get_z(cIdx) // refinement

                        # TODO@Max - XLB is flattening the z dimension in 3D, while neon uses the y dimension
                        if _d == 2:
                            ly, lz = lz, ly

                        local_mask = bc_mask_warp[0, lx, ly, lz]
                        wp.neon_write(bc_mask_hdl, gridIdx, 0, local_mask)

                        for q in range(self.velocity_set.q):
                            is_missing = wp.uint8(missing_mask_warp[q, lx, ly, lz])
                            wp.neon_write(missing_mask_hdl, gridIdx, q, is_missing)

                            if q != lattice_central_index and is_missing == wp.uint8(False):
                                wp.neon_write(f_1_hdl, gridIdx, q, f_1_warp[q, lx, ly, lz])

                    loader.declare_kernel(masker)

                return loading_step

            c = container(f_1_warp, bc_mask_warp, missing_mask_warp, f_1, bc_mask, missing_mask)
            c.run(0)
            wp.synchronize()

            del f_1_warp
            del bc_mask_warp
            del missing_mask_warp

        return f_1, bc_mask, missing_mask
