import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import MeshMaskerAABB
from xlb.operator.operator import Operator
import neon


class MultiresMeshMaskerAABB(MeshMaskerAABB):
    """
    Operator for creating boundary missing_mask from mesh using Axis-Aligned Bounding Box (AABB) voxelization in multiresolution simulations.

    This implementation uses warp.mesh_query_aabb for efficient mesh-voxel intersection testing,
    providing approximate 1-voxel thick surface detection around the mesh geometry.
    Suitable for scenarios where fast, approximate boundary detection is sufficient.
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.WARP,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="MeshMaskerAABB")
        def container(
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            needs_mesh_distance: Any,
            level: Any,
        ):
            def aabb_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                distances_pn = loader.get_mres_write_handle(distances)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)

                @wp.func
                def aabb_kernel(index: Any):
                    # apply the functional
                    functional(
                        index,
                        mesh_id,
                        id_number,
                        distances_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        needs_mesh_distance,
                    )

                loader.declare_kernel(aabb_kernel)

            return aabb_launcher

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
        stream=0,
    ):
        # Prepare inputs
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            # Launch the neon container
            c = self.neon_container(mesh_id, bc_id, distances, bc_mask, missing_mask, wp.static(bc.needs_mesh_distance), level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return distances, bc_mask, missing_mask
