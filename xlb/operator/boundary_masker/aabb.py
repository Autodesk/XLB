import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker.mesh_boundary_masker import MeshBoundaryMasker
from xlb.operator.operator import Operator
import neon


class MeshMaskerAABB(MeshBoundaryMasker):
    """
    Operator for creating boundary missing_mask from mesh using Axis-Aligned Bounding Box (AABB) voxelization.

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

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        # Set local constants
        lattice_central_index = self.velocity_set.center_index

        @wp.func
        def functional(
            index: Any,
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            needs_mesh_distance: Any,
        ):
            # position of the point
            cell_center_pos = self.index_to_position(bc_mask, index)
            HALF_VOXEL = wp.vec3(0.5, 0.5, 0.5)

            if self.read_field(bc_mask, index, 0) == wp.uint8(255) or self.mesh_voxel_intersect(mesh_id=mesh_id, low=cell_center_pos - HALF_VOXEL):
                # Make solid voxel
                self.write_field(bc_mask, index, 0, wp.uint8(255))
            else:
                # Find the boundary voxels and their missing directions
                for direction_idx in range(_q):
                    if direction_idx == lattice_central_index:
                        # Skip the central index as it is not relevant for boundary masking
                        continue

                    # Get the lattice direction vector
                    direction_vec = wp.vec3f(wp.float32(_c[0, direction_idx]), wp.float32(_c[1, direction_idx]), wp.float32(_c[2, direction_idx]))

                    # Check to see if this neighbor is solid
                    if self.mesh_voxel_intersect(mesh_id=mesh_id, low=cell_center_pos + direction_vec - HALF_VOXEL):
                        # We know we have a solid neighbor
                        # Set the boundary id and missing_mask
                        self.write_field(bc_mask, index, 0, wp.uint8(id_number))
                        self.write_field(missing_mask, index, _opp_indices[direction_idx], wp.uint8(True))

                        # If we don't need the mesh distance, we can return early
                        if not needs_mesh_distance:
                            continue

                        # Find the fractional distance to the mesh in each direction
                        # We increase max_length to find intersections in neighboring cells
                        max_length = wp.length(direction_vec)
                        query = wp.mesh_query_ray(mesh_id, cell_center_pos, direction_vec / max_length, 1.5 * max_length)
                        if query.result:
                            # get position of the mesh triangle that intersects with the ray
                            pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                            # We reduce the distance to give some wall thickness
                            dist = wp.length(pos_mesh - cell_center_pos) - 0.5 * max_length
                            weight = dist / max_length
                            self.write_field(distances, index, direction_idx, self.store_dtype(weight))
                        else:
                            # Expected an intersection in this direction but none was found.
                            # Assume the solid extends one lattice unit beyond the BC voxel leading to a distance fraction of 1.
                            self.write_field(distances, index, direction_idx, self.store_dtype(1.0))

        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            needs_mesh_distance: bool,
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            functional(
                index,
                mesh_id,
                id_number,
                distances,
                bc_mask,
                missing_mask,
                needs_mesh_distance,
            )

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
    ):
        return self.warp_implementation_base(
            bc,
            distances,
            bc_mask,
            missing_mask,
        )

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
        ):
            def aabb_launcher(loader: neon.Loader):
                loader.set_grid(bc_mask.get_grid())
                bc_mask_pn = loader.get_write_handle(bc_mask)
                missing_mask_pn = loader.get_write_handle(missing_mask)
                distances_pn = loader.get_write_handle(distances)

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
    ):
        # Prepare inputs
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        # Launch the appropriate neon container
        c = self.neon_container(mesh_id, bc_id, distances, bc_mask, missing_mask, wp.static(bc.needs_mesh_distance))
        c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
        return distances, bc_mask, missing_mask
