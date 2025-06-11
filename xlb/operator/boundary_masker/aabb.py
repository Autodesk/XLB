import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker.mesh_boundary_masker import MeshBoundaryMasker
from xlb.operator.operator import Operator


class MeshMaskerAABB(MeshBoundaryMasker):
    """
    Operator for creating a boundary missing_mask from an STL file
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

        # Do voxelization mesh query (warp.mesh_query_aabb) to find solid voxels
        #  - this gives an approximate 1 voxel thick surface around mesh
        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # position of the point
            pos_bc_cell = self.index_to_position(index)
            half = wp.vec3(0.5, 0.5, 0.5)

            if bc_mask[0, index[0], index[1], index[2]] == wp.uint8(255) or self.mesh_voxel_intersect(mesh_id=mesh_id, low=pos_bc_cell - half):
                # Make solid voxel
                bc_mask[0, index[0], index[1], index[2]] = wp.uint8(255)
            else:
                # Find the boundary voxels and their missing directions
                for l in range(1, _q):
                    _dir = wp.vec3f(wp.float32(_c[0, l]), wp.float32(_c[1, l]), wp.float32(_c[2, l]))

                    # Check to see if this neighbor is solid - this is super inefficient TODO: make it way better
                    if self.mesh_voxel_intersect(mesh_id=mesh_id, low=pos_bc_cell + _dir - half):
                        # We know we have a solid neighbor
                        # Set the boundary id and missing_mask
                        bc_mask[0, index[0], index[1], index[2]] = wp.uint8(id_number)
                        missing_mask[_opp_indices[l], index[0], index[1], index[2]] = True

        @wp.kernel
        def kernel_with_distance(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # position of the point
            pos_bc_cell = self.index_to_position(index)
            half = wp.vec3(0.5, 0.5, 0.5)

            if bc_mask[0, index[0], index[1], index[2]] == wp.uint8(255) or self.mesh_voxel_intersect(mesh_id=mesh_id, low=pos_bc_cell - half):
                # Make solid voxel
                bc_mask[0, index[0], index[1], index[2]] = wp.uint8(255)
            else:
                # Find the boundary voxels and their missing directions
                for l in range(1, _q):
                    _dir = wp.vec3f(wp.float32(_c[0, l]), wp.float32(_c[1, l]), wp.float32(_c[2, l]))

                    # Check to see if this neighbor is solid - this is super inefficient TODO: make it way better
                    if self.mesh_voxel_intersect(mesh_id=mesh_id, low=pos_bc_cell + _dir - half):
                        # We know we have a solid neighbor
                        # Set the boundary id and missing_mask
                        bc_mask[0, index[0], index[1], index[2]] = wp.uint8(id_number)
                        missing_mask[_opp_indices[l], index[0], index[1], index[2]] = True

                        # Find the fractional distance to the mesh in each direction
                        # We increase max_length to find intersections in neighboring cells
                        max_length = wp.length(_dir)
                        query = wp.mesh_query_ray(mesh_id, pos_bc_cell, _dir / max_length, 1.5 * max_length)
                        if query.result:
                            # get position of the mesh triangle that intersects with the ray
                            pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                            # We reduce the distance to give some wall thickness
                            dist = wp.length(pos_mesh - pos_bc_cell) - 0.5 * max_length
                            weight = self.store_dtype(dist / max_length)
                            distances[l, index[0], index[1], index[2]] = weight
                            # if weight <= 0.0 or weight > 1.0:
                            #     wp.printf("Got bad weight %f at %d,%d,%d\n", weight, index[0], index[1], index[2])
                        else:
                            # We didn't have an intersection in the given direction but we know we should so we assume the solid is slightly thicker
                            # and one lattice direction away from the BC voxel
                            distances[l, index[0], index[1], index[2]] = self.store_dtype(1.0)

        return None, [kernel, kernel_with_distance]

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
