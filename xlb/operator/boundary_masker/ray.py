import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker.mesh_boundary_masker import MeshBoundaryMasker
from xlb.operator.operator import Operator


class MeshMaskerRay(MeshBoundaryMasker):
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

            # position of the point
            cell_center_pos = self.helper_masker.index_to_position(bc_mask, index)

            # Find the fractional distance to the mesh in each direction
            for direction_idx in range(1, _q):
                direction_vec = wp.vec3f(wp.float32(_c[0, direction_idx]), wp.float32(_c[1, direction_idx]), wp.float32(_c[2, direction_idx]))
                # Max length depends on ray direction (diagonals are longer)
                max_length = wp.length(direction_vec)
                query = wp.mesh_query_ray(mesh_id, cell_center_pos, direction_vec / max_length, max_length)
                if query.result:
                    # Set the boundary id and missing_mask
                    bc_mask[0, index[0], index[1], index[2]] = wp.uint8(id_number)
                    missing_mask[_opp_indices[direction_idx], index[0], index[1], index[2]] = wp.uint8(True)

                    # If we don't need the mesh distance, we can return early
                    if not needs_mesh_distance:
                        continue

                    # get position of the mesh triangle that intersects with the ray
                    pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                    dist = wp.length(pos_mesh - cell_center_pos)
                    weight = self.store_dtype(dist / max_length)
                    distances[direction_idx, index[0], index[1], index[2]] = weight

        return None, kernel

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
