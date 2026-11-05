"""
AABB-Close boundary masker with morphological close operation.

Identifies solid voxels via axis-aligned bounding-box (AABB) intersection,
then applies a morphological *close* (dilate followed by erode) to fill
thin gaps and small cavities in the mesh surface.  The resulting solid
mask is used to determine boundary voxels and their missing population
directions.

Supports both Warp (single-resolution) and Neon (multi-resolution)
backends.
"""

import numpy as np
import warp as wp
import jax
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_masker.mesh_boundary_masker import MeshBoundaryMasker
from xlb.cell_type import BC_SOLID


class MeshMaskerAABBClose(MeshBoundaryMasker):
    """Boundary masker using AABB voxelization with morphological close.

    The *close* operation (dilate then erode) thickens the raw solid mask
    by ``close_voxels`` layers before shrinking it back, sealing small
    holes and thin slits in the mesh surface.

    Parameters
    ----------
    velocity_set : VelocitySet, optional
    precision_policy : PrecisionPolicy, optional
    compute_backend : ComputeBackend, optional
    close_voxels : int
        Half-width of the morphological structuring element.  Must be
        provided explicitly.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        close_voxels: int = None,
    ):
        assert close_voxels is not None, (
            "Please provide the number of close voxels using the 'close_voxels' argument! e.g., MeshVoxelizationMethod('AABB_CLOSE', close_voxels=3)"
        )
        self.close_voxels = close_voxels
        # Call super
        self.tile_half = close_voxels
        self.tile_size = self.tile_half * 2 + 1
        super().__init__(velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices
        TILE_SIZE = wp.constant(self.tile_size)
        TILE_HALF = wp.constant(self.tile_half)
        lattice_central_index = self.velocity_set.center_index

        # Erode the solid mask in mask_field, removing a layer of outer solid voxels, storing output in mask_field_out
        @wp.kernel
        def erode_tile(mask_field: wp.array3d(dtype=Any), mask_field_out: wp.array3d(dtype=Any)):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            if not self.helper_masker.is_in_bounds(index, wp.vec3i(mask_field.shape[0], mask_field.shape[1], mask_field.shape[2]), TILE_HALF):
                mask_field_out[i, j, k] = mask_field[i, j, k]
                return
            t = wp.tile_load(mask_field, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE), offset=(i - TILE_HALF, j - TILE_HALF, k - TILE_HALF))
            min_val = wp.tile_min(t)
            mask_field_out[i, j, k] = min_val[0]

        # Dilate the solid mask in mask_field, adding a layer of outer solid voxels, storing output in mask_field_out
        @wp.kernel
        def dilate_tile(mask_field: wp.array3d(dtype=Any), mask_field_out: wp.array3d(dtype=Any)):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            if not self.helper_masker.is_in_bounds(index, wp.vec3i(mask_field.shape[0], mask_field.shape[1], mask_field.shape[2]), TILE_HALF):
                mask_field_out[i, j, k] = mask_field[i, j, k]
                return
            t = wp.tile_load(mask_field, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE), offset=(i - TILE_HALF, j - TILE_HALF, k - TILE_HALF))
            max_val = wp.tile_max(t)
            mask_field_out[i, j, k] = max_val[0]

        # Erode the solid mask in mask_field, removing a layer of outer solid voxels, storing output in mask_field_out
        @wp.func
        def functional_erode(index: Any, mask_field: Any, mask_field_out: Any):
            min_val = wp.uint8(BC_SOLID)
            for l in range(_q):
                if l == lattice_central_index:
                    continue
                is_valid = wp.bool(False)
                ngh = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                ngh_val = wp.neon_read_ngh(mask_field, index, ngh, 0, wp.uint8(0), is_valid)
                if is_valid:
                    # Take the min value of all neighbors in bounds
                    min_val = wp.min(min_val, ngh_val)
            self.write_field(mask_field_out, index, 0, min_val)

        # Dilate the solid mask in mask_field, adding a layer of outer solid voxels, storing output in mask_field_out
        @wp.func
        def functional_dilate(index: Any, mask_field: Any, mask_field_out: Any):
            max_val = wp.uint8(0)
            for l in range(_q):
                if l == lattice_central_index:
                    continue
                is_valid = wp.bool(False)
                ngh = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                ngh_val = wp.neon_read_ngh(mask_field, index, ngh, 0, wp.uint8(0), is_valid)
                if is_valid:
                    max_val = wp.max(max_val, ngh_val)
            self.write_field(mask_field_out, index, 0, max_val)

        # Construct the warp kernel
        # Find solid voxels that intersect the mesh
        @wp.func
        def functional_solid(index: Any, mesh_id: Any, solid_mask: Any, offset: Any):
            # position of the point
            cell_center_pos = self.helper_masker.index_to_position(solid_mask, index) + offset
            half = wp.vec3(0.5, 0.5, 0.5)

            if self.mesh_voxel_intersect(mesh_id=mesh_id, low=cell_center_pos - half):
                # Make solid voxel
                self.write_field(solid_mask, index, 0, wp.uint8(BC_SOLID))

        @wp.kernel
        def kernel_solid(
            mesh_id: wp.uint64,
            solid_mask: wp.array3d(dtype=wp.int32),
            offset: wp.vec3f,
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            functional_solid(index, mesh_id, solid_mask, offset)

            return

        @wp.func
        def functional_aabb(
            index: Any,
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            solid_mask: wp.array3d(dtype=wp.uint8),
            needs_mesh_distance: bool,
        ):
            # position of the point
            cell_center_pos = self.helper_masker.index_to_position(bc_mask, index)
            HALF_VOXEL = wp.vec3(0.5, 0.5, 0.5)

            if self.read_field(solid_mask, index, 0) == wp.uint8(BC_SOLID) or self.read_field(bc_mask, index, 0) == wp.uint8(BC_SOLID):
                # Make solid voxel
                self.write_field(bc_mask, index, 0, wp.uint8(BC_SOLID))
            else:
                # Find the boundary voxels and their missing directions
                for direction_idx in range(_q):
                    if direction_idx == lattice_central_index:
                        # Skip the central index as it is not relevant for boundary masking
                        continue

                    # Get the lattice direction vector
                    direction_vec = wp.vec3f(wp.float32(_c[0, direction_idx]), wp.float32(_c[1, direction_idx]), wp.float32(_c[2, direction_idx]))

                    # Check to see if this neighbor is solid
                    if self.helper_masker.is_in_bounds(index, wp.vec3i(solid_mask.shape[0], solid_mask.shape[1], solid_mask.shape[2]), 1):
                        if self.read_field(solid_mask, index + direction_idx, 0) == wp.uint8(BC_SOLID):
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

        # Assign the bc_mask and distances based on the solid_mask we already computed
        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            solid_mask: wp.array3d(dtype=wp.uint8),
            needs_mesh_distance: bool,
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # position of the point
            cell_center_pos = self.helper_masker.index_to_position(bc_mask, index)

            if solid_mask[i, j, k] == wp.uint8(BC_SOLID) or bc_mask[0, index[0], index[1], index[2]] == wp.uint8(BC_SOLID):
                # Make solid voxel
                bc_mask[0, index[0], index[1], index[2]] = wp.uint8(BC_SOLID)
            else:
                # Find the boundary voxels and their missing directions
                for direction_idx in range(_q):
                    if direction_idx == lattice_central_index:
                        # Skip the central index as it is not relevant for boundary masking
                        continue
                    direction_vec = wp.vec3f(wp.float32(_c[0, direction_idx]), wp.float32(_c[1, direction_idx]), wp.float32(_c[2, direction_idx]))

                    # Check to see if this neighbor is solid - this is super inefficient TODO: make it way better
                    # if solid_mask[i,j,k] == wp.uint8(BC_SOLID):
                    if solid_mask[i + _c[0, direction_idx], j + _c[1, direction_idx], k + _c[2, direction_idx]] == wp.uint8(BC_SOLID):
                        # We know we have a solid neighbor
                        # Set the boundary id and missing_mask
                        bc_mask[0, index[0], index[1], index[2]] = wp.uint8(id_number)
                        missing_mask[_opp_indices[direction_idx], index[0], index[1], index[2]] = wp.uint8(True)

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
                            weight = self.store_dtype(dist / max_length)
                            distances[direction_idx, index[0], index[1], index[2]] = weight
                        else:
                            # We didn't have an intersection in the given direction but we know we should so we assume the solid is slightly thicker
                            # and one lattice direction away from the BC voxel
                            distances[direction_idx, index[0], index[1], index[2]] = self.store_dtype(1.0)

        functional_dict = {
            "functional_erode": functional_erode,
            "functional_dilate": functional_dilate,
            "functional_solid": functional_solid,
            "functional_aabb": functional_aabb,
        }
        kernel_dict = {
            "kernel": kernel,
            "kernel_solid": kernel_solid,
            "erode_tile": erode_tile,
            "dilate_tile": dilate_tile,
        }
        return functional_dict, kernel_dict

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
    ):
        assert bc.mesh_vertices is not None, f'Please provide the mesh vertices for {bc.__class__.__name__} BC using keyword "mesh_vertices"!'
        assert bc.indices is None, f"Please use IndicesBoundaryMasker operator if {bc.__class__.__name__} is imposed on known indices of the grid!"
        assert bc.mesh_vertices.shape[1] == self.velocity_set.d, (
            "Mesh points must be reshaped into an array (N, 3) where N indicates number of points!"
        )

        domain_shape = bc_mask.shape[1:]  # (nx, ny, nz)
        mesh_vertices = bc.mesh_vertices
        mesh_min = np.min(mesh_vertices, axis=0)
        mesh_max = np.max(mesh_vertices, axis=0)

        if any(mesh_min < 0) or any(mesh_max >= domain_shape):
            raise ValueError(
                f"Mesh extents ({mesh_min}, {mesh_max}) exceed domain dimensions {domain_shape}. The mesh must be fully contained within the domain."
            )

        # We are done with bc.mesh_vertices. Remove them from BC objects
        bc.__dict__.pop("mesh_vertices", None)

        mesh_indices = np.arange(mesh_vertices.shape[0])
        mesh = wp.Mesh(
            points=wp.array(mesh_vertices, dtype=wp.vec3),
            indices=wp.array(mesh_indices, dtype=wp.int32),
        )
        mesh_id = wp.uint64(mesh.id)
        bc_id = bc.id

        # Create a padded mask for the solid voxels to account for the tile size
        # It needs to be padded by twice the tile size on each side since we run two tile operations
        tile_length = 2 * self.tile_half
        offset = wp.vec3f(-tile_length, -tile_length, -tile_length)
        pad = 2 * tile_length
        nx, ny, nz = domain_shape
        solid_mask = wp.zeros((nx + pad, ny + pad, nz + pad), dtype=wp.int32)
        solid_mask_out = wp.zeros((nx + pad, ny + pad, nz + pad), dtype=wp.int32)

        # Prepare the warp kernel dictionary
        kernel_dict = self.warp_kernel

        # Launch all required kernels for creating the solid mask
        wp.launch(
            kernel=kernel_dict["kernel_solid"],
            inputs=[
                mesh_id,
                solid_mask,
                offset,
            ],
            dim=solid_mask.shape,
        )
        wp.launch_tiled(
            kernel=kernel_dict["dilate_tile"],
            dim=solid_mask.shape,
            block_dim=32,
            inputs=[solid_mask, solid_mask_out],
        )
        wp.launch_tiled(
            kernel=kernel_dict["erode_tile"],
            dim=solid_mask.shape,
            block_dim=32,
            inputs=[solid_mask_out, solid_mask],
        )
        solid_mask_cropped = wp.array(
            solid_mask[tile_length:-tile_length, tile_length:-tile_length, tile_length:-tile_length],
            dtype=wp.uint8,
        )

        # Launch the main kernel for boundary masker
        wp.launch(
            kernel_dict["kernel"],
            inputs=[mesh_id, bc_id, distances, bc_mask, missing_mask, solid_mask_cropped, wp.static(bc.needs_mesh_distance)],
            dim=bc_mask.shape[1:],
        )

        # Resolve out of bound indices
        wp.launch(
            self.resolve_out_of_bound_kernel,
            inputs=[bc_id, bc_mask, missing_mask],
            dim=bc_mask.shape[1:],
        )
        return distances, bc_mask, missing_mask
