# Base class for all equilibriums

import numpy as np
import warp as wp
import jax
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_masker.mesh_boundary_masker import MeshBoundaryMasker


class MeshMaskerAABBFill(MeshBoundaryMasker):
    """
    Operator for creating a boundary missing_mask from an STL file
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.WARP,
        fill_in_voxels: int = 3,
    ):
        # Call super
        self.tile_half = fill_in_voxels
        self.tile_size = self.tile_half * 2 + 1
        super().__init__(velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices
        TILE_SIZE = wp.constant(self.tile_size)
        TILE_HALF = wp.constant(self.tile_half)

        # Erode the solid mask in f_field, removing a layer of outer solid voxels, storing output in f_field_out
        @wp.kernel
        def erode_tile(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any)):
            i, j, k = wp.tid()
            if (
                i < TILE_HALF
                or i >= f_field.shape[0] - TILE_HALF
                or j < TILE_HALF
                or j >= f_field.shape[1] - TILE_HALF
                or k < TILE_HALF
                or k >= f_field.shape[2] - TILE_HALF
            ):
                f_field_out[i, j, k] = f_field[i, j, k]
                return
            t = wp.tile_load(f_field, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE), offset=(i - TILE_HALF, j - TILE_HALF, k - TILE_HALF))
            min_val = wp.tile_min(t)
            f_field_out[i, j, k] = min_val[0]

        # Dilate the solid mask in f_field, adding a layer of outer solid voxels, storing output in f_field_out
        @wp.kernel
        def dilate_tile(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any)):
            i, j, k = wp.tid()
            if (
                i < TILE_HALF
                or i >= f_field.shape[0] - TILE_HALF
                or j < TILE_HALF
                or j >= f_field.shape[1] - TILE_HALF
                or k < TILE_HALF
                or k >= f_field.shape[2] - TILE_HALF
            ):
                f_field_out[i, j, k] = f_field[i, j, k]
                return
            t = wp.tile_load(f_field, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE), offset=(i - TILE_HALF, j - TILE_HALF, k - TILE_HALF))
            max_val = wp.tile_max(t)
            f_field_out[i, j, k] = max_val[0]

        # Construct the warp kernel
        # Find solid voxels that intersect the mesh
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

            # position of the point
            pos_bc_cell = self.index_to_position(index) + offset
            half = wp.vec3(0.5, 0.5, 0.5)

            if self.mesh_voxel_intersect(mesh_id=mesh_id, low=pos_bc_cell - half):
                # Make solid voxel
                solid_mask[index[0], index[1], index[2]] = wp.int32(255)

        # Assign the bc_mask based on the solid_mask we already computed
        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            solid_mask: wp.array3d(dtype=wp.uint8),
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            if solid_mask[i, j, k] == wp.uint8(255):
                # Make solid voxel
                bc_mask[0, index[0], index[1], index[2]] = wp.uint8(255)
            else:
                # Find the boundary voxels and their missing directions
                for l in range(1, _q):
                    # Check to see if this neighbor is solid - this is super inefficient TODO: make it way better
                    if solid_mask[i + _c[0, l], j + _c[1, l], k + _c[2, l]] == wp.uint8(255):
                        # We know we have a solid neighbor
                        # Set the boundary id and missing_mask
                        bc_mask[0, index[0], index[1], index[2]] = wp.uint8(id_number)
                        missing_mask[_opp_indices[l], index[0], index[1], index[2]] = True

        # Assign the bc_mask and distances based on the solid_mask we already computed
        @wp.kernel
        def kernel_with_distance(
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            solid_mask: wp.array3d(dtype=wp.uint8),
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # position of the point
            pos_bc_cell = self.index_to_position(index)

            if solid_mask[i, j, k] == wp.uint8(255) or bc_mask[0, index[0], index[1], index[2]] == wp.uint8(255):
                # Make solid voxel
                bc_mask[0, index[0], index[1], index[2]] = wp.uint8(255)
            else:
                # Find the boundary voxels and their missing directions
                for l in range(1, _q):
                    _dir = wp.vec3f(wp.float32(_c[0, l]), wp.float32(_c[1, l]), wp.float32(_c[2, l]))

                    # Check to see if this neighbor is solid - this is super inefficient TODO: make it way better
                    # if solid_mask[i,j,k] == wp.uint8(255):
                    if solid_mask[i + _c[0, l], j + _c[1, l], k + _c[2, l]] == wp.uint8(255):
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

        kernel_dict = {
            "kernel": kernel,
            "kernel_with_distance": kernel_with_distance,
            "kernel_solid": kernel_solid,
            "erode_tile": erode_tile,
            "dilate_tile": dilate_tile,
        }
        return None, kernel_dict

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
        if bc.needs_mesh_distance:
            wp.launch(
                kernel_dict["kernel_with_distance"],
                inputs=[mesh_id, bc_id, distances, bc_mask, missing_mask, solid_mask_cropped],
                dim=bc_mask.shape[1:],
            )
        else:
            wp.launch(
                kernel_dict["kernel"],
                inputs=[mesh_id, bc_id, bc_mask, missing_mask, solid_mask_cropped],
                dim=bc_mask.shape[1:],
            )

        # Resolve out of bound indices
        wp.launch(
            self.resolve_out_of_bound_kernel,
            inputs=[bc_id, bc_mask, missing_mask],
            dim=bc_mask.shape[1:],
        )
        return distances, bc_mask, missing_mask
