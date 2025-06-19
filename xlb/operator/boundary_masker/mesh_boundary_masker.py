# Base class for mesh masker operators

import numpy as np
import warp as wp
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_masker.helper_functions_masker import HelperFunctionsMasker


class MeshBoundaryMasker(Operator):
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

        assert self.compute_backend in [ComputeBackend.WARP, ComputeBackend.NEON], (
            f"MeshBoundaryMasker is only implemented for {ComputeBackend.WARP} and {ComputeBackend.NEON} backends!"
        )

        assert self.velocity_set.d == 3, "MeshBoundaryMasker is only implemented for 3D velocity sets!"
        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This Operator is not implemented in 2D!")

        # Make constants for warp
        _c = self.velocity_set.c
        _q = self.velocity_set.q

        if self.compute_backend in [ComputeBackend.WARP, ComputeBackend.NEON]:
            # Define masker helper functions
            self.helper_masker = HelperFunctionsMasker(
                velocity_set=self.velocity_set,
                precision_policy=self.precision_policy,
                compute_backend=self.compute_backend,
            )

        @wp.func
        def out_of_bound_pull_index(
            lattice_dir: wp.int32,
            index: wp.vec3i,
            field: wp.array4d(dtype=wp.uint8),
        ):
            # Get the index of the streaming direction
            pull_index = wp.vec3i()
            for d in range(self.velocity_set.d):
                pull_index[d] = index[d] - _c[d, lattice_dir]

            # check if pull index is out of bound
            # These directions will have missing information after streaming
            missing = not self.helper_masker.is_in_bounds(pull_index, field)
            return missing

        # Function to precompute useful values per triangle, assuming spacing is (1,1,1)
        # inputs: verts: triangle vertices, normal: triangle unit normal
        # outputs: dist1, dist2, normal_edge0, normal_edge1, dist_edge
        @wp.func
        def pre_compute(
            verts: wp.mat33f,  # triangle vertices
            normal: wp.vec3f,  # triangle normal
        ):
            corner = wp.vec3f(float(normal[0] > 0.0), float(normal[1] > 0.0), float(normal[2] > 0.0))

            dist1 = wp.dot(normal, corner - verts[0])
            dist2 = wp.dot(normal, wp.vec3f(1.0, 1.0, 1.0) - corner - verts[0])

            edges = wp.transpose(wp.mat33(verts[1] - verts[0], verts[2] - verts[1], verts[0] - verts[2]))
            normal_edge0 = wp.mat33f(0.0)
            normal_edge1 = wp.mat33f(0.0)
            dist_edge = wp.mat33f(0.0)

            for axis0 in range(0, 3):
                axis1 = (axis0 + 1) % 3
                axis2 = (axis0 + 2) % 3

                sgn = 1.0
                if normal[axis2] < 0.0:
                    sgn = -1.0

                for i in range(0, 3):
                    normal_edge0[i, axis0] = -1.0 * sgn * edges[i, axis1]
                    normal_edge1[i, axis0] = sgn * edges[i, axis0]

                    dist_edge[i, axis0] = (
                        -1.0 * (normal_edge0[i, axis0] * verts[i, axis0] + normal_edge1[i, axis0] * verts[i, axis1])
                        + wp.max(0.0, normal_edge0[i, axis0])
                        + wp.max(0.0, normal_edge1[i, axis0])
                    )

            return dist1, dist2, normal_edge0, normal_edge1, dist_edge

        # Check whether this triangle intersects the unit cube at position low
        @wp.func
        def triangle_box_intersect(
            low: wp.vec3f,
            normal: wp.vec3f,
            dist1: wp.float32,
            dist2: wp.float32,
            normal_edge0: wp.mat33f,
            normal_edge1: wp.mat33f,
            dist_edge: wp.mat33f,
        ):
            if (wp.length(normal) > 0.0) and (wp.dot(normal, low) + dist1) * (wp.dot(normal, low) + dist2) <= 0.0:
                intersect = True
                #  Loop over primary axis for projection
                for ax0 in range(0, 3):
                    ax1 = (ax0 + 1) % 3
                    for i in range(0, 3):
                        intersect = intersect and (normal_edge0[i, ax0] * low[ax0] + normal_edge1[i, ax0] * low[ax1] + dist_edge[i, ax0] >= 0.0)

                return intersect
            else:
                return False

        # Check whether the unit voxel at position low intersects the warp mesh, assumes mesh has valid normals
        #  inputs: mesh_id: mesh id, low: position of the voxel
        #  outputs: True if intersection, False otherwise
        @wp.func
        def mesh_voxel_intersect(mesh_id: wp.uint64, low: wp.vec3):
            query = wp.mesh_query_aabb(mesh_id, low, low + wp.vec3f(1.0, 1.0, 1.0))

            for f in query:
                v0 = wp.mesh_eval_position(mesh_id, f, 1.0, 0.0)
                v1 = wp.mesh_eval_position(mesh_id, f, 0.0, 1.0)
                v2 = wp.mesh_eval_position(mesh_id, f, 0.0, 0.0)
                normal = wp.mesh_eval_face_normal(mesh_id, f)

                v = wp.transpose(wp.mat33f(v0, v1, v2))

                # TODO: run this on triangles in advance
                dist1, dist2, normal_edge0, normal_edge1, dist_edge = pre_compute(verts=v, normal=normal)

                if triangle_box_intersect(
                    low=low, normal=normal, dist1=dist1, dist2=dist2, normal_edge0=normal_edge0, normal_edge1=normal_edge1, dist_edge=dist_edge
                ):
                    return True

            return False

        @wp.kernel
        def resolve_out_of_bound_kernel(
            id_number: wp.int32,
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # domain shape to check for out of bounds
            domain_shape = wp.vec3i(bc_mask.shape[1], bc_mask.shape[2], bc_mask.shape[3])

            # Find the fractional distance to the mesh in each direction
            if bc_mask[0, index[0], index[1], index[2]] == wp.uint8(id_number):
                for l in range(1, _q):
                    # Ensuring out of bound pull indices are properly considered in the missing_mask
                    if out_of_bound_pull_index(l, index, missing_mask):
                        missing_mask[l, index[0], index[1], index[2]] = wp.uint8(True)

        # Construct some helper warp functions
        self.mesh_voxel_intersect = mesh_voxel_intersect
        self.resolve_out_of_bound_kernel = resolve_out_of_bound_kernel

    def _prepare_kernel_inputs(
        self,
        bc,
        bc_mask,
    ):
        assert bc.mesh_vertices is not None, f'Please provide the mesh vertices for {bc.__class__.__name__} BC using keyword "mesh_vertices"!'
        assert bc.indices is None, f"Please use IndicesBoundaryMasker operator if {bc.__class__.__name__} is imposed on known indices of the grid!"
        assert bc.mesh_vertices.shape[1] == self.velocity_set.d, (
            "Mesh points must be reshaped into an array (N, 3) where N indicates number of points!"
        )

        grid_shape = self.helper_masker.get_grid_shape(bc_mask)  # (nx, ny, nz)
        mesh_vertices = bc.mesh_vertices
        mesh_min = np.min(mesh_vertices, axis=0)
        mesh_max = np.max(mesh_vertices, axis=0)

        if any(mesh_min < 0) or any(mesh_max >= grid_shape):
            raise ValueError(
                f"Mesh extents ({mesh_min}, {mesh_max}) exceed domain dimensions {grid_shape}. The mesh must be fully contained within the domain."
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
        return mesh_id, bc_id

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(
        self,
        bc,
        bc_mask,
        missing_mask,
    ):
        raise NotImplementedError(f"Operation {self.__class__.__name__} not implemented in JAX!")

    def warp_implementation_base(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
    ):
        # Prepare inputs
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        # Launch the appropriate warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[mesh_id, bc_id, distances, bc_mask, missing_mask, wp.static(bc.needs_mesh_distance)],
            dim=bc_mask.shape[1:],
        )
        wp.launch(
            self.resolve_out_of_bound_kernel,
            inputs=[bc_id, bc_mask, missing_mask],
            dim=bc_mask.shape[1:],
        )
        return distances, bc_mask, missing_mask
