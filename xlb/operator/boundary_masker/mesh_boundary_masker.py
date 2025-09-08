# Base class for all equilibriums

import numpy as np
import warp as wp
import jax
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


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

        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This Operator is not implemented in 2D!")

        # Also using Warp kernels for JAX implementation
        if self.compute_backend == ComputeBackend.JAX:
            self.warp_functional, self.warp_kernel = self._construct_warp()

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(
        self,
        bc,
        bc_mask,
        missing_mask,
    ):
        raise NotImplementedError(f"Operation {self.__class__.__name} not implemented in JAX!")
        # Use Warp backend even for this particular operation.
        wp.init()
        bc_mask = wp.from_jax(bc_mask)
        missing_mask = wp.from_jax(missing_mask)
        bc_mask, missing_mask = self.warp_implementation(bc, bc_mask, missing_mask)
        return wp.to_jax(bc_mask), wp.to_jax(missing_mask)

    def _construct_warp(self):
        # Make constants for warp
        _c_float = self.velocity_set.c_float
        _q = wp.constant(self.velocity_set.q)
        _opp_indices = self.velocity_set.opp_indices

        @wp.func
        def index_to_position(index: wp.vec3i):
            # position of the point
            ijk = wp.vec3(wp.float32(index[0]), wp.float32(index[1]), wp.float32(index[2]))
            pos = ijk + wp.vec3(0.5, 0.5, 0.5)  # cell center
            return pos

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
                axis2 = (axis0 + 2) % 3

                sgn = 1.0
                if normal[axis2] < 0.0:
                    sgn = -1.0

                for i in range(0, 3):
                    normal_edge0[i][axis0] = -1.0 * sgn * edges[i][axis0]
                    normal_edge1[i][axis0] = sgn * edges[i][axis0]

                    dist_edge[i][axis0] = (
                        -1.0 * (normal_edge0[i][axis0] * verts[i][axis0] + normal_edge1[i][axis0] * verts[i][axis0])
                        + wp.max(0.0, normal_edge0[i][axis0])
                        + wp.max(0.0, normal_edge1[i][axis0])
                    )

            return dist1, dist2, normal_edge0, normal_edge1, dist_edge

        # Check whether this triangle intersects the unit cube at position low
        #  inputs: low: position of the cube, normal: triangle unit normal, dist1, dist2, normal_edge0, normal_edge1, dist_edge: precomputed values
        #  outputs: True if intersection, False otherwise
        #  reference: Fast parallel surface and solid voxelization on GPUs, M. Schwarz, H-P. Siedel, https://dl.acm.org/doi/10.1145/1882261.1866201
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
                        intersect = intersect and (normal_edge0[i][ax0] * low[ax0] + normal_edge1[i][ax0] * low[ax1] + dist_edge[i][ax0] >= 0.0)

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

        # Construct the warp kernel
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
            pos_bc_cell = index_to_position(index)
            half = wp.vec3(0.5, 0.5, 0.5)

            if mesh_voxel_intersect(mesh_id=mesh_id, low=pos_bc_cell - half):
                # Make solid voxel
                bc_mask[0, index[0], index[1], index[2]] = wp.uint8(255)
            else:
                # Find the fractional distance to the mesh in each direction
                for l in range(1, _q):
                    _dir = wp.vec3f(_c_float[0, l], _c_float[1, l], _c_float[2, l])

                    # Check to see if this neighbor is solid - this is super inefficient TODO: make it way better
                    if mesh_voxel_intersect(mesh_id=mesh_id, low=pos_bc_cell + _dir - half):
                        # We know we have a solid neighbor
                        # Set the boundary id and missing_mask
                        bc_mask[0, index[0], index[1], index[2]] = wp.uint8(id_number)
                        missing_mask[_opp_indices[l], index[0], index[1], index[2]] = True

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        bc,
        bc_mask,
        missing_mask,
    ):
        assert bc.mesh_vertices is not None, f'Please provide the mesh vertices for {bc.__class__.__name__} BC using keyword "mesh_vertices"!'
        assert bc.indices is None, f"Please use IndicesBoundaryMasker operator if {bc.__class__.__name__} is imposed on known indices of the grid!"
        assert bc.mesh_vertices.shape[1] == self.velocity_set.d, (
            "Mesh points must be reshaped into an array (N, 3) where N indicates number of points!"
        )
        mesh_vertices = bc.mesh_vertices
        id_number = bc.id

        # Check mesh extents against domain dimensions
        domain_shape = bc_mask.shape[1:]  # (nx, ny, nz)
        mesh_min = np.min(mesh_vertices, axis=0)
        mesh_max = np.max(mesh_vertices, axis=0)

        if any(mesh_min < 0) or any(mesh_max >= domain_shape):
            raise ValueError(
                f"Mesh extents ({mesh_min}, {mesh_max}) exceed domain dimensions {domain_shape}. The mesh must be fully contained within the domain."
            )

        # We are done with bc.mesh_vertices. Remove them from BC objects
        bc.__dict__.pop("mesh_vertices", None)

        # Ensure this masker is called only for BCs that need implicit distance to the mesh
        assert not bc.needs_mesh_distance, 'Please use "MeshDistanceBoundaryMasker" if this BC needs mesh distance!'

        mesh_indices = np.arange(mesh_vertices.shape[0])
        mesh = wp.Mesh(
            points=wp.array(mesh_vertices, dtype=wp.vec3),
            indices=wp.array(mesh_indices, dtype=int),
        )

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                mesh.id,
                id_number,
                bc_mask,
                missing_mask,
            ],
            dim=bc_mask.shape[1:],
        )

        return bc_mask, missing_mask
