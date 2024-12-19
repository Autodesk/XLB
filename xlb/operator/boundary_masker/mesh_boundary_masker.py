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

            vox_query = wp.mesh_query_aabb(mesh_id, pos_bc_cell - half, pos_bc_cell + half)
            face = wp.int32(0)
            if wp.mesh_query_aabb_next(vox_query, face):
                # Make solid voxel
                bc_mask[0, index[0], index[1], index[2]] = wp.uint8(255)
            else:
                # Find the fractional distance to the mesh in each direction
                for l in range(1, _q):
                    _dir = wp.vec3f(_c_float[0, l], _c_float[1, l], _c_float[2, l])

                    # Check to see if this neighbor is solid
                    vox_query_dir = wp.mesh_query_aabb(mesh_id, pos_bc_cell + _dir - half, pos_bc_cell + _dir + half)
                    face = wp.int32(0)
                    if wp.mesh_query_aabb_next(vox_query_dir, face):
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
        assert (
            bc.mesh_vertices.shape[1] == self.velocity_set.d
        ), "Mesh points must be reshaped into an array (N, 3) where N indicates number of points!"
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
