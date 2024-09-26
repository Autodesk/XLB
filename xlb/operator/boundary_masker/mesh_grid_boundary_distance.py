# Base class for all equilibriums

import numpy as np
import warp as wp
import jax
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class MeshGridBoundaryDistance(Operator):
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

        # Also using Warp kernels for JAX implementation
        if self.compute_backend == ComputeBackend.JAX:
            self.warp_functional, self.warp_kernel = self._construct_warp()

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(
        self,
        mesh_vertices,
        origin,
        spacing,
        missing_mask,
        boundary_distance,
        start_index=(0, 0, 0),
    ):
        raise NotImplementedError(f"Operation {self.__class__.__name} not implemented in JAX!")

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _q = wp.constant(self.velocity_set.q)

        @wp.func
        def index_to_position(index: wp.vec3i, origin: wp.vec3, spacing: wp.vec3):
            # position of the point
            ijk = wp.vec3(wp.float32(index[0]), wp.float32(index[1]), wp.float32(index[2]))
            ijk = ijk + wp.vec3(0.5, 0.5, 0.5)  # cell center
            pos = wp.cw_mul(ijk, spacing) + origin
            return pos

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            origin: wp.vec3,
            spacing: wp.vec3,
            missing_mask: wp.array4d(dtype=wp.bool),
            boundary_distance: wp.array4d(dtype=wp.float32),
            start_index: wp.vec3i,
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = i - start_index[0]
            index[1] = j - start_index[1]
            index[2] = k - start_index[2]

            # position of the point
            pos_solid_cell = index_to_position(index, origin, spacing)

            # Compute the maximum length
            max_length = wp.sqrt(
                (spacing[0] * wp.float32(missing_mask.shape[1])) ** 2.0
                + (spacing[1] * wp.float32(missing_mask.shape[2])) ** 2.0
                + (spacing[2] * wp.float32(missing_mask.shape[3])) ** 2.0
            )

            # evaluate if point is inside mesh
            query = wp.mesh_query_point_sign_winding_number(mesh_id, pos_solid_cell, max_length)
            if query.result:
                # set point to be solid
                if query.sign <= 0:  # TODO: fix this
                    # get position of the mesh triangle that intersects with the solid cell
                    pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)

                    # Stream indices
                    for l in range(1, _q):
                        # Get the index of the streaming direction
                        push_index = wp.vec3i()
                        for d in range(self.velocity_set.d):
                            push_index[d] = index[d] + _c[d, l]

                        # Set the boundary id and missing_mask
                        if missing_mask[l, push_index[0], push_index[1], push_index[2]]:
                            pos_fluid_cell = index_to_position(push_index, origin, spacing)
                            query = wp.mesh_query_point_sign_winding_number(mesh_id, pos_fluid_cell, max_length)
                            if query.result and query.sign > 0:
                                # get signed-distance field of the fluid voxel (i.e. sdf_f)
                                pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                                weight = wp.length(pos_fluid_cell - pos_mesh) / wp.length(pos_fluid_cell - pos_solid_cell)
                                boundary_distance[l, push_index[0], push_index[1], push_index[2]] = weight

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        mesh_vertices,
        origin,
        spacing,
        missing_mask,
        boundary_distance,
        start_index=(0, 0, 0),
    ):
        assert mesh_vertices is not None, "Please provide the mesh vertices for which the boundary_distace wrt grid is sought!"
        assert mesh_vertices.shape[1] == self.velocity_set.d, "Mesh points must be reshaped into an array (N, 3) where N indicates number of points!"
        assert (
            boundary_distance is not None and boundary_distance.shape == missing_mask.shape
        ), 'To compute "boundary_distance" for this BC a field with the same shape as "missing_mask" must be prvided!'

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
                origin,
                spacing,
                missing_mask,
                boundary_distance,
                start_index,
            ],
            dim=missing_mask.shape[1:],
        )

        return boundary_distance
