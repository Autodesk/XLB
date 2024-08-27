# Base class for all equilibriums

import numpy as np
from stl import mesh as np_mesh
import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class STLBoundaryMasker(Operator):
    """
    Operator for creating a boundary missing_mask from an STL file
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.JAX,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(self, stl_file, origin, spacing, id_number, boundary_id, missing_mask, start_index=(0, 0, 0)):
        # Use Warp backend even for this particular operation.
        boundary_id, missing_mask = self.warp_implementation(stl_file, origin, spacing, id_number, boundary_id, missing_mask, start_index=(0, 0, 0))
        return wp.to_jax(boundary_id), wp.to_jax(missing_mask)

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.wp_c
        _q = wp.constant(self.velocity_set.q)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            origin: wp.vec3,
            spacing: wp.vec3,
            id_number: wp.int32,
            boundary_id: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
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
            ijk = wp.vec3(wp.float32(index[0]), wp.float32(index[1]), wp.float32(index[2]))
            ijk = ijk + wp.vec3(0.5, 0.5, 0.5)  # cell center
            pos = wp.cw_mul(ijk, spacing) + origin

            # Compute the maximum length
            max_length = wp.sqrt(
                (spacing[0] * wp.float32(boundary_id.shape[1])) ** 2.0
                + (spacing[1] * wp.float32(boundary_id.shape[2])) ** 2.0
                + (spacing[2] * wp.float32(boundary_id.shape[3])) ** 2.0
            )

            # evaluate if point is inside mesh
            face_index = int(0)
            face_u = float(0.0)
            face_v = float(0.0)
            sign = float(0.0)
            if wp.mesh_query_point_sign_winding_number(mesh_id, pos, max_length, sign, face_index, face_u, face_v):
                # set point to be solid
                if sign <= 0:  # TODO: fix this
                    # Stream indices
                    for l in range(_q):
                        # Get the index of the streaming direction
                        push_index = wp.vec3i()
                        for d in range(self.velocity_set.d):
                            push_index[d] = index[d] + _c[d, l]

                        # Set the boundary id and missing_mask
                        boundary_id[0, push_index[0], push_index[1], push_index[2]] = wp.uint8(id_number)
                        missing_mask[l, push_index[0], push_index[1], push_index[2]] = True

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        stl_file,
        origin,
        spacing,
        id_number,
        boundary_id,
        missing_mask,
        start_index=(0, 0, 0),
    ):
        # Load the mesh
        mesh = np_mesh.Mesh.from_file(stl_file)
        mesh_points = mesh.points.reshape(-1, 3)
        mesh_indices = np.arange(mesh_points.shape[0])
        mesh = wp.Mesh(
            points=wp.array(mesh_points, dtype=wp.vec3),
            indices=wp.array(mesh_indices, dtype=int),
        )

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                mesh.id,
                origin,
                spacing,
                id_number,
                boundary_id,
                missing_mask,
                start_index,
            ],
            dim=boundary_id.shape[1:],
        )

        return boundary_id, missing_mask
