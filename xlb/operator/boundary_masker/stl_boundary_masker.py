# Base class for all equilibriums

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Tuple

from xlb.global_config import GlobalConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.stream.stream import Stream


class STLBoundaryMasker(Operator):
    """
    Operator for creating a boundary mask from an STL file
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.JAX,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

        # TODO: Implement this
        raise NotImplementedError

        # Make stream operator
        self.stream = Stream(velocity_set, precision_policy, compute_backend)

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(
        self, mesh, id_number, boundary_id, mask, start_index=(0, 0, 0)
    ):
        # TODO: Implement this
        raise NotImplementedError

    def _construct_warp(self):
        # Make constants for warp
        _opp_indices = wp.constant(
            self._warp_int_lattice_vec(self.velocity_set.opp_indices)
        )
        _q = wp.constant(self.velocity_set.q)
        _d = wp.constant(self.velocity_set.d)
        _id = wp.constant(self.id)

        # Construct the warp kernel
        @wp.kernel
        def _voxelize_mesh(
            voxels: wp.array3d(dtype=wp.uint8),
            mesh: wp.uint64,
            spacing: wp.vec3,
            origin: wp.vec3,
            shape: wp.vec(3, wp.uint32),
            max_length: float,
            material_id: int,
        ):
            # get index of voxel
            i, j, k = wp.tid()

            # position of voxel
            ijk = wp.vec3(wp.float(i), wp.float(j), wp.float(k))
            ijk = ijk + wp.vec3(0.5, 0.5, 0.5)  # cell center
            pos = wp.cw_mul(ijk, spacing) + origin

            # Only evaluate voxel if not set yet
            if voxels[i, j, k] != wp.uint8(0):
                return

            # evaluate distance of point
            face_index = int(0)
            face_u = float(0.0)
            face_v = float(0.0)
            sign = float(0.0)
            if wp.mesh_query_point(
                mesh, pos, max_length, sign, face_index, face_u, face_v
            ):
                p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
                delta = pos - p
                norm = wp.sqrt(wp.dot(delta, delta))

                # set point to be solid
                if norm < wp.min(spacing):
                    voxels[i, j, k] = wp.uint8(255)
                elif sign < 0:  # TODO: fix this
                    voxels[i, j, k] = wp.uint8(material_id)
                else:
                    pass

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, start_index, boundary_id, mask, id_number):
        # Reuse the jax implementation, TODO: implement a warp version
        # Convert to jax
        boundary_id = wp.jax.to_jax(boundary_id)
        mask = wp.jax.to_jax(mask)

        # Call jax implementation
        boundary_id, mask = self.jax_implementation(
            start_index, boundary_id, mask, id_number
        )

        # Convert back to warp
        boundary_id = wp.jax.to_warp(boundary_id)
        mask = wp.jax.to_warp(mask)

        return boundary_id, mask
