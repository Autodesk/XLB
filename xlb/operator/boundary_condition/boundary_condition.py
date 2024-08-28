"""
Base class for boundary conditions in a LBM simulation.
"""

from enum import Enum, auto
import warp as wp
from typing import Any
from jax import jit
from functools import partial

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb import DefaultConfig


# Enum for implementation step
class ImplementationStep(Enum):
    COLLISION = auto()
    STREAMING = auto()


class BoundaryCondition(Operator):
    """
    Base class for boundary conditions in a LBM simulation.
    """

    def __init__(
        self,
        implementation_step: ImplementationStep,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_points=None,
    ):
        velocity_set = velocity_set or DefaultConfig.velocity_set
        precision_policy = precision_policy or DefaultConfig.default_precision_policy
        compute_backend = compute_backend or DefaultConfig.default_backend

        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set the BC indices
        self.indices = indices
        self.mesh_points = mesh_points

        # Set the implementation step
        self.implementation_step = implementation_step

        if self.compute_backend == ComputeBackend.WARP:
            # Set local constants TODO: This is a hack and should be fixed with warp update
            _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
            _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)  # TODO fix vec bool

        @wp.func
        def functional_postcollision(
            f_pre: Any,
            f_post: Any,
            f_aux: Any,
            missing_mask: Any,
        ):
            return f_post

        @wp.func
        def functional_poststream(
            f_pre: Any,
            f_post: Any,
            f_aux: Any,
            missing_mask: Any,
        ):
            return f_post

        @wp.func
        def _get_thread_data_2d(
            f_pre: wp.array3d(dtype=Any),
            f_post: wp.array3d(dtype=Any),
            boundary_map: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.bool),
            index: wp.vec2i,
        ):
            # Get the boundary id and missing mask
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _boundary_map = boundary_map[0, index[0], index[1]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of populations
                _f_pre[l] = f_pre[l, index[0], index[1]]
                _f_post[l] = f_post[l, index[0], index[1]]

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)
            return _f_pre, _f_post, _boundary_map, _missing_mask

        @wp.func
        def _get_thread_data_3d(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            boundary_map: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            index: wp.vec3i,
        ):
            # Get the boundary id and missing mask
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _boundary_map = boundary_map[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of populations
                _f_pre[l] = f_pre[l, index[0], index[1], index[2]]
                _f_post[l] = f_post[l, index[0], index[1], index[2]]

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)
            return _f_pre, _f_post, _boundary_map, _missing_mask

        # Construct some helper warp functions for getting tid data
        if self.compute_backend == ComputeBackend.WARP:
            self._get_thread_data_2d = _get_thread_data_2d
            self._get_thread_data_3d = _get_thread_data_3d
            self.warp_functional_poststream = functional_poststream
            self.warp_functional_postcollision = functional_postcollision

    @partial(jit, static_argnums=(0,), inline=True)
    def prepare_bc_auxilary_data(self, f_pre, f_post, boundary_map, missing_mask):
        """
        A placeholder function for prepare the auxilary distribution functions for the boundary condition.
        currently being called after collision only.
        """
        return f_post
