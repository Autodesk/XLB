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
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry


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
        mesh_vertices=None,
    ):
        self.id = boundary_condition_registry.register_boundary_condition(self.__class__.__name__ + "_" + str(hash(self)))
        velocity_set = velocity_set or DefaultConfig.velocity_set
        precision_policy = precision_policy or DefaultConfig.default_precision_policy
        compute_backend = compute_backend or DefaultConfig.default_backend

        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set the BC indices
        self.indices = indices
        self.mesh_vertices = mesh_vertices

        # Set the implementation step
        self.implementation_step = implementation_step

        # A flag to indicate whether bc indices need to be padded in both normal directions to identify missing directions
        # when inside/outside of the geoemtry is not known
        self.needs_padding = False

        # A flag for BCs that need implicit boundary distance between the grid and a mesh (to be set to True if applicable inside each BC)
        self.needs_mesh_distance = False

        if self.compute_backend == ComputeBackend.WARP:
            # Set local constants TODO: This is a hack and should be fixed with warp update
            _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
            _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)  # TODO fix vec bool

        @wp.func
        def prepare_bc_auxilary_data(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            return f_post

        @wp.func
        def _get_thread_data(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            index: wp.vec3i,
        ):
            # Get the boundary id and missing mask
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _boundary_id = bc_mask[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of populations
                _f_pre[l] = self.compute_dtype(f_pre[l, index[0], index[1], index[2]])
                _f_post[l] = self.compute_dtype(f_post[l, index[0], index[1], index[2]])

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)
            return _f_pre, _f_post, _boundary_id, _missing_mask

        # Construct some helper warp functions for getting tid data
        if self.compute_backend == ComputeBackend.WARP:
            self._get_thread_data = _get_thread_data
            self.prepare_bc_auxilary_data = prepare_bc_auxilary_data

    @partial(jit, static_argnums=(0,), inline=True)
    def prepare_bc_auxilary_data(self, f_pre, f_post, bc_mask, missing_mask):
        """
        A placeholder function for prepare the auxilary distribution functions for the boundary condition.
        currently being called after collision only.
        """
        return f_post

    def _construct_kernel(self, functional):
        """
        Constructs the warp kernel for the boundary condition.
        The functional is specific to each boundary condition and should be passed as an argument.
        """
        _id = wp.uint8(self.id)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data(f_pre, f_post, bc_mask, missing_mask, index)

            # Apply the boundary condition
            if _boundary_id == _id:
                timestep = 0
                _f = functional(index, timestep, _missing_mask, f_pre, f_post, _f_pre, _f_post)
            else:
                _f = _f_post

            # Write the result
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1], index[2]] = self.store_dtype(_f[l])

        return kernel
