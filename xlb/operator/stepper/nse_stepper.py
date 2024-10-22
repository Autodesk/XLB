# Base class for all stepper operators

from functools import partial
from jax import jit
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.collision import ForcedCollision


class IncompressibleNavierStokesStepper(Stepper):
    def __init__(self, omega, boundary_conditions=[], collision_type="BGK", forcing_scheme="exact_difference", force_vector=None):
        velocity_set = DefaultConfig.velocity_set
        precision_policy = DefaultConfig.default_precision_policy
        compute_backend = DefaultConfig.default_backend

        # Construct the collision operator
        if collision_type == "BGK":
            self.collision = BGK(omega, velocity_set, precision_policy, compute_backend)
        elif collision_type == "KBC":
            self.collision = KBC(omega, velocity_set, precision_policy, compute_backend)

        if force_vector is not None:
            self.collision = ForcedCollision(collision_operator=self.collision, forcing_scheme=forcing_scheme, force_vector=force_vector)

        # Construct the operators
        self.stream = Stream(velocity_set, precision_policy, compute_backend)
        self.equilibrium = QuadraticEquilibrium(velocity_set, precision_policy, compute_backend)
        self.macroscopic = Macroscopic(velocity_set, precision_policy, compute_backend)

        operators = [self.macroscopic, self.equilibrium, self.collision, self.stream]

        super().__init__(operators, boundary_conditions)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_0, f_1, bc_mask, missing_mask, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """
        # Cast to compute precisioni
        f_0 = self.precision_policy.cast_to_compute_jax(f_0)
        f_1 = self.precision_policy.cast_to_compute_jax(f_1)

        # Apply streaming
        f_post_stream = self.stream(f_0)

        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                f_post_stream = bc(
                    f_0,
                    f_post_stream,
                    bc_mask,
                    missing_mask,
                )

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_post_stream)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(f_post_stream, feq, rho, u)

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            f_post_collision = bc.prepare_bc_auxilary_data(f_post_stream, f_post_collision, bc_mask, missing_mask)
            if bc.implementation_step == ImplementationStep.COLLISION:
                f_post_collision = bc(
                    f_post_stream,
                    f_post_collision,
                    bc_mask,
                    missing_mask,
                )

        # Copy back to store precision
        f_1 = self.precision_policy.cast_to_store_jax(f_post_collision)

        return f_0, f_1

    def _construct_warp(self):
        # Set local constants
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id
        id_to_bc = boundary_condition_registry.id_to_bc

        # Gather IDs of ExtrapolationOutflowBC boundary conditions
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)
        # Group active boundary conditions
        active_bcs = set(boundary_condition_registry.id_to_bc[bc.id] for bc in self.boundary_conditions)

        @wp.func
        def apply_bc(
            index: Any,
            timestep: Any,
            _boundary_id: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
            is_post_streaming: bool,
        ):
            f_result = f_post

            # Unroll the loop over boundary conditions
            for i in range(wp.static(len(self.boundary_conditions))):
                if is_post_streaming:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.STREAMING):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, missing_mask, f_0, f_1, f_pre, f_post)
                else:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.COLLISION):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, missing_mask, f_0, f_1, f_pre, f_post)
                    if wp.static(self.boundary_conditions[i].id in extrapolation_outflow_bc_ids):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].prepare_bc_auxilary_data)(
                                index, timestep, missing_mask, f_0, f_1, f_pre, f_post
                            )
            return f_result

        @wp.func
        def get_thread_data(
            f0_buffer: wp.array4d(dtype=Any),
            f1_buffer: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            index: Any,
        ):
            # Read thread data for populations
            _f0_thread = _f_vec()
            _f1_thread = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                _f0_thread[l] = self.compute_dtype(f0_buffer[l, index[0], index[1], index[2]])
                _f1_thread[l] = self.compute_dtype(f1_buffer[l, index[0], index[1], index[2]])
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            return _f0_thread, _f1_thread, _missing_mask

        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            timestep: int,
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _boundary_id = bc_mask[0, index[0], index[1], index[2]]
            if _boundary_id == wp.uint8(255):
                return

            # Apply streaming
            _f_post_stream = self.stream.warp_functional(f_0, index)

            _f0_thread, _f1_thread, _missing_mask = get_thread_data(f_0, f_1, missing_mask, index)
            _f_post_collision = _f0_thread

            # Apply post-streaming boundary conditions
            _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0, f_1, _f_post_collision, _f_post_stream, True)

            _rho, _u = self.macroscopic.warp_functional(_f_post_stream)
            _feq = self.equilibrium.warp_functional(_rho, _u)
            _f_post_collision = self.collision.warp_functional(_f_post_stream, _feq, _rho, _u)

            # Apply post-collision boundary conditions
            _f_post_collision = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0, f_1, _f_post_stream, _f_post_collision, False)

            # Store the result in f_1
            for l in range(self.velocity_set.q):
                # TODO: Improve this later
                if wp.static("GradsApproximationBC" in active_bcs):
                    if _boundary_id == wp.static(boundary_condition_registry.bc_to_id["GradsApproximationBC"]):
                        if _missing_mask[l] == wp.uint8(1):
                            f_0[_opp_indices[l], index[0], index[1], index[2]] = self.store_dtype(_f1_thread[_opp_indices[l]])
                f_1[l, index[0], index[1], index[2]] = self.store_dtype(_f_post_collision[l])

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, bc_mask, missing_mask, timestep):
        wp.launch(
            self.warp_kernel,
            inputs=[f_0, f_1, bc_mask, missing_mask, timestep],
            dim=f_0.shape[1:],
        )
        return f_0, f_1
