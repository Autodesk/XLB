# Base class for all stepper operators

from functools import partial
from jax import jit
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.collision import ForcedCollision
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker
from xlb.helper import check_bc_overlaps
from xlb.helper.nse_solver import create_nse_fields


class IncompressibleNavierStokesStepper(Stepper):
    def __init__(
        self,
        omega,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
        forcing_scheme="exact_difference",
        force_vector=None,
    ):
        super().__init__(grid, boundary_conditions)

        # Construct the collision operator
        if collision_type == "BGK":
            self.collision = BGK(omega, self.velocity_set, self.precision_policy, self.compute_backend)
        elif collision_type == "KBC":
            self.collision = KBC(omega, self.velocity_set, self.precision_policy, self.compute_backend)

        if force_vector is not None:
            self.collision = ForcedCollision(collision_operator=self.collision, forcing_scheme=forcing_scheme, force_vector=force_vector)

        # Construct the operators
        self.stream = Stream(self.velocity_set, self.precision_policy, self.compute_backend)
        self.equilibrium = QuadraticEquilibrium(self.velocity_set, self.precision_policy, self.compute_backend)
        self.macroscopic = Macroscopic(self.velocity_set, self.precision_policy, self.compute_backend)

    def prepare_fields(self, initializer=None):
        """Prepare the fields required for the stepper.

        Args:
            initializer: Optional operator to initialize the distribution functions.
                        If provided, it should be a callable that takes (grid, velocity_set,
                        precision_policy, compute_backend) as arguments and returns initialized f_0.
                        If None, default equilibrium initialization is used with rho=1 and u=0.

        Returns:
            Tuple of (f_0, f_1, bc_mask, missing_mask):
                - f_0: Initial distribution functions
                - f_1: Copy of f_0 for double-buffering
                - bc_mask: Boundary condition mask indicating which BC applies to each node
                - missing_mask: Mask indicating which populations are missing at boundary nodes
        """
        # Create fields using the helper function
        _, f_0, f_1, missing_mask, bc_mask = create_nse_fields(
            grid=self.grid, velocity_set=self.velocity_set, compute_backend=self.compute_backend, precision_policy=self.precision_policy
        )

        # Initialize distribution functions if initializer is provided
        if initializer is not None:
            f_0 = initializer(self.grid, self.velocity_set, self.precision_policy, self.compute_backend)
        else:
            from xlb.helper.initializers import initialize_eq

            f_0 = initialize_eq(f_0, self.grid, self.velocity_set, self.precision_policy, self.compute_backend)

        # Copy f_0 using backend-specific copy to f_1
        if self.compute_backend == ComputeBackend.JAX:
            f_1 = f_0.copy()
        else:
            wp.copy(f_1, f_0)

        # Process boundary conditions and update masks
        bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, bc_mask, missing_mask)
        # Initialize auxiliary data if needed
        f_0, f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_0, f_1, bc_mask, missing_mask)

        return f_0, f_1, bc_mask, missing_mask

    @classmethod
    def _process_boundary_conditions(cls, boundary_conditions, bc_mask, missing_mask):
        """Process boundary conditions and update boundary masks."""
        # Check for boundary condition overlaps
        check_bc_overlaps(boundary_conditions, DefaultConfig.velocity_set.d, DefaultConfig.default_backend)
        # Create boundary maskers
        indices_masker = IndicesBoundaryMasker(
            velocity_set=DefaultConfig.velocity_set,
            precision_policy=DefaultConfig.default_precision_policy,
            compute_backend=DefaultConfig.default_backend,
        )
        # Split boundary conditions by type
        bc_with_vertices = [bc for bc in boundary_conditions if bc.mesh_vertices is not None]
        bc_with_indices = [bc for bc in boundary_conditions if bc.indices is not None]
        # Process indices-based boundary conditions
        if bc_with_indices:
            bc_mask, missing_mask = indices_masker(bc_with_indices, bc_mask, missing_mask)
        # Process mesh-based boundary conditions for 3D
        if DefaultConfig.velocity_set.d == 3 and bc_with_vertices:
            mesh_masker = MeshBoundaryMasker(
                velocity_set=DefaultConfig.velocity_set,
                precision_policy=DefaultConfig.default_precision_policy,
                compute_backend=DefaultConfig.default_backend,
            )
            for bc in bc_with_vertices:
                bc_mask, missing_mask = mesh_masker(bc, bc_mask, missing_mask)

        return bc_mask, missing_mask

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_0, f_1, bc_mask, missing_mask):
        """Initialize auxiliary data for boundary conditions that require it."""
        for bc in boundary_conditions:
            if bc.needs_aux_init and not bc.is_initialized_with_aux_data:
                f_0, f_1 = bc.aux_data_init(f_0, f_1, bc_mask, missing_mask)
        return f_0, f_1

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
            f_post_collision = bc.update_bc_auxilary_data(f_post_stream, f_post_collision, bc_mask, missing_mask)
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

        _opp_indices = self.velocity_set.opp_indices

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
                            f_result = wp.static(self.boundary_conditions[i].update_bc_auxilary_data)(
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

        @wp.func
        def apply_aux_recovery_bc(
            index: Any,
            _boundary_id: Any,
            _missing_mask: Any,
            f_0: Any,
            _f1_thread: Any,
        ):
            # Unroll the loop over boundary conditions
            for i in range(wp.static(len(self.boundary_conditions))):
                if wp.static(self.boundary_conditions[i].needs_aux_recovery):
                    if _boundary_id == wp.static(self.boundary_conditions[i].id):
                        # Perform the swapping of data
                        for l in range(self.velocity_set.q):
                            if _missing_mask[l] == wp.uint8(1):
                                f_0[_opp_indices[l], index[0], index[1], index[2]] = self.store_dtype(_f1_thread[_opp_indices[l]])

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

            # Apply auxiliary recovery for boundary conditions (swapping)
            apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0, _f1_thread)

            # Store the result in f_1
            for l in range(self.velocity_set.q):
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
