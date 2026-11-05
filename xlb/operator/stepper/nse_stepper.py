"""
Single-resolution incompressible Navier-Stokes stepper.

Implements the full LBM step (stream, collide, apply BCs) for a single-
resolution grid.  Supports pull and push streaming schemes on JAX, a
pull-only fused kernel on Warp, and a pull-only Neon container.
"""

from functools import partial

from jax import jit
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC, SmagorinskyLESBGK
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.collision import ForcedCollision
from xlb.operator.boundary_masker import (
    IndicesBoundaryMasker,
    MeshVoxelizationMethod,
    MeshMaskerAABB,
    MeshMaskerRay,
    MeshMaskerWinding,
    MeshMaskerAABBClose,
)
from xlb.helper import check_bc_overlaps
from xlb.helper.nse_fields import create_nse_fields
from xlb.operator.boundary_condition.helper_functions_bc import EncodeAuxiliaryData
from xlb.cell_type import BC_SOLID


class IncompressibleNavierStokesStepper(Stepper):
    """Single-resolution incompressible Navier-Stokes LBM stepper.

    Composes streaming, collision, equilibrium, macroscopic, and boundary-
    condition operators into a complete timestep.

    Parameters
    ----------
    grid : Grid
        Computational grid.
    boundary_conditions : list of BoundaryCondition
        Boundary conditions to apply each step.
    collision_type : str
        ``"BGK"``, ``"KBC"``, or ``"SmagorinskyLESBGK"``.
    streaming_scheme : str
        ``"pull"`` (default) or ``"push"`` (JAX only).
    forcing_scheme : str
        Forcing scheme name (used when *force_vector* is given).
    force_vector : array-like, optional
        External body force vector.
    backend_config : dict
        Backend-specific options (e.g. Neon OCC configuration).
    """

    def __init__(
        self,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
        streaming_scheme="pull",
        forcing_scheme="exact_difference",
        force_vector=None,
        backend_config={},
    ):
        super().__init__(grid, boundary_conditions)
        self.backend_config = backend_config

        # Construct the collision operator
        if collision_type == "BGK":
            self.collision = BGK(self.velocity_set, self.precision_policy, self.compute_backend)
        elif collision_type == "KBC":
            self.collision = KBC(self.velocity_set, self.precision_policy, self.compute_backend)
        elif collision_type == "SmagorinskyLESBGK":
            self.collision = SmagorinskyLESBGK(self.velocity_set, self.precision_policy, self.compute_backend)

        if force_vector is not None:
            self.collision = ForcedCollision(collision_operator=self.collision, forcing_scheme=forcing_scheme, force_vector=force_vector)

        # Choose the implementation based on backend and streaming scheme
        self.streaming_scheme = streaming_scheme
        if self.compute_backend != ComputeBackend.JAX:
            assert streaming_scheme == "pull", f"Unknown or unimplemented streaming scheme for backend: {self.compute_backend}"

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

        # Copy f_0 using backend-specific copy to f_1
        if self.compute_backend == ComputeBackend.JAX:
            f_1 = f_0.copy()
        if self.compute_backend == ComputeBackend.WARP:
            wp.copy(f_1, f_0)
        if self.compute_backend == ComputeBackend.NEON:
            f_1.copy_from_run(f_0, 0)

        # Important note: XLB uses f_1 buffer (center index and missing directions) to store auxiliary data for boundary conditions.
        # Process boundary conditions and update masks
        f_1, bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, f_1, bc_mask, missing_mask)

        # Initialize auxiliary data if needed
        f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_1, bc_mask, missing_mask)
        # bc_mask.update_host(0)
        # missing_mask.update_host(0)
        wp.synchronize()
        # bc_mask.export_vti("bc_mask.vti", 'bc_mask')
        # missing_mask.export_vti("missing_mask.vti", 'missing_mask')

        # Initialize distribution functions if initializer is provided
        if initializer is not None:
            f_0 = initializer(bc_mask, f_0)
        else:
            from xlb.helper.initializers import initialize_eq

            f_0 = initialize_eq(f_0, self.grid, self.velocity_set, self.precision_policy, self.compute_backend)

        return f_0, f_1, bc_mask, missing_mask

    def _process_boundary_conditions(self, boundary_conditions, f_1, bc_mask, missing_mask):
        """Process boundary conditions and update boundary masks."""

        # Check for boundary condition overlaps
        check_bc_overlaps(boundary_conditions, DefaultConfig.velocity_set.d, DefaultConfig.default_backend)

        # Create boundary maskers
        indices_masker = IndicesBoundaryMasker(
            velocity_set=DefaultConfig.velocity_set,
            precision_policy=DefaultConfig.default_precision_policy,
            compute_backend=DefaultConfig.default_backend,
            grid=self.grid,
        )

        # Split boundary conditions by type
        bc_with_vertices = [bc for bc in boundary_conditions if bc.mesh_vertices is not None]
        bc_with_indices = [bc for bc in boundary_conditions if bc.indices is not None]

        # Process indices-based boundary conditions
        if bc_with_indices:
            bc_mask, missing_mask = indices_masker(bc_with_indices, bc_mask, missing_mask)

        # Process mesh-based boundary conditions for 3D
        if DefaultConfig.velocity_set.d == 3 and bc_with_vertices:
            for bc in bc_with_vertices:
                if bc.voxelization_method.id is MeshVoxelizationMethod("AABB").id:
                    mesh_masker = MeshMaskerAABB(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                elif bc.voxelization_method.id is MeshVoxelizationMethod("RAY").id:
                    mesh_masker = MeshMaskerRay(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                elif bc.voxelization_method.id is MeshVoxelizationMethod("WINDING").id:
                    mesh_masker = MeshMaskerWinding(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                elif bc.voxelization_method.id is MeshVoxelizationMethod("AABB_CLOSE").id:
                    mesh_masker = MeshMaskerAABBClose(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                        close_voxels=bc.voxelization_method.options.get("close_voxels"),
                    )
                else:
                    raise ValueError(f"Unsupported voxelization method: {bc.voxelization_method}")
                # Apply the mesh masker to the boundary condition
                f_1, bc_mask, missing_mask = mesh_masker(bc, f_1, bc_mask, missing_mask)

        return f_1, bc_mask, missing_mask

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_1, bc_mask, missing_mask):
        """Initialize auxiliary data for boundary conditions that require it."""
        for bc in boundary_conditions:
            if bc.needs_aux_init and not bc.is_initialized_with_aux_data:
                # Create the encoder operator for storing the auxiliary data
                encode_auxiliary_data = EncodeAuxiliaryData(
                    bc.id,
                    bc.num_of_aux_data,
                    bc.profile,
                    velocity_set=bc.velocity_set,
                    precision_policy=bc.precision_policy,
                    compute_backend=bc.compute_backend,
                )

                # Encode the auxiliary data in f_1
                f_1 = encode_auxiliary_data(f_1, bc_mask, missing_mask)
                bc.is_initialized_with_aux_data = True
        return f_1

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        if self.streaming_scheme == "pull":
            return self.jax_implementation_pull(f_0, f_1, bc_mask, missing_mask, omega, timestep)
        elif self.streaming_scheme == "push":
            return self.jax_implementation_push(f_0, f_1, bc_mask, missing_mask, omega, timestep)
        else:
            raise ValueError(f"Unknown streaming scheme: {self.streaming_scheme}")

    @partial(jit, static_argnums=(0,))
    def jax_implementation_pull(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """
        # Cast to compute precision
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
        f_post_collision = self.collision(f_post_stream, feq, omega)

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            f_post_collision = bc.assemble_auxiliary_data(f_post_stream, f_post_collision, bc_mask, missing_mask)
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

    @partial(jit, static_argnums=(0,))
    def jax_implementation_push(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """
        # Cast to compute precision
        f_0 = self.precision_policy.cast_to_compute_jax(f_0)
        f_1 = self.precision_policy.cast_to_compute_jax(f_1)

        # Assign f_post_stream
        f_post_stream = f_0

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_post_stream)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(f_post_stream, feq, omega)

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            f_post_collision = bc.update_bc_auxiliary_data(f_post_stream, f_post_collision, bc_mask, missing_mask)
            if bc.implementation_step == ImplementationStep.COLLISION:
                f_post_collision = bc(
                    f_post_stream,
                    f_post_collision,
                    bc_mask,
                    missing_mask,
                )

        # Apply streaming
        f_post_stream = self.stream(f_post_collision)

        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                f_post_stream = bc(
                    f_post_collision,
                    f_post_stream,
                    bc_mask,
                    missing_mask,
                )

        # Copy back to store precision
        f_0 = self.precision_policy.cast_to_store_jax(f_post_collision)
        f_1 = self.precision_policy.cast_to_store_jax(f_post_stream)

        return f_0, f_1

    def _construct_warp(self):
        # Set local constants
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices
        lattice_central_index = self.velocity_set.center_index

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id

        # Gather IDs of ExtrapolationOutflowBC boundary conditions
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)

        @wp.func
        def apply_bc(
            index: Any,
            timestep: Any,
            _boundary_id: Any,
            _missing_mask: Any,
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
                            f_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, _missing_mask, f_0, f_1, f_pre, f_post)
                else:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.COLLISION):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, _missing_mask, f_0, f_1, f_pre, f_post)
                    if wp.static(self.boundary_conditions[i].id in extrapolation_outflow_bc_ids):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].assemble_auxiliary_data)(
                                index, timestep, _missing_mask, f_0, f_1, f_pre, f_post
                            )
            return f_result

        @wp.func
        def get_thread_data(
            f0_buffer: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            index: Any,
        ):
            # Read thread data for populations
            _f0_thread = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                _f0_thread[l] = self.compute_dtype(f0_buffer[l, index[0], index[1], index[2]])
                _missing_mask[l] = missing_mask[l, index[0], index[1], index[2]]

            return _f0_thread, _missing_mask

        @wp.func
        def apply_aux_recovery_bc(
            index: Any,
            _boundary_id: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
        ):
            # Note:
            # In XLB, the BC auxiliary data (e.g. prescribed values of pressure or normal velocity) are stored in (i) central index of f_1 and/or
            # (ii) missing directions of f_1. Some BCs may or may not need all these available storage space. This function checks whether
            # the BC needs recovery of auxiliary data and then recovers the information for the next iteration (due to buffer swapping) by
            # writting the values of f_1 into f_0.

            # Unroll the loop over boundary conditions
            for i in range(wp.static(len(self.boundary_conditions))):
                if wp.static(self.boundary_conditions[i].needs_aux_recovery):
                    if _boundary_id == wp.static(self.boundary_conditions[i].id):
                        for l in range(self.velocity_set.q):
                            # Perform the swapping of data
                            if l == lattice_central_index:
                                # (i) Recover the values stored in the central index of f_1
                                f_0[l, index[0], index[1], index[2]] = self.store_dtype(f_1[l, index[0], index[1], index[2]])
                            elif _missing_mask[l] == wp.uint8(1):
                                # (ii) Recover the values stored in the missing directions of f_1
                                f_0[_opp_indices[l], index[0], index[1], index[2]] = self.store_dtype(
                                    f_1[_opp_indices[l], index[0], index[1], index[2]]
                                )

        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            omega: Any,
            timestep: int,
        ):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            _boundary_id = bc_mask[0, index[0], index[1], index[2]]
            if _boundary_id == wp.uint8(BC_SOLID):
                return

            # Apply streaming
            _f_post_stream = self.stream.warp_functional(f_0, index)

            _f0_thread, _missing_mask = get_thread_data(f_0, missing_mask, index)
            _f_post_collision = _f0_thread

            # Apply post-streaming boundary conditions
            _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0, f_1, _f_post_collision, _f_post_stream, True)

            _rho, _u = self.macroscopic.warp_functional(_f_post_stream)
            _feq = self.equilibrium.warp_functional(_rho, _u)
            _f_post_collision = self.collision.warp_functional(_f_post_stream, _feq, omega)

            # Apply post-collision boundary conditions
            _f_post_collision = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0, f_1, _f_post_stream, _f_post_collision, False)

            # Apply auxiliary recovery for boundary conditions (swapping)
            apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0, f_1)

            # Store the result in f_1
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1], index[2]] = self.store_dtype(_f_post_collision[l])

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        wp.launch(
            self.warp_kernel,
            inputs=[f_0, f_1, bc_mask, missing_mask, omega, timestep],
            dim=f_0.shape[1:],
            device=f_0.device,
        )
        return f_0, f_1

    def _construct_neon(self):
        import neon

        # Set local constants
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices
        lattice_central_index = self.velocity_set.center_index

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id

        # Gather IDs of ExtrapolationOutflowBC boundary conditions
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)

        @wp.func
        def apply_bc(
            index: Any,
            timestep: Any,
            _boundary_id: Any,
            _missing_mask: Any,
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
                            f_result = wp.static(self.boundary_conditions[i].neon_functional)(index, timestep, _missing_mask, f_0, f_1, f_pre, f_post)
                else:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.COLLISION):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].neon_functional)(index, timestep, _missing_mask, f_0, f_1, f_pre, f_post)
                    if wp.static(self.boundary_conditions[i].id in extrapolation_outflow_bc_ids):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].assemble_auxiliary_data)(
                                index, timestep, _missing_mask, f_0, f_1, f_pre, f_post
                            )
            return f_result

        @wp.func
        def neon_get_thread_data(
            f0_pn: Any,
            missing_mask_pn: Any,
            index: Any,
        ):
            # Read thread data for populations
            _f0_thread = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                _f0_thread[l] = self.compute_dtype(wp.neon_read(f0_pn, index, l))
                _missing_mask[l] = wp.neon_read(missing_mask_pn, index, l)

            return _f0_thread, _missing_mask

        @wp.func
        def neon_apply_aux_recovery_bc(
            index: Any,
            _boundary_id: Any,
            _missing_mask: Any,
            f_0_pn: Any,
            f_1_pn: Any,
        ):
            # Note:
            # In XLB, the BC auxiliary data (e.g. prescribed values of pressure or normal velocity) are stored in (i) central index of f_1 and/or
            # (ii) missing directions of f_1. Some BCs may or may not need all these available storage space. This function checks whether
            # the BC needs recovery of auxiliary data and then recovers the information for the next iteration (due to buffer swapping) by
            # writting the values of f_1 into f_0.

            # Unroll the loop over boundary conditions
            for i in range(wp.static(len(self.boundary_conditions))):
                if wp.static(self.boundary_conditions[i].needs_aux_recovery):
                    if _boundary_id == wp.static(self.boundary_conditions[i].id):
                        for l in range(self.velocity_set.q):
                            # Perform the swapping of data
                            if l == lattice_central_index:
                                # (i) Recover the values stored in the central index of f_1
                                _f1_thread = wp.neon_read(f_1_pn, index, l)
                                wp.neon_write(f_0_pn, index, l, self.store_dtype(_f1_thread))
                            elif _missing_mask[l] == wp.uint8(1):
                                # (ii) Recover the values stored in the missing directions of f_1
                                _f1_thread = wp.neon_read(f_1_pn, index, _opp_indices[l])
                                wp.neon_write(f_0_pn, index, _opp_indices[l], self.store_dtype(_f1_thread))

        @neon.Container.factory(name="nse_stepper")
        def container(
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: int,
        ):
            def nse_stepper_ll(loader: neon.Loader):
                loader.set_grid(bc_mask_fd.get_grid())

                f_0_pn = loader.get_read_handle(
                    f_0_fd,
                    operation=neon.Loader.Operation.stencil,
                    discretization=neon.Loader.Discretization.lattice,
                )
                bc_mask_pn = loader.get_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_read_handle(missing_mask_fd)

                f_1_pn = loader.get_write_handle(f_1_fd)

                @wp.func
                def nse_stepper_cl(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    # Apply streaming
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread

                    # Apply post-streaming boundary conditions
                    _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream, True)

                    _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                    _feq = self.equilibrium.neon_functional(_rho, _u)
                    _f_post_collision = self.collision.neon_functional(_f_post_stream, _feq, omega)

                    # Apply post-collision boundary conditions
                    _f_post_collision = apply_bc(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision, False
                    )

                    # Apply auxiliary recovery for boundary conditions (swapping)
                    neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

                    # Store the result in f_1
                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_collision[l]))

                loader.declare_kernel(nse_stepper_cl)

            return nse_stepper_ll

        return None, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_launch(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        if timestep == 0:
            self.prepare_skeleton(f_0, f_1, bc_mask, missing_mask, omega)
        self.sk[self.sk_iter].run()
        self.sk_iter = (self.sk_iter + 1) % 2
        return f_0, f_1

    def prepare_skeleton(self, f_0, f_1, bc_mask, missing_mask, omega):
        """Build the Neon odd/even skeletons for double-buffered time stepping."""
        import neon

        grid = f_0.get_grid()
        bk = grid.backend
        self.neon_skeleton = {"odd": {}, "even": {}}
        self.neon_skeleton["odd"]["container"] = self.neon_container(f_0, f_1, bc_mask, missing_mask, omega, 0)
        self.neon_skeleton["even"]["container"] = self.neon_container(f_1, f_0, bc_mask, missing_mask, omega, 1)
        # check if 'occ' is a valid key
        if "occ" not in self.backend_config:
            occ = neon.SkeletonConfig.OCC.none()
        else:
            occ = self.backend_config["occ"]
            # check that occ is of type neon.SkeletonConfig.OCC
            if not isinstance(occ, neon.SkeletonConfig.OCC):
                print(type(occ))
                raise ValueError("occ must be of type neon.SkeletonConfig.OCC")

        for key in self.neon_skeleton:
            self.neon_skeleton[key]["app"] = [self.neon_skeleton[key]["container"]]
            self.neon_skeleton[key]["skeleton"] = neon.Skeleton(backend=bk)
            self.neon_skeleton[key]["skeleton"].sequence(name="mres_nse_stepper", containers=self.neon_skeleton[key]["app"], occ=occ)

        self.sk = [self.neon_skeleton["odd"]["skeleton"], self.neon_skeleton["even"]["skeleton"]]
        self.sk_iter = 0
