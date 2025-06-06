import warp as wp
import neon
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC
from xlb.operator.equilibrium import MultiresQuadraticEquilibrium
from xlb.operator.macroscopic import MultiresMacroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.collision import ForcedCollision
from xlb.operator.boundary_masker import MultiresBoundaryMasker


class MultiresIncompressibleNavierStokesStepper(Stepper):
    def __init__(
        self,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
        forcing_scheme="exact_difference",
        force_vector=None,
    ):
        super().__init__(grid, boundary_conditions)

        # Construct the collision operator
        if collision_type == "BGK":
            self.collision = BGK(self.velocity_set, self.precision_policy, self.compute_backend)
        elif collision_type == "KBC":
            self.collision = KBC(self.velocity_set, self.precision_policy, self.compute_backend)

        if force_vector is not None:
            self.collision = ForcedCollision(collision_operator=self.collision, forcing_scheme=forcing_scheme, force_vector=force_vector)

        # Construct the operators
        self.stream = Stream(self.velocity_set, self.precision_policy, self.compute_backend)
        self.equilibrium = MultiresQuadraticEquilibrium(self.velocity_set, self.precision_policy, self.compute_backend)
        self.macroscopic = MultiresMacroscopic(self.velocity_set, self.precision_policy, self.compute_backend)

    def prepare_fields(self, rho, u, initializer=None):
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

        f_0 = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        f_1 = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        missing_mask = self.grid.create_field(cardinality=self.velocity_set.q, dtype=Precision.UINT8)
        bc_mask = self.grid.create_field(cardinality=1, dtype=Precision.UINT8)

        from xlb.helper.initializers import initialize_multires_eq

        f_0 = initialize_multires_eq(f_0, self.grid, self.velocity_set, self.precision_policy, self.compute_backend, rho=rho, u=u)

        for level in range(self.grid.count_levels):
            f_1.copy_from_run(level, f_0, 0)

        # Process boundary conditions and update masks
        bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, bc_mask, missing_mask, xlb_grid=self.grid)
        # Initialize auxiliary data if needed
        f_0, f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_0, f_1, bc_mask, missing_mask)
        # bc_mask.update_host(0)
        bc_mask.update_host(0)
        f_0.update_host(0)
        wp.synchronize()
        bc_mask.export_vti("bc_mask.vti", "bc_mask")

        return f_0, f_1, bc_mask, missing_mask

    def prepare_coalescence_count(self, coalescence_factor, bc_mask):
        """Prepare coalescence factors for multires operations."""
        lattice_central_index = self.velocity_set.center_index
        num_levels = coalescence_factor.get_grid().get_num_levels()

        self._compute_coalescence_factors(coalescence_factor, bc_mask, num_levels)
        self._invert_coalescence_factors(coalescence_factor, bc_mask, num_levels, lattice_central_index)

    def _compute_coalescence_factors(self, coalescence_factor, bc_mask, num_levels):
        """Compute initial coalescence factors."""

        @neon.Container.factory(name="compute_coalescence")
        def compute_coalescence(level):
            def loading_step(loader: neon.Loader):
                loader.set_mres_grid(coalescence_factor.get_grid(), level)
                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                _c = self.velocity_set.c

                @wp.func
                def compute_kernel(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    if not wp.neon_has_child(coalescence_factor_pn, index):
                        for l in range(self.velocity_set.q):
                            if level < num_levels - 1:
                                push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                                val = self.compute_dtype(1)
                                wp.neon_mres_lbm_store_op(coalescence_factor_pn, index, l, push_direction, val)

                loader.declare_kernel(compute_kernel)

            return loading_step

        for level in range(num_levels):
            compute_coalescence(level).run(0)

    def _invert_coalescence_factors(self, coalescence_factor, bc_mask, num_levels, lattice_central_index):
        """Invert coalescence factors for proper weighting."""

        @neon.Container.factory(name="invert_coalescence")
        def invert_coalescence(level):
            def loading_step(loader: neon.Loader):
                loader.set_mres_grid(coalescence_factor.get_grid(), level)
                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                _c = self.velocity_set.c

                @wp.func
                def invert_kernel(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    if not wp.neon_has_child(coalescence_factor_pn, index):
                        for l in range(self.velocity_set.q):
                            if l == lattice_central_index:
                                continue

                            pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))
                            has_ngh_at_same_level = wp.bool(False)
                            coalescence_factor = wp.neon_read_ngh(
                                coalescence_factor_pn, index, pull_direction, l, self.compute_dtype(0), has_ngh_at_same_level
                            )

                            if wp.neon_has_finer_ngh(coalescence_factor_pn, index, pull_direction):
                                if has_ngh_at_same_level:
                                    if coalescence_factor > self.compute_dtype(0):
                                        coalescence_factor = self.compute_dtype(1) / (self.compute_dtype(2) * coalescence_factor)
                                        wp.neon_write(coalescence_factor_pn, index, l, coalescence_factor)
                                else:
                                    wp.print("Error: Expected neighbor at same level")

                loader.declare_kernel(invert_kernel)

            return loading_step

        for level in range(num_levels):
            invert_coalescence(level).run(0)

    @classmethod
    def _process_boundary_conditions(cls, boundary_conditions, bc_mask, missing_mask, xlb_grid=None):
        """Process boundary conditions and update boundary masks."""
        # Check for boundary condition overlaps
        # TODO! check_bc_overlaps(boundary_conditions, DefaultConfig.velocity_set.d, DefaultConfig.default_backend)
        # Create boundary maskers
        mres_masker = MultiresBoundaryMasker(
            velocity_set=DefaultConfig.velocity_set,
            precision_policy=DefaultConfig.default_precision_policy,
            compute_backend=DefaultConfig.default_backend,
        )
        # Split boundary conditions by type
        bc_with_vertices = [bc for bc in boundary_conditions if bc.mesh_vertices is not None]
        bc_with_indices = [bc for bc in boundary_conditions if bc.indices is not None]
        # Process indices-based boundary conditions
        if bc_with_indices:
            bc_mask, missing_mask = mres_masker(bc_with_indices, bc_mask, missing_mask, xlb_grid=xlb_grid)
        # Process mesh-based boundary conditions for 3D
        if DefaultConfig.velocity_set.d == 3 and bc_with_vertices:
            raise Exception("Mesh-based boundary conditions are not implemented yet")
        return bc_mask, missing_mask

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_0, f_1, bc_mask, missing_mask):
        """Initialize auxiliary data for boundary conditions that require it."""
        for bc in boundary_conditions:
            if bc.needs_aux_init and not bc.is_initialized_with_aux_data:
                for level in range(bc_mask.get_grid().get_num_levels()):
                    # Initialize auxiliary data for each level
                    f_0, f_1 = bc.multires_aux_data_init(f_0, f_1, bc_mask, missing_mask, level=level, stream=0)
        return f_0, f_1

    def _construct_neon(self):
        # Set local constants
        lattice_central_index = self.velocity_set.center_index
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices

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
                            f_result = wp.static(self.boundary_conditions[i].neon_functional)(index, timestep, missing_mask, f_0, f_1, f_pre, f_post)
                else:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.COLLISION):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].neon_functional)(index, timestep, missing_mask, f_0, f_1, f_pre, f_post)
                    if wp.static(self.boundary_conditions[i].id in extrapolation_outflow_bc_ids):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            f_result = wp.static(self.boundary_conditions[i].prepare_bc_auxilary_data)(
                                index, timestep, missing_mask, f_0, f_1, f_pre, f_post
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
            # Unroll the loop over boundary conditions
            for i in range(wp.static(len(self.boundary_conditions))):
                if wp.static(self.boundary_conditions[i].needs_aux_recovery):
                    if _boundary_id == wp.static(self.boundary_conditions[i].id):
                        for l in range(self.velocity_set.q):
                            # Perform the swapping of data
                            if l == lattice_central_index:
                                # (i) Recover the values stored in the central index of f_1
                                _f1_thread = wp.neon_read(f_1_pn, index, l)
                                wp.neon_write(f_0_pn, index, l, _f1_thread)
                            elif _missing_mask[l] == wp.uint8(1):
                                # (ii) Recover the values stored in the missing directions of f_1
                                _f1_thread = wp.neon_read(f_1_pn, index, _opp_indices[l])
                                wp.neon_write(f_0_pn, index, _opp_indices[l], _f1_thread)

        # Main kernel functions
        @neon.Container.factory(name="collide_coarse")
        def collide_coarse(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: int,
        ):
            num_levels = f_0_fd.get_grid().get_num_levels()

            def collision_step(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                if level + 1 < num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                _c = self.velocity_set.c

                @wp.func
                def collision_kernel(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    if not wp.neon_has_child(f_0_pn, index):
                        _perform_collision_step(
                            index, f_0_pn, f_1_pn, bc_mask_pn, missing_mask_pn, omega, timestep, _boundary_id, _c, level, num_levels
                        )
                    else:
                        # Initialize halo cells
                        for l in range(self.velocity_set.q):
                            wp.neon_write(f_1_pn, index, l, self.compute_dtype(0))

                loader.declare_kernel(collision_kernel)

            return collision_step

        @wp.func
        def _perform_collision_step(
            index: Any,
            f_0_pn: Any,
            f_1_pn: Any,
            bc_mask_pn: Any,
            missing_mask_pn: Any,
            omega: Any,
            timestep: int,
            _boundary_id: Any,
            _c: Any,
            level: int,
            num_levels: int,
        ):
            # Read thread data for populations
            _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
            _f_post_stream = _f0_thread

            # Compute macroscopic properties and collision
            _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
            _feq = self.equilibrium.neon_functional(_rho, _u)
            _f_post_collision = self.collision.neon_functional(_f_post_stream, _feq, _rho, _u, omega)

            # Apply post-collision boundary conditions
            _f_post_collision = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision, False)

            # Apply auxiliary recovery for boundary conditions
            neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

            # Store results with multires handling
            for l in range(self.velocity_set.q):
                push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                if level < num_levels - 1:
                    val = _f_post_collision[l]
                    wp.neon_mres_lbm_store_op(f_1_pn, index, l, push_direction, val)
                wp.neon_write(f_1_pn, index, l, _f_post_collision[l])

        @neon.Container.factory(name="stream_coarse_step_ABC")
        def stream_coarse_step_ABC(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: int,
        ):
            def streaming_step(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                coalescence_factor_pn = loader.get_mres_read_handle(omega)
                _c = self.velocity_set.c

                @wp.func
                def streaming_kernel(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    if not wp.neon_has_child(f_0_pn, index):
                        _perform_streaming_step(
                            index,
                            f_0_pn,
                            f_1_pn,
                            bc_mask_pn,
                            missing_mask_pn,
                            coalescence_factor_pn,
                            timestep,
                            _boundary_id,
                            _c,
                            lattice_central_index,
                        )

                loader.declare_kernel(streaming_kernel)

            return streaming_step

        @wp.func
        def _perform_streaming_step(
            index: Any,
            f_0_pn: Any,
            f_1_pn: Any,
            bc_mask_pn: Any,
            missing_mask_pn: Any,
            coalescence_factor_pn: Any,
            timestep: int,
            _boundary_id: Any,
            _c: Any,
            lattice_central_index: int,
        ):
            # Get thread data and perform streaming
            _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
            _f_post_collision = _f0_thread
            _f_post_stream = self.stream.neon_functional(f_0_pn, index)

            # Handle multires refinement operations
            for l in range(self.velocity_set.q):
                if l == lattice_central_index:
                    continue

                pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))
                has_ngh_at_same_level = wp.bool(False)
                accumulated = wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.compute_dtype(0), has_ngh_at_same_level)

                # Handle explosion (coarse to fine)
                if not wp.neon_has_finer_ngh(f_0_pn, index, pull_direction):
                    if not has_ngh_at_same_level and wp.neon_has_parent(f_0_pn, index):
                        has_a_coarser_ngh = wp.bool(False)
                        exploded_pop = wp.neon_lbm_read_coarser_ngh(f_0_pn, index, pull_direction, l, self.compute_dtype(0), has_a_coarser_ngh)
                        if has_a_coarser_ngh:
                            _f_post_stream[l] = exploded_pop
                # Handle coalescence (fine to coarse)
                else:
                    if has_ngh_at_same_level:
                        coalescence_factor = wp.neon_read(coalescence_factor_pn, index, l)
                        accumulated = accumulated * coalescence_factor
                        _f_post_stream[l] = accumulated

            # Apply post-streaming boundary conditions
            _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream, True)

            # Apply auxiliary recovery
            neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

            # Store results
            for l in range(self.velocity_set.q):
                wp.neon_write(f_1_pn, index, l, _f_post_stream[l])

        return None, {
            "collide_coarse": collide_coarse,
            "stream_coarse_step_ABC": stream_coarse_step_ABC,
        }

    def launch_container(self, streamId, op_name, mres_level, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        self.neon_container[op_name](mres_level, f_0, f_1, bc_mask, missing_mask, omega, timestep).run(0)

    def add_to_app(self, app, op_name, mres_level, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        app.append(self.neon_container[op_name](mres_level, f_0, f_1, bc_mask, missing_mask, omega, timestep))

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_launch(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        c = self.neon_container(f_0, f_1, bc_mask, missing_mask, omega, timestep)
        c.run(0)
        return f_0, f_1
