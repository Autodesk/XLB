# Base class for all stepper operators

from functools import partial

from docutils.nodes import container
from jax import jit
import warp as wp
import neon
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
        self.equilibrium = QuadraticEquilibrium(self.velocity_set, self.precision_policy, self.compute_backend)
        self.macroscopic = Macroscopic(self.velocity_set, self.precision_policy, self.compute_backend)

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
        # f_0.update_host(0)
        # wp.synchronize()
        # f_0.export_vti("f0_eq_init.vti", "init_f0")

        # Process boundary conditions and update masks
        bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, bc_mask, missing_mask, xlb_grid=self.grid)
        # Initialize auxiliary data if needed
        f_0, f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_0, f_1, bc_mask, missing_mask)
        # bc_mask.update_host(0)
        bc_mask.update_host(0)
        f_0.update_host(0)
        wp.synchronize()
        bc_mask.export_vti("bc_mask.vti", "bc_mask")
        # f_0.export_vti("init_f0.vti", 'init_f0')
        # missing_mask.export_vti("missing_mask.vti", 'missing_mask')

        return f_0, f_1, bc_mask, missing_mask

    def prepare_coalescence_count(self, coalescence_factor, bc_mask):
        num_levels = coalescence_factor.get_grid().get_num_levels()

        @neon.Container.factory(name="sum_kernel_by_level")
        def sum_kernel_by_level(level):
            def ll_coalescence_count(loader: neon.Loader):
                loader.set_mres_grid(coalescence_factor.get_grid(), level)

                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)

                _c = self.velocity_set.c
                _w = self.velocity_set.w
                import typing

                @wp.func
                def cl_collide_coarse(index: typing.Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return
                    if not wp.neon_has_child(coalescence_factor_pn, index):
                        for l in range(self.velocity_set.q):
                            if level < num_levels - 1:
                                push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                                val = self.compute_dtype(1)
                                wp.neon_mres_lbm_store_op(coalescence_factor_pn, index, l, push_direction, val)

                loader.declare_kernel(cl_collide_coarse)

            return ll_coalescence_count

        for level in range(num_levels):
            sum_kernel = sum_kernel_by_level(level)
            sum_kernel.run(0)

        @neon.Container.factory(name="sum_kernel_by_level")
        def invert_count(level):
            def loading(loader: neon.Loader):
                loader.set_mres_grid(coalescence_factor.get_grid(), level)

                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)

                _c = self.velocity_set.c
                _w = self.velocity_set.w
                import typing

                @wp.func
                def compute(index: typing.Any):
                    # _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    # if _boundary_id == wp.uint8(255):
                    #     return
                    # for l in range(self.velocity_set.q):
                    #     val = wp.neon_read(coalescence_factor_pn, index, l)
                    #     if val > 0:
                    #         val = self.compute_dtype(1) / val
                    #     wp.neon_write(coalescence_factor_pn, index, l, val)
                    #####
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    are_we_a_halo_cell = wp.neon_has_child(coalescence_factor_pn, index)
                    if are_we_a_halo_cell:
                        # HERE: we are a halo cell so we just exit
                        return

                    for l in range(self.velocity_set.q):
                        if l == 9:
                            # HERE, we skip the center direction
                            continue

                        pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                        has_ngh_at_same_level = wp.bool(False)
                        coalescence_factor = wp.neon_read_ngh(
                            coalescence_factor_pn, index, pull_direction, l, self.compute_dtype(0), has_ngh_at_same_level
                        )

                        # if (!pin.hasChildren(cell, dir)) {
                        if not wp.neon_has_finer_ngh(coalescence_factor_pn, index, pull_direction):
                            pass
                        else:
                            # HERE -> I have a finer ngh. in direction pull (opposite l)
                            # Then I have to read from the halo on top of my finer ngh.
                            if has_ngh_at_same_level:
                                # if l == 10:
                                #     wp.print(accumulated)
                                #     glob = wp.neon_global_idx(f_1_pn, index)
                                #     wp.neon_cuda_info()
                                #     wp.neon_print(glob)
                                #     wp.neon_level(f_1_pn)
                                # accumulated = _w[l]
                                # Full State
                                # YES finer ngh. in the pull direction (opposite of l)
                                # YES ngh. at the same level
                                # -> **Coalescence**
                                if coalescence_factor > self.compute_dtype(0):
                                    coalescence_factor = self.compute_dtype(1) / (self.compute_dtype(2) * coalescence_factor)
                                    wp.neon_write(coalescence_factor_pn, index, l, coalescence_factor)

                            else:
                                wp.print("ERRRRRRORRRRRRRRRRRRRR")

                loader.declare_kernel(compute)

            return loading

        for level in range(num_levels):
            sum_kernel = invert_count(level)
            sum_kernel.run(0)
        return

    @classmethod
    def _process_boundary_conditions(cls, boundary_conditions, bc_mask, missing_mask, xlb_grid=None):
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
            bc_mask, missing_mask = indices_masker(bc_with_indices, bc_mask, missing_mask, xlb_grid=xlb_grid)
        # Process mesh-based boundary conditions for 3D
        if DefaultConfig.velocity_set.d == 3 and bc_with_vertices:
            # throw an exception because this option is not implemented yet
            raise Exception("Mesh-based boundary conditions are not implemented yet")
            # mesh_masker = MeshBoundaryMasker(
            #     velocity_set=DefaultConfig.velocity_set,
            #     precision_policy=DefaultConfig.default_precision_policy,
            #     compute_backend=DefaultConfig.default_backend,
            # )
            # for bc in bc_with_vertices:
            #     bc_mask, missing_mask = mesh_masker(bc, bc_mask, missing_mask)

        return bc_mask, missing_mask

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_0, f_1, bc_mask, missing_mask):
        """Initialize auxiliary data for boundary conditions that require it."""
        for bc in boundary_conditions:
            if bc.needs_aux_init and not bc.is_initialized_with_aux_data:
                f_0, f_1 = bc.aux_data_init(f_0, f_1, bc_mask, missing_mask)
        return f_0, f_1

    def _construct_neon(self):
        # Set local constants
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices
        # _cast_to_store_dtype = self.store_dtype()

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id
        id_to_bc = boundary_condition_registry.id_to_bc
        _zero = self.compute_dtype(0)
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
            f1_pn: Any,
            missing_mask_pn: Any,
            index: Any,
        ):
            # Read thread data for populations
            _f0_thread = _f_vec()
            _f1_thread = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                _f0_thread[l] = self.compute_dtype(wp.neon_read(f0_pn, index, l))
                _f1_thread[l] = self.compute_dtype(wp.neon_read(f1_pn, index, l))
                _missing_mask[l] = wp.neon_read(missing_mask_pn, index, l)

            return _f0_thread, _f1_thread, _missing_mask

        import typing

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

            def ll_collide_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                if level + 1 < f_0_fd.get_grid().get_num_levels():
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)

                # fake loading to enforce sequential step

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                _c = self.velocity_set.c
                _w = self.velocity_set.w

                @wp.func
                def device(index: typing.Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    """
                    The c++ version starts with the following, which I am not sure is right:
                        if (type(cell, 0) == CellType::bulk ) {
                    CB type cells should do collide too  
                    """
                    if _boundary_id == wp.uint8(255):
                        return

                    if not wp.neon_has_child(f_0_pn, index):
                        # Read thread data for populations, these are post streaming
                        _f0_thread, _f1_thread, _missing_mask = neon_get_thread_data(f_0_pn, f_1_pn, missing_mask_pn, index)
                        _f_post_stream = _f0_thread

                        _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                        _feq = self.equilibrium.neon_functional(_rho, _u)
                        _f_post_collision = self.collision.neon_functional(_f_post_stream, _feq, _rho, _u, omega)

                        # Apply post-collision boundary conditions
                        # _f_post_collision = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision, False)

                        for l in range(self.velocity_set.q):
                            push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                            if level < num_levels - 1:
                                val = _f_post_collision[l]
                                wp.neon_mres_lbm_store_op(f_1_pn, index, l, push_direction, val)
                                wp.neon_mres_lbm_store_op(f_0_pn, index, l, push_direction, val)

                            wp.neon_write(f_1_pn, index, l, _f_post_collision[l])
                    else:
                        for l in range(self.velocity_set.q):
                            wp.neon_write(f_1_pn, index, l, self.compute_dtype(0))
                            wp.neon_write(f_0_pn, index, l, self.compute_dtype(0))

                loader.declare_kernel(device)

            return ll_collide_coarse

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
            num_levels = f_0_fd.get_grid().get_num_levels()

            # if level != 0:
            #     # throw an exception
            #     raise Exception("Only the finest level is supported for now")

            # module op to define odd of even iteration
            # od_or_even = wp.module("odd_or_even", "even")

            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                _c = self.velocity_set.c

                coalescence_factor_fd = omega
                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor_fd)

                @wp.func
                def cl_stream_coarse(index: typing.Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    are_we_a_halo_cell = wp.neon_has_child(f_0_pn, index)
                    if are_we_a_halo_cell:
                        # HERE: we are a halo cell so we just exit
                        return

                    # do stream normally
                    _missing_mask = _missing_mask_vec()
                    _f0_thread, _f1_thread, _missing_mask = neon_get_thread_data(f_0_pn, f_1_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                    for l in range(self.velocity_set.q):
                        if l == 9:
                            # HERE, we skip the center direction
                            continue

                        pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                        has_ngh_at_same_level = wp.bool(False)
                        accumulated = wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.compute_dtype(0), has_ngh_at_same_level)

                        # if (!pin.hasChildren(cell, dir)) {
                        if not wp.neon_has_finer_ngh(f_0_pn, index, pull_direction):
                            # NO finer ngh. in the pull direction (opposite of l)
                            if not has_ngh_at_same_level:
                                # NO ngh. at the same level
                                # COULD we have a ngh. at the courser level?
                                if wp.neon_has_parent(f_0_pn, index):
                                    # YES halo cell on top of us
                                    has_a_courser_ngh = wp.bool(False)
                                    exploded_pop = wp.neon_lbm_read_coarser_ngh(
                                        f_0_pn, index, pull_direction, l, self.compute_dtype(0), has_a_courser_ngh
                                    )
                                    if has_a_courser_ngh:
                                        # Full state:
                                        # NO finer ngh. in the pull direction (opposite of l)
                                        # NO ngh. at the same level
                                        # YES ghost cell on top of us
                                        # YES courser ngh.
                                        # -> **Explosion**
                                        # wp.neon_write(f_1_pn, index, l, exploded_pop)
                                        _f_post_stream[l] = exploded_pop
                        else:
                            # HERE -> I have a finer ngh. in direction pull (opposite l)
                            # Then I have to read from the halo on top of my finer ngh.
                            if has_ngh_at_same_level:
                                # if l == 10:
                                #     wp.print(accumulated)
                                #     glob = wp.neon_global_idx(f_1_pn, index)
                                #     wp.neon_cuda_info()
                                #     wp.neon_print(glob)
                                #     wp.neon_level(f_1_pn)
                                # accumulated = _w[l]
                                # Full State
                                # YES finer ngh. in the pull direction (opposite of l)
                                # YES ngh. at the same level
                                # -> **Coalescence**
                                coalescence_factor = wp.neon_read(coalescence_factor_pn, index, l)
                                accumulated = accumulated * coalescence_factor
                                # wp.neon_write(f_1_pn, index, l, accumulated)
                                _f_post_stream[l] = accumulated
                            else:
                                wp.print("ERRRRRRORRRRRRRRRRRRRR")

                    # do non mres post-streaming corrections
                    _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream, True)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, _f_post_stream[l])
                    # wp.print("stream_coarse")

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        @neon.Container.factory(name="stream_coarse_step_A")
        def stream_coarse_step_A(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: int,
        ):
            num_levels = f_0_fd.get_grid().get_num_levels()

            # if level != 0:
            #     # throw an exception
            #     raise Exception("Only the finest level is supported for now")

            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                _c = self.velocity_set.c

                @wp.func
                def cl_stream_coarse(index: typing.Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    are_we_a_halo_cell = wp.neon_has_child(f_0_pn, index)
                    if are_we_a_halo_cell:
                        # HERE: we are a halo cell so we just exit
                        return

                    # do stream normally
                    _missing_mask = _missing_mask_vec()
                    _f0_thread, _f1_thread, _missing_mask = neon_get_thread_data(f_0_pn, f_1_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, _f_post_stream[l])
                    # wp.print("stream_coarse")

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        @neon.Container.factory(name="stream_coarse_step_B")
        def stream_coarse_step_B(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: int,
        ):
            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                coalescence_factor_fd = omega
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor_fd)

                _c = self.velocity_set.c
                _w = self.velocity_set.w

                @wp.func
                def cl_stream_coarse(index: typing.Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    are_we_a_halo_cell = wp.neon_has_child(f_0_pn, index)
                    if are_we_a_halo_cell:
                        # HERE: we are a halo cell so we just exit
                        return

                    for l in range(self.velocity_set.q):
                        if l == 9:
                            # HERE, we skip the center direction
                            continue

                        pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                        has_ngh_at_same_level = wp.bool(False)
                        accumulated = wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.compute_dtype(0), has_ngh_at_same_level)

                        # if (!pin.hasChildren(cell, dir)) {
                        if not wp.neon_has_finer_ngh(f_0_pn, index, pull_direction):
                            # NO finer ngh. in the pull direction (opposite of l)
                            if not has_ngh_at_same_level:
                                # NO ngh. at the same level
                                # COULD we have a ngh. at the courser level?
                                if wp.neon_has_parent(f_0_pn, index):
                                    # YES halo cell on top of us
                                    has_a_courser_ngh = wp.bool(False)
                                    exploded_pop = wp.neon_lbm_read_coarser_ngh(
                                        f_0_pn, index, pull_direction, l, self.compute_dtype(0), has_a_courser_ngh
                                    )
                                    if has_a_courser_ngh:
                                        # Full state:
                                        # NO finer ngh. in the pull direction (opposite of l)
                                        # NO ngh. at the same level
                                        # YES ghost cell on top of us
                                        # YES courser ngh.
                                        # -> **Explosion**
                                        wp.neon_write(f_1_pn, index, l, exploded_pop)
                        else:
                            # HERE -> I have a finer ngh. in direction pull (opposite l)
                            # Then I have to read from the halo on top of my finer ngh.
                            if has_ngh_at_same_level:
                                # if l == 10:
                                #     wp.print(accumulated)
                                #     glob = wp.neon_global_idx(f_1_pn, index)
                                #     wp.neon_cuda_info()
                                #     wp.neon_print(glob)
                                #     wp.neon_level(f_1_pn)
                                # accumulated = _w[l]
                                # Full State
                                # YES finer ngh. in the pull direction (opposite of l)
                                # YES ngh. at the same level
                                # -> **Coalescence**
                                coalescence_factor = wp.neon_read(coalescence_factor_pn, index, l)
                                accumulated = accumulated * coalescence_factor
                                wp.neon_write(f_1_pn, index, l, accumulated)

                            else:
                                wp.print("ERRRRRRORRRRRRRRRRRRRR")

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        @neon.Container.factory(name="stream_coarse_step_C")
        def stream_coarse_step_C(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: int,
        ):
            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                _c = self.velocity_set.c

                @wp.func
                def cl_stream_coarse(index: typing.Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    are_we_a_halo_cell = wp.neon_has_child(f_0_pn, index)
                    if are_we_a_halo_cell:
                        # HERE: we are a halo cell so we just exit
                        return

                    _missing_mask = _missing_mask_vec()
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    # do stream normally
                    _f0_thread, _f1_thread, _missing_mask = neon_get_thread_data(f_0_pn, f_1_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = _f1_thread

                    # do non mres post-streaming corrections
                    _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream, True)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, _f_post_stream[l])

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        return None, {
            # "single_step_finest": single_step_finest,
            "collide_coarse": collide_coarse,
            "stream_coarse_step_ABC": stream_coarse_step_ABC,
            "stream_coarse_step_A": stream_coarse_step_A,
            "stream_coarse_step_B": stream_coarse_step_B,
            "stream_coarse_step_C": stream_coarse_step_C,
        }

    def init_containers(self):
        self.containers = None
        _, self.containers = self._construct_neon()

    def launch_container(self, streamId, op_name, mres_level, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        self.containers[op_name](mres_level, f_0, f_1, bc_mask, missing_mask, omega, timestep).run(0)

    def add_to_app(self, app, op_name, mres_level, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        app.append(self.containers[op_name](mres_level, f_0, f_1, bc_mask, missing_mask, omega, timestep))

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_launch(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        c = self.neon_container(f_0, f_1, bc_mask, missing_mask, omega, timestep)
        c.run(0)
        return f_0, f_1
