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
from xlb.helper.nse_solver import create_nse_fields


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
        self.odd_or_even='even'
        self.c_even = None
        self.c_odd = None

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
            # throw an exception because this option is not implemented yet
            raise Exception("Initializer is not implemented yet")
            #f_0 = initializer(self.grid, self.velocity_set, self.precision_policy, self.compute_backend)
        else:
            from xlb.helper.initializers import initialize_multires_eq
            f_0 = initialize_multires_eq(f_0, self.grid, self.velocity_set, self.precision_policy, self.compute_backend)

        if self.compute_backend == ComputeBackend.NEON:
            for level in range(self.grid.count_levels):
                f_1.copy_from_run(level, f_0, 0)

        # Process boundary conditions and update masks
        bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, bc_mask, missing_mask, xlb_grid=self.grid)
        # Initialize auxiliary data if needed
        f_0, f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_0, f_1, bc_mask, missing_mask)
        bc_mask.update_host(0)
        missing_mask.update_host(0)
        wp.synchronize()
        bc_mask.export_vti("bc_mask.vti", 'bc_mask')
        #missing_mask.export_vti("missing_mask.vti", 'missing_mask')

        return f_0, f_1, bc_mask, missing_mask

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

    def _construct_neon(self):
        # Set local constants
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices
        #_cast_to_store_dtype = self.store_dtype()

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
        @neon.Container.factory(name="finest_collide")
        def single_step_finest(
                level: int,
                f_0_fd: Any,
                f_1_fd: Any,
                bc_mask_fd: Any,
                missing_mask_fd: Any,
                omega: Any,
                timestep: int,
        ):
            # if level != 0:
            #     # throw an exception
            #     raise Exception("Only the finest level is supported for now")

            # module op to define odd of even iteration
            od_or_even = wp.module("odd_or_even", "even")

            def ll_single_step_finest(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn=loader.get_mres_read_handle(f_0_fd)
                bc_mask_pn=loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn=loader.get_mres_read_handle(missing_mask_fd)

                f_1_pn =loader.get_mres_write_handle(f_1_fd)

                @wp.func
                def cl_single_step_finest(index: typing.Any):
                    _c = self.velocity_set.c
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(255):
                        return

                    # Read thread data for populations, these are post streaming
                    _f0_thread, _f1_thread, _missing_mask = neon_get_thread_data(f_0_pn, f_1_pn, missing_mask_pn, index)
                    _f_post_stream = _f0_thread

                    _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                    _feq = self.equilibrium.neon_functional(_rho, _u)
                    _f_post_collision = self.collision.neon_functional(_f_post_stream, _feq, _rho, _u, omega)

                    # Apply post-collision boundary conditions
                    _f_post_collision = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision, False)

                    # Apply streaming boundary conditions
                    _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision, True)
                    _opposite_c_idx = self.velocity_set.self.opp_indices


                    for l in range(self.velocity_set.q):
                        push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]),
                                                   wp.int8(_c[1, l]),
                                                   wp.int8(_c[2, l]))
                        ## Store
                        if od_or_even == 0:
                            wp.neon_mres_lbm_store_op(f_0_pn, index, l, push_direction, _f_post_stream[l])
                        else:
                            wp.neon_mres_lbm_store_op(f_1_pn, index, l, push_direction,_f_post_stream[l])

                        ## Push stream
                        is_active = wp.neon_is_active(f_0_pn, index, push_direction)
                        if is_active:
                            ngh_gidx = wp.neon_ngh_idx(f_0_pn, index, push_direction)
                            ngh_boundary_id = wp.neon_read(bc_mask_pn, ngh_gidx, 0)
                            ## WHAT IS BULK?
                            if ngh_boundary_id == BULK:
                                wp.neon_write(f_1_pn, ngh_gidx, l, _f_post_stream[l])
                            else:
                                opposite_l = _opp_indices[l]
                                wp.neon_write(f_1_pn, index, opposite_l, _f_post_stream[l])
                        else:
                            if wp.int8(_c[0, l]) != 0 and wp.int8(_c[1, l]) != 0 and wp.int8(_c[2, l]) != 0:
                                opposite_l = _opp_indices[l]
                                is_valid = False
                                value = self.compute_dtype(0)
                                if od_or_even == 0:
                                    value = wp.neon_uncle_read(f_1_pn, index, push_direction, opposite_l, value, is_valid)
                                else:
                                    value = wp.neon_uncle_read(f_0_pn, index, push_direction, opposite_l, value, is_valid)
                                if is_valid:
                                    wp.neon_write(f_1_pn, index, l, _f_post_stream[l], value)


                loader.declare_kernel(cl_single_step_finest)
            return ll_single_step_finest


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

            # module op to define odd of even iteration
            od_or_even = wp.mod(timestep, 2)

            def ll_collide_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn=loader.get_mres_read_handle(f_0_fd)
                bc_mask_pn=loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn=loader.get_mres_read_handle(missing_mask_fd)
                f_1_pn =loader.get_mres_write_handle(f_1_fd)

                _c = self.velocity_set.c

                @wp.func
                def cl_collide_coarse(index: typing.Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    """
                    The c++ version starts with the following, which I am not sure is right:
                        if (type(cell, 0) == CellType::bulk ) {
                    CB type cells should do collide too  
                    """
                    if _boundary_id == wp.uint8(255):
                        return

                    if not wp.neon_has_children(f_0_pn, index):

                        # Read thread data for populations, these are post streaming
                        _f0_thread, _f1_thread, _missing_mask = neon_get_thread_data(f_0_pn, f_1_pn, missing_mask_pn, index)
                        _f_post_stream = _f0_thread

                        _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                        _feq = self.equilibrium.neon_functional(_rho, _u)
                        _f_post_collision = self.collision.neon_functional(_f_post_stream, _feq, _rho, _u, omega)

                        # Apply post-collision boundary conditions
                        _f_post_collision = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision, False)

                        for l in range(self.velocity_set.q):
                            push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                            if(level < num_levels - 1):
                                ## Store
                                if od_or_even == 0:
                                    wp.neon_mres_lbm_store_op(f_0_pn, index, l, push_direction, _f_post_collision[l])
                                else:
                                    wp.neon_mres_lbm_store_op(f_1_pn, index, l, push_direction, _f_post_collision[l])

                            wp.neon_write(f_1_pn, index, l, _f_post_collision[l])
                    else:
                        for l in range(self.velocity_set.q):
                            wp.neon_write(f_1_pn, index, l, self.compute_dtype(0))

                    wp.print("collide_coarse")



                loader.declare_kernel(cl_collide_coarse)
            return ll_collide_coarse

        @neon.Container.factory(name="stream_coarse")
        def stream_coarse(
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
            #od_or_even = wp.module("odd_or_even", "even")

            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                _c = self.velocity_set.c

                @wp.func
                def cl_stream_coarse(index: typing.Any):
                    _missing_mask = _missing_mask_vec()
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id != wp.uint8(255):
                        if not wp.neon_has_children(f_0_pn, index):
                            # do stream normally
                            _f0_thread, _f1_thread, _missing_mask = neon_get_thread_data(f_0_pn, f_1_pn, missing_mask_pn, index)
                            _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                            # do mres corrections
                            for l in range(self.velocity_set.q):
                                pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))
                                _missing_mask[l] = wp.neon_read(missing_mask_pn, index, l)
                                if wp.neon_has_children(f_0_pn, index, pull_direction):
                                    is_valid = wp.bool(False)
                                    read_accumulate_date = wp.neon_ngh_data(f_1_pn, index, pull_direction, l, self.compute_dtype(0),is_valid)
                                    if is_valid:
                                        wp.print("read_accumulate_date")
                                        _f_post_stream[l] = self.compute_dtype(33) #read_accumulate_date * self.compute_dtype(0.5)

                            # do non mres post-streaming corrections
                            _f_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_stream, True)

                            for l in range(self.velocity_set.q):
                                wp.neon_write(f_1_pn, index, l, _f_post_stream[l])
                    wp.print("stream_coarse")

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        return None, {
            #"single_step_finest": single_step_finest,
            "collide_coarse": collide_coarse,
            "stream_coarse": stream_coarse}



    def get_containers(self, target_level,  f_0, f_1, bc_mask, missing_mask,  omega, timestep):
        containers = {'even': {}, 'odd': {}}
        _, container = self._construct_neon()
        for key in container.keys():
            containers['odd'][key] = container[key](target_level, f_1, f_0, bc_mask, missing_mask, omega, 1)
            containers['even'][key] = container[key](target_level, f_0, f_1, bc_mask, missing_mask, omega, 0)
        return containers

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_launch(self, f_0, f_1, bc_mask, missing_mask,  omega, timestep):
        #if self.c is None:
        #    self.c = self.neon_container(f_0, f_1, bc_mask, missing_mask, timestep)
        # c = None
        # if self.odd_or_even == 'even':
        #     c = self.c_even
        # else:
        #     c = self.c_odd
        #
        # if c is None:
        #     pass
        c = self.neon_container(f_0, f_1, bc_mask, missing_mask, omega, timestep)
        c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
        #
        # if self.odd_or_even == 'even':
        #     c = self.c_even
        # else:
        #     c = self.c_odd
        #
        # if self.odd_or_even == 'even':
        #     self.odd_or_even = 'odd'

        return f_0, f_1
