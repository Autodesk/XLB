"""
Multi-Resolution Navier-Stokes Stepper for the NEON Backend

This module implements the multi-resolution LBM stepper using Warp kernels on the
Neon multi-GPU runtime. It uses several programming patterns specific to Warp's
compile-time code generation model.

Compile-Time Specialization Pattern
-----------------------------------
Warp's @wp.func decorator traces Python code at kernel compilation time, not runtime.
This means runtime boolean parameters cause Warp to emit branching code for both paths,
increasing register pressure even when only one path is ever taken.

To generate optimized, branch-free kernels, we use a **factory pattern** that captures
boolean configuration at function-definition time:

    def make_specialized_func(do_feature: bool):
        @wp.func
        def impl(...):
            if wp.static(do_feature):  # Evaluated at compile time
                # This code is only emitted when do_feature=True
                ...
            else:
                # This code is only emitted when do_feature=False
                ...
        return impl

    # Generate specialized variants
    func_with_feature = make_specialized_func(do_feature=True)
    func_without_feature = make_specialized_func(do_feature=False)

The `wp.static()` call evaluates its argument during Warp's tracing phase. Since
`do_feature` is a Python bool captured in the closure, Warp sees a constant and
eliminates the dead branch entirely.

This pattern is used for:
- `apply_bc_post_streaming` / `apply_bc_post_collision`: Specialized BC application
  for streaming vs collision implementation steps
- `collide_bc_accum` / `collide_simple`: Collision pipeline variants with/without
  BC application and multi-resolution accumulation

Closure Capture for Self Attributes
-----------------------------------
Warp cannot resolve `self.X` in plain assignments inside @wp.func bodies (e.g.,
`_c = self.velocity_set.c` fails with "Invalid external reference type"). However,
it can resolve `self.X` in:
- Function call contexts: `self.stream.neon_functional(...)`
- Range arguments: `range(self.velocity_set.q)`
- Type casts: `self.compute_dtype(0)`

For other uses, we pre-capture attributes at the Python level before defining the
@wp.func, making them available as simple closure variables:

    _c = self.velocity_set.c  # Captured in Python scope

    @wp.func
    def my_kernel(...):
        # Use _c directly — Warp sees it as a closure variable
        direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), ...)

Cell Type Constants
-------------------
Cell types are defined in `xlb.cell_type`:
- BC_SFV (254): Simple Fluid Voxel — no BC, no explosion/coalescence
- BC_SOLID (255): Solid obstacle voxel
- BC_NONE (0): Regular fluid voxel with potential BCs or multi-res interactions
"""

import nvtx
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC, SmagorinskyLESBGK
from xlb.operator.equilibrium import MultiresQuadraticEquilibrium
from xlb.operator.macroscopic import MultiresMacroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.collision import ForcedCollision
from xlb.helper import check_bc_overlaps
from xlb.operator.boundary_masker import (
    MeshVoxelizationMethod,
    MultiresMeshMaskerAABB,
    MultiresMeshMaskerAABBClose,
    MultiresIndicesBoundaryMasker,
    MultiresMeshMaskerRay,
)
from xlb.operator.boundary_condition.helper_functions_bc import MultiresEncodeAuxiliaryData
from xlb.cell_type import BC_SFV, BC_SOLID

"""
SFV = Simple Fluid Voxel: a fluid voxel that is not a BC nor is involved in explosion or coalescence
CFV = Complex Fluid Voxel: a fluid voxel that is not a SFV
"""


class MultiresIncompressibleNavierStokesStepper(Stepper):
    """Multi-resolution incompressible Navier-Stokes stepper for the Neon backend.

    Implements the full LBM step (stream, collide, boundary conditions) across
    a hierarchy of grid levels using Neon containers.  Each container is a
    compile-time specialized Warp kernel wrapped in a Neon execution-graph
    node.

    The stepper supports several performance optimization strategies (see
    :class:`MresPerfOptimizationType`):

    * **NAIVE_COLLIDE_STREAM** — separate collide and stream containers at
      every level.
    * **FUSION_AT_FINEST** — fused stream+collide at the finest level.
    * **FUSION_AT_FINEST_SFV** — additionally splits SFV / CFV voxels at
      the finest level for reduced branching.
    * **FUSION_AT_FINEST_SFV_ALL** — SFV / CFV splitting at all levels.

    Parameters
    ----------
    grid : NeonMultiresGrid
        The multi-resolution grid.
    boundary_conditions : list of BoundaryCondition
        Boundary conditions to apply.
    collision_type : str
        Collision operator type: ``"BGK"`` or ``"KBC"`` or ``"SmagorinskyLESBGK"``.
    forcing_scheme : str
        Forcing scheme name (only used when *force_vector* is given).
    force_vector : array-like, optional
        External body force vector.
    """

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
        elif collision_type == "SmagorinskyLESBGK":
            self.collision = SmagorinskyLESBGK(self.velocity_set, self.precision_policy, self.compute_backend)

        if force_vector is not None:
            self.collision = ForcedCollision(collision_operator=self.collision, forcing_scheme=forcing_scheme, force_vector=force_vector)

        # Construct the operators
        self.stream = Stream(self.velocity_set, self.precision_policy, self.compute_backend)
        self.equilibrium = MultiresQuadraticEquilibrium(self.velocity_set, self.precision_policy, self.compute_backend)
        self.macroscopic = MultiresMacroscopic(self.velocity_set, self.precision_policy, self.compute_backend)

    def prepare_fields(self, rho, u, initializer=None):
        import neon

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

        f_0 = self.grid.create_field(
            cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision, neon_memory_type=neon.MemoryType.device()
        )

        f_1 = self.grid.create_field(
            cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision, neon_memory_type=neon.MemoryType.device()
        )

        missing_mask = self.grid.create_field(cardinality=self.velocity_set.q, dtype=Precision.UINT8)
        bc_mask = self.grid.create_field(cardinality=1, dtype=Precision.UINT8)

        for level in range(self.grid.count_levels):
            f_1.copy_from_run(level, f_0, 0)

        # Process boundary conditions and update masks
        f_1, bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, f_1, bc_mask, missing_mask)
        # Initialize auxiliary data if needed
        f_1 = self._initialize_auxiliary_data(self.boundary_conditions, f_1, bc_mask, missing_mask)

        # Initialize distribution functions if initializer is provided
        if initializer is not None:
            # Refer to xlb.helper.initializers for available initializers
            f_0 = initializer(bc_mask, f_0)
        else:
            from xlb.helper.initializers import initialize_multires_eq

            f_0 = initialize_multires_eq(f_0, self.grid, self.velocity_set, self.precision_policy, self.compute_backend, rho=rho, u=u)

        return f_0, f_1, bc_mask, missing_mask

    def prepare_coalescence_count(self, coalescence_factor, bc_mask):
        """Precompute coalescence weighting factors for multi-resolution streaming.

        For each non-halo voxel at every level, this method accumulates
        the number of finer neighbours that contribute populations via
        coalescence (child-to-parent transfer), then inverts the count
        so that the streaming kernel can apply the correct averaging weight.

        Parameters
        ----------
        coalescence_factor : field
            Multi-resolution field to store the per-direction coalescence
            weights (modified in-place).
        bc_mask : field
            Boundary-condition mask used to skip solid voxels.
        """
        import neon

        lattice_central_index = self.velocity_set.center_index
        num_levels = coalescence_factor.get_grid().num_levels

        @neon.Container.factory(name="sum_kernel_by_level")
        def sum_kernel_by_level(level):
            def ll_coalescence_count(loader: neon.Loader):
                loader.set_mres_grid(coalescence_factor.get_grid(), level)

                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)

                _c = self.velocity_set.c
                _w = self.velocity_set.w

                @wp.func
                def cl_collide_coarse(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if not wp.neon_has_child(coalescence_factor_pn, index):
                        for l in range(self.velocity_set.q):
                            if level < num_levels - 1:
                                push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                                val = self.store_dtype(1)
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

                @wp.func
                def compute(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return

                    if wp.neon_has_child(coalescence_factor_pn, index):
                        # we are a halo cell so we just exit
                        return

                    for l in range(self.velocity_set.q):
                        if l == lattice_central_index:
                            continue

                        pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                        has_ngh_at_same_level = wp.bool(False)
                        coalescence_factor = self.compute_dtype(
                            wp.neon_read_ngh(coalescence_factor_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)
                        )

                        if not wp.neon_has_finer_ngh(coalescence_factor_pn, index, pull_direction):
                            pass
                        else:
                            # Finer neighbour exists in the pull direction (opposite of l).
                            # Read from the halo sitting on top of that finer neighbour.
                            if has_ngh_at_same_level:
                                # Finer ngh in pull direction: YES
                                # Same-level ngh:              YES
                                # Compute coalescence factor
                                if coalescence_factor > self.compute_dtype(0):
                                    coalescence_factor = self.compute_dtype(1) / (self.compute_dtype(2) * coalescence_factor)
                                    wp.neon_write(coalescence_factor_pn, index, l, self.store_dtype(coalescence_factor))

                loader.declare_kernel(compute)

            return loading

        for level in range(num_levels):
            sum_kernel = invert_count(level)
            sum_kernel.run(0)
        return

    @classmethod
    def _process_boundary_conditions(cls, boundary_conditions, f_1, bc_mask, missing_mask):
        """Process boundary conditions and update boundary masks."""

        # Check for boundary condition overlaps
        # TODO! check_bc_overlaps(boundary_conditions, DefaultConfig.velocity_set.d, DefaultConfig.default_backend)

        # Create boundary maskers
        indices_masker = MultiresIndicesBoundaryMasker(
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
            for bc in bc_with_vertices:
                if bc.voxelization_method.id is MeshVoxelizationMethod("AABB").id:
                    mesh_masker = MultiresMeshMaskerAABB(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                elif bc.voxelization_method.id is MeshVoxelizationMethod("RAY").id:
                    mesh_masker = MultiresMeshMaskerRay(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                    )
                elif bc.voxelization_method.id is MeshVoxelizationMethod("AABB_CLOSE").id:
                    mesh_masker = MultiresMeshMaskerAABBClose(
                        velocity_set=DefaultConfig.velocity_set,
                        precision_policy=DefaultConfig.default_precision_policy,
                        compute_backend=DefaultConfig.default_backend,
                        close_voxels=bc.voxelization_method.options.get("close_voxels"),
                    )
                else:
                    raise ValueError(f"Unsupported voxelization method for multi-res: {bc.voxelization_method}")
                # Apply the mesh masker to the boundary condition
                f_1, bc_mask, missing_mask = mesh_masker(bc, f_1, bc_mask, missing_mask)

        return f_1, bc_mask, missing_mask

    @staticmethod
    def _initialize_auxiliary_data(boundary_conditions, f_1, bc_mask, missing_mask):
        """Initialize auxiliary data for boundary conditions that require it."""
        for bc in boundary_conditions:
            if bc.needs_aux_init and not bc.is_initialized_with_aux_data:
                # Create the encoder operator for storing the auxiliary data
                encode_auxiliary_data = MultiresEncodeAuxiliaryData(
                    bc.id,
                    bc.num_of_aux_data,
                    bc.profile,
                    velocity_set=bc.velocity_set,
                    precision_policy=bc.precision_policy,
                    compute_backend=bc.compute_backend,
                )

                # Encode the auxiliary data in f_1
                f_1 = encode_auxiliary_data(f_1, bc_mask, missing_mask, stream=0)
                bc.is_initialized_with_aux_data = True
        return f_1

    def _construct_neon(self):
        import neon

        # Pre-capture self attributes that Warp cannot resolve inside @wp.func bodies.
        # Warp rejects `self` as an "Invalid external reference type" when it appears
        # in a plain assignment (e.g. `_c = self.velocity_set.c`).  Capturing here
        # makes these values available as simple closure variables.
        lattice_central_index = self.velocity_set.center_index
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices
        _c = self.velocity_set.c

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id

        # Gather IDs of ExtrapolationOutflowBC boundary conditions
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)

        # Factory for apply_bc: generates compile-time specialized variants
        def make_apply_bc(is_post_streaming: bool):
            @wp.func
            def apply_bc_impl(
                index: Any,
                timestep: Any,
                _boundary_id: Any,
                _missing_mask: Any,
                f_0: Any,
                f_1: Any,
                f_pre: Any,
                f_post: Any,
            ):
                f_result = f_post

                for i in range(wp.static(len(self.boundary_conditions))):
                    if wp.static(is_post_streaming):
                        if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.STREAMING):
                            if _boundary_id == wp.static(self.boundary_conditions[i].id):
                                f_result = wp.static(self.boundary_conditions[i].neon_functional)(
                                    index, timestep, _missing_mask, f_0, f_1, f_pre, f_post
                                )
                    else:
                        if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.COLLISION):
                            if _boundary_id == wp.static(self.boundary_conditions[i].id):
                                f_result = wp.static(self.boundary_conditions[i].neon_functional)(
                                    index, timestep, _missing_mask, f_0, f_1, f_pre, f_post
                                )
                        if wp.static(self.boundary_conditions[i].id in extrapolation_outflow_bc_ids):
                            if _boundary_id == wp.static(self.boundary_conditions[i].id):
                                f_result = wp.static(self.boundary_conditions[i].assemble_auxiliary_data)(
                                    index, timestep, _missing_mask, f_0, f_1, f_pre, f_post
                                )
                return f_result

            return apply_bc_impl

        # Compile-time specialized BC application variants
        apply_bc_post_streaming = make_apply_bc(is_post_streaming=True)
        apply_bc_post_collision = make_apply_bc(is_post_streaming=False)

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

        # Factory for neon_collide_pipeline: generates compile-time specialized variants
        def make_collide_pipeline(do_bc: bool, do_accumulation: bool):
            @wp.func
            def collide_pipeline_impl(
                index: Any,
                timestep: Any,
                _boundary_id: Any,
                _missing_mask: Any,
                f_0_pn: Any,
                f_1_pn: Any,
                _f_post_stream: Any,
                omega: Any,
                num_levels: int,
                level: int,
                accumulation_pn: Any,
            ):
                _rho, _u = self.macroscopic.neon_functional(_f_post_stream)
                _feq = self.equilibrium.neon_functional(_rho, _u)
                _f_post_collision = self.collision.neon_functional(_f_post_stream, _feq, omega)

                if wp.static(do_bc):
                    _f_post_collision = apply_bc_post_collision(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_stream, _f_post_collision
                    )
                    neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

                if wp.static(do_accumulation):
                    for l in range(self.velocity_set.q):
                        push_direction = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                        if level < num_levels - 1:
                            wp.neon_mres_lbm_store_op(accumulation_pn, index, l, push_direction, self.store_dtype(_f_post_collision[l]))
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_collision[l]))
                else:
                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_collision[l]))

                return _f_post_collision

            return collide_pipeline_impl

        # Compile-time specialized collision pipeline variants
        collide_bc_accum = make_collide_pipeline(do_bc=True, do_accumulation=True)
        collide_bc_only = make_collide_pipeline(do_bc=True, do_accumulation=False)
        collide_simple = make_collide_pipeline(do_bc=False, do_accumulation=False)

        @wp.func
        def neon_stream_explode_coalesce(
            index: Any,
            f_0_pn: Any,
            coalescence_factor_pn: Any,
        ):
            _f_post_stream = self.stream.neon_functional(f_0_pn, index)

            for l in range(self.velocity_set.q):
                if l == lattice_central_index:
                    continue

                pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                has_ngh_at_same_level = wp.bool(False)
                accumulated = wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)

                if not wp.neon_has_finer_ngh(f_0_pn, index, pull_direction):
                    # No finer ngh in the pull direction (opposite of l)
                    if not has_ngh_at_same_level:
                        # No same-level ngh — could we have a coarser-level ngh?
                        if wp.neon_has_parent(f_0_pn, index):
                            # Halo cell on top of us (parent exists)
                            has_a_coarser_ngh = wp.bool(False)
                            exploded_pop = wp.neon_lbm_read_coarser_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_a_coarser_ngh)
                            if has_a_coarser_ngh:
                                # No finer ngh in pull direction, no same-level ngh,
                                # but a parent (ghost cell) exists with a coarser ngh
                                # -> Explosion: read the exploded population from the
                                #    coarser level's halo.
                                _f_post_stream[l] = self.compute_dtype(exploded_pop)
                else:
                    # Finer ngh exists in the pull direction (opposite of l).
                    # Read from the halo on top of that finer ngh.
                    if has_ngh_at_same_level:
                        # Finer ngh in pull direction: YES
                        # Same-level ngh:              YES
                        # -> Coalescence
                        coalescence_factor = wp.neon_read(coalescence_factor_pn, index, l)
                        accumulated = accumulated * coalescence_factor
                        _f_post_stream[l] = self.compute_dtype(accumulated)

            return _f_post_stream

        @neon.Container.factory(name="collide_coarse")
        def collide_coarse(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, omega: Any, timestep: int):
            num_levels = f_0_fd.get_grid().num_levels

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                if level + 1 < f_0_fd.get_grid().num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if not wp.neon_has_child(f_0_pn, index):
                        _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                        collide_bc_accum(
                            index,
                            timestep,
                            _boundary_id,
                            _missing_mask,
                            f_0_pn,
                            f_1_pn,
                            _f0_thread,
                            omega,
                            num_levels,
                            level,
                            f_1_pn,
                        )
                    else:
                        for l in range(self.velocity_set.q):
                            wp.neon_write(f_1_pn, index, l, self.store_dtype(0))

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="SFV_collide_coarse")
        def SFV_collide_coarse(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, omega: Any, timestep: int):
            """Collision on SFV voxels only — no BCs, no multi-resolution accumulation."""

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id != wp.uint8(BC_SFV):
                        return
                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    collide_simple(
                        index,
                        0,
                        _boundary_id,
                        _missing_mask,
                        f_0_pn,
                        f_1_pn,
                        _f0_thread,
                        omega,
                        0,
                        level,
                        f_1_pn,
                    )

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="CFV_collide_coarse")
        def CFV_collide_coarse(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, omega: Any, timestep: int):
            """Collision on CFV voxels only — skips both solid and SFV."""
            num_levels = f_0_fd.get_grid().num_levels

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                if level + 1 < f_0_fd.get_grid().num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if _boundary_id == wp.uint8(BC_SFV):
                        return
                    if not wp.neon_has_child(f_0_pn, index):
                        _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                        collide_bc_accum(
                            index,
                            timestep,
                            _boundary_id,
                            _missing_mask,
                            f_0_pn,
                            f_1_pn,
                            _f0_thread,
                            omega,
                            num_levels,
                            level,
                            f_1_pn,
                        )
                    else:
                        for l in range(self.velocity_set.q):
                            wp.neon_write(f_1_pn, index, l, self.store_dtype(0))

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="stream_coarse_step_ABC")
        def stream_coarse_step_ABC(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            coalescence_factor: Any,
            timestep: int,
        ):
            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if wp.neon_has_child(f_0_pn, index):
                        return

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = neon_stream_explode_coalesce(index, f_0_pn, coalescence_factor_pn)

                    _f_post_stream = apply_bc_post_streaming(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream
                    )
                    neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_stream[l]))

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="SFV_stream_coarse_step_ABC")
        def SFV_stream_coarse_step_ABC(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            coalescence_factor: Any,
            timestep: int,
        ):
            """Stream on CFV voxels only — skips SFV and solid."""

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                coalescence_factor_pn = loader.get_mres_read_handle(coalescence_factor)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SFV):
                        return
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if wp.neon_has_child(f_0_pn, index):
                        return

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = neon_stream_explode_coalesce(index, f_0_pn, coalescence_factor_pn)

                    _f_post_stream = apply_bc_post_streaming(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream
                    )
                    neon_apply_aux_recovery_bc(index, _boundary_id, _missing_mask, f_0_pn, f_1_pn)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_stream[l]))

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="SFV_reset_bc_mask")
        def SFV_reset_bc_mask(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
        ):
            """
            Setting the BC type to BC_SFV
            """

            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                _c = self.velocity_set.c

                @wp.func
                def cl_stream_coarse(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if _boundary_id != 0:
                        return

                    if wp.neon_has_child(f_0_pn, index):
                        # we are a halo cell so we just exit
                        return

                    # do stream normally
                    _missing_mask = _missing_mask_vec()
                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                    for l in range(self.velocity_set.q):
                        if l == lattice_central_index:
                            continue

                        pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                        has_ngh_at_same_level = wp.bool(False)
                        wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)

                        if not wp.neon_has_finer_ngh(f_0_pn, index, pull_direction):
                            if not has_ngh_at_same_level:
                                if wp.neon_has_parent(f_0_pn, index):
                                    has_a_coarser_ngh = wp.bool(False)
                                    wp.neon_lbm_read_coarser_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_a_coarser_ngh)
                                    if has_a_coarser_ngh:
                                        # Explosion: not an SFV
                                        return
                        else:
                            if has_ngh_at_same_level:
                                # Coalescence: not an SFV
                                return

                    # Voxel is a pure fluid cell with no multi-resolution interactions — mark as SFV
                    wp.neon_write(bc_mask_pn, index, 0, wp.uint8(BC_SFV))

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        @neon.Container.factory(name="SFV_stream_coarse_step")
        def SFV_stream_coarse_step(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any):
            def ll_stream_coarse(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)

                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)

                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                _c = self.velocity_set.c

                @wp.func
                def cl_stream_coarse(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id != wp.uint8(BC_SFV):
                        return
                    # BC_SFV voxel type:
                    #   - They are not BC voxels
                    #   - They are not on a resolution jump -> they do not do coalescence or explosion
                    #   - They are not mr halo cells

                    _missing_mask = _missing_mask_vec()
                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)

                    for l in range(self.velocity_set.q):
                        wp.neon_write(f_1_pn, index, l, self.store_dtype(_f_post_stream[l]))

                loader.declare_kernel(cl_stream_coarse)

            return ll_stream_coarse

        @wp.func
        def neon_stream_finest_with_explosion(
            index: Any,
            f_0_pn: Any,
            explosion_src_pn: Any,
        ):
            _f_post_stream = self.stream.neon_functional(f_0_pn, index)

            for l in range(self.velocity_set.q):
                if l == lattice_central_index:
                    continue

                pull_direction = wp.neon_ngh_idx(wp.int8(-_c[0, l]), wp.int8(-_c[1, l]), wp.int8(-_c[2, l]))

                has_ngh_at_same_level = wp.bool(False)
                wp.neon_read_ngh(f_0_pn, index, pull_direction, l, self.store_dtype(0), has_ngh_at_same_level)

                if not has_ngh_at_same_level:
                    # No same-level ngh — could we have a coarser-level ngh?
                    if wp.neon_has_parent(f_0_pn, index):
                        # Parent exists — try to read the exploded population from the coarser level
                        has_a_coarser_ngh = wp.bool(False)
                        exploded_pop = wp.neon_lbm_read_coarser_ngh(
                            explosion_src_pn, index, pull_direction, l, self.store_dtype(0), has_a_coarser_ngh
                        )
                        if has_a_coarser_ngh:
                            # No finer ngh in pull direction, no same-level ngh,
                            # but a parent (ghost cell) exists with a coarser ngh
                            # -> Explosion: read the exploded population from the
                            #    coarser level's halo.
                            _f_post_stream[l] = self.compute_dtype(exploded_pop)

            return _f_post_stream

        @neon.Container.factory(name="finest_fused_pull")
        def finest_fused_pull(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: Any,
            is_f1_the_explosion_src_field: bool,
        ):
            if level != 0:
                raise Exception("Only the finest level is supported for now")
            num_levels = f_0_fd.get_grid().num_levels

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                if level + 1 < f_0_fd.get_grid().num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                explosion_src_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn
                accumulation_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if wp.neon_has_child(f_0_pn, index):
                        return

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = neon_stream_finest_with_explosion(index, f_0_pn, explosion_src_pn)

                    _f_post_stream = apply_bc_post_streaming(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream
                    )

                    collide_bc_accum(
                        index,
                        timestep,
                        _boundary_id,
                        _missing_mask,
                        f_0_pn,
                        f_1_pn,
                        _f_post_stream,
                        omega,
                        num_levels,
                        level,
                        accumulation_pn,
                    )

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="CFV_finest_fused_pull")
        def CFV_finest_fused_pull(
            level: int,
            f_0_fd: Any,
            f_1_fd: Any,
            bc_mask_fd: Any,
            missing_mask_fd: Any,
            omega: Any,
            timestep: Any,
            is_f1_the_explosion_src_field: bool,
        ):
            """Fused stream+collide on CFV voxels at the finest level — skips SFV and solid."""
            if level != 0:
                raise Exception("Only the finest level is supported for now")
            num_levels = f_0_fd.get_grid().num_levels

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                if level + 1 < f_0_fd.get_grid().num_levels:
                    f_0_pn = loader.get_mres_write_handle(f_0_fd, neon.Loader.Operation.stencil_up)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd, neon.Loader.Operation.stencil_up)
                else:
                    f_0_pn = loader.get_mres_read_handle(f_0_fd)
                    f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)
                explosion_src_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn
                accumulation_pn = f_1_pn if is_f1_the_explosion_src_field else f_0_pn

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id == wp.uint8(BC_SOLID):
                        return
                    if _boundary_id == wp.uint8(BC_SFV):
                        return
                    if wp.neon_has_child(f_0_pn, index):
                        return

                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_collision = _f0_thread
                    _f_post_stream = neon_stream_finest_with_explosion(index, f_0_pn, explosion_src_pn)

                    _f_post_stream = apply_bc_post_streaming(
                        index, timestep, _boundary_id, _missing_mask, f_0_pn, f_1_pn, _f_post_collision, _f_post_stream
                    )

                    collide_bc_accum(
                        index,
                        timestep,
                        _boundary_id,
                        _missing_mask,
                        f_0_pn,
                        f_1_pn,
                        _f_post_stream,
                        omega,
                        num_levels,
                        level,
                        accumulation_pn,
                    )

                loader.declare_kernel(device)

            return ll

        @neon.Container.factory(name="SFV_finest_fused_pull")
        def SFV_finest_fused_pull(level: int, f_0_fd: Any, f_1_fd: Any, bc_mask_fd: Any, missing_mask_fd: Any, omega: Any):
            """Fused stream+collide on SFV voxels at the finest level — no BCs, no explosion."""
            if level != 0:
                raise Exception("Only the finest level is supported for now")

            def ll(loader: neon.Loader):
                loader.set_mres_grid(bc_mask_fd.get_grid(), level)
                f_0_pn = loader.get_mres_read_handle(f_0_fd)
                f_1_pn = loader.get_mres_write_handle(f_1_fd)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask_fd)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask_fd)

                @wp.func
                def device(index: Any):
                    _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
                    if _boundary_id != wp.uint8(BC_SFV):
                        return
                    _f0_thread, _missing_mask = neon_get_thread_data(f_0_pn, missing_mask_pn, index)
                    _f_post_stream = self.stream.neon_functional(f_0_pn, index)
                    collide_simple(
                        index,
                        0,
                        _boundary_id,
                        _missing_mask,
                        f_0_pn,
                        f_1_pn,
                        _f_post_stream,
                        omega,
                        0,
                        0,
                        f_1_pn,
                    )

                loader.declare_kernel(device)

            return ll

        return None, {
            "collide_coarse": collide_coarse,
            "stream_coarse_step_ABC": stream_coarse_step_ABC,
            "finest_fused_pull": finest_fused_pull,
            "CFV_finest_fused_pull": CFV_finest_fused_pull,
            "SFV_finest_fused_pull": SFV_finest_fused_pull,
            "SFV_reset_bc_mask": SFV_reset_bc_mask,
            "CFV_collide_coarse": CFV_collide_coarse,
            "SFV_collide_coarse": SFV_collide_coarse,
            "SFV_stream_coarse_step_ABC": SFV_stream_coarse_step_ABC,
            "SFV_stream_coarse_step": SFV_stream_coarse_step,
        }

    def add_to_app(self, **kwargs):
        """Append a container invocation to the Neon skeleton application list.

        Required keyword arguments are ``op_name`` (str) and ``app`` (list).
        All remaining keyword arguments are forwarded to the container
        factory for the given ``op_name``.  Argument validation is performed
        before the call, and a ``ValueError`` is raised on mismatch.
        """
        import inspect

        def validate_kwargs_forward(func, kwargs):
            """
            Check whether `func(**kwargs)` would be valid,
            and return *all* the issues instead of raising on the first one.

            Returns a dict; empty dict means "everything is OK".
            """
            sig = inspect.signature(func)
            params = sig.parameters

            errors = {}

            # --- 1. Positional-only required params (cannot be given via kwargs) ---
            pos_only_required = [name for name, p in params.items() if p.kind == inspect.Parameter.POSITIONAL_ONLY and p.default is inspect._empty]
            if pos_only_required:
                errors["positional_only_required"] = pos_only_required

            # --- 2. Unexpected kwargs (if no **kwargs in target) ---
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not has_var_kw:
                allowed_kw = {
                    name
                    for name, p in params.items()
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                unexpected = sorted(set(kwargs) - allowed_kw)
                if unexpected:
                    errors["unexpected_kwargs"] = unexpected

            # --- 3. Missing required keyword-passable params ---
            missing_required = [
                name
                for name, p in params.items()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and p.default is inspect._empty  # no default
                and name not in kwargs  # not provided
            ]
            if missing_required:
                errors["missing_required"] = missing_required

            return errors

        container_generator = None
        try:
            op_name = kwargs.pop("op_name")
            app = kwargs.pop("app")
        except KeyError:
            raise ValueError("op_name and app must be provided as keyword arguments")

        try:
            container_generator = self.neon_container[op_name]
        except KeyError:
            raise ValueError(f"Operator {op_name} not found in neon container. Available operators: {list(self.neon_container.keys())}")

        errors = validate_kwargs_forward(container_generator, kwargs)
        if errors:
            raise ValueError(f"Cannot forward kwargs to target: {errors}")

        nvtx.push_range(f"New Container {op_name}", color="yellow")
        app.append(container_generator(**kwargs))
        nvtx.pop_range()

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_launch(self, f_0, f_1, bc_mask, missing_mask, omega, timestep):
        raise NotImplementedError("Use MultiresSimulationManager.step() instead of launching this stepper directly.")
