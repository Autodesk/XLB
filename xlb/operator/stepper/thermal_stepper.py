"""
Single-resolution scalar-transport (thermal) stepper.

Implements the full LBM step (stream, collide, apply BCs, add source) for a set of scalar
populations on a single-resolution grid. The scalar obeys

    d(phi)/dt + div(phi * u) = div(diffusivity * grad(phi)) + source

which covers the heat equation (``u = 0``), passive scalar advection-diffusion, and any
volumetric production term. Supports pull streaming on JAX and a pull-only fused kernel on
Warp.
"""

from functools import partial

from jax import jit
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK
from xlb.operator.equilibrium import LinearEquilibrium
from xlb.operator.macroscopic import ZeroMoment
from xlb.operator.source import ScalarSource
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.helper import check_bc_overlaps
from xlb.helper.thermal_fields import create_thermal_fields
from xlb.cell_type import BC_SOLID


class ThermalStepper(Stepper):
    """Single-resolution scalar-transport (thermal) LBM stepper.

    Composes streaming, collision, the linear scalar equilibrium, the scalar moment, the
    volumetric source, and boundary-condition operators into a complete timestep.

    The transported scalar is the zeroth moment of the populations, so it is recovered with
    :class:`~xlb.operator.macroscopic.ZeroMoment` exactly as the density is recovered from
    the flow populations.

    The stepper carries its own populations (``g_0``, ``g_1``) and its own boundary masks,
    so it is completely independent of the flow solver. Coupling to a flow field is
    one-way and explicit: the caller recovers the velocity from the Navier-Stokes
    populations and passes it in as the *u* argument. Passing a zero velocity field reduces
    the transport equation to the heat equation.

    Parameters
    ----------
    grid : Grid
        Computational grid. Must be the same grid as the flow solver when coupling.
    boundary_conditions : list of BoundaryCondition
        Boundary conditions to apply each step. These must be separate objects from the
        flow boundary conditions since they carry their own registry ids and act on the
        scalar populations (e.g. :class:`ScalarDirichletBC`, :class:`ScalarNeumannBC`).
    velocity_set : VelocitySet, optional
        Stencil for the scalar populations. Defaults to ``DefaultConfig.velocity_set``, but
        a reduced stencil such as ``D2Q5`` is cheaper and sufficient in 2D.
    collision_type : str
        ``"BGK"``. Higher-order collision models are not meaningful for the linear scalar
        equilibrium.
    """

    def __init__(
        self,
        grid,
        boundary_conditions=[],
        velocity_set=None,
        collision_type="BGK",
    ):
        # Note: Stepper.__init__ is deliberately bypassed here. It forces the velocity set to
        # DefaultConfig.velocity_set, whereas the scalar populations usually live on a
        # reduced stencil (e.g. D2Q5) while the flow keeps the default one (e.g. D2Q9).
        self.grid = grid
        self.boundary_conditions = boundary_conditions

        velocity_set = velocity_set or DefaultConfig.velocity_set
        Operator.__init__(self, velocity_set, DefaultConfig.default_precision_policy, DefaultConfig.default_backend)

        # Boundary conditions are constructed by the user before the stepper exists, so they
        # default to DefaultConfig.velocity_set, which is the *flow* stencil whenever the two
        # solvers are coupled. Re-bind them to the scalar stencil here so that callers never
        # have to pass the velocity set twice.
        self._adopt_boundary_conditions()

        # Construct the collision operator
        if collision_type == "BGK":
            self.collision = BGK(self.velocity_set, self.precision_policy, self.compute_backend)
        else:
            raise ValueError(f"Unsupported collision type for scalar transport: {collision_type}")

        # Construct the operators
        self.stream = Stream(self.velocity_set, self.precision_policy, self.compute_backend)
        self.equilibrium = LinearEquilibrium(self.velocity_set, self.precision_policy, self.compute_backend)
        self.zero_moment = ZeroMoment(self.velocity_set, self.precision_policy, self.compute_backend)
        self.source = ScalarSource(self.velocity_set, self.precision_policy, self.compute_backend)

    def _adopt_boundary_conditions(self):
        """Re-bind the boundary conditions to the stepper's stencil and precision.

        Re-running ``Operator.__init__`` also rebuilds the backend functionals and kernels,
        which is required because they close over the stencil at construction time. The BC
        registry id and its indices are set outside of ``Operator.__init__`` and so survive.
        """
        for bc in self.boundary_conditions:
            if bc.velocity_set is self.velocity_set and bc.compute_backend == self.compute_backend:
                continue
            Operator.__init__(bc, self.velocity_set, self.precision_policy, self.compute_backend)

    def prepare_fields(self, phi_init=0.0):
        """Prepare the scalar populations and boundary masks required for the stepper.

        Args:
            phi_init: Initial scalar value. Either a scalar (uniform initialization) or a
                      field of cardinality 1 holding the initial scalar distribution.

        Returns:
            Tuple of (g_0, g_1, bc_mask, missing_mask):
                - g_0: Initial scalar distribution functions
                - g_1: Copy of g_0 for double-buffering
                - bc_mask: Boundary condition mask indicating which BC applies to each node
                - missing_mask: Mask indicating which populations are missing at boundary nodes
        """
        # Create fields using the helper function
        _, g_0, g_1, missing_mask, bc_mask = create_thermal_fields(
            grid=self.grid, velocity_set=self.velocity_set, compute_backend=self.compute_backend, precision_policy=self.precision_policy
        )

        # The scalar boundary conditions provided by XLB are all local in the populations and
        # therefore need none of the auxiliary-data machinery of the Navier-Stokes stepper.
        for bc in self.boundary_conditions:
            if bc.needs_aux_init:
                raise ValueError(f"Boundary condition {bc.__class__.__name__} requires auxiliary data and is not supported for scalar transport")

        # Process boundary conditions and update masks
        bc_mask, missing_mask = self._process_boundary_conditions(self.boundary_conditions, bc_mask, missing_mask)

        # Initialize the populations at equilibrium with a zero advecting velocity, which for
        # the linear scalar equilibrium is simply g[l] = w[l] * phi_init
        g_0 = self._initialize_eq(g_0, phi_init)

        # Copy g_0 using backend-specific copy to g_1
        if self.compute_backend == ComputeBackend.JAX:
            g_1 = g_0.copy()
        if self.compute_backend == ComputeBackend.WARP:
            wp.copy(g_1, g_0)

        return g_0, g_1, bc_mask, missing_mask

    def prepare_aux_fields(self):
        """Allocate the advecting velocity and volumetric source fields.

        Both are zero-filled, which makes the stepper solve the pure heat equation out of
        the box. Fill the velocity field from the flow solver to enable advection, and the
        source field with ``q''' * dt / (rho * cp)`` to enable volumetric heating.

        Returns:
            Tuple of (u, source) fields, of cardinality d and 1 respectively.
        """
        u = self.grid.create_field(cardinality=self.velocity_set.d, dtype=self.precision_policy.store_precision)
        source = self.grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        return u, source

    def _initialize_eq(self, g_0, phi_init):
        """Initialize the scalar populations at rest equilibrium for the given scalar value."""
        if isinstance(phi_init, (int, float)):
            phi = self.grid.create_field(cardinality=1, fill_value=float(phi_init), dtype=self.precision_policy.compute_precision)
        else:
            phi = phi_init
        u = self.grid.create_field(cardinality=self.velocity_set.d, fill_value=0.0, dtype=self.precision_policy.compute_precision)

        if self.compute_backend == ComputeBackend.JAX:
            g_0 = self.equilibrium(phi, u)
        else:
            g_0 = self.equilibrium(phi, u, g_0)

        del u
        return g_0

    def _process_boundary_conditions(self, boundary_conditions, bc_mask, missing_mask):
        """Process boundary conditions and update boundary masks."""

        # Check for boundary condition overlaps
        check_bc_overlaps(boundary_conditions, self.velocity_set.d, self.compute_backend)

        # Create boundary masker using the scalar velocity set rather than the default one
        indices_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
            grid=self.grid,
        )

        bc_with_indices = [bc for bc in boundary_conditions if bc.indices is not None]
        if len(bc_with_indices) != len(boundary_conditions):
            raise ValueError("Scalar boundary conditions must be specified with indices; mesh-based scalar BCs are not supported yet")

        if bc_with_indices:
            bc_mask, missing_mask = indices_masker(bc_with_indices, bc_mask, missing_mask)

        return bc_mask, missing_mask

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, g_0, g_1, u, source, bc_mask, missing_mask, omega, timestep):
        """
        Perform a single scalar-transport step of the lattice Boltzmann method
        """
        # Cast to compute precision
        g_0 = self.precision_policy.cast_to_compute_jax(g_0)
        u = self.precision_policy.cast_to_compute_jax(u)
        source = self.precision_policy.cast_to_compute_jax(source)

        # Apply streaming
        g_post_stream = self.stream(g_0)

        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                g_post_stream = bc(
                    g_0,
                    g_post_stream,
                    bc_mask,
                    missing_mask,
                )

        # Compute the macroscopic scalar
        phi = self.zero_moment(g_post_stream)

        # Compute equilibrium
        geq = self.equilibrium(phi, u)

        # Apply collision
        g_post_collision = self.collision(g_post_stream, geq, omega)

        # Apply the volumetric source term
        g_post_collision = self.source(g_post_collision, source)

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.COLLISION:
                g_post_collision = bc(
                    g_post_stream,
                    g_post_collision,
                    bc_mask,
                    missing_mask,
                )

        # Copy back to store precision
        g_1 = self.precision_policy.cast_to_store_jax(g_post_collision)

        return g_0, g_1

    def _construct_warp(self):
        # Set local constants
        _g_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        @wp.func
        def apply_bc(
            index: Any,
            timestep: Any,
            _boundary_id: Any,
            _missing_mask: Any,
            g_0: Any,
            g_1: Any,
            g_pre: Any,
            g_post: Any,
            is_post_streaming: bool,
        ):
            g_result = g_post

            # Unroll the loop over boundary conditions
            for i in range(wp.static(len(self.boundary_conditions))):
                if is_post_streaming:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.STREAMING):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            g_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, _missing_mask, g_0, g_1, g_pre, g_post)
                else:
                    if wp.static(self.boundary_conditions[i].implementation_step == ImplementationStep.COLLISION):
                        if _boundary_id == wp.static(self.boundary_conditions[i].id):
                            g_result = wp.static(self.boundary_conditions[i].warp_functional)(index, timestep, _missing_mask, g_0, g_1, g_pre, g_post)
            return g_result

        @wp.func
        def get_thread_data(
            g0_buffer: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            index: Any,
        ):
            # Read thread data for populations
            _g0_thread = _g_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                _g0_thread[l] = self.compute_dtype(g0_buffer[l, index[0], index[1], index[2]])
                _missing_mask[l] = missing_mask[l, index[0], index[1], index[2]]

            return _g0_thread, _missing_mask

        @wp.kernel
        def kernel(
            g_0: wp.array4d(dtype=Any),
            g_1: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            source: wp.array4d(dtype=Any),
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
            _g_post_stream = self.stream.warp_functional(g_0, index)

            _g0_thread, _missing_mask = get_thread_data(g_0, missing_mask, index)
            _g_post_collision = _g0_thread

            # Apply post-streaming boundary conditions
            _g_post_stream = apply_bc(index, timestep, _boundary_id, _missing_mask, g_0, g_1, _g_post_collision, _g_post_stream, True)

            # Read the advecting velocity and the volumetric source
            _u = _u_vec()
            for d in range(self.velocity_set.d):
                _u[d] = self.compute_dtype(u[d, index[0], index[1], index[2]])
            _source = self.compute_dtype(source[0, index[0], index[1], index[2]])

            _phi = self.zero_moment.warp_functional(_g_post_stream)
            _geq = self.equilibrium.warp_functional(_phi, _u)
            _g_post_collision = self.collision.warp_functional(_g_post_stream, _geq, omega)
            _g_post_collision = self.source.warp_functional(_g_post_collision, _source)

            # Apply post-collision boundary conditions
            _g_post_collision = apply_bc(index, timestep, _boundary_id, _missing_mask, g_0, g_1, _g_post_stream, _g_post_collision, False)

            # Store the result in g_1
            for l in range(self.velocity_set.q):
                g_1[l, index[0], index[1], index[2]] = self.store_dtype(_g_post_collision[l])

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, g_0, g_1, u, source, bc_mask, missing_mask, omega, timestep):
        wp.launch(
            self.warp_kernel,
            inputs=[g_0, g_1, u, source, bc_mask, missing_mask, omega, timestep],
            dim=g_0.shape[1:],
            device=g_0.device,
        )
        return g_0, g_1
