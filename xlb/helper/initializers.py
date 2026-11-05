"""
Initializers for distribution function fields.

Provides helper functions and Operator subclasses that populate
distribution-function fields with equilibrium values.  Two usage patterns
are supported:

* **Functional helpers** (`initialize_eq`, `initialize_multires_eq`) —
  one-shot initialization used during simulation setup.
* **Operator classes** (`CustomInitializer`, `CustomMultiresInitializer`) —
  reusable operators that can target the whole domain or a single boundary
  condition region, with support for JAX, Warp, and Neon backends.
"""

import warp as wp
from typing import Any
from xlb import DefaultConfig
from xlb.operator import Operator
from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.equilibrium import MultiresQuadraticEquilibrium


def initialize_eq(f, grid, velocity_set, precision_policy, compute_backend, rho=None, u=None):
    """Initialize a distribution-function field to equilibrium.

    Computes the quadratic equilibrium for the given density and velocity
    fields and writes it into *f*.  When *rho* or *u* are ``None`` the
    defaults are uniform density 1 and zero velocity.

    Parameters
    ----------
    f : field
        Distribution-function field to populate (modified in-place for
        Warp / Neon backends; replaced for JAX).
    grid : Grid
        Computational grid used to allocate temporary fields.
    velocity_set : VelocitySet
        Lattice velocity set (e.g. D3Q19).
    precision_policy : PrecisionPolicy
        Precision policy for compute / store dtypes.
    compute_backend : ComputeBackend
        Active compute backend (JAX, WARP, or NEON).
    rho : field, optional
        Density field.  Defaults to uniform 1.0.
    u : field, optional
        Velocity field.  Defaults to uniform 0.0.

    Returns
    -------
    field
        The initialized distribution-function field.
    """
    if rho is None:
        rho = grid.create_field(cardinality=1, fill_value=1.0, dtype=precision_policy.compute_precision)
    if u is None:
        u = grid.create_field(cardinality=velocity_set.d, fill_value=0.0, dtype=precision_policy.compute_precision)
    equilibrium = QuadraticEquilibrium()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(rho, u)
    elif compute_backend == ComputeBackend.WARP:
        f = equilibrium(rho, u, f)
    elif compute_backend == ComputeBackend.NEON:
        f = equilibrium(rho, u, f)
    else:
        raise NotImplementedError(f"Backend {compute_backend} not implemented")

    del rho, u

    return f


def initialize_multires_eq(f, grid, velocity_set, precision_policy, backend, rho, u):
    """Initialize a multi-resolution distribution-function field to equilibrium.

    Parameters
    ----------
    f : field
        Multi-resolution distribution-function field to populate.
    grid : NeonMultiresGrid
        Multi-resolution grid.
    velocity_set : VelocitySet
        Lattice velocity set.
    precision_policy : PrecisionPolicy
        Precision policy.
    backend : ComputeBackend
        Compute backend (expected to be NEON).
    rho : field
        Density field across all grid levels.
    u : field
        Velocity field across all grid levels.

    Returns
    -------
    field
        The initialized multi-resolution distribution-function field.
    """
    equilibrium = MultiresQuadraticEquilibrium()
    return equilibrium(rho, u, f, stream=0)


class CustomInitializer(Operator):
    """Operator that initializes distribution functions to equilibrium.

    When ``bc_id == -1`` (default) the entire domain is initialized with the
    given constant velocity and density.  Otherwise only voxels whose
    ``bc_mask`` matches *bc_id* are set while the rest receive the
    weight-only equilibrium (zero velocity, unit density).

    Supports JAX, Warp, and Neon backends.

    Parameters
    ----------
    constant_velocity_vector : list of float
        Macroscopic velocity [ux, uy, uz] used for initialization.
    constant_density : float
        Macroscopic density used for initialization.
    bc_id : int
        Boundary-condition ID to target.  ``-1`` means the whole domain.
    initialization_operator : Operator, optional
        Equilibrium operator to use.  Defaults to ``QuadraticEquilibrium``.
    velocity_set : VelocitySet, optional
    precision_policy : PrecisionPolicy, optional
    compute_backend : ComputeBackend, optional
    """

    def __init__(
        self,
        constant_velocity_vector=[0.0, 0.0, 0.0],
        constant_density: float = 1.0,
        bc_id: int = -1,
        initialization_operator=None,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        self.bc_id = bc_id
        self.constant_velocity_vector = constant_velocity_vector
        self.constant_density = constant_density
        if initialization_operator is None:
            compute_backend = compute_backend or DefaultConfig.default_backend
            self.initialization_operator = QuadraticEquilibrium(
                velocity_set=velocity_set or DefaultConfig.velocity_set,
                precision_policy=precision_policy or DefaultConfig.precision_policy,
                compute_backend=compute_backend if compute_backend == ComputeBackend.JAX else ComputeBackend.WARP,
            )
        super().__init__(velocity_set, precision_policy, compute_backend)

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(self, bc_mask, f_field):
        from xlb.grid import grid_factory
        import jax.numpy as jnp

        grid_shape = f_field.shape[1:]
        grid = grid_factory(grid_shape)
        rho_init = grid.create_field(cardinality=1, fill_value=self.constant_density, dtype=self.precision_policy.compute_precision)
        u_init = grid.create_field(cardinality=self.velocity_set.d, fill_value=0.0, dtype=self.precision_policy.compute_precision)
        _vel = jnp.array(self.constant_velocity_vector)[(...,) + (None,) * self.velocity_set.d]
        if self.bc_id == -1:
            u_init += _vel
        else:
            u_init = jnp.where(bc_mask[0] == self.bc_id, u_init + _vel, u_init)
        return self.initialization_operator(rho_init, u_init)

    def _construct_warp(self):
        _q = self.velocity_set.q
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _u = _u_vec(self.constant_velocity_vector[0], self.constant_velocity_vector[1], self.constant_velocity_vector[2])
        _rho = self.compute_dtype(self.constant_density)
        _w = self.velocity_set.w
        bc_id = self.bc_id

        @wp.func
        def functional_local(index: Any, bc_mask: Any, f_field: Any):
            # Check if the index corresponds to the outlet
            if self.read_field(bc_mask, index, 0) == bc_id:
                _f_init = self.initialization_operator.warp_functional(_rho, _u)
                for l in range(_q):
                    self.write_field(f_field, index, l, self.store_dtype(_f_init[l]))
            else:
                # In the rest of the domain, we assume zero velocity and equilibrium distribution.
                for l in range(_q):
                    self.write_field(f_field, index, l, self.store_dtype(_w[l]))

        @wp.func
        def functional_domain(index: Any, bc_mask: Any, f_field: Any):
            # If bc_id is -1, initialize the entire domain according to the custom initialization operator for the given velocity
            _f_init = self.initialization_operator.warp_functional(_rho, _u)
            for l in range(_q):
                self.write_field(f_field, index, l, self.store_dtype(_f_init[l]))

        # Set the functional based on whether we are initializing a specific BC or the entire domain
        functional = functional_local if self.bc_id != -1 else functional_domain

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            bc_mask: wp.array4d(dtype=wp.uint8),
            f_field: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Set the velocity at the outlet (i.e. where i = nx-1)
            functional(index, bc_mask, f_field)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, bc_mask, f_field):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[bc_mask, f_field],
            dim=f_field.shape[1:],
        )
        return f_field

    def _construct_neon(self):
        import neon

        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="CustomInitializer")
        def container(
            bc_mask: Any,
            f_field: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_grid(f_field.get_grid())
                f_field_pn = loader.get_write_handle(f_field)
                bc_mask_pn = loader.get_read_handle(bc_mask)

                @wp.func
                def kernel(index: Any):
                    # apply the functional
                    functional(index, bc_mask_pn, f_field_pn)

                loader.declare_kernel(kernel)

            return launcher

        return _, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bc_mask, f_field, stream=0):
        import neon

        # Launch the neon container
        c = self.neon_container(bc_mask, f_field)
        c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_field


class CustomMultiresInitializer(CustomInitializer):
    """Multi-resolution variant of :class:`CustomInitializer`.

    Iterates over all grid levels and initializes distribution functions
    using the Neon multi-resolution container API.
    """

    def __init__(
        self,
        constant_velocity_vector=[0.0, 0.0, 0.0],
        constant_density: float = 1.0,
        bc_id: int = -1,
        initialization_operator=None,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(constant_velocity_vector, constant_density, bc_id, initialization_operator, velocity_set, precision_policy, compute_backend)

    def _construct_neon(self):
        import neon

        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="CustomMultiresInitializer")
        def container(
            bc_mask: Any,
            f_field: Any,
            level: Any,
        ):
            def launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_write_handle(f_field)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)

                @wp.func
                def kernel(index: Any):
                    # apply the functional
                    functional(index, bc_mask_pn, f_field_pn)

                loader.declare_kernel(kernel)

            return launcher

        return _, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bc_mask, f_field, stream=0):
        import neon

        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            # Launch the neon container
            c = self.neon_container(bc_mask, f_field, level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_field
