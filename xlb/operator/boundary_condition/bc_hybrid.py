"""
Hybrid boundary condition combining interpolated bounce-back with regularization.

Provides three wall-treatment strategies, selectable via *bc_method*:

* ``"bounceback_regularized"`` — interpolated bounce-back + Latt regularization.
* ``"bounceback_grads"`` — interpolated bounce-back + Grad's approximation.
* ``"nonequilibrium_regularized"`` — Tao non-equilibrium bounce-back + Latt
  regularization.

All variants optionally support:

* Moving walls (via *prescribed_value* or *profile*).
* Curved boundaries with fractional distance to the mesh surface (via
  *use_mesh_distance*).
"""

import inspect
from jax import jit
from functools import partial
import warp as wp
from typing import Any, Union, Tuple, Callable
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
    HelperFunctionsBC,
)
from xlb.operator.boundary_masker.mesh_voxelization_method import MeshVoxelizationMethod


class HybridBC(BoundaryCondition):
    """
    The hybrid BC methods in this boundary condition have been originally developed by H. Salehipour and are inspired from
    various previous publications, in particular [1]. The reformulations are aimed to provide local formulations that are
    computationally efficient and numerically stable at high Reynolds numbers.

    [1] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
        stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.
    """

    def __init__(
        self,
        bc_method,
        profile: Callable = None,
        prescribed_value: Union[float, Tuple[float, ...], np.ndarray] = None,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
        voxelization_method: MeshVoxelizationMethod = None,
        use_mesh_distance=False,
    ):
        """
        Parameters
        ----------
        bc_method : str
            Wall-treatment strategy.  One of ``"bounceback_regularized"``,
            ``"bounceback_grads"``, or ``"nonequilibrium_regularized"``.
        profile : callable, optional
            Warp function ``(index) -> u_vec`` or ``(index, timestep) -> u_vec``
            defining the wall velocity.  Mutually exclusive with *prescribed_value*.
        prescribed_value : float or array-like, optional
            Constant wall velocity vector.  Mutually exclusive with *profile*.
            If neither is given, a no-slip wall is assumed.
        velocity_set : VelocitySet, optional
        precision_policy : PrecisionPolicy, optional
        compute_backend : ComputeBackend, optional
        indices : list of array-like, optional
            Boundary voxel indices (use this **or** *mesh_vertices*, not both).
        mesh_vertices : np.ndarray, optional
            Mesh triangle vertices for mesh-based voxelization.
        voxelization_method : MeshVoxelizationMethod, optional
            Voxelization strategy (AABB, RAY, AABB_CLOSE, etc.).
        use_mesh_distance : bool
            If ``True``, fractional distances to the mesh surface are
            computed and stored for interpolated boundary schemes.
        """
        assert bc_method in [
            "bounceback_regularized",
            "bounceback_grads",
            "nonequilibrium_regularized",
        ], f"type = {bc_method} not supported! Use 'bounceback_regularized', 'bounceback_grads' or 'nonequilibrium_regularized'."
        self.bc_method = bc_method

        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
            voxelization_method,
        )

        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This BC is not implemented in 2D!")

        # Check if the compute backend is Warp
        assert self.compute_backend in (ComputeBackend.WARP, ComputeBackend.NEON), "This BC is currently not supported by JAX backend!"

        # Instantiate the operator for computing macroscopic values
        # Explicitly using the WARP backend for these operators as they may also be called by the Neon backend.
        self.macroscopic = Macroscopic(compute_backend=ComputeBackend.WARP)
        self.equilibrium = QuadraticEquilibrium(compute_backend=ComputeBackend.WARP)

        # Define BC helper functions. Explicitly using the WARP backend for helper functions as it may also be called by the Neon backend.
        self.bc_helper = HelperFunctionsBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=ComputeBackend.WARP,
            distance_decoder_function=self._construct_distance_decoder_function(),
        )

        # A flag to enable moving wall treatment when either "prescribed_value" or "profile" are provided.
        self.needs_moving_wall_treatment = False

        if (profile is not None) or (prescribed_value is not None):
            self.needs_moving_wall_treatment = True

        # Handle no-slip BCs if neither prescribed_value or profile are provided.
        if prescribed_value is None and profile is None:
            print(f"WARNING! Assuming no-slip condition for BC type = {self.__class__.__name__}_{self.bc_method}!")
            prescribed_value = [0, 0, 0]

        # Handle prescribed value if provided
        if prescribed_value is not None:
            assert profile is None, "Cannot specify both profile and prescribed_value"

            # Ensure prescribed_value is a NumPy array of floats
            if isinstance(prescribed_value, (tuple, list, np.ndarray)):
                prescribed_value = np.asarray(prescribed_value, dtype=np.float64)
            else:
                raise ValueError("Velocity prescribed_value must be a tuple, list, or array")

            # Handle 2D velocity sets
            if self.velocity_set.d == 2:
                assert len(prescribed_value) == 2, "For 2D velocity set, prescribed_value must be a tuple or array of length 2!"
                prescribed_value = np.array([prescribed_value[0], prescribed_value[1], 0.0], dtype=np.float64)

            # create a constant prescribed profile
            _u_vec = wp.vec(3, dtype=self.compute_dtype)
            prescribed_value = _u_vec(prescribed_value)

            @wp.func
            def prescribed_profile_warp(index: Any):
                return _u_vec(prescribed_value[0], prescribed_value[1], prescribed_value[2])

            profile = prescribed_profile_warp

        # Inspect the function signature and add time parameter if needed
        self.is_time_dependent = False
        sig = inspect.signature(profile)
        if len(sig.parameters) > 1:
            # We assume the profile function takes only the index as input and is hence time-independent.
            # In case it is defined with more than 1 input, we assume the second input is time and create
            # a wrapper function that also accepts time as a parameter.
            self.is_time_dependent = True

        # This BC class accepts both constant prescribed values of velocity with keyword "prescribed_value" or
        # velocity profiles given by keyword "profile" which must be a callable function.
        self.profile = profile

        # Set whether this BC needs mesh distance
        self.needs_mesh_distance = use_mesh_distance

        # This BC needs normalized distance to the mesh
        if self.needs_mesh_distance:
            # This BC needs auxiliary data recovery after streaming
            self.needs_aux_recovery = True

        # If this BC is defined using indices, it would need padding in order to find missing directions
        # when imposed on a geometry that is in the domain interior
        if self.mesh_vertices is None:
            assert self.indices is not None
            assert self.needs_mesh_distance is False, 'To use mesh distance, please provide the mesh vertices using keyword "mesh_vertices"!'
            assert self.voxelization_method is None, "Voxelization method is only applicable when using mesh vertices!"
            self.needs_padding = True
        else:
            assert self.indices is None, "Cannot use indices with mesh vertices! Please provide mesh vertices only."

        # Define the profile functional
        self.profile_functional = self._construct_profile_functional()

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        raise NotImplementedError(f"Operation {self.__class__.__name__} not implemented in JAX!")

    def _construct_distance_decoder_function(self):
        """
        Constructs the distance decoder function for this BC.
        """
        # Get the opposite indices for the velocity set
        _opp_indices = self.velocity_set.opp_indices

        # Define the distance decoder function for this BC
        @wp.func
        def distance_decoder_function(f_1: Any, index: Any, direction: Any):
            return self.read_field(f_1, index, _opp_indices[direction])

        return distance_decoder_function

    def _construct_profile_functional(self):
        """
        Get the profile functional for this BC.
        TODO@Hesam:
        Right now, we can impose a profile on a boundary which requires mesh-distance only if that boundary lives on the finest level.
        In order to extract "level" from the "neon_field_hdl" we can use the function wp.neon_level(neon_field_hdl). This will allow us
        to do the following and get rid of the above limitation.
               cIdx = wp.neon_global_idx(field_neon_hdl, index)
               gx = wp.neon_get_x(cIdx) // 2 ** level
               gy = wp.neon_get_y(cIdx) // 2 ** level
               gz = wp.neon_get_z(cIdx) // 2 ** level
        """

        @wp.func
        def profile_functional_neon(f_1: Any, index: Any, timestep: Any):
            # Convert neon index to warp index
            warp_index = self.bc_helper.neon_index_to_warp(f_1, index)
            if wp.static(self.is_time_dependent):
                return self.profile(warp_index, timestep)
            else:
                return self.profile(warp_index)

        @wp.func
        def profile_functional_warp(f_1: Any, index: Any, timestep: Any):
            if wp.static(self.is_time_dependent):
                return self.profile(index, timestep)
            else:
                return self.profile(index)

        return profile_functional_warp if self.compute_backend == ComputeBackend.WARP else profile_functional_neon

    def _construct_warp(self):
        # Construct the functionals for this BC
        @wp.func
        def hybrid_bounceback_regularized(
            index: Any,
            timestep: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Using regularization technique [1] to represent fpop using macroscopic values derived from interpolated bounceback scheme of [2].
            # missing data in lattice Boltzmann.
            # [1] Latt, J., Chopard, B., Malaspinas, O., Deville, M., Michler, A., 2008. Straight velocity
            #     boundaries in the lattice Boltzmann method. Physical Review E 77, 056703.
            # [2] Yu, D., Mei, R., Shyy, W., 2003. A unified boundary treatment in lattice boltzmann method,
            #     in: 41st aerospace sciences meeting and exhibit, p. 953.

            # Apply interpolated bounceback first to find missing populations at the boundary
            u_wall = self.profile_functional(f_1, index, timestep)
            f_post = self.bc_helper.interpolated_bounceback(
                index,
                _missing_mask,
                f_0,
                f_1,
                f_pre,
                f_post,
                u_wall,
                wp.static(self.needs_moving_wall_treatment),
                wp.static(self.needs_mesh_distance),
            )

            # Compute density, velocity using all f_post-streaming values
            rho, u = self.macroscopic.warp_functional(f_post)

            # Regularize the resulting populations
            feq = self.equilibrium.warp_functional(rho, u)
            f_post = self.bc_helper.regularize_fpop(f_post, feq)
            return f_post

        @wp.func
        def hybrid_bounceback_grads(
            index: Any,
            timestep: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Using Grad's approximation [1] to represent fpop using macroscopic values derived from interpolated bounceback scheme of [2].
            # missing data in lattice Boltzmann.
            # [1] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
            #    stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.
            # [2] Yu, D., Mei, R., Shyy, W., 2003. A unified boundary treatment in lattice boltzmann method,
            #     in: 41st aerospace sciences meeting and exhibit, p. 953.

            # Apply interpolated bounceback first to find missing populations at the boundary
            u_wall = self.profile_functional(f_1, index, timestep)
            f_post = self.bc_helper.interpolated_bounceback(
                index,
                _missing_mask,
                f_0,
                f_1,
                f_pre,
                f_post,
                u_wall,
                wp.static(self.needs_moving_wall_treatment),
                wp.static(self.needs_mesh_distance),
            )

            # Compute density, velocity using all f_post-streaming values
            rho, u = self.macroscopic.warp_functional(f_post)

            # Compute Grad's approximation using full equation as in Eq (10) of Dorschner et al.
            f_post = self.bc_helper.grads_approximate_fpop(_missing_mask, rho, u, f_post)
            return f_post

        @wp.func
        def hybrid_nonequilibrium_regularized(
            index: Any,
            timestep: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # This boundary condition uses the method of Tao et al (2018) [1] to get unknown populations on curved boundaries (denoted here by
            # interpolated_nonequilibrium_bounceback method). To further stabilize this BC, we add regularization technique of [2].
            # [1] Tao, Shi, et al. "One-point second-order curved boundary condition for lattice Boltzmann simulation of suspended particles."
            #     Computers & Mathematics with Applications 76.7 (2018): 1593-1607.
            # [2] Latt, J., Chopard, B., Malaspinas, O., Deville, M., Michler, A., 2008. Straight velocity
            #     boundaries in the lattice Boltzmann method. Physical Review E 77, 056703.

            # Apply interpolated bounceback first to find missing populations at the boundary
            u_wall = self.profile_functional(f_1, index, timestep)
            f_post = self.bc_helper.interpolated_nonequilibrium_bounceback(
                index,
                _missing_mask,
                f_0,
                f_1,
                f_pre,
                f_post,
                u_wall,
                wp.static(self.needs_moving_wall_treatment),
                wp.static(self.needs_mesh_distance),
            )

            # Compute density, velocity using all f_post-streaming values
            rho, u = self.macroscopic.warp_functional(f_post)

            # Regularize the resulting populations
            feq = self.equilibrium.warp_functional(rho, u)
            f_post = self.bc_helper.regularize_fpop(f_post, feq)
            return f_post

        if self.bc_method == "bounceback_regularized":
            functional = hybrid_bounceback_regularized
        elif self.bc_method == "bounceback_grads":
            functional = hybrid_bounceback_grads
        elif self.bc_method == "nonequilibrium_regularized":
            functional = hybrid_nonequilibrium_regularized

        kernel = self._construct_kernel(functional)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, _missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, _missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post

    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
