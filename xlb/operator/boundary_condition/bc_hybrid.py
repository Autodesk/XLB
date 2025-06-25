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

        # Check if the compute backend is Warp
        assert self.compute_backend == ComputeBackend.WARP or ComputeBackend.NEON, "This BC is currently not supported by JAX backend!"

        # Instantiate the operator for computing macroscopic values
        # Explicitly using the WARP backend for these operators as they may also be called by the Neon backend.
        self.macroscopic = Macroscopic(compute_backend=ComputeBackend.WARP)
        self.equilibrium = QuadraticEquilibrium(compute_backend=ComputeBackend.WARP)

        # This BC class accepts both constant prescribed values of velocity with keyword "prescribed_value" or
        # velocity profiles given by keyword "profile" which must be a callable function.
        self.profile = profile

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
            if profile is not None:
                raise ValueError("Cannot specify both profile and prescribed_value")

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
            prescribed_value = wp.vec(3, dtype=self.compute_dtype)(prescribed_value)

            @wp.func
            def prescribed_profile_warp(index: Any, time: Any):
                return wp.vec3(prescribed_value[0], prescribed_value[1], prescribed_value[2])

            self.profile = prescribed_profile_warp

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

        # Define BC helper functions. Explicitly using the WARP backend for helper functions as it may also be called by the Neon backend.
        self.bc_helper = HelperFunctionsBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=ComputeBackend.WARP,
            distance_decoder_function=self._construct_distance_decoder_function(),
        )

        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This BC is not implemented in 2D!")

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
        if self.compute_backend == ComputeBackend.WARP:

            @wp.func
            def distance_decoder_function(f_1: Any, index: Any, direction: Any):
                return f_1[_opp_indices[direction], index[0], index[1], index[2]]

        elif self.compute_backend == ComputeBackend.NEON:

            @wp.func
            def distance_decoder_function(f_1_pn: Any, index: Any, direction: Any):
                return wp.neon_read(f_1_pn, index, _opp_indices[direction])

        return distance_decoder_function

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
            u_wall = self.profile(index, timestep)
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
            u_wall = self.profile(index, timestep)
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
            u_wall = self.profile(index, timestep)
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
        # rise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")
