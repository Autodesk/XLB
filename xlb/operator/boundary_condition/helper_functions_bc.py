"""
Warp/Neon helper functions shared by multiple boundary conditions.

:class:`HelperFunctionsBC` exposes ``@wp.func`` helpers for bounce-back,
regularization, Grad's approximation, moving-wall corrections,
interpolated BCs, and BC thread-data loading.  These are used as building
blocks by the concrete BC classes.

Also contains :class:`EncodeAuxiliaryData` and
:class:`MultiresEncodeAuxiliaryData` operators for writing user-prescribed
BC profiles into the ``f_1`` buffer during initialization.
"""

import inspect
from typing import Any, Callable

import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb import DefaultConfig, ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import SecondMoment as MomentumFlux
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium


class HelperFunctionsBC(object):
    """Collection of Warp/Neon ``@wp.func`` helpers for boundary conditions.

    Parameters
    ----------
    velocity_set : VelocitySet, optional
    precision_policy : PrecisionPolicy, optional
    compute_backend : ComputeBackend, optional
        Must be ``WARP`` or ``NEON`` (JAX not supported).
    distance_decoder_function : callable, optional
        Function to decode wall-distance data for interpolated BCs.
    """

    def __init__(self, velocity_set=None, precision_policy=None, compute_backend=None, distance_decoder_function=None):
        if compute_backend == ComputeBackend.JAX:
            raise ValueError("This helper class contains helper functions only for the WARP implementation of some BCs not JAX!")

        # Set the default values from the global config
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.default_precision_policy
        self.compute_backend = compute_backend or DefaultConfig.default_backend
        self.distance_decoder_function = distance_decoder_function

        # Set the compute and Store dtypes
        compute_dtype = self.precision_policy.compute_precision.wp_dtype
        store_dtype = self.precision_policy.store_precision.wp_dtype

        # Set local constants
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices
        _w = self.velocity_set.w
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float
        _qi = self.velocity_set.qi
        _u_vec = wp.vec(_d, dtype=compute_dtype)
        _f_vec = wp.vec(_q, dtype=compute_dtype)
        _missing_mask_vec = wp.vec(_q, dtype=wp.uint8)  # TODO fix vec bool

        # Define the operator needed for computing equilibrium
        equilibrium = QuadraticEquilibrium(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing macroscopic variables
        macroscopic = Macroscopic(velocity_set, precision_policy, compute_backend)

        # Define the operator needed for computing the momentum flux
        momentum_flux = MomentumFlux(velocity_set, precision_policy, compute_backend)

        @wp.func
        def get_bc_thread_data(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            index: wp.vec3i,
        ):
            # Get the boundary id and missing mask
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _boundary_id = bc_mask[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(_q):
                # q-sized vector of populations
                _f_pre[l] = compute_dtype(f_pre[l, index[0], index[1], index[2]])
                _f_post[l] = compute_dtype(f_post[l, index[0], index[1], index[2]])

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)
            return _f_pre, _f_post, _boundary_id, _missing_mask

        @wp.func
        def neon_get_bc_thread_data(
            f_pre_pn: Any,
            f_post_pn: Any,
            bc_mask_pn: Any,
            missing_mask_pn: Any,
            index: Any,
        ):
            # Get the boundary id and missing mask
            _f_pre = _f_vec()
            _f_post = _f_vec()
            _boundary_id = wp.neon_read(bc_mask_pn, index, 0)
            _missing_mask = _missing_mask_vec()
            for l in range(_q):
                # q-sized vector of populations
                _f_pre[l] = compute_dtype(wp.neon_read(f_pre_pn, index, l))
                _f_post[l] = compute_dtype(wp.neon_read(f_post_pn, index, l))
                _missing_mask[l] = wp.neon_read(missing_mask_pn, index, l)

            return _f_pre, _f_post, _boundary_id, _missing_mask

        @wp.func
        def get_bc_fsum(
            fpop: Any,
            _missing_mask: Any,
        ):
            fsum_known = compute_dtype(0.0)
            fsum_middle = compute_dtype(0.0)
            for l in range(_q):
                if _missing_mask[_opp_indices[l]] == wp.uint8(1):
                    fsum_known += compute_dtype(2.0) * fpop[l]
                elif _missing_mask[l] != wp.uint8(1):
                    fsum_middle += fpop[l]
            return fsum_known + fsum_middle

        @wp.func
        def get_normal_vectors(
            _missing_mask: Any,
        ):
            if wp.static(_d == 3):
                for l in range(_q):
                    if _missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                        return -_u_vec(_c_float[0, l], _c_float[1, l], _c_float[2, l])
            else:
                for l in range(_q):
                    if _missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                        return -_u_vec(_c_float[0, l], _c_float[1, l])

        @wp.func
        def bounceback_nonequilibrium(
            fpop: Any,
            feq: Any,
            _missing_mask: Any,
        ):
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
            return fpop

        @wp.func
        def regularize_fpop(
            fpop: Any,
            feq: Any,
        ):
            """
            Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.
            """
            # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
            f_neq = fpop - feq
            PiNeq = momentum_flux.warp_functional(f_neq)

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            nt = _d * (_d + 1) // 2
            for l in range(_q):
                QiPi1 = compute_dtype(0.0)
                for t in range(nt):
                    QiPi1 += _qi[l, t] * PiNeq[t]

                # assign all populations based on eq 45 of Latt et al (2008)
                # fneq ~ f^1
                fpop1 = compute_dtype(4.5) * _w[l] * QiPi1
                fpop[l] = feq[l] + fpop1
            return fpop

        @wp.func
        def grads_approximate_fpop(
            _missing_mask: Any,
            rho: Any,
            u: Any,
            f_post: Any,
        ):
            # Purpose: Using Grad's approximation to represent fpop based on macroscopic inputs used for outflow [1] and
            # Dirichlet BCs [2]
            # [1] S. Chikatax`marla, S. Ansumali, and I. Karlin, "Grad's approximation for missing data in lattice Boltzmann
            #   simulations", Europhys. Lett. 74, 215 (2006).
            # [2] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
            #    stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.

            # Note: See also self.regularize_fpop function which is somewhat similar.

            # Compute pressure tensor Pi using all f_post-streaming values
            Pi = momentum_flux.warp_functional(f_post)

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            nt = _d * (_d + 1) // 2
            for l in range(_q):
                if _missing_mask[l] == wp.uint8(1):
                    # compute dot product of qi and Pi
                    QiPi = compute_dtype(0.0)
                    for t in range(nt):
                        if t == 0 or t == 3 or t == 5:
                            QiPi += _qi[l, t] * (Pi[t] - rho / compute_dtype(3.0))
                        else:
                            QiPi += _qi[l, t] * Pi[t]

                    # Compute c.u
                    cu = compute_dtype(0.0)
                    for d in range(_d):
                        if _c[d, l] == 1:
                            cu += u[d]
                        elif _c[d, l] == -1:
                            cu -= u[d]
                    cu *= compute_dtype(3.0)

                    # change f_post using the Grad's approximation
                    f_post[l] = rho * _w[l] * (compute_dtype(1.0) + cu) + _w[l] * compute_dtype(4.5) * QiPi

            return f_post

        @wp.func
        def moving_wall_fpop_correction(
            u_wall: Any,
            lattice_direction: Any,
        ):
            # Add forcing term necessary to account for the local density changes caused by the mass displacement
            # as the object moves with velocity u_wall.
            # [1] L.-S. Luo, Unified theory of lattice Boltzmann models for nonideal gases, Phys. Rev. Lett. 81 (1998) 1618-1621.
            # [2] L.-S. Luo, Theory of the lattice Boltzmann method: Lattice Boltzmann models for nonideal gases, Phys. Rev. E 62 (2000) 4982-4996.
            #
            # Note: this function must be called within a for-loop over all lattice directions and the populations to be modified must
            # be only those in the missing direction (the check for missing direction must be outside of this function).
            cu = compute_dtype(0.0)
            l = lattice_direction
            for d in range(_d):
                if _c[d, l] == 1:
                    cu += u_wall[d]
                elif _c[d, l] == -1:
                    cu -= u_wall[d]
            cu *= compute_dtype(6.0) * _w[l]
            return cu

        @wp.func
        def interpolated_bounceback(
            index: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
            u_wall: Any,
            needs_moving_wall_treatment: bool,
            needs_mesh_distance: bool,
        ):
            # A local single-node version of the interpolated bounce-back boundary condition due to Bouzidi for a lattice
            # Boltzmann method simulation.
            # Ref:
            # [1] Yu, D., Mei, R., Shyy, W., 2003. A unified boundary treatment in lattice boltzmann method,
            # in: 41st aerospace sciences meeting and exhibit, p. 953.

            one = compute_dtype(1.0)
            for l in range(_q):
                # If the mask is missing then take the opposite index
                if _missing_mask[l] == wp.uint8(1):
                    # The normalized distance to the mesh or "weights" have been stored in known directions of f_1
                    if needs_mesh_distance:
                        # use weights associated with curved boundaries that are properly stored in f_1.
                        weight = compute_dtype(self.distance_decoder_function(f_1, index, l))

                        # Use differentiable interpolated BB to find f_missing:
                        f_post[l] = ((one - weight) * f_post[_opp_indices[l]] + weight * (f_pre[l] + f_pre[_opp_indices[l]])) / (one + weight)
                    else:
                        # Use regular halfway bounceback
                        f_post[l] = f_pre[_opp_indices[l]]

                    if _missing_mask[_opp_indices[l]] == wp.uint8(1):
                        # These are cases where the boundary is sandwiched between 2 solid cells and so both opposite directions are missing.
                        f_post[l] = f_pre[_opp_indices[l]]

                    # Add contribution due to moving_wall to f_missing as is usual in regular Bouzidi BC
                    if needs_moving_wall_treatment:
                        f_post[l] += moving_wall_fpop_correction(u_wall, l)
            return f_post

        @wp.func
        def interpolated_nonequilibrium_bounceback(
            index: Any,
            _missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
            u_wall: Any,
            needs_moving_wall_treatment: bool,
            needs_mesh_distance: bool,
        ):
            # Compute density, velocity using all f_post-collision values
            rho, u = macroscopic.warp_functional(f_pre)
            feq = equilibrium.warp_functional(rho, u)

            # Compute equilibrium distribution at the wall
            if needs_moving_wall_treatment:
                feq_wall = equilibrium.warp_functional(rho, u_wall)
            else:
                feq_wall = _f_vec()

            # Apply method in Tao et al (2018) [1] to find missing populations at the boundary
            one = compute_dtype(1.0)
            for l in range(_q):
                # If the mask is missing then take the opposite index
                if _missing_mask[l] == wp.uint8(1):
                    # The normalized distance to the mesh or "weights" have been stored in known directions of f_1
                    if needs_mesh_distance:
                        # use weights associated with curved boundaries that are properly stored in f_1.
                        weight = compute_dtype(self.distance_decoder_function(f_1, index, l))
                    else:
                        weight = compute_dtype(0.5)

                    # Use non-equilibrium bounceback to find f_missing:
                    fneq = f_pre[_opp_indices[l]] - feq[_opp_indices[l]]

                    # Compute equilibrium distribution at the wall
                    # Same quadratic equilibrium but accounting for zero velocity (no-slip)
                    if not needs_moving_wall_treatment:
                        feq_wall[l] = _w[l] * rho

                    # Assemble wall population for doing interpolation at the boundary
                    f_wall = feq_wall[l] + fneq
                    f_post[l] = (f_wall + weight * f_pre[l]) / (one + weight)

            return f_post

        @wp.func
        def neon_index_to_warp(neon_field_hdl: Any, index: Any):
            # Unpack the global index in Neon at the finest level and convert it to a warp vector
            cIdx = wp.neon_global_idx(neon_field_hdl, index)
            gx = wp.neon_get_x(cIdx)
            gy = wp.neon_get_y(cIdx)
            gz = wp.neon_get_z(cIdx)

            # XLB is flattening the z dimension in 3D, while neon uses the y dimension
            if _d == 2:
                gy, gz = gz, gy

            # Get warp indices
            index_wp = wp.vec3i(gx, gy, gz)
            return index_wp

        self.get_bc_thread_data = get_bc_thread_data
        self.get_bc_fsum = get_bc_fsum
        self.get_normal_vectors = get_normal_vectors
        self.bounceback_nonequilibrium = bounceback_nonequilibrium
        self.regularize_fpop = regularize_fpop
        self.grads_approximate_fpop = grads_approximate_fpop
        self.moving_wall_fpop_correction = moving_wall_fpop_correction
        self.interpolated_bounceback = interpolated_bounceback
        self.interpolated_nonequilibrium_bounceback = interpolated_nonequilibrium_bounceback
        self.neon_get_bc_thread_data = neon_get_bc_thread_data
        self.neon_index_to_warp = neon_index_to_warp


class EncodeAuxiliaryData(Operator):
    """
    Operator for encoding boundary auxiliary data during initialization.
    """

    def __init__(
        self,
        boundary_id: int,
        num_of_aux_data: int,
        user_defined_functional: Callable,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        self.user_defined_functional = user_defined_functional
        self.boundary_id = wp.uint8(boundary_id)
        self.num_of_aux_data = num_of_aux_data

        super().__init__(velocity_set, precision_policy, compute_backend)

        # Inspect the signature of the user-defined functional.
        # We assume the profile function takes only the index as input and is hence time-independent.
        sig = inspect.signature(user_defined_functional)
        assert self.compute_backend != ComputeBackend.JAX, "Encoding/decoding of auxiliary data are not required for boundary conditions in JAX"
        assert len(sig.parameters) == 1, f"User-defined functional must take exactly one argument (the index), it received {len(sig.parameters)}."

        # Define a HelperFunctionsBC instance
        self.bc_helper = HelperFunctionsBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

        # TODO: Somehow raise an error if the number of prescribed values does not match the number of missing directions

    def _construct_warp(self):
        """
        Constructs the warp kernel for the auxiliary data recovery.
        """
        # Find velocity index for (0, 0, 0)
        lattice_central_index = self.velocity_set.center_index
        _opp_indices = self.velocity_set.opp_indices
        _id = self.boundary_id
        _num_of_aux_data = self.num_of_aux_data
        _aux_vec = wp.vec(_num_of_aux_data, dtype=self.compute_dtype)

        @wp.func
        def encoder_functional(
            index: Any,
            _missing_mask: Any,
            field_storage: Any,
            prescribed_values: Any,
        ):
            if len(prescribed_values) != _num_of_aux_data:
                wp.printf("Error: User-defined profile must return a vector of size %d\n", _num_of_aux_data)
                return

            # Write the result for all q directions, but only store up to _num_of_aux_data
            counter = wp.int32(0)
            for l in range(self.velocity_set.q):
                # Only store up to _num_of_aux_data
                if counter == _num_of_aux_data:
                    return

                if l == lattice_central_index:
                    # The first BC auxiliary data is stored in the zero'th index of f_1 associated with its center.
                    self.write_field(field_storage, index, l, self.store_dtype(prescribed_values[l]))
                    counter += 1
                elif _missing_mask[l] == wp.uint8(1):
                    # The other remaining BC auxiliary data are stored in missing directions of f_1.
                    self.write_field(field_storage, index, _opp_indices[l], self.store_dtype(prescribed_values[l]))
                    counter += 1

        @wp.func
        def decoder_functional(
            field_storage: Any,
            index: Any,
            _missing_mask: Any,
        ):
            """
            Decode the encoded values needed for the boundary condition treatment from the center location in field_storage.
            """

            # Define a vector to hold prescribed_values
            prescribed_values = _aux_vec()

            # Read all q directions, but only retrieve up to _num_of_aux_data
            counter = wp.int32(0)
            for l in range(self.velocity_set.q):
                # Only retrieve up to _num_of_aux_data
                if counter == _num_of_aux_data:
                    return prescribed_values

                if l == lattice_central_index:
                    # The first BC auxiliary data is stored in the zero'th index of f_1 associated with its center.
                    value = self.read_field(field_storage, index, l)
                    prescribed_values[counter] = self.compute_dtype(value)
                    counter += 1
                elif _missing_mask[l] == wp.uint8(1):
                    # The other remaining BC auxiliary data are stored in missing directions of f_1.
                    value = self.read_field(field_storage, index, _opp_indices[l])
                    prescribed_values[counter] = self.compute_dtype(value)
                    counter += 1

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_1: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _, _, _boundary_id, _missing_mask = self.bc_helper.get_bc_thread_data(f_1, f_1, bc_mask, missing_mask, index)

            # Apply the functional
            # change this to use central location
            if _boundary_id == _id:
                # prescribed_values is a q-sized vector of type wp.vec
                prescribed_values = self.user_defined_functional(index)

                # call the functional
                encoder_functional(index, _missing_mask, f_1, prescribed_values)

        functional_dict = {"encoder": encoder_functional, "decoder": decoder_functional}
        return functional_dict, kernel

    def _construct_neon(self):
        import neon

        """
        Constructs the Neon container for encoding auxiliary data recovery.
        """
        # Use the warp functional for the Neon backend
        functional_dict, _ = self._construct_warp()
        encoder_functional = functional_dict["encoder"]
        _id = self.boundary_id

        # Construct the Neon container
        @neon.Container.factory(name="EncodingAuxData_" + str(_id))
        def aux_data_init_container(
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
        ):
            def aux_data_init_ll(loader: neon.Loader):
                loader.set_grid(f_1.get_grid())

                f_1_pn = loader.get_write_handle(f_1)
                bc_mask_pn = loader.get_read_handle(bc_mask)
                missing_mask_pn = loader.get_read_handle(missing_mask)

                @wp.func
                def aux_data_init_cl(index: Any):
                    # read tid data
                    _, _, _boundary_id, _missing_mask = self.bc_helper.neon_get_bc_thread_data(f_1_pn, f_1_pn, bc_mask_pn, missing_mask_pn, index)

                    # Apply the functional
                    if _boundary_id == _id:
                        warp_index = self.bc_helper.neon_index_to_warp(f_1_pn, index)
                        prescribed_values = self.user_defined_functional(warp_index)

                        # Call the functional
                        encoder_functional(index, _missing_mask, f_1_pn, prescribed_values)

                # Declare the kernel in the Neon loader
                loader.declare_kernel(aux_data_init_cl)

            return aux_data_init_ll

        return functional_dict, aux_data_init_container

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_1, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_1, bc_mask, missing_mask],
            dim=f_1.shape[1:],
        )
        return f_1

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_1, bc_mask, missing_mask):
        import neon

        c = self.neon_container(f_1, bc_mask, missing_mask)
        c.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_1


class MultiresEncodeAuxiliaryData(EncodeAuxiliaryData):
    """
    Operator for encoding boundary auxiliary data during initialization.
    """

    def __init__(
        self,
        boundary_id: int,
        num_of_aux_data: int,
        user_defined_functional: Callable,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        super().__init__(
            boundary_id=boundary_id,
            num_of_aux_data=num_of_aux_data,
            user_defined_functional=user_defined_functional,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

        assert self.compute_backend == ComputeBackend.NEON, f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend."

    def _construct_neon(self):
        """
        Constructs the Neon container for encoding auxiliary data recovery.
        """
        import neon

        # Borrow the functional from the warp implementation
        functional_dict, _ = self._construct_warp()
        encoder_functional = functional_dict["encoder"]
        _id = self.boundary_id

        # Construct the Neon container
        @neon.Container.factory(name="MultiresEncodingAuxData_" + str(_id))
        def aux_data_init_container(
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            level: Any,
        ):
            def aux_data_init_ll(loader: neon.Loader):
                loader.set_mres_grid(f_1.get_grid(), level)

                f_1_pn = loader.get_mres_write_handle(f_1)
                bc_mask_pn = loader.get_mres_read_handle(bc_mask)
                missing_mask_pn = loader.get_mres_read_handle(missing_mask)

                @wp.func
                def aux_data_init_cl(index: Any):
                    # read tid data
                    _, _, _boundary_id, _missing_mask = self.bc_helper.neon_get_bc_thread_data(f_1_pn, f_1_pn, bc_mask_pn, missing_mask_pn, index)

                    # Apply the functional
                    if _boundary_id == _id:
                        # IMPORTANT: XLB assumes the user_defined_functional in multi-res
                        # simulations uses finest-level indices, enabling BCs that span
                        # multiple levels.
                        warp_index = self.bc_helper.neon_index_to_warp(f_1_pn, index)
                        prescribed_values = self.user_defined_functional(warp_index)

                        # Call the functional
                        encoder_functional(index, _missing_mask, f_1_pn, prescribed_values)

                # Declare the kernel in the Neon loader
                loader.declare_kernel(aux_data_init_cl)

            return aux_data_init_ll

        return functional_dict, aux_data_init_container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_1, bc_mask, missing_mask, stream):
        import neon

        grid = bc_mask.get_grid()
        for level in range(grid.num_levels):
            c = self.neon_container(f_1, bc_mask, missing_mask, level)
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return f_1
