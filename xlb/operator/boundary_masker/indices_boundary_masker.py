from typing import Any
import copy

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.operator import Operator
from xlb.operator.stream.stream import Stream
from xlb.precision_policy import Precision
from xlb.operator.boundary_masker.helper_functions_masker import HelperFunctionsMasker
import neon


class IndicesBoundaryMasker(Operator):
    """
    Operator for creating a boundary mask
    """

    def __init__(
        self,
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

        if self.compute_backend in [ComputeBackend.WARP, ComputeBackend.NEON]:
            # Define masker helper functions
            self.helper_masker = HelperFunctionsMasker(
                velocity_set=self.velocity_set,
                precision_policy=self.precision_policy,
                compute_backend=self.compute_backend,
            )
        else:
            # Make stream operator
            self.stream = Stream(velocity_set, precision_policy, compute_backend)

    def are_indices_in_interior(self, indices, shape):
        """
        Check if each 2D or 3D index is inside the bounds of the domain with the given shape and not
        at its boundary.

        :param indices: Array of indices, where each column contains indices for each dimension.
        :param shape: Tuple representing the shape of the domain (nx, ny) for 2D or (nx, ny, nz) for 3D.
        :return: Array of boolean flags where each flag indicates whether the corresponding index is inside the bounds.
        """
        _d = self.velocity_set.d
        shape_array = np.array(shape)
        return np.all((indices[:_d] > 0) & (indices[:_d] < shape_array[:_d, np.newaxis] - 1), axis=0)

    def _find_bclist_interior(self, bclist, grid_shape):
        bc_interior = []
        for bc in bclist:
            if any(self.are_indices_in_interior(np.array(bc.indices), grid_shape)):
                bc_copy = copy.copy(bc)  # shallow copy of the whole object
                bc_copy.indices = copy.deepcopy(bc.pad_indices())  # deep copy only the modified part
                bc_interior.append(bc_copy)
        return bc_interior

    @Operator.register_backend(ComputeBackend.JAX)
    # TODO HS: figure out why uncommenting the line below fails unlike other operators!
    # @partial(jit, static_argnums=(0))
    def jax_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        # Extend the missing mask by padding to identify out of bound boundaries
        # Set padded region to True (i.e. boundary)
        dim = missing_mask.ndim - 1
        grid_shape = bc_mask[0].shape
        nDevices = jax.device_count()
        pad_x, pad_y, pad_z = nDevices, 1, 1

        # Shift indices due to padding
        shift = np.array((pad_x, pad_y) if dim == 2 else (pad_x, pad_y, pad_z))[:, np.newaxis]
        if start_index is None:
            start_index = (0,) * dim

        # TODO MEHDI: There is sometimes a halting problem here when padding is used in a multi-GPU setting since we're not jitting this function.
        # For now, we compute the bc_mask_extended on GPU zero.
        if dim == 2:
            bc_mask_extended = jnp.pad(bc_mask[0], ((pad_x, pad_x), (pad_y, pad_y)), constant_values=0)
            missing_mask_extended = jnp.pad(missing_mask, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), constant_values=True)
        if dim == 3:
            bc_mask_extended = jnp.pad(bc_mask[0], ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), constant_values=0)
            missing_mask_extended = jnp.pad(missing_mask, ((0, 0), (pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), constant_values=True)

        # Iterate over boundary conditions and set the mask
        for bc in bclist:
            assert bc.indices is not None, f"Please specify indices associated with the {bc.__class__.__name__} BC!"
            assert bc.mesh_vertices is None, (
                f"Please use operators based on MeshBoundaryMasker if {bc.__class__.__name__} is imposed on a mesh (e.g. STL)!"
            )
            id_number = bc.id
            bc_indices = np.array(bc.indices)
            indices_origin = np.array(start_index)[:, np.newaxis]
            if any(self.are_indices_in_interior(bc_indices, grid_shape)):
                # If the indices are in the interior, we assume the usre specified indices are solid indices
                solid_indices = bc_indices - indices_origin
                solid_indices_shifted = solid_indices + shift

                # We obtain the boundary indices by padding the solid indices in all lattice directions
                indices_padded = bc.pad_indices() - indices_origin
                indices_shifted = indices_padded + shift

                # The missing mask is set to True meaning (exterior or solid nodes) using the original indices.
                # This is because of the following streaming step which will assign missing directions for the boundary nodes.
                missing_mask_extended = missing_mask_extended.at[:, solid_indices_shifted[0], solid_indices_shifted[1], solid_indices_shifted[2]].set(
                    True
                )
            else:
                indices_shifted = bc_indices - indices_origin + shift

            # Assign the boundary id to the shifted (and possibly padded) indices
            bc_mask_extended = bc_mask_extended.at[tuple(indices_shifted)].set(id_number)

            # We are done with bc.indices. Remove them from BC objects
            bc.__dict__.pop("indices", None)

        # Stream the missing mask to identify missing directions
        missing_mask_extended = self.stream(missing_mask_extended)

        # Crop the extended masks to remove padding
        if dim == 2:
            missing_mask = missing_mask_extended[:, pad_x:-pad_x, pad_y:-pad_y]
            bc_mask = bc_mask.at[0].set(bc_mask_extended[pad_x:-pad_x, pad_y:-pad_y])
        if dim == 3:
            missing_mask = missing_mask_extended[:, pad_x:-pad_x, pad_y:-pad_y, pad_z:-pad_z]
            bc_mask = bc_mask.at[0].set(bc_mask_extended[pad_x:-pad_x, pad_y:-pad_y, pad_z:-pad_z])
        return bc_mask, missing_mask

    def _construct_warp(self):
        # Make constants for warp
        _q = self.velocity_set.q

        @wp.func
        def functional_domain_bounds(
            index: Any,
            bc_indices: Any,
            id_number: Any,
            is_interior: Any,
            bc_mask: Any,
            missing_mask: Any,
            grid_shape: Any,
        ):
            for ii in range(bc_indices.shape[1]):
                # If the current index does not match the boundary condition index, we skip it
                if not self.helper_masker.is_in_bc_indices(bc_mask, index, bc_indices, ii):
                    continue

                if is_interior[ii] == wp.uint8(True):
                    # If the index is in the interior, we set that index to be a solid node (identified by 255)
                    # This information will be used in the next kernel to identify missing directions using the
                    # padded indices of the solid node that are associated with the boundary condition.
                    self.write_field(bc_mask, index, 0, wp.uint8(255))
                    return

                # Set bc_mask for all bc indices
                self.write_field(bc_mask, index, 0, wp.uint8(id_number[ii]))

                # Stream indices
                for l in range(_q):
                    # Get the pull index which is the index of the neighboring node where information is pulled from
                    pull_index, _ = self.helper_masker.get_pull_index(bc_mask, l, index)

                    # Check if pull index is out of bound
                    # These directions will have missing information after streaming
                    if not self.helper_masker.is_in_bounds(pull_index, grid_shape, missing_mask):
                        # Set the missing mask
                        self.write_field(missing_mask, index, l, wp.uint8(True))

        @wp.func
        def functional_interior_bc_mask(
            index: Any,
            bc_indices: Any,
            id_number: Any,
            bc_mask: Any,
        ):
            for ii in range(bc_indices.shape[1]):
                # If the current index does not match the boundary condition index, we skip it
                if not self.helper_masker.is_in_bc_indices(bc_mask, index, bc_indices, ii):
                    continue
                # Set bc_mask for all interior bc indices
                self.write_field(bc_mask, index, 0, wp.uint8(id_number[ii]))

        @wp.func
        def functional_interior_missing_mask(
            index: Any,
            bc_indices: Any,
            bc_mask: Any,
            missing_mask: Any,
            grid_shape: Any,
        ):
            for ii in range(bc_indices.shape[1]):
                # If the current index does not match the boundary condition index, we skip it
                if not self.helper_masker.is_in_bc_indices(bc_mask, index, bc_indices, ii):
                    continue
                for l in range(_q):
                    # Get the index of the streaming direction
                    pull_index, offset = self.helper_masker.get_pull_index(bc_mask, l, index)

                    # Check if pull index is a fluid node (bc_mask is zero for fluid nodes)
                    bc_mask_ngh = self.read_field_neighbor(bc_mask, index, offset, 0)
                    if (self.helper_masker.is_in_bounds(pull_index, grid_shape, missing_mask)) and (bc_mask_ngh == wp.uint8(255)):
                        self.write_field(missing_mask, index, l, wp.uint8(True))

        # Construct the warp 3D kernel
        @wp.kernel
        def kernel_domain_bounds(
            bc_indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            is_interior: wp.array1d(dtype=wp.uint8),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            grid_shape: wp.vec3i,
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # Call the functional
            functional_domain_bounds(
                index,
                bc_indices,
                id_number,
                is_interior,
                bc_mask,
                missing_mask,
                grid_shape,
            )

        @wp.kernel
        def kernel_interior_bc_mask(
            bc_indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            bc_mask: wp.array4d(dtype=wp.uint8),
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            # Set bc_mask for all interior bc indices
            functional_interior_bc_mask(
                index,
                bc_indices,
                id_number,
                bc_mask,
            )
            return

        @wp.kernel
        def kernel_interior_missing_mask(
            bc_indices: wp.array2d(dtype=wp.int32),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            grid_shape: wp.vec3i,
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i(i, j, k)

            functional_interior_missing_mask(index, bc_indices, bc_mask, missing_mask, grid_shape)

        functional_dict = {
            "functional_domain_bounds": functional_domain_bounds,
            "functional_interior_bc_mask": functional_interior_bc_mask,
            "functional_interior_missing_mask": functional_interior_missing_mask,
        }
        kernel_dict = {
            "kernel_domain_bounds": kernel_domain_bounds,
            "kernel_interior_bc_mask": kernel_interior_bc_mask,
            "kernel_interior_missing_mask": kernel_interior_missing_mask,
        }
        return functional_dict, kernel_dict

    def _prepare_kernel_inputs(self, bclist, grid_shape):
        """
        Prepare the inputs for the warp kernel by pre-allocating arrays and filling them with boundary condition information.
        """

        # Pre-allocate arrays with maximum possible size
        max_size = sum(
            len(bc.indices[0]) if isinstance(bc.indices, (list, tuple)) else bc.indices.shape[1] for bc in bclist if bc.indices is not None
        )
        indices = np.zeros((3, max_size), dtype=np.int32)
        id_numbers = np.zeros(max_size, dtype=np.uint8)
        is_interior = np.zeros(max_size, dtype=np.uint8)

        current_index = 0
        for bc in bclist:
            assert bc.indices is not None, f'Please specify indices associated with the {bc.__class__.__name__} BC using keyword "indices"!'
            assert bc.mesh_vertices is None, (
                f"Please use operators based on MeshBoundaryMasker if {bc.__class__.__name__} is imposed on a mesh (e.g. STL)!"
            )
            bc_indices = np.asarray(bc.indices)
            num_indices = bc_indices.shape[1]

            # Ensure indices are 3D
            if bc_indices.shape[0] == 2:
                bc_indices = np.vstack([bc_indices, np.zeros(num_indices, dtype=int)])

            # Add indices to the pre-allocated array
            indices[:, current_index : current_index + num_indices] = bc_indices

            # Set id numbers
            id_numbers[current_index : current_index + num_indices] = bc.id

            # Set is_interior flags
            is_interior[current_index : current_index + num_indices] = self.are_indices_in_interior(bc_indices, grid_shape)

            current_index += num_indices

            # Remove indices from BC objects
            bc.__dict__.pop("indices", None)

        # Trim arrays to actual size
        total_index = current_index
        indices = indices[:, :total_index]
        id_numbers = id_numbers[:total_index]
        is_interior = is_interior[:total_index]

        # Convert to Warp arrays
        wp_bc_indices = wp.array(indices, dtype=wp.int32)
        wp_id_numbers = wp.array(id_numbers, dtype=wp.uint8)
        wp_is_interior = wp.array(is_interior, dtype=wp.uint8)
        return wp_bc_indices, wp_id_numbers, wp_is_interior

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        # get the grid shape
        grid_shape = self.helper_masker.get_grid_shape(bc_mask)

        # Find interior boundary conditions
        bc_interior = self._find_bclist_interior(bclist, grid_shape)

        # Prepare the first kernel inputs for all items in boundary condition list
        wp_bc_indices, wp_id_numbers, wp_is_interior = self._prepare_kernel_inputs(bclist, grid_shape)

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel["kernel_domain_bounds"],
            dim=bc_mask.shape[1:],
            inputs=[wp_bc_indices, wp_id_numbers, wp_is_interior, bc_mask, missing_mask, grid_shape],
        )

        # If there are no interior boundary conditions, skip the rest and retun early
        if not bc_interior:
            return bc_mask, missing_mask

        # Prepare the second and third kernel inputs for only a subset of boundary conditions associated with the interior
        # Note 1: launching order of the following kernels are important here!
        # Note 2: Due to race conditioning, the two kernels cannot be fused together.
        wp_bc_indices, wp_id_numbers, _ = self._prepare_kernel_inputs(bc_interior, grid_shape)
        wp.launch(
            self.warp_kernel["kernel_interior_missing_mask"],
            dim=bc_mask.shape[1:],
            inputs=[wp_bc_indices, bc_mask, missing_mask, grid_shape],
        )
        wp.launch(
            self.warp_kernel["kernel_interior_bc_mask"],
            dim=bc_mask.shape[1:],
            inputs=[
                wp_bc_indices,
                wp_id_numbers,
                bc_mask,
            ],
        )

        return bc_mask, missing_mask

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional_dict, _ = self._construct_warp()
        functional_domain_bounds = functional_dict.get("functional_domain_bounds")
        functional_interior_bc_mask = functional_dict.get("functional_interior_bc_mask")
        functional_interior_missing_mask = functional_dict.get("functional_interior_missing_mask")

        @neon.Container.factory(name="IndicesBoundaryMasker_DomainBounds")
        def container_domain_bounds(
            wp_bc_indices,
            wp_id_numbers,
            wp_is_interior,
            bc_mask,
            missing_mask,
            grid_shape,
        ):
            def domain_bounds_launcher(loader: neon.Loader):
                loader.set_grid(bc_mask.get_grid())
                bc_mask_pn = loader.get_write_handle(bc_mask)
                missing_mask_pn = loader.get_write_handle(missing_mask)

                @wp.func
                def domain_bounds_kernel(index: Any):
                    # apply the functional
                    functional_domain_bounds(
                        index,
                        wp_bc_indices,
                        wp_id_numbers,
                        wp_is_interior,
                        bc_mask_pn,
                        missing_mask_pn,
                        grid_shape,
                    )

                loader.declare_kernel(domain_bounds_kernel)

            return domain_bounds_launcher

        @neon.Container.factory(name="IndicesBoundaryMasker_InteriorBcMask")
        def container_interior_bc_mask(
            wp_bc_indices,
            wp_id_numbers,
            bc_mask,
        ):
            def interior_bc_mask_launcher(loader: neon.Loader):
                loader.set_grid(bc_mask.get_grid())
                bc_mask_pn = loader.get_write_handle(bc_mask)

                @wp.func
                def interior_bc_mask_kernel(index: Any):
                    # apply the functional
                    functional_interior_bc_mask(
                        index,
                        wp_bc_indices,
                        wp_id_numbers,
                        bc_mask_pn,
                    )

                loader.declare_kernel(interior_bc_mask_kernel)

            return interior_bc_mask_launcher

        @neon.Container.factory(name="IndicesBoundaryMasker_InteriorMissingMask")
        def container_interior_missing_mask(
            wp_bc_indices,
            bc_mask,
            missing_mask,
            grid_shape,
        ):
            def interior_bc_mask_launcher(loader: neon.Loader):
                loader.set_grid(bc_mask.get_grid())
                bc_mask_pn = loader.get_write_handle(bc_mask)
                missing_mask_pn = loader.get_write_handle(missing_mask)

                @wp.func
                def interior_missing_mask_kernel(index: Any):
                    # apply the functional
                    functional_interior_missing_mask(
                        index,
                        wp_bc_indices,
                        bc_mask_pn,
                        missing_mask_pn,
                        grid_shape,
                    )

                loader.declare_kernel(interior_missing_mask_kernel)

            return interior_bc_mask_launcher

        container_dict = {
            "container_domain_bounds": container_domain_bounds,
            "container_interior_bc_mask": container_interior_bc_mask,
            "container_interior_missing_mask": container_interior_missing_mask,
        }

        return functional_dict, container_dict

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        # get the grid shape
        grid_shape = self.helper_masker.get_grid_shape(bc_mask)

        # Find interior boundary conditions
        bc_interior = self._find_bclist_interior(bclist, grid_shape)

        # Prepare the first kernel inputs for all items in boundary condition list
        wp_bc_indices, wp_id_numbers, wp_is_interior = self._prepare_kernel_inputs(bclist, grid_shape)

        # Launch the first container
        container_domain_bounds = self.neon_container["container_domain_bounds"](
            wp_bc_indices,
            wp_id_numbers,
            wp_is_interior,
            bc_mask,
            missing_mask,
            grid_shape,
        )
        container_domain_bounds.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        # If there are no interior boundary conditions, skip the rest and retun early
        if not bc_interior:
            return bc_mask, missing_mask

        # Prepare the second and third kernel inputs for only a subset of boundary conditions associated with the interior
        # Note 1: launching order of the following kernels are important here!
        # Note 2: Due to race conditioning, the two kernels cannot be fused together.
        wp_bc_indices, wp_id_numbers, _ = self._prepare_kernel_inputs(bc_interior, grid_shape)
        container_interior_missing_mask = self.neon_container["container_interior_missing_mask"](wp_bc_indices, bc_mask, missing_mask, grid_shape)
        container_interior_missing_mask.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        # Launch the third container
        container_interior_bc_mask = self.neon_container["container_interior_bc_mask"](
            wp_bc_indices,
            wp_id_numbers,
            bc_mask,
        )
        container_interior_bc_mask.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        return bc_mask, missing_mask
