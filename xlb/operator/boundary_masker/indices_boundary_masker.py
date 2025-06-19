from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
import copy

from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.operator import Operator
from xlb.operator.stream.stream import Stream
from xlb.precision_policy import Precision


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
        # Make stream operator
        self.stream = Stream(velocity_set, precision_policy, compute_backend)

        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

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
        _c = self.velocity_set.c
        _q = wp.constant(self.velocity_set.q)

        @wp.func
        def is_in_bounds(index: wp.vec3i, shape: wp.vec3i):
            return index[0] >= 0 and index[0] < shape[0] and index[1] >= 0 and index[1] < shape[1] and index[2] >= 0 and index[2] < shape[2]

        # Construct the warp 3D kernel
        @wp.kernel
        def kernel_domain_bounds(
            indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            is_interior: wp.array1d(dtype=wp.uint8),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
        ):
            # Get the index of indices
            ii = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = indices[0, ii]
            index[1] = indices[1, ii]
            index[2] = indices[2, ii]

            if is_interior[ii] == wp.uint8(True):
                # If the index is in the interior, we set that index to be a solid node (identified by 255)
                # This information will be used in the next kernel to identify missing directions using the
                # padded indices of the solid node that are associated with the boundary condition.
                self.write_field(bc_mask, index, 0, wp.uint8(255))
                return

            # Check if index is in bounds
            shape = wp.vec3i(missing_mask.shape[1], missing_mask.shape[2], missing_mask.shape[3])
            if is_in_bounds(index, shape):
                # Set bc_mask for all bc indices
                self.write_field(bc_mask, index, 0, wp.uint8(id_number[ii]))

                # Stream indices
                for l in range(_q):
                    # Get the index of the streaming direction
                    pull_index = wp.vec3i()
                    for d in range(self.velocity_set.d):
                        pull_index[d] = index[d] - _c[d, l]

                    # Check if pull index is out of bound
                    # These directions will have missing information after streaming
                    if not is_in_bounds(pull_index, shape):
                        # Set the missing mask
                        self.write_field(missing_mask, index, l, wp.uint8(True))

        @wp.kernel
        def kernel_interior_bc_mask(
            indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            bc_mask: wp.array4d(dtype=wp.uint8),
        ):
            # Get the index of indices
            ii = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = indices[0, ii]
            index[1] = indices[1, ii]
            index[2] = indices[2, ii]

            # Set bc_mask for all interior bc indices
            self.write_field(bc_mask, index, 0, wp.uint8(id_number[ii]))
            return

        @wp.kernel
        def kernel_interior_missing_mask(
            indices: wp.array2d(dtype=wp.int32),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
        ):
            # Get the index of indices
            ii = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = indices[0, ii]
            index[1] = indices[1, ii]
            index[2] = indices[2, ii]

            shape = wp.vec3i(missing_mask.shape[1], missing_mask.shape[2], missing_mask.shape[3])
            for l in range(_q):
                # Get the index of the streaming direction
                pull_index = wp.vec3i()
                for d in range(self.velocity_set.d):
                    pull_index[d] = index[d] - _c[d, l]

                # Check if pull index is a fluid node (bc_mask is zero for fluid nodes)
                if is_in_bounds(pull_index, shape) and self.read_field(bc_mask, pull_index, 0) == wp.uint8(255):
                    self.write_field(missing_mask, index, l, wp.uint8(True))

        kernel_dic = {
            "kernel_domain_bounds": kernel_domain_bounds,
            "kernel_interior_bc_mask": kernel_interior_bc_mask,
            "kernel_interior_missing_mask": kernel_interior_missing_mask,
        }
        return None, kernel_dic

    # a helper for this operator
    def _prepare_kernel_inputs(self, bclist, grid_shape):
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
        wp_indices = wp.array(indices, dtype=wp.int32)
        wp_id_numbers = wp.array(id_numbers, dtype=wp.uint8)
        wp_is_interior = wp.array(is_interior, dtype=wp.uint8)
        return total_index, wp_indices, wp_id_numbers, wp_is_interior

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        # prepare warp kernel inputs
        bc_interior = []
        grid_shape = self.get_grid_shape(bc_mask)
        for bc in bclist:
            if any(self.are_indices_in_interior(np.array(bc.indices), grid_shape)):
                bc_copy = copy.copy(bc)  # shallow copy of the whole object
                bc_copy.indices = copy.deepcopy(bc.pad_indices())  # deep copy only the modified part
                bc_interior.append(bc_copy)

        # Prepare the first kernel inputs for all items in boundary condition list
        total_index, wp_indices, wp_id_numbers, wp_is_interior = self._prepare_kernel_inputs(bclist, grid_shape)

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel["kernel_domain_bounds"],
            dim=total_index,
            inputs=[
                wp_indices,
                wp_id_numbers,
                wp_is_interior,
                bc_mask,
                missing_mask,
            ],
        )
        # Prepare the second and third kernel inputs for only a subset of boundary conditions associated with the interior
        # Note 1: launching order of the following kernels are important here!
        # Note 2: Due to race conditioning, the two kernels cannot be fused together.
        total_index, wp_indices, wp_id_numbers, _ = self._prepare_kernel_inputs(bc_interior, grid_shape)
        wp.launch(
            self.warp_kernel["kernel_interior_missing_mask"],
            dim=total_index,
            inputs=[
                wp_indices,
                bc_mask,
                missing_mask,
            ],
        )
        wp.launch(
            self.warp_kernel["kernel_interior_bc_mask"],
            dim=total_index,
            inputs=[
                wp_indices,
                wp_id_numbers,
                bc_mask,
            ],
        )

        return bc_mask, missing_mask

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        import neon

        # Make constants
        _d = self.velocity_set.d

        # Pre-allocate arrays with maximum possible size
        grid_shape = bc_mask.get_grid().dim.x, bc_mask.get_grid().dim.y, bc_mask.get_grid().dim.z
        grid_warp = grid_factory(grid_shape, compute_backend=ComputeBackend.WARP, velocity_set=self.velocity_set)
        missing_mask_warp = grid_warp.create_field(cardinality=self.velocity_set.q, dtype=Precision.UINT8)
        bc_mask_warp = grid_warp.create_field(cardinality=1, dtype=Precision.UINT8)

        # Use indices masker with the warp backend to build bc_mask_warp and missing_mask_warp before writing in Neon DS.
        indices_masker_warp = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=ComputeBackend.WARP,
        )
        bc_mask_warp, missing_mask_warp = indices_masker_warp(bclist, bc_mask_warp, missing_mask_warp, start_index)
        wp.synchronize()

        @neon.Container.factory("")
        def container(
            bc_mask_warp: Any,
            missing_mask_warp: Any,
            bc_mask_field: Any,
            missing_mask_field: Any,
        ):
            def loading_step(loader: neon.Loader):
                loader.set_grid(bc_mask_field.get_grid())
                bc_mask_hdl = loader.get_write_handle(bc_mask_field)
                missing_mask_hdl = loader.get_write_handle(missing_mask_field)

                @wp.func
                def masker(gridIdx: Any):
                    cIdx = wp.neon_global_idx(bc_mask_hdl, gridIdx)
                    gx = wp.neon_get_x(cIdx)
                    gy = wp.neon_get_y(cIdx)
                    gz = wp.neon_get_z(cIdx)

                    # TODO@Max - XLB is flattening the z dimension in 3D, while neon uses the y dimension
                    if _d == 2:
                        gy, gz = gz, gy

                    local_mask = bc_mask_warp[0, gx, gy, gz]
                    wp.neon_write(bc_mask_hdl, gridIdx, 0, local_mask)

                    for q in range(self.velocity_set.q):
                        is_missing = wp.uint8(missing_mask_warp[q, gx, gy, gz])
                        wp.neon_write(missing_mask_hdl, gridIdx, q, is_missing)

                loader.declare_kernel(masker)

            return loading_step

        c = container(bc_mask_warp, missing_mask_warp, bc_mask, missing_mask)
        c.run(0)
        wp.synchronize()

        del bc_mask_warp
        del missing_mask_warp

        return bc_mask, missing_mask
