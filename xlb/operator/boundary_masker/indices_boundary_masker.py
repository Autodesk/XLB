from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

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
        d = self.velocity_set.d
        shape_array = np.array(shape)
        return np.all((indices[:d] > 0) & (indices[:d] < shape_array[:d, np.newaxis] - 1), axis=0)

    @Operator.register_backend(ComputeBackend.JAX)
    # TODO HS: figure out why uncommenting the line below fails unlike other operators!
    # @partial(jit, static_argnums=(0))
    def jax_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        # Pad the missing mask to create a grid mask to identify out of bound boundaries
        # Set padded regin to True (i.e. boundary)
        dim = missing_mask.ndim - 1
        nDevices = jax.device_count()
        pad_x, pad_y, pad_z = nDevices, 1, 1
        # TODO MEHDI: There is sometimes a halting problem here when padding is used in a multi-GPU setting since we're not jitting this function.
        # For now, we compute the bmap on GPU zero.
        if dim == 2:
            bmap = jnp.zeros((pad_x * 2 + bc_mask[0].shape[0], pad_y * 2 + bc_mask[0].shape[1]), dtype=jnp.uint8)
            bmap = bmap.at[pad_x:-pad_x, pad_y:-pad_y].set(bc_mask[0])
            grid_mask = jnp.pad(missing_mask, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), constant_values=True)
            # bmap = jnp.pad(bc_mask[0], ((pad_x, pad_x), (pad_y, pad_y)), constant_values=0)
        if dim == 3:
            bmap = jnp.zeros((pad_x * 2 + bc_mask[0].shape[0], pad_y * 2 + bc_mask[0].shape[1], pad_z * 2 + bc_mask[0].shape[2]), dtype=jnp.uint8)
            bmap = bmap.at[pad_x:-pad_x, pad_y:-pad_y, pad_z:-pad_z].set(bc_mask[0])
            grid_mask = jnp.pad(missing_mask, ((0, 0), (pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), constant_values=True)
            # bmap = jnp.pad(bc_mask[0], ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), constant_values=0)

        # shift indices
        shift_tup = (pad_x, pad_y) if dim == 2 else (pad_x, pad_y, pad_z)
        if start_index is None:
            start_index = (0,) * dim

        domain_shape = bc_mask[0].shape
        for bc in bclist:
            assert bc.indices is not None, f"Please specify indices associated with the {bc.__class__.__name__} BC!"
            assert bc.mesh_vertices is None, f"Please use MeshBoundaryMasker operator if {bc.__class__.__name__} is imposed on a mesh (e.g. STL)!"
            id_number = bc.id
            bc_indices = np.array(bc.indices)
            local_indices = bc_indices - np.array(start_index)[:, np.newaxis]
            padded_indices = local_indices + np.array(shift_tup)[:, np.newaxis]
            bmap = bmap.at[tuple(padded_indices)].set(id_number)
            if any(self.are_indices_in_interior(bc_indices, domain_shape)) and bc.needs_padding:
                # checking if all indices associated with this BC are in the interior of the domain.
                # This flag is needed e.g. if the no-slip geometry is anywhere but at the boundaries of the computational domain.
                if dim == 2:
                    grid_mask = grid_mask.at[:, padded_indices[0], padded_indices[1]].set(True)
                if dim == 3:
                    grid_mask = grid_mask.at[:, padded_indices[0], padded_indices[1], padded_indices[2]].set(True)

                # Assign the boundary id to the push indices
                push_indices = padded_indices[:, :, None] + self.velocity_set.c[:, None, :]
                push_indices = push_indices.reshape(dim, -1)
                bmap = bmap.at[tuple(push_indices)].set(id_number)

            # We are done with bc.indices. Remove them from BC objects
            bc.__dict__.pop("indices", None)

        grid_mask = self.stream(grid_mask)
        if dim == 2:
            missing_mask = grid_mask[:, pad_x:-pad_x, pad_y:-pad_y]
            bc_mask = bc_mask.at[0].set(bmap[pad_x:-pad_x, pad_y:-pad_y])
        if dim == 3:
            missing_mask = grid_mask[:, pad_x:-pad_x, pad_y:-pad_y, pad_z:-pad_z]
            bc_mask = bc_mask.at[0].set(bmap[pad_x:-pad_x, pad_y:-pad_y, pad_z:-pad_z])
        return bc_mask, missing_mask

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _q = wp.constant(self.velocity_set.q)

        @wp.func
        def check_index_bounds(index: wp.vec3i, shape: wp.vec3i):
            is_in_bounds = index[0] >= 0 and index[0] < shape[0] and index[1] >= 0 and index[1] < shape[1] and index[2] >= 0 and index[2] < shape[2]
            return is_in_bounds

        # Construct the warp 3D kernel
        @wp.kernel
        def kernel(
            indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            is_interior: wp.array1d(dtype=wp.bool),
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

            # Check if index is in bounds
            shape = wp.vec3i(missing_mask.shape[1], missing_mask.shape[2], missing_mask.shape[3])
            if check_index_bounds(index, shape):
                # Stream indices
                for l in range(_q):
                    # Get the index of the streaming direction
                    pull_index = wp.vec3i()
                    push_index = wp.vec3i()
                    for d in range(self.velocity_set.d):
                        pull_index[d] = index[d] - _c[d, l]
                        push_index[d] = index[d] + _c[d, l]

                    # set bc_mask for all bc indices
                    bc_mask[0, index[0], index[1], index[2]] = id_number[ii]

                    # check if pull index is out of bound
                    # These directions will have missing information after streaming
                    if not check_index_bounds(pull_index, shape):
                        # Set the missing mask
                        missing_mask[l, index[0], index[1], index[2]] = wp.uint8(True)

                    # handling geometries in the interior of the computational domain
                    elif check_index_bounds(pull_index, shape) and is_interior[ii]:
                        # Set the missing mask
                        missing_mask[l, push_index[0], push_index[1], push_index[2]] = wp.uint8(True)
                        bc_mask[0, push_index[0], push_index[1], push_index[2]] = id_number[ii]

        return None, kernel

    # a helper for this operator
    def _prepare_warp_kernel_inputs(self, bclist, bc_mask):
        # Pre-allocate arrays with maximum possible size
        max_size = sum(len(bc.indices[0]) if isinstance(bc.indices, list) else bc.indices.shape[1] for bc in bclist if bc.indices is not None)
        indices = np.zeros((3, max_size), dtype=np.int32)
        id_numbers = np.zeros(max_size, dtype=np.uint8)
        is_interior = np.zeros(max_size, dtype=bool)

        current_index = 0
        for bc in bclist:
            assert bc.indices is not None, f'Please specify indices associated with the {bc.__class__.__name__} BC using keyword "indices"!'
            assert bc.mesh_vertices is None, f"Please use MeshBoundaryMasker operator if {bc.__class__.__name__} is imposed on a mesh (e.g. STL)!"

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
            if bc.needs_padding:
                is_interior[current_index : current_index + num_indices] = self.are_indices_in_interior(bc_indices, bc_mask[0].shape)
            else:
                is_interior[current_index : current_index + num_indices] = False

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
        wp_is_interior = wp.array(is_interior, dtype=wp.bool)
        return total_index, wp_indices, wp_id_numbers, wp_is_interior

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        # prepare warp kernel inputs
        total_index, wp_indices, wp_id_numbers, wp_is_interior = self._prepare_warp_kernel_inputs(bclist, bc_mask)

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            dim=total_index,
            inputs=[
                wp_indices,
                wp_id_numbers,
                wp_is_interior,
                bc_mask,
                missing_mask,
            ],
        )

        return bc_mask, missing_mask

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bclist, bc_mask, missing_mask, start_index=None):
        import neon

        # Make constants
        _d = self.velocity_set.d

        # Pre-allocate arrays with maximum possible size
        grid_shape = bc_mask.shape[1:]  # (nx, ny) for 2D or (nx, ny, nz) for 3D
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
