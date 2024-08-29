import numpy as np
import warp as wp
import jax
import jax.numpy as jnp
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.stream.stream import Stream
from xlb.grid import grid_factory
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

    @Operator.register_backend(ComputeBackend.JAX)
    # TODO HS: figure out why uncommenting the line below fails unlike other operators!
    # @partial(jit, static_argnums=(0))
    def jax_implementation(self, bclist, boundary_map, missing_mask, start_index=None):
        # Pad the missing mask to create a grid mask to identify out of bound boundaries
        # Set padded regin to True (i.e. boundary)
        dim = missing_mask.ndim - 1
        nDevices = jax.device_count()
        pad_x, pad_y, pad_z = nDevices, 1, 1
        if dim == 2:
            grid_mask = jnp.pad(missing_mask, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), constant_values=True)
        if dim == 3:
            grid_mask = jnp.pad(missing_mask, ((0, 0), (pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), constant_values=True)

        # shift indices
        shift_tup = (pad_x, pad_y) if dim == 2 else (pad_x, pad_y, pad_z)
        if start_index is None:
            start_index = (0,) * dim

        bid = boundary_map[0]
        for bc in bclist:
            assert bc.indices is not None, f"Please specify indices associated with the {bc.__class__.__name__} BC!"
            assert bc.mesh_vertices is None, f"Please use MeshBoundaryMasker operator if {bc.__class__.__name__} is imposed on a mesh (e.g. STL)!"
            id_number = bc.id
            local_indices = np.array(bc.indices) - np.array(start_index)[:, np.newaxis]
            padded_indices = local_indices + np.array(shift_tup)[:, np.newaxis]
            bid = bid.at[tuple(local_indices)].set(id_number)
            # if dim == 2:
            #     grid_mask = grid_mask.at[:, padded_indices[0], padded_indices[1]].set(True)
            # if dim == 3:
            #     grid_mask = grid_mask.at[:, padded_indices[0], padded_indices[1], padded_indices[2]].set(True)
            # We are done with bc.indices. Remove them from BC objects
            bc.__dict__.pop("indices", None)

        boundary_map = boundary_map.at[0].set(bid)
        grid_mask = self.stream(grid_mask)
        if dim == 2:
            missing_mask = grid_mask[:, pad_x:-pad_x, pad_y:-pad_y]
        if dim == 3:
            missing_mask = grid_mask[:, pad_x:-pad_x, pad_y:-pad_y, pad_z:-pad_z]
        return boundary_map, missing_mask

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.wp_c
        _q = wp.constant(self.velocity_set.q)

        # Construct the warp 2D kernel
        @wp.kernel
        def kernel2d(
            indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            boundary_map: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.bool),
            start_index: wp.vec2i,
        ):
            # Get the index of indices
            ii = wp.tid()

            # Get local indices
            index = wp.vec2i()
            index[0] = indices[0, ii] - start_index[0]
            index[1] = indices[1, ii] - start_index[1]

            # Check if index is in bounds
            if index[0] >= 0 and index[0] < missing_mask.shape[1] and index[1] >= 0 and index[1] < missing_mask.shape[2]:
                # Stream indices
                for l in range(_q):
                    # Get the index of the streaming direction
                    pull_index = wp.vec2i()
                    for d in range(self.velocity_set.d):
                        pull_index[d] = index[d] - _c[d, l]

                    # check if pull index is out of bound
                    # These directions will have missing information after streaming
                    if pull_index[0] < 0 or pull_index[0] >= missing_mask.shape[1] or pull_index[1] < 0 or pull_index[1] >= missing_mask.shape[2]:
                        # Set the missing mask
                        missing_mask[l, index[0], index[1]] = True

                boundary_map[0, index[0], index[1]] = id_number[ii]

        # Construct the warp 3D kernel
        @wp.kernel
        def kernel3d(
            indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            boundary_map: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            start_index: wp.vec3i,
        ):
            # Get the index of indices
            ii = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = indices[0, ii] - start_index[0]
            index[1] = indices[1, ii] - start_index[1]
            index[2] = indices[2, ii] - start_index[2]

            # Check if index is in bounds
            if (
                index[0] >= 0
                and index[0] < missing_mask.shape[1]
                and index[1] >= 0
                and index[1] < missing_mask.shape[2]
                and index[2] >= 0
                and index[2] < missing_mask.shape[3]
            ):
                # Stream indices
                for l in range(_q):
                    # Get the index of the streaming direction
                    pull_index = wp.vec3i()
                    for d in range(self.velocity_set.d):
                        pull_index[d] = index[d] - _c[d, l]

                    # check if pull index is out of bound
                    # These directions will have missing information after streaming
                    if (
                        pull_index[0] < 0
                        or pull_index[0] >= missing_mask.shape[1]
                        or pull_index[1] < 0
                        or pull_index[1] >= missing_mask.shape[2]
                        or pull_index[2] < 0
                        or pull_index[2] >= missing_mask.shape[3]
                    ):
                        # Set the missing mask
                        missing_mask[l, index[0], index[1], index[2]] = True

                boundary_map[0, index[0], index[1], index[2]] = id_number[ii]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, bclist, boundary_map, missing_mask, start_index=None):
        dim = self.velocity_set.d
        index_list = [[] for _ in range(dim)]
        id_list = []
        for bc in bclist:
            assert bc.indices is not None, f'Please specify indices associated with the {bc.__class__.__name__} BC using keyword "indices"!'
            assert bc.mesh_vertices is None, f"Please use MeshBoundaryMasker operator if {bc.__class__.__name__} is imposed on a mesh (e.g. STL)!"
            for d in range(dim):
                index_list[d] += bc.indices[d]
            id_list += [bc.id] * len(bc.indices[0])
            # We are done with bc.indices. Remove them from BC objects
            bc.__dict__.pop("indices", None)

        indices = wp.array2d(index_list, dtype=wp.int32)
        id_number = wp.array1d(id_list, dtype=wp.uint8)

        if start_index is None:
            start_index = (0,) * dim

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                indices,
                id_number,
                boundary_map,
                missing_mask,
                start_index,
            ],
            dim=indices.shape[1],
        )

        return boundary_map, missing_mask
