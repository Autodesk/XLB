import numpy as np
import warp as wp
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.stream.stream import Stream


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
    def jax_implementation(self, bclist, boundary_mask, mask, start_index=None):
        # define a helper function
        def compute_boundary_id_and_mask(boundary_mask, mask):
            if dim == 2:
                boundary_mask = boundary_mask.at[0, local_indices[0], local_indices[1]].set(id_number)
                mask = mask.at[:, local_indices[0], local_indices[1]].set(True)

            if dim == 3:
                boundary_mask = boundary_mask.at[0, local_indices[0], local_indices[1], local_indices[2]].set(id_number)
                mask = mask.at[:, local_indices[0], local_indices[1], local_indices[2]].set(True)
            return boundary_mask, mask

        dim = mask.ndim - 1
        if start_index is None:
            start_index = (0,) * dim

        for bc in bclist:
            assert bc.indices is not None, f"Please specify indices associated with the {bc.__class__.__name__} BC!"
            id_number = bc.id
            local_indices = np.array(bc.indices) - np.array(start_index)[:, np.newaxis]
            boundary_mask, mask = compute_boundary_id_and_mask(boundary_mask, mask)
            # We are done with bc.indices. Remove them from BC objects
            bc.__dict__.pop("indices", None)

        mask = self.stream(mask)
        return boundary_mask, mask

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.wp_c
        _q = wp.constant(self.velocity_set.q)

        # Construct the warp 2D kernel
        @wp.kernel
        def kernel2d(
            indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            boundary_mask: wp.array3d(dtype=wp.uint8),
            mask: wp.array3d(dtype=wp.bool),
            start_index: wp.vec2i,
        ):
            # Get the index of indices
            ii = wp.tid()

            # Get local indices
            index = wp.vec2i()
            index[0] = indices[0, ii] - start_index[0]
            index[1] = indices[1, ii] - start_index[1]

            # Check if in bounds
            if index[0] >= 0 and index[0] < mask.shape[1] and index[1] >= 0 and index[1] < mask.shape[2]:
                # Stream indices
                for l in range(_q):
                    # Get the index of the streaming direction
                    push_index = wp.vec2i()
                    for d in range(self.velocity_set.d):
                        push_index[d] = index[d] + _c[d, l]

                    # Set the boundary id and mask
                    mask[l, push_index[0], push_index[1]] = True

                boundary_mask[0, index[0], index[1]] = id_number[ii]

        # Construct the warp 3D kernel
        @wp.kernel
        def kernel3d(
            indices: wp.array2d(dtype=wp.int32),
            id_number: wp.array1d(dtype=wp.uint8),
            boundary_mask: wp.array4d(dtype=wp.uint8),
            mask: wp.array4d(dtype=wp.bool),
            start_index: wp.vec3i,
        ):
            # Get the index of indices
            ii = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = indices[0, ii] - start_index[0]
            index[1] = indices[1, ii] - start_index[1]
            index[2] = indices[2, ii] - start_index[2]

            # Check if in bounds
            if (
                index[0] >= 0
                and index[0] < mask.shape[1]
                and index[1] >= 0
                and index[1] < mask.shape[2]
                and index[2] >= 0
                and index[2] < mask.shape[3]
            ):
                # Stream indices
                for l in range(_q):
                    # Get the index of the streaming direction
                    push_index = wp.vec3i()
                    for d in range(self.velocity_set.d):
                        push_index[d] = index[d] + _c[d, l]

                    # Set the mask
                    mask[l, push_index[0], push_index[1], push_index[2]] = True

                boundary_mask[0, index[0], index[1], index[2]] = id_number[ii]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, bclist, boundary_mask, missing_mask, start_index=None):
        dim = self.velocity_set.d
        index_list = [[] for _ in range(dim)]
        id_list = []
        for bc in bclist:
            assert bc.indices is not None, f'Please specify indices associated with the {bc.__class__.__name__} BC using keyword "indices"!'
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
                boundary_mask,
                missing_mask,
                start_index,
            ],
            dim=indices.shape[1],
        )

        return boundary_mask, missing_mask
