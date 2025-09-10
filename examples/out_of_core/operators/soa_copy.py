from typing import Any
import warp as wp

class SOACopy:
    """
    SOACopy is an operator for copying data from a source array to a destination array.
    This is currently just used to speed up the copy as warps copy is not optimized for
    non-contiguous arrays.
    """

    @wp.kernel
    def soa_copy_3d(
        dest: wp.array4d(dtype=Any),
        src: wp.array4d(dtype=Any),
        q: wp.int32,
    ):
        """
        Copy data from a 3D source array to a 3D destination array.

        Parameters
        ----------
        dest : wp.array4d
            The destination array where data will be copied to.
        src : wp.array4d
            The source array from which data will be copied.
        q : wp.int32
            The number of elements to copy along the first dimension.
        """
        # Get the global index
        i, j, k = wp.tid()

        # Copy the data
        for ii in range(q):
            dest[ii, i, j, k] = src[ii, i, j, k]

    def __call__(
        self,
        dest: wp.array,
        src: wp.array,
    ):
        """
        Parameters
        ----------
        dest : wp.array
            The destination array where data will be copied to.
        src : wp.array
            The source array from which data will be copied.

        Returns
        -------
        wp.array
            The destination array with copied data.
        """
        # Launch the warp kernel
        wp.launch(
            self.soa_copy_3d,
            inputs=[
                dest,
                src,
                dest.shape[0]
            ],
            dim=dest.shape[1:],
        )
        return dest
