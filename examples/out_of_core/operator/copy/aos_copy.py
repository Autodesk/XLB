from typing import Any
import warp as wp

from operator.operator import Operator


class AOSCopy(Operator):
    """
    AOSCopy is an operator for copying data from a source array to a destination array.
    It supports both 2D and 3D arrays using the Array of Structures (AoS) format.
    """

    @wp.kernel
    def aos_copy_3d(
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
            dest[i, j, k, ii] = src[i, j, k, ii]

    @wp.kernel
    def aos_copy_2d(
        dest: wp.array3d(dtype=Any),
        src: wp.array3d(dtype=Any),
        q: wp.int32,
    ):
        """
        Copy data from a 2D source array to a 2D destination array.

        Parameters
        ----------
        dest : wp.array3d
            The destination array where data will be copied to.
        src : wp.array3d
            The source array from which data will be copied.
        q : wp.int32
            The number of elements to copy along the first dimension.
        """
        # Get the global index
        i, j = wp.tid()

        # Copy the data
        for ii in range(q):
            dest[i, j, ii] = src[i, j, ii]

    @wp.kernel
    def aos_copy_1d(
        dest: wp.array2d(dtype=Any),
        src: wp.array2d(dtype=Any),
        q: wp.int32,
    ):
        """
        Copy data from a 1D source array to a 1D destination array.

        Parameters
        ----------
        dest : wp.array2d
            The destination array where data will be copied to.
        src : wp.array2d
            The source array from which data will be copied.
        q : wp.int32
            The number of elements to copy.
        """
        # Get the global index
        i = wp.tid()

        # Copy the data
        for ii in range(q):
            dest[i, ii] = src[i, ii]

    def __call__(
        self,
        dest: wp.array,
        src: wp.array,
    ):
        """
        Execute the appropriate kernel to copy data from the source to the destination array.

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
        kernel = (
            self.aos_copy_3d if len(dest.shape) == 4 else
            self.aos_copy_2d if len(dest.shape) == 3 else
            self.aos_copy_1d
        )
        wp.launch(
            kernel,
            inputs=[
                dest,
                src,
                dest.shape[-1]
            ],
            dim=dest.shape[:-1],
        )
        return dest 
