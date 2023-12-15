import numpy as np
import cupy as cp
import itertools
from dataclasses import dataclass


class Tile:
    """Base class for Tile with ghost cells. This tile is used to build a distributed array.

    Attributes
    ----------
    shape : tuple
        Shape of the tile. This will be the shape of the array without padding/ghost cells.
    dtype : cp.dtype
        Data type the tile represents. Note that the data data may be stored in a different
        data type. For example, if it is stored in compressed form.
    padding : tuple
        Number of padding/ghost cells in each dimension.
    """

    def __init__(self, shape, dtype, padding, codec=None):
        # Store parameters
        self.shape = shape
        self.dtype = dtype
        self.padding = padding
        self.dtype_itemsize = cp.dtype(self.dtype).itemsize
        self.nbytes = 0  # Updated when array is allocated
        self.codec = (
            codec  # Codec to use for compression TODO: Find better abstraction for this
        )

        # Make center array
        self._array = self.allocate_array(self.shape)

        # Make padding indices
        pad_dir = []
        for i in range(len(self.shape)):
            if self.padding[i] == 0:
                pad_dir.append((0,))
            else:
                pad_dir.append((-1, 0, 1))
        self.pad_ind = list(itertools.product(*pad_dir))
        self.pad_ind.remove((0,) * len(self.shape))

        # Make padding and padding buffer arrays
        self._padding = {}
        self._buf_padding = {}
        for ind in self.pad_ind:
            # determine array shape
            shape = []
            for i in range(len(self.shape)):
                if ind[i] == -1 or ind[i] == 1:
                    shape.append(self.padding[i])
                else:
                    shape.append(self.shape[i])

            # Make padding and padding buffer
            self._padding[ind] = self.allocate_array(shape)
            self._buf_padding[ind] = self.allocate_array(shape)

        # Get slicing for array copies
        self._slice_center = tuple(
            [slice(pad, pad + shape) for (pad, shape) in zip(self.padding, self.shape)]
        )
        self._slice_padding_to_array = {}
        self._slice_array_to_padding = {}
        self._padding_shape = {}
        for pad_ind in self.pad_ind:
            slice_padding_to_array = []
            slice_array_to_padding = []
            padding_shape = []
            for pad, ind, s in zip(self.padding, pad_ind, self.shape):
                if ind == -1:
                    slice_padding_to_array.append(slice(0, pad))
                    slice_array_to_padding.append(slice(pad, 2 * pad))
                    padding_shape.append(pad)
                elif ind == 0:
                    slice_padding_to_array.append(slice(pad, s + pad))
                    slice_array_to_padding.append(slice(pad, s + pad))
                    padding_shape.append(s)
                else:
                    slice_padding_to_array.append(slice(s + pad, s + 2 * pad))
                    slice_array_to_padding.append(slice(s, s + pad))
                    padding_shape.append(pad)
            self._slice_padding_to_array[pad_ind] = tuple(slice_padding_to_array)
            self._slice_array_to_padding[pad_ind] = tuple(slice_array_to_padding)
            self._padding_shape[pad_ind] = tuple(padding_shape)

    def size(self):
        """Returns the number of bytes allocated for the tile."""
        raise NotImplementedError

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        raise NotImplementedError

    def copy_tile(self, dst_tile):
        """Copy a tile from one tile to another."""
        raise NotImplementedError

    def to_array(self, array):
        """Copy a tile to a full array."""
        raise NotImplementedError

    def from_array(self, array):
        """Copy a full array to a tile."""
        raise NotImplementedError

    def swap_buf_padding(self):
        """Swap the padding buffer pointer with the padding pointer."""
        for index in self.pad_ind:
            (self._buf_padding[index], self._padding[index]) = (
                self._padding[index],
                self._buf_padding[index],
            )
