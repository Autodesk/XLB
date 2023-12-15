import numpy as np
import cupy as cp
import itertools
from dataclasses import dataclass

from xlb.experimental.ooc.tiles.tile import Tile


class DenseTile(Tile):
    """A Tile where the data is stored in a dense array of the requested dtype."""

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        raise NotImplementedError

    def to_array(self, array):
        """Copy a tile to a full array."""
        # TODO: This can be done with a single kernel call, profile to see if it is faster and needs to be done.

        # Copy center array
        array[self._slice_center] = self._array

        # Copy padding
        for pad_ind in self.pad_ind:
            array[self._slice_padding_to_array[pad_ind]] = self._padding[pad_ind]

    def from_array(self, array):
        """Copy a full array to tile."""
        # TODO: This can be done with a single kernel call, profile to see if it is faster and needs to be done.

        # Copy center array
        self._array[...] = array[self._slice_center]

        # Copy padding
        for pad_ind in self.pad_ind:
            self._padding[pad_ind][...] = array[self._slice_array_to_padding[pad_ind]]


class DenseCPUTile(DenseTile):
    """A dense tile with cells on the CPU."""

    def __init__(self, shape, dtype, padding, codec=None):
        super().__init__(shape, dtype, padding, None)

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        # TODO: Seems hacky, but it works. Is there a better way?
        mem = cp.cuda.alloc_pinned_memory(np.prod(shape) * self.dtype_itemsize)
        array = np.frombuffer(mem, dtype=self.dtype, count=np.prod(shape)).reshape(
            shape
        )
        self.nbytes += mem.size()
        return array

    def to_gpu_tile(self, dst_gpu_tile):
        """Copy tile to a GPU tile."""

        # Check that the destination tile is on the GPU
        assert isinstance(dst_gpu_tile, DenseGPUTile), "Destination tile must be on GPU"

        # Copy array
        dst_gpu_tile._array.set(self._array)

        # Copy padding
        for src_array, dst_gpu_array in zip(
            self._padding.values(), dst_gpu_tile._padding.values()
        ):
            dst_gpu_array.set(src_array)


class DenseGPUTile(DenseTile):
    """A sub-array with ghost cells on the GPU."""

    def __init__(self, shape, dtype, padding, codec=None):
        super().__init__(shape, dtype, padding, None)

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        array = cp.zeros(shape, dtype=self.dtype)
        self.nbytes += array.nbytes
        return array

    def to_cpu_tile(self, dst_cpu_tile):
        """Copy tile to a CPU tile."""

        # Check that the destination tile is on the CPU
        assert isinstance(dst_cpu_tile, DenseCPUTile), "Destination tile must be on CPU"

        # Copy arra
        self._array.get(out=dst_cpu_tile._array)

        # Copy padding
        for src_array, dst_array in zip(
            self._padding.values(), dst_cpu_tile._padding.values()
        ):
            src_array.get(out=dst_array)
