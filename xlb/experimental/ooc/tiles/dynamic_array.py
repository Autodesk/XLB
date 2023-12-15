# Dynamic array class for pinned memory allocation

import math
import cupy as cp
import numpy as np
import time


class DynamicArray:
    """
    Dynamic pinned memory array class.

    Attributes
    ----------
    nbytes : int
        The number of bytes in the array.
    bytes_resize : int
        The number of bytes to resize the array by if the number of bytes requested exceeds the allocated number of bytes.
    """

    def __init__(self, nbytes, bytes_resize_factor=0.025):
        # Set the number of bytes
        self.nbytes = nbytes
        self.bytes_resize_factor = bytes_resize_factor
        self.bytes_resize = math.ceil(bytes_resize_factor * nbytes)

        # Set the number of bytes
        self.allocated_bytes = math.ceil(nbytes / self.bytes_resize) * self.bytes_resize


class DynamicPinnedArray(DynamicArray):
    def __init__(self, nbytes, bytes_resize_factor=0.05):
        super().__init__(nbytes, bytes_resize_factor)

        # Allocate the memory
        self.mem = cp.cuda.alloc_pinned_memory(self.allocated_bytes)

        # Make np array that points to the pinned memory
        self.array = np.frombuffer(self.mem, dtype=np.uint8, count=int(self.nbytes))

    def size(self):
        return self.mem.size()

    def resize(self, nbytes):
        # Set the new number of bytes
        self.nbytes = nbytes

        # Check if the number of bytes requested is less than 2xbytes_resize or if the number of bytes requested exceeds the allocated number of bytes
        if (
            nbytes < (self.allocated_bytes - 2 * self.bytes_resize)
            or nbytes > self.allocated_bytes
        ):
            ## Free the memory
            # del self.mem

            # Set the new number of allocated bytes
            self.allocated_bytes = (
                math.ceil(nbytes / self.bytes_resize) * self.bytes_resize
            )

            # Allocate the memory
            self.mem = cp.cuda.alloc_pinned_memory(self.allocated_bytes)

            # Make np array that points to the pinned memory
            self.array = np.frombuffer(self.mem, dtype=np.uint8, count=int(self.nbytes))

            # Set new resize number of bytes
            self.bytes_resize = math.ceil(self.bytes_resize_factor * nbytes)

        # Otherwise change numpy array size
        else:
            self.array = np.frombuffer(self.mem, dtype=np.uint8, count=int(self.nbytes))
