import warp as wp
import gc

class MemoryPool:

    def __init__(self):
        self.pool = {}

    def clear(self):
        for key in list(self.pool.keys()):
            for array in self.pool[key]:
                del array
            del self.pool[key]
            self.pool[key] = []
        wp.synchronize()
        gc.collect()

    def get(self, shape, dtype):
        key = (tuple(shape), dtype)
        if key not in self.pool:
            self.pool[key] = []
        if len(self.pool[key]) == 0:
            self.pool[key].append(wp.zeros(shape, dtype=dtype))
        return self.pool[key].pop()

    def ret(self, array, zero=True):
        key = (tuple(array.shape), array.dtype)
        #if zero:
        #    array.zero_()
        array.zero_()
        self.pool[key].append(array)

    @property
    def nbytes(self):
        nbytes = 0
        for key in self.pool.keys():
            for array in self.pool[key]:
                nbytes += array.capacity
        return nbytes

    def print(self):

        print("Memory Pool")
        for key in self.pool.keys():
            for array in self.pool[key]:
                print(f"Array {array.shape}, {array.dtype}, {array.capacity}")
