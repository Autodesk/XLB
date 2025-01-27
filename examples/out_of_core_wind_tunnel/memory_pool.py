import warp as wp

class MemoryPool:

    def __init__(self):
        self.pool = {}

    def clear(self):
        for key in self.pool.keys():
            for array in self.pool[key]:
                del array
            self.pool[key] = []

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
