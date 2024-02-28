import warp as wp

from xlb.grid import Grid
from xlb.operator import Operator

class WarpGrid(Grid):
    def __init__(self, shape):
        super().__init__(shape)

    def parallelize_operator(self, operator: Operator):
        # TODO: Implement parallelization of the operator
        raise NotImplementedError("Parallelization of the operator is not implemented yet for the WarpGrid")

    def create_field(self, cardinality: int, dtype, callback=None):
        # Get shape of the field
        shape = (cardinality,) + (self.shape)

        # Create the field
        f = wp.zeros(shape, dtype=dtype)

        # Raise error on callback
        if callback is not None:
            raise ValueError("Callback is not supported in the WarpGrid")

        # Add field to the field dictionary
        return f
