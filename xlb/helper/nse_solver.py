import xlb
from xlb.compute_backend import ComputeBackend
from xlb import DefaultConfig
from xlb.grid import grid_factory
from xlb.precision_policy import Precision
from typing import Tuple


def create_nse_fields(
    grid_shape: Tuple[int, int, int], distribute=False, velocity_set=None, compute_backend=None, precision_policy=None
):
    velocity_set = velocity_set if velocity_set else DefaultConfig.velocity_set
    compute_backend = (
        compute_backend if compute_backend else DefaultConfig.default_backend
    )
    precision_policy = (
        precision_policy if precision_policy else DefaultConfig.default_precision_policy
    )
    grid = grid_factory(grid_shape, distribute=distribute, compute_backend=compute_backend)

    # Create fields
    f_0 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    missing_mask = grid.create_field(cardinality=velocity_set.q, dtype=Precision.UINT8)
    boundary_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8)

    return grid, f_0, f_1, missing_mask, boundary_mask

