from xlb import DefaultConfig
from xlb.grid import grid_factory
from xlb.precision_policy import Precision
from typing import Tuple


def create_nse_fields(
    grid_shape: Tuple[int, int, int] = None,
    grid=None,
    velocity_set=None,
    compute_backend=None,
    precision_policy=None,
):
    """Create fields for Navier-Stokes equation solver.

    Args:
        grid_shape: Tuple of grid dimensions. Required if grid is not provided.
        grid: Optional Grid object. If provided, will be used instead of creating new grid.
        velocity_set: Optional velocity set. Defaults to DefaultConfig.velocity_set.
        compute_backend: Optional compute backend. Defaults to DefaultConfig.default_backend.
        precision_policy: Optional precision policy. Defaults to DefaultConfig.default_precision_policy.

    Returns:
        Tuple of (grid, f_0, f_1, missing_mask, bc_mask)
    """
    velocity_set = velocity_set or DefaultConfig.velocity_set
    compute_backend = compute_backend or DefaultConfig.default_backend
    precision_policy = precision_policy or DefaultConfig.default_precision_policy

    if grid is None:
        if grid_shape is None:
            raise ValueError("grid_shape must be provided when grid is None")
        grid = grid_factory(grid_shape, compute_backend=compute_backend)

    # Create fields
    f_0 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    missing_mask = grid.create_field(cardinality=velocity_set.q, dtype=Precision.BOOL)
    bc_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8)

    return grid, f_0, f_1, missing_mask, bc_mask
