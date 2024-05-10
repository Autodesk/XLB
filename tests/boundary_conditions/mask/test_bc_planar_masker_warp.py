import pytest
import numpy as np
import xlb
import warp as wp

from xlb.compute_backend import ComputeBackend
from xlb.default_config import DefaultConfig
from xlb.grid import grid_factory


def init_xlb_env(velocity_set):
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=velocity_set,
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape,lower_bound,upper_bound,direction",
    [
        # 2D Grids - Different directions
        (
            2,
            xlb.velocity_set.D2Q9,
            (4, 4),
            (0, 0),
            (2, 4),
            (1, 0),
        ),  # Horizontal direction
        (
            2,
            xlb.velocity_set.D2Q9,
            (50, 50),
            (0, 0),
            (50, 25),
            (0, 1),
        ),  # Vertical direction
        (
            2,
            xlb.velocity_set.D2Q9,
            (100, 100),
            (50, 0),
            (100, 50),
            (0, 1),
        ),  # Vertical direction
        # 3D Grids - Different directions
        (
            3,
            xlb.velocity_set.D3Q19,
            (50, 50, 50),
            (0, 0, 0),
            (25, 50, 50),
            (1, 0, 0),
        ),  # Along x-axis
        (
            3,
            xlb.velocity_set.D3Q19,
            (100, 100, 100),
            (0, 50, 0),
            (50, 100, 100),
            (0, 1, 0),
        ),  # Along y-axis
        (
            3,
            xlb.velocity_set.D3Q27,
            (50, 50, 50),
            (0, 0, 0),
            (50, 25, 50),
            (0, 0, 1),
        ),  # Along z-axis
        (
            3,
            xlb.velocity_set.D3Q27,
            (100, 100, 100),
            (0, 0, 0),
            (50, 100, 50),
            (1, 0, 0),
        ),  # Along x-axis
    ],
)
def test_planar_masker_warp(
    dim, velocity_set, grid_shape, lower_bound, upper_bound, direction
):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)
    velocity_set = DefaultConfig.velocity_set

    # Create required fields
    missing_mask = my_grid.create_field(
        cardinality=velocity_set.q, dtype=xlb.Precision.BOOL
    )
    fill_value = 0
    boundary_id_field = my_grid.create_field(
        cardinality=1, dtype=xlb.Precision.UINT8, fill_value=fill_value
    )

    planar_boundary_masker = xlb.operator.boundary_masker.PlanarBoundaryMasker()
    start_index = (0,) * dim
    id_number = 1

    boundary_id_field, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        id_number,
        boundary_id_field,
        missing_mask,
        start_index,
    )

    boundary_id_field = boundary_id_field.numpy()

    # Assertions to verify boundary settings
    expected_slice = (slice(None),) + tuple(
        slice(lb, ub) for lb, ub in zip(lower_bound, upper_bound)
    )
    assert np.all(
        boundary_id_field[expected_slice] == id_number
    ), "Boundary not set correctly"

    # Assertions for non-affected areas
    full_slice = tuple(slice(None) for _ in grid_shape)
    mask = np.ones_like(boundary_id_field, dtype=bool)
    mask[expected_slice] = False
    assert np.all(
        boundary_id_field[full_slice][mask] == fill_value
    ), "Rest of domain incorrectly affected"
