import pytest
import warp as wp
import numpy as np
import xlb
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
    "dim,velocity_set,grid_shape",
    [
        (2, xlb.velocity_set.D2Q9, (50, 50)),
        (2, xlb.velocity_set.D2Q9, (50, 50)),
        (3, xlb.velocity_set.D3Q19, (20, 20, 20)),
        (3, xlb.velocity_set.D3Q19, (20, 20, 20)),
    ],
)
def test_indices_masker_warp(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)
    velocity_set = DefaultConfig.velocity_set

    missing_mask = my_grid.create_field(
        cardinality=velocity_set.q, dtype=xlb.Precision.BOOL
    )

    boundary_id_field = my_grid.create_field(cardinality=1, dtype=xlb.Precision.UINT8)

    indices_boundary_masker = xlb.operator.boundary_masker.IndicesBoundaryMasker()

    # Make indices for boundary conditions (sphere)
    sphere_radius = grid_shape[0] // 4
    nr = grid_shape[0]
    x = np.arange(nr)
    y = np.arange(nr)
    z = np.arange(nr)
    if dim == 2:
        X, Y = np.meshgrid(x, y)
        indices = np.where((X - nr // 2) ** 2 + (Y - nr // 2) ** 2 < sphere_radius**2)
    else:
        X, Y, Z = np.meshgrid(x, y, z)
        indices = np.where(
            (X - nr // 2) ** 2 + (Y - nr // 2) ** 2 + (Z - nr // 2) ** 2
            < sphere_radius**2
        )

    indices = wp.array(indices, dtype=wp.int32)

    assert indices.shape[0] == dim
    test_id = 5
    boundary_id_field, missing_mask = indices_boundary_masker(
        indices,
        test_id,
        boundary_id_field,
        missing_mask,
        start_index=(0, 0, 0) if dim == 3 else (0, 0),
    )
    assert missing_mask.dtype == xlb.Precision.BOOL.wp_dtype

    assert boundary_id_field.dtype == xlb.Precision.UINT8.wp_dtype

    boundary_id_field = boundary_id_field.numpy()
    missing_mask = missing_mask.numpy()
    indices = indices.numpy()

    assert boundary_id_field.shape == (1,) + grid_shape

    assert missing_mask.shape == (velocity_set.q,) + grid_shape

    if dim == 2:
        assert np.all(boundary_id_field[0, indices[0], indices[1]] == test_id)
        # assert that the rest of the boundary_id_field is zero
        boundary_id_field[0, indices[0], indices[1]]= 0
        assert np.all(boundary_id_field == 0)
    if dim == 3:
        assert np.all(
            boundary_id_field[0, indices[0], indices[1], indices[2]] == test_id
        )
        # assert that the rest of the boundary_id_field is zero
        boundary_id_field[0, indices[0], indices[1], indices[2]] = 0
        assert np.all(boundary_id_field == 0)

if __name__ == "__main__":
    pytest.main()
