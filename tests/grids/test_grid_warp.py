import pytest
import warp as wp
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.precision_policy import Precision


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=vel_set,
    )


@pytest.mark.parametrize("grid_size", [50, 100, 150])
def test_warp_grid_create_field(grid_size):
    for grid_shape in [(grid_size, grid_size), (grid_size, grid_size, grid_size)]:
        init_xlb_env(xlb.velocity_set.D3Q19)
        my_grid = grid_factory(grid_shape)
        f = my_grid.create_field(cardinality=9, dtype=Precision.FP32)
        if len(grid_shape) == 2:
            assert f.shape == (9,) + grid_shape + (1,), "Field shape is incorrect got {}".format(f.shape)
        else:
            assert f.shape == (9,) + grid_shape, "Field shape is incorrect got {}".format(f.shape)
        assert isinstance(f, wp.array), "Field should be a Warp ndarray"


def test_warp_grid_create_field_fill_value():
    init_xlb_env(xlb.velocity_set.D2Q9)
    grid_shape = (100, 100)
    fill_value = 3.14
    my_grid = grid_factory(grid_shape)

    f = my_grid.create_field(cardinality=9, dtype=Precision.FP32, fill_value=fill_value)
    assert isinstance(f, wp.array), "Field should be a Warp ndarray"

    f = f.numpy()
    assert np.allclose(f, fill_value), "Field not properly initialized with fill_value"


@pytest.fixture(autouse=True)
def setup_xlb_env():
    init_xlb_env(xlb.velocity_set.D2Q9)


if __name__ == "__main__":
    pytest.main()
