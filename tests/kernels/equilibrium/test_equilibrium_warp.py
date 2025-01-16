import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.grid import grid_factory
from xlb import DefaultConfig


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=vel_set,
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape",
    [
        (2, xlb.velocity_set.D2Q9, (50, 50)),
        (2, xlb.velocity_set.D2Q9, (100, 100)),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50)),
        (3, xlb.velocity_set.D3Q19, (100, 100, 100)),
        (3, xlb.velocity_set.D3Q27, (50, 50, 50)),
        (3, xlb.velocity_set.D3Q27, (100, 100, 100)),
    ],
)
def test_quadratic_equilibrium_warp(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)

    rho = my_grid.create_field(cardinality=1, fill_value=1.0)
    u = my_grid.create_field(cardinality=dim, fill_value=0.0)

    f_eq = my_grid.create_field(cardinality=DefaultConfig.velocity_set.q)

    compute_macro = QuadraticEquilibrium()
    f_eq = compute_macro(rho, u, f_eq)

    f_eq_np = f_eq.numpy()

    sum_f_eq = np.sum(f_eq_np, axis=0)
    assert np.allclose(sum_f_eq, 1.0), "Sum of f_eq should be 1.0 across all directions at each grid point"

    weights = DefaultConfig.velocity_set.w
    for i, weight in enumerate(weights):
        assert np.allclose(f_eq_np[i, ...], weight), f"Direction {i} in f_eq does not match the expected weight"


# @pytest.fixture(autouse=True)
# def setup_xlb_env(request):
#     dim, velocity_set, grid_shape = request.param
#     init_xlb_env(velocity_set)

if __name__ == "__main__":
    pytest.main()
