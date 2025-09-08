import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.grid import grid_factory
from xlb import DefaultConfig


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.JAX)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.JAX,
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
def test_quadratic_equilibrium_jax(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)

    rho = my_grid.create_field(cardinality=1, fill_value=1.0)
    u = my_grid.create_field(cardinality=dim, fill_value=0.0)

    # Compute equilibrium
    compute_macro = QuadraticEquilibrium()
    f_eq = compute_macro(rho, u)

    # Test sum of f_eq across cardinality at each point
    sum_f_eq = np.sum(f_eq, axis=0)
    assert np.allclose(sum_f_eq, 1.0), "Sum of f_eq should be 1.0 across all directions at each grid point"

    # Test that each direction matches the expected weights
    weights = DefaultConfig.velocity_set.w
    for i, weight in enumerate(weights):
        assert np.allclose(f_eq[i, ...], weight), f"Direction {i} in f_eq does not match the expected weight"


if __name__ == "__main__":
    pytest.main()
