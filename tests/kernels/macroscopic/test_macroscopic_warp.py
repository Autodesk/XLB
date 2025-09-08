import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
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
    "dim,velocity_set,grid_shape,rho,velocity",
    [
        (2, xlb.velocity_set.D2Q9, (100, 100), 1.0, 0.0),
        (2, xlb.velocity_set.D2Q9, (100, 100), 1.1, 1.0),
        (2, xlb.velocity_set.D2Q9, (100, 100), 1.1, 2.0),
        (2, xlb.velocity_set.D2Q9, (50, 50), 1.1, 2.0),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.0, 0.0),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.1, 1.0),  # TODO: Uncommenting will cause a Warp error. Needs investigation.
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.1, 2.0),  # TODO: Uncommenting will cause a Warp error. Needs investigation.
    ],
)
def test_macroscopic_warp(dim, velocity_set, grid_shape, rho, velocity):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)

    rho_field = my_grid.create_field(cardinality=1, fill_value=rho)
    velocity_field = my_grid.create_field(cardinality=dim, fill_value=velocity)

    f_eq = my_grid.create_field(cardinality=DefaultConfig.velocity_set.q)
    f_eq = QuadraticEquilibrium()(rho_field, velocity_field, f_eq)

    compute_macro = Macroscopic()
    rho_calc = my_grid.create_field(cardinality=1)
    u_calc = my_grid.create_field(cardinality=dim)

    rho_calc, u_calc = compute_macro(f_eq, rho_calc, u_calc)

    assert np.allclose(rho_calc.numpy(), rho), f"Computed density should be close to initialized density {rho}"
    assert np.allclose(u_calc.numpy(), velocity), f"Computed velocity should be close to initialized velocity {velocity}"


if __name__ == "__main__":
    pytest.main()
