import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.grid import grid_factory


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.JAX)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.JAX,
        velocity_set=vel_set,
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape,rho,velocity",
    [
        (2, xlb.velocity_set.D2Q9, (100, 100), 1.0, 0.0),
        (2, xlb.velocity_set.D2Q9, (100, 100), 1.1, 1.0),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.0, 0.0),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.1, 1.0),
        (3, xlb.velocity_set.D3Q27, (50, 50, 50), 1.0, 0.0),
        (3, xlb.velocity_set.D3Q27, (50, 50, 50), 1.1, 1.0),
    ],
)
def test_macroscopic_jax(dim, velocity_set, grid_shape, rho, velocity):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)

    rho_field = my_grid.create_field(cardinality=1, fill_value=rho)
    velocity_field = my_grid.create_field(cardinality=dim, fill_value=velocity)

    # Compute equilibrium
    f_eq = QuadraticEquilibrium()(rho_field, velocity_field)

    compute_macro = Macroscopic()

    rho_calc, u_calc = compute_macro(f_eq)

    # Test sum of f_eq which should be 1.0 for rho and 0.0 for u
    assert np.allclose(rho_calc, rho), "Sum of f_eq should be {rho} for rho"
    assert np.allclose(u_calc, velocity), "Sum of f_eq should be {velocity} for u"


if __name__ == "__main__":
    pytest.main()
