import pytest
import warp as wp
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.collision import BGK
from xlb.grid import grid
from xlb.default_config import DefaultConfig
from xlb.precision_policy import Precision

def init_xlb_warp_env(velocity_set):
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=velocity_set,
    )

@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape,omega",
    [
        (2, xlb.velocity_set.D2Q9, (100, 100), 0.6),
        (2, xlb.velocity_set.D2Q9, (100, 100), 1.0),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 0.6),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.0),
    ],
)
def test_bgk_collision_warp(dim, velocity_set, grid_shape, omega):
    init_xlb_warp_env(velocity_set)
    my_grid = grid(grid_shape)

    rho = my_grid.create_field(cardinality=1, init_val=1.0)
    u = my_grid.create_field(cardinality=dim, init_val=0.0)

    compute_macro = QuadraticEquilibrium()

    f_eq = my_grid.create_field(cardinality=DefaultConfig.velocity_set.q)
    f_eq = compute_macro(rho, u, f_eq)


    compute_collision = BGK(omega=omega)
    f_orig = my_grid.create_field(cardinality=DefaultConfig.velocity_set.q)

    f_out = my_grid.create_field(cardinality=DefaultConfig.velocity_set.q)
    f_out = compute_collision(f_orig, f_eq, f_out)

    f_eq = f_eq.numpy()
    f_out = f_out.numpy()
    f_orig = f_orig.numpy()

    assert np.allclose(f_out, f_orig - omega * (f_orig - f_eq), atol=1e-5)

if __name__ == "__main__":
    pytest.main()
