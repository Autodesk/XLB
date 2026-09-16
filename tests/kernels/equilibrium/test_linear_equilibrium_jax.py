import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import LinearEquilibrium
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
        (2, xlb.velocity_set.D2Q5, (20, 20)),
        (2, xlb.velocity_set.D2Q9, (20, 20)),
        (3, xlb.velocity_set.D3Q19, (20, 20, 20)),
        (3, xlb.velocity_set.D3Q27, (20, 20, 20)),
    ],
)
def test_linear_equilibrium_at_rest(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)

    phi = my_grid.create_field(cardinality=1, fill_value=2.5)
    u = my_grid.create_field(cardinality=dim, fill_value=0.0)

    g_eq = LinearEquilibrium()(phi, u)

    # The zeroth moment must return the scalar itself
    assert np.allclose(np.sum(g_eq, axis=0), 2.5), "Sum of g_eq should recover the scalar"

    # At rest the equilibrium reduces to the lattice weights times the scalar
    weights = DefaultConfig.velocity_set.w
    for i, weight in enumerate(weights):
        assert np.allclose(g_eq[i, ...], 2.5 * weight), f"Direction {i} in g_eq does not match the expected weight"


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape",
    [
        (2, xlb.velocity_set.D2Q5, (20, 20)),
        (2, xlb.velocity_set.D2Q9, (20, 20)),
        (3, xlb.velocity_set.D3Q19, (20, 20, 20)),
    ],
)
def test_linear_equilibrium_moments(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)

    phi_value = 1.7
    u_value = 0.05
    phi = my_grid.create_field(cardinality=1, fill_value=phi_value)
    u = my_grid.create_field(cardinality=dim, fill_value=u_value)

    g_eq = LinearEquilibrium()(phi, u)
    c = np.array(DefaultConfig.velocity_set.c)

    # Zeroth moment: sum(g_eq) = phi
    assert np.allclose(np.sum(g_eq, axis=0), phi_value, atol=1e-5)

    # First moment: sum(c * g_eq) = phi * u, i.e. the advective flux
    for d in range(dim):
        flux = np.einsum("l,l...->...", c[d], np.asarray(g_eq))
        assert np.allclose(flux, phi_value * u_value, atol=1e-5), f"First moment along {d} does not recover the advective flux"

    # The linear equilibrium is symmetric in the sense g_eq[l] + g_eq[opp[l]] = 2 * w[l] * phi,
    # which is what makes the anti-bounce-back Dirichlet condition exact.
    opp = np.array(DefaultConfig.velocity_set.opp_indices)
    w = np.array(DefaultConfig.velocity_set.w)
    for l in range(len(w)):
        pair_sum = np.asarray(g_eq)[l] + np.asarray(g_eq)[opp[l]]
        assert np.allclose(pair_sum, 2.0 * w[l] * phi_value, atol=1e-5)


if __name__ == "__main__":
    pytest.main()
