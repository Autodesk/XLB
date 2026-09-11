"""
Cross-backend consistency tests for the scalar-transport (thermal) stepper.

The Warp path runs one fused kernel instead of the chain of operators used by JAX, so it
is checked against the same analytical solutions and against the JAX result.
"""

import pytest
import numpy as np
import warp as wp
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.helper import omega_from_diffusivity
from xlb.operator.boundary_condition import ScalarDirichletBC, ScalarNeumannBC
from xlb.operator.macroscopic import ZeroMoment
from xlb.operator.stepper import ThermalStepper


def init_xlb_env(compute_backend, velocity_set=xlb.velocity_set.D2Q5):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP64FP64, compute_backend=compute_backend)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP64FP64,
        default_backend=compute_backend,
        velocity_set=vel_set,
    )


def run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps):
    for i in range(num_steps):
        g_0, g_1 = stepper(g_0, g_1, u, source, bc_mask, missing_mask, omega, i)
        g_0, g_1 = g_1, g_0
    return g_0


def temperature_of(g, compute_backend):
    """Recover the scalar and return it as a 2D numpy array."""
    if compute_backend == ComputeBackend.JAX:
        return np.asarray(ZeroMoment()(g))[0]

    phi = wp.zeros((1,) + g.shape[1:], dtype=g.dtype)
    phi = ZeroMoment()(g, phi)
    return phi.numpy()[0, :, :, 0]


def steady_conduction(compute_backend, nx=32, ny=20, num_steps=20000):
    init_xlb_env(compute_backend)
    grid = grid_factory((nx, ny), compute_backend=compute_backend)

    t_bot, t_top = 1.0, 0.0
    box = grid.bounding_box_indices()
    stepper = ThermalStepper(
        grid=grid,
        boundary_conditions=[
            ScalarDirichletBC(value=t_bot, indices=box["bottom"]),
            ScalarDirichletBC(value=t_top, indices=box["top"]),
        ],
    )

    omega = omega_from_diffusivity(1.0 / 6.0)
    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=0.5)
    u, source = stepper.prepare_aux_fields()

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps)
    return temperature_of(g_0, compute_backend)


def test_steady_conduction_warp_matches_analytical():
    nx, ny = 32, 20
    temperature = steady_conduction(ComputeBackend.WARP, nx, ny)[nx // 2]

    xi = (np.arange(ny) + 0.5) / ny
    expected = 1.0 + (0.0 - 1.0) * xi

    assert np.allclose(temperature, expected, atol=1e-3)


def test_steady_conduction_warp_matches_jax():
    nx, ny = 32, 20
    warp_result = steady_conduction(ComputeBackend.WARP, nx, ny)
    jax_result = steady_conduction(ComputeBackend.JAX, nx, ny)

    assert np.allclose(warp_result, jax_result, atol=1e-8)


def test_uniform_source_warp():
    """With a uniform source and no boundaries the scalar grows by exactly source per step."""
    init_xlb_env(ComputeBackend.WARP)
    grid = grid_factory((8, 8), compute_backend=ComputeBackend.WARP)

    stepper = ThermalStepper(grid=grid, boundary_conditions=[])

    omega = omega_from_diffusivity(1.0 / 6.0)
    source_value = 1e-3
    num_steps = 250

    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=0.0)
    u, source = stepper.prepare_aux_fields()
    source.fill_(source_value)

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps)

    assert np.allclose(temperature_of(g_0, ComputeBackend.WARP), source_value * num_steps, rtol=1e-9)


def test_adiabatic_walls_conserve_the_scalar_warp():
    """With zero-flux walls all around, the total amount of scalar must be conserved."""
    init_xlb_env(ComputeBackend.WARP)
    nx, ny = 16, 16
    grid = grid_factory((nx, ny), compute_backend=ComputeBackend.WARP)

    box = grid.bounding_box_indices()
    walls = [box["bottom"][i] + box["top"][i] + box["left"][i] + box["right"][i] for i in range(2)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    stepper = ThermalStepper(grid=grid, boundary_conditions=[ScalarNeumannBC(flux=0.0, indices=walls)])

    omega = omega_from_diffusivity(1.0 / 6.0)
    u, source = stepper.prepare_aux_fields()

    phi_init = np.zeros((1, nx, ny, 1))
    phi_init[0, nx // 2, ny // 2, 0] = 1.0
    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=wp.array(phi_init, dtype=wp.float64))

    initial_total = temperature_of(g_0, ComputeBackend.WARP).sum()
    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, 4000)
    temperature = temperature_of(g_0, ComputeBackend.WARP)

    assert np.isclose(temperature.sum(), initial_total, rtol=1e-9)
    assert np.allclose(temperature, initial_total / (nx * ny), atol=1e-6)


if __name__ == "__main__":
    pytest.main()
