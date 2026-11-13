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


def test_repeated_stepper_construction_warp():
    """Steppers built one after the other in a process must not reuse each other's kernels.

    Every boundary condition takes a fresh registry id, so the second stepper's BCs have
    different ids from the first's. Before apply_bc was typed, Warp hashed both steppers'
    kernels identically and ran the first, stale binary, which ignored the new BCs.
    """
    nx, ny = 32, 20
    first = steady_conduction(ComputeBackend.WARP, nx, ny, num_steps=3000)
    second = steady_conduction(ComputeBackend.WARP, nx, ny, num_steps=3000)

    assert first[nx // 2].max() > 0.9
    assert np.array_equal(first, second)


def test_repeated_nse_stepper_construction_warp():
    """Same regression as above, for the Navier-Stokes stepper (lid-driven cavity)."""
    from xlb.operator.boundary_condition import EquilibriumBC, HalfwayBounceBackBC
    from xlb.operator.macroscopic import Macroscopic
    from xlb.operator.stepper import IncompressibleNavierStokesStepper

    def lid_driven_cavity():
        precision = xlb.PrecisionPolicy.FP64FP64
        vel_set = xlb.velocity_set.D2Q9(precision_policy=precision, compute_backend=ComputeBackend.WARP)
        xlb.init(velocity_set=vel_set, default_backend=ComputeBackend.WARP, default_precision_policy=precision)
        grid = grid_factory((16, 16), compute_backend=ComputeBackend.WARP)
        box = grid.bounding_box_indices()
        box_no_edge = grid.bounding_box_indices(remove_edges=True)
        walls = np.unique(np.array([box["bottom"][i] + box["left"][i] + box["right"][i] for i in range(2)]), axis=-1).tolist()
        stepper = IncompressibleNavierStokesStepper(
            grid=grid,
            boundary_conditions=[HalfwayBounceBackBC(indices=walls), EquilibriumBC(rho=1.0, u=(0.05, 0.0), indices=box_no_edge["top"])],
            collision_type="BGK",
        )
        f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
        for i in range(200):
            f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, 1.0, i)
            f_0, f_1 = f_1, f_0
        rho = grid.create_field(cardinality=1, dtype=precision.store_precision)
        u = grid.create_field(cardinality=2, dtype=precision.store_precision)
        rho, u = Macroscopic()(f_0, rho, u)
        return u.numpy()

    first = lid_driven_cavity()
    second = lid_driven_cavity()

    assert np.abs(first).max() > 0.04
    assert np.array_equal(first, second)


def advected_channel(compute_backend, u_conv, nx=40, ny=12, u_x=0.05, num_steps=800):
    """Scalar advected through a channel at a uniform velocity, leaving through an outlet.

    The default number of steps stops the run while the front is crossing the outlet,
    which is when the convective and extrapolation variants differ the most; once the
    outlet region is uniform the two coincide.
    """
    from xlb.operator.boundary_condition import ScalarOutflowBC

    init_xlb_env(compute_backend)
    grid = grid_factory((nx, ny), compute_backend=compute_backend)
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)

    stepper = ThermalStepper(
        grid=grid,
        boundary_conditions=[
            ScalarNeumannBC(flux=0.0, indices=[box_no_edge["bottom"][i] + box_no_edge["top"][i] for i in range(2)]),
            ScalarDirichletBC(value=1.0, indices=box["left"]),
            ScalarOutflowBC(u_conv=u_conv, flow_direction=0, indices=box["right"]),
        ],
    )

    omega = omega_from_diffusivity(0.02)
    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=0.0)
    u, source = stepper.prepare_aux_fields()
    if compute_backend == ComputeBackend.WARP:
        u_np = np.zeros(u.shape)
        u_np[0] = u_x
        wp.copy(u, wp.array(u_np, dtype=wp.float64))
    else:
        u = u.at[0].set(u_x)

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps)
    return temperature_of(g_0, compute_backend)


@pytest.mark.parametrize("u_conv", [0.0, 0.05])
def test_scalar_outflow_warp_matches_jax(u_conv):
    warp_result = advected_channel(ComputeBackend.WARP, u_conv)
    jax_result = advected_channel(ComputeBackend.JAX, u_conv)

    # The front is crossing the outlet, which neither reflects nor amplifies it
    assert jax_result[-1].min() > 0.5
    assert jax_result.max() < 1.0 + 1e-6

    assert np.allclose(warp_result, jax_result, atol=1e-10)


def test_convective_outflow_differs_from_extrapolation_warp():
    """Guards against the convective branch silently falling back to plain extrapolation."""
    plain = advected_channel(ComputeBackend.WARP, 0.0)
    convective = advected_channel(ComputeBackend.WARP, 0.05)
    assert np.abs(convective - plain).max() > 1e-4


if __name__ == "__main__":
    pytest.main()
