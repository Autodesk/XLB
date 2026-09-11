"""
Validation tests for the scalar-transport (thermal) stepper against analytical solutions.

The cases below are the standard benchmarks of the heat equation and of passive scalar
advection-diffusion, run on the JAX backend.

Note on the wall-bounded cases: XLB flags every lattice link that leaves the domain as
missing, so a boundary node sitting where a wall row meets a periodic direction also gets
the wall treatment applied along that direction. This pollutes the two outermost columns,
so the profiles below are compared on the mid-domain column of a sufficiently wide grid.
"""

import pytest
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.helper import omega_from_diffusivity, diffusivity_from_omega
from xlb.operator.boundary_condition import ScalarDirichletBC, ScalarNeumannBC
from xlb.operator.macroscopic import ZeroMoment
from xlb.operator.stepper import ThermalStepper


def init_xlb_env(velocity_set=xlb.velocity_set.D2Q5):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP64FP64, compute_backend=ComputeBackend.JAX)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP64FP64,
        default_backend=ComputeBackend.JAX,
        velocity_set=vel_set,
    )


def run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps):
    for i in range(num_steps):
        g_0, g_1 = stepper(g_0, g_1, u, source, bc_mask, missing_mask, omega, i)
        g_0, g_1 = g_1, g_0
    return g_0


def temperature_of(g):
    return np.asarray(ZeroMoment()(g))[0]


def test_omega_diffusivity_roundtrip():
    for alpha in [0.01, 1.0 / 6.0, 0.5]:
        omega = omega_from_diffusivity(alpha)
        assert np.isclose(diffusivity_from_omega(omega), alpha)


@pytest.mark.parametrize("velocity_set", [xlb.velocity_set.D2Q5, xlb.velocity_set.D2Q9])
def test_pure_diffusion_recovers_the_heat_kernel(velocity_set):
    """A blob spreading in a periodic domain must follow the analytical heat kernel.

    This is the cleanest check of the omega-to-diffusivity mapping since no boundary
    conditions are involved at all.
    """
    init_xlb_env(velocity_set)
    nx, ny = 100, 3
    grid = grid_factory((nx, ny))

    stepper = ThermalStepper(grid=grid, boundary_conditions=[])

    alpha = 1.0 / 6.0
    omega = omega_from_diffusivity(alpha)
    num_steps = 150

    x = np.arange(nx)
    sigma, x0 = 4.0, nx // 2
    phi_init = np.exp(-0.5 * ((x - x0) / sigma) ** 2)[None, :, None] * np.ones((1, nx, ny))

    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=phi_init)
    u, source = stepper.prepare_aux_fields()

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps)
    temperature = temperature_of(g_0)[:, 1]

    sigma_t = np.sqrt(sigma**2 + 2.0 * alpha * num_steps)
    expected = sigma / sigma_t * np.exp(-0.5 * ((x - x0) / sigma_t) ** 2)

    assert np.allclose(temperature, expected, atol=1e-3)
    assert np.isclose(temperature.sum(), phi_init[0, :, 1].sum(), rtol=1e-9)


def test_uniform_source_heats_at_the_prescribed_rate():
    """With a uniform source and no boundaries the scalar grows by exactly source per step."""
    init_xlb_env()
    grid = grid_factory((8, 8))

    stepper = ThermalStepper(grid=grid, boundary_conditions=[])

    omega = omega_from_diffusivity(1.0 / 6.0)
    source_value = 1e-3
    num_steps = 250

    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=0.0)
    u, source = stepper.prepare_aux_fields()
    source = source + source_value

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps)

    assert np.allclose(temperature_of(g_0), source_value * num_steps, rtol=1e-9)


@pytest.mark.parametrize("velocity_set", [xlb.velocity_set.D2Q5, xlb.velocity_set.D2Q9])
def test_steady_conduction_linear_profile(velocity_set):
    """Steady conduction between two walls at fixed temperature gives a linear profile.

    This also pins down the effective wall location of the anti-bounce-back condition,
    which must be exactly halfway between the boundary node and its solid neighbour.
    """
    init_xlb_env(velocity_set)
    nx, ny = 32, 20
    grid = grid_factory((nx, ny))

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

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, 20000)
    temperature = temperature_of(g_0)[nx // 2]

    # The halfway walls sit at y = -1/2 and y = ny - 1/2, so the wall-to-wall distance is ny
    xi = (np.arange(ny) + 0.5) / ny
    expected = t_bot + (t_top - t_bot) * xi

    assert np.allclose(temperature, expected, atol=1e-3)


def test_steady_conduction_with_volumetric_source():
    """Conduction with a uniform source between two cold walls gives a parabolic profile."""
    init_xlb_env()
    nx, ny = 32, 20
    grid = grid_factory((nx, ny))

    box = grid.bounding_box_indices()
    stepper = ThermalStepper(
        grid=grid,
        boundary_conditions=[
            ScalarDirichletBC(value=0.0, indices=box["bottom"]),
            ScalarDirichletBC(value=0.0, indices=box["top"]),
        ],
    )

    alpha = 1.0 / 6.0
    omega = omega_from_diffusivity(alpha)
    source_value = 1e-4

    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=0.0)
    u, source = stepper.prepare_aux_fields()
    source = source + source_value

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, 20000)
    temperature = temperature_of(g_0)[nx // 2]

    # alpha * T'' + S = 0 with T = 0 on both walls gives T = S * L^2 / (2 alpha) * xi * (1 - xi)
    xi = (np.arange(ny) + 0.5) / ny
    expected = source_value * ny**2 / (2.0 * alpha) * xi * (1.0 - xi)

    # The anti-bounce-back wall carries a residual defect of order source (see the note in
    # ScalarDirichletBC), which is second order in dx under diffusive scaling. It is
    # therefore measured against the peak of the profile rather than pointwise.
    assert np.abs(temperature - expected).max() < 5e-3 * expected.max()


def test_prescribed_wall_flux():
    """A prescribed flux on one wall drives the analytical temperature gradient."""
    init_xlb_env()
    nx, ny = 256, 20
    grid = grid_factory((nx, ny))

    box = grid.bounding_box_indices()
    flux = 1e-3
    stepper = ThermalStepper(
        grid=grid,
        boundary_conditions=[
            ScalarNeumannBC(flux=flux, indices=box["bottom"]),
            ScalarDirichletBC(value=0.0, indices=box["top"]),
        ],
    )

    alpha = 1.0 / 6.0
    omega = omega_from_diffusivity(alpha)
    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=0.0)
    u, source = stepper.prepare_aux_fields()

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, 40000)
    temperature = temperature_of(g_0)[nx // 2]

    # At steady state alpha * dT/dy = -flux throughout the slab, with T = 0 imposed halfway
    # past the last node
    expected = flux / alpha * (ny - 0.5 - np.arange(ny))

    assert np.allclose(np.diff(temperature), -flux / alpha, rtol=1e-3)
    assert np.allclose(temperature, expected, rtol=1e-3)


def test_adiabatic_walls_conserve_the_scalar():
    """With zero-flux walls all around, the total amount of scalar must be conserved."""
    init_xlb_env()
    nx, ny = 16, 16
    grid = grid_factory((nx, ny))

    box = grid.bounding_box_indices()
    walls = [box["bottom"][i] + box["top"][i] + box["left"][i] + box["right"][i] for i in range(2)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    stepper = ThermalStepper(grid=grid, boundary_conditions=[ScalarNeumannBC(flux=0.0, indices=walls)])

    omega = omega_from_diffusivity(1.0 / 6.0)
    u, source = stepper.prepare_aux_fields()

    # Start from a hot spot in the middle of an otherwise cold domain
    phi_init = np.zeros((1, nx, ny))
    phi_init[0, nx // 2, ny // 2] = 1.0
    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=phi_init)

    initial_total = temperature_of(g_0).sum()
    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, 4000)
    temperature = temperature_of(g_0)

    assert np.isclose(temperature.sum(), initial_total, rtol=1e-9)

    # Diffusion with insulated walls must relax towards the well-mixed state
    assert np.allclose(temperature, initial_total / (nx * ny), atol=1e-6)


def test_advection_of_a_gaussian_blob():
    """A blob advected by a uniform periodic flow translates at the imposed velocity."""
    init_xlb_env()
    nx, ny = 64, 3
    grid = grid_factory((nx, ny))

    # No boundary conditions: streaming is periodic in both directions
    stepper = ThermalStepper(grid=grid, boundary_conditions=[])

    alpha = 1e-2
    omega = omega_from_diffusivity(alpha)
    u_x = 0.05
    num_steps = 400

    x = np.arange(nx)
    sigma, x0 = 5.0, nx // 4
    phi_init = np.exp(-0.5 * ((x - x0) / sigma) ** 2)[None, :, None] * np.ones((1, nx, ny))

    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=phi_init)
    u, source = stepper.prepare_aux_fields()
    u = u.at[0].set(u_x)

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps)
    temperature = temperature_of(g_0)[:, 1]

    # The scalar is conserved by advection-diffusion on a periodic domain
    assert np.isclose(temperature.sum(), phi_init[0, :, 1].sum(), rtol=1e-9)

    # The blob translates at u_x and spreads as a Gaussian, so compare its first two
    # moments. The periodic coordinate is unwrapped around the expected blob position so
    # that its tails are not folded to the opposite side of the domain.
    x_expected = x0 + u_x * num_steps
    x_unwrapped = (x - x_expected + nx / 2) % nx - nx / 2
    weights = temperature / temperature.sum()
    centroid = np.sum(weights * x_unwrapped)
    variance = np.sum(weights * (x_unwrapped - centroid) ** 2)

    assert np.isclose(centroid, 0.0, atol=5e-2)
    assert np.isclose(variance, sigma**2 + 2.0 * alpha * num_steps, rtol=1e-2)


def test_scalar_stencil_may_differ_from_the_default_one():
    """The scalar may run on D2Q5 while the global default stays on the flow stencil.

    This is the configuration of every coupled flow-thermal run, so the stepper has to
    re-bind its boundary conditions to its own stencil rather than the default one.
    """
    # The global default is the flow stencil, as it would be when coupling to the flow
    init_xlb_env(xlb.velocity_set.D2Q9)

    nx, ny = 32, 20
    grid = grid_factory((nx, ny))
    thermal_velocity_set = xlb.velocity_set.D2Q5(precision_policy=xlb.PrecisionPolicy.FP64FP64, compute_backend=ComputeBackend.JAX)

    box = grid.bounding_box_indices()
    stepper = ThermalStepper(
        grid=grid,
        boundary_conditions=[
            ScalarDirichletBC(value=1.0, indices=box["bottom"]),
            ScalarDirichletBC(value=0.0, indices=box["top"]),
        ],
        velocity_set=thermal_velocity_set,
    )

    assert stepper.velocity_set.q == 5
    assert all(bc.velocity_set.q == 5 for bc in stepper.boundary_conditions)

    omega = omega_from_diffusivity(1.0 / 6.0)
    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=0.5)
    u, source = stepper.prepare_aux_fields()

    assert g_0.shape[0] == 5

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, 20000)
    temperature = temperature_of(g_0)[nx // 2]

    # Steady conduction between the two walls is linear across the slab
    xi = (np.arange(ny) + 0.5) / ny
    assert np.allclose(temperature, 1.0 - xi, atol=1e-3)


def test_velocity_sets_report_the_speed_of_sound_of_their_stencil():
    """cs^2 is a property of the stencil and must be derived from it, not assumed.

    Every stencil XLB uses for Navier-Stokes has cs^2 = 1/3, and so does the D2Q5 scalar
    stencil, but the minimal 3D scalar stencil D3Q7 has cs^2 = 1/4. Hard-coding 1/3 would
    silently give the wrong diffusivity on D3Q7.
    """
    for velocity_set, expected_cs2 in [
        (xlb.velocity_set.D2Q5, 1.0 / 3.0),
        (xlb.velocity_set.D2Q9, 1.0 / 3.0),
        (xlb.velocity_set.D3Q7, 1.0 / 4.0),
        (xlb.velocity_set.D3Q19, 1.0 / 3.0),
        (xlb.velocity_set.D3Q27, 1.0 / 3.0),
    ]:
        vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP64FP64, compute_backend=ComputeBackend.JAX)

        # cs^2 follows from sum_i w_i c_ia c_ib = cs^2 delta_ab
        c, w = np.array(vel_set._c), np.array(vel_set._w)
        second_moment = np.einsum("i,ai,bi->ab", w, c, c)

        assert np.isclose(w.sum(), 1.0)
        assert np.allclose(second_moment, expected_cs2 * np.eye(vel_set.d))
        assert float(vel_set.cs2) == expected_cs2
        assert float(vel_set.inv_cs2) == 1.0 / expected_cs2


def test_omega_diffusivity_roundtrip_accepts_a_velocity_set():
    """Passing the velocity set is the safe way to get cs^2 right on any stencil."""
    for velocity_set in [xlb.velocity_set.D2Q5, xlb.velocity_set.D3Q7, xlb.velocity_set.D3Q19]:
        vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP64FP64, compute_backend=ComputeBackend.JAX)
        for alpha in [0.01, 1.0 / 6.0, 0.5]:
            omega = omega_from_diffusivity(alpha, cs2=vel_set)
            assert np.isclose(diffusivity_from_omega(omega, cs2=vel_set), alpha)


def test_d3q7_pure_diffusion_recovers_the_heat_kernel():
    """D3Q7 must reproduce the analytical heat kernel, as D2Q5 does in 2D.

    The blob spreads as ``sigma^2(t) = sigma_0^2 + 2 * alpha * t``, so measuring the
    variance directly tests the omega-to-diffusivity mapping, which is what makes D3Q7
    easy to get wrong: its cs^2 is 1/4 rather than the 1/3 of every other XLB stencil.
    """
    init_xlb_env(xlb.velocity_set.D3Q7)
    nx, ny, nz = 100, 3, 3
    grid = grid_factory((nx, ny, nz))

    stepper = ThermalStepper(grid=grid, boundary_conditions=[])
    assert stepper.velocity_set.q == 7

    alpha = 1.0 / 60.0
    # D3Q7 has cs^2 = 1/4, so the stencil's own value must be used here
    omega = omega_from_diffusivity(alpha, cs2=stepper.velocity_set)
    num_steps = 400

    x = np.arange(nx)
    sigma, x0 = 6.0, nx // 2
    phi_init = np.exp(-0.5 * ((x - x0) / sigma) ** 2)[None, :, None, None] * np.ones((1, nx, ny, nz))

    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=phi_init)
    u, source = stepper.prepare_aux_fields()

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps)
    temperature = temperature_of(g_0)[:, ny // 2, nz // 2]

    # The variance must grow at exactly 2 * alpha per step. D3Q7 is a smaller stencil than
    # D2Q5 and carries slightly more higher-order error, hence the 1% tolerance.
    weights = temperature / temperature.sum()
    mean = (weights * x).sum()
    variance = (weights * (x - mean) ** 2).sum()
    assert np.isclose(variance, sigma**2 + 2.0 * alpha * num_steps, rtol=0.01)

    # The profile itself must stay Gaussian, and the scalar must be conserved exactly
    sigma_t = np.sqrt(sigma**2 + 2.0 * alpha * num_steps)
    expected = sigma / sigma_t * np.exp(-0.5 * ((x - x0) / sigma_t) ** 2)
    assert np.allclose(temperature, expected, atol=5e-3 * expected.max())
    assert np.isclose(temperature.sum(), phi_init[0, :, ny // 2, nz // 2].sum(), rtol=1e-9)


def test_d3q7_diffusion_is_isotropic():
    """A point source on D3Q7 must spread identically along x, y and z."""
    init_xlb_env(xlb.velocity_set.D3Q7)
    n = 31
    grid = grid_factory((n, n, n))

    stepper = ThermalStepper(grid=grid, boundary_conditions=[])
    omega = omega_from_diffusivity(1.0 / 6.0, cs2=stepper.velocity_set)

    phi_init = np.zeros((1, n, n, n))
    phi_init[0, n // 2, n // 2, n // 2] = 1.0

    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=phi_init)
    u, source = stepper.prepare_aux_fields()

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, 120)
    temperature = temperature_of(g_0)

    axis = np.arange(n)
    variances = []
    for d in range(3):
        profile = temperature.sum(axis=tuple(i for i in range(3) if i != d))
        weights = profile / profile.sum()
        mean = (weights * axis).sum()
        variances.append((weights * (axis - mean) ** 2).sum())

    assert np.allclose(variances, variances[0], rtol=1e-10)


def test_d3q7_advects_a_blob_at_the_prescribed_velocity():
    """With a uniform velocity the blob must translate at exactly u_x while diffusing."""
    init_xlb_env(xlb.velocity_set.D3Q7)
    nx, ny, nz = 120, 3, 3
    grid = grid_factory((nx, ny, nz))

    stepper = ThermalStepper(grid=grid, boundary_conditions=[])
    alpha = 1.0 / 60.0
    omega = omega_from_diffusivity(alpha, cs2=stepper.velocity_set)
    u_x, num_steps = 0.05, 400

    x = np.arange(nx)
    sigma, x0 = 5.0, nx // 4
    phi_init = np.exp(-0.5 * ((x - x0) / sigma) ** 2)[None, :, None, None] * np.ones((1, nx, ny, nz))

    g_0, g_1, bc_mask, missing_mask = stepper.prepare_fields(phi_init=phi_init)
    u, source = stepper.prepare_aux_fields()
    u = u.at[0].set(u_x)

    g_0 = run_thermal(stepper, g_0, g_1, u, source, bc_mask, missing_mask, omega, num_steps)
    temperature = temperature_of(g_0)[:, ny // 2, nz // 2]

    # Unwrap around the expected centre so the periodic tails are not folded across
    expected_centre = x0 + u_x * num_steps
    x_unwrapped = (x - expected_centre + nx / 2) % nx - nx / 2
    weights = temperature / temperature.sum()
    centroid = (weights * x_unwrapped).sum() + expected_centre
    variance = (weights * (x_unwrapped - (weights * x_unwrapped).sum()) ** 2).sum()

    assert np.isclose(centroid, expected_centre, atol=0.1)
    assert np.isclose(variance, sigma**2 + 2.0 * alpha * num_steps, rtol=0.02)


if __name__ == "__main__":
    pytest.main()
