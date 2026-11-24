"""
Tests for the hybrid thermal LBM solver: lattice difference stencils
(grad_x, laplacian_x), Runge-Kutta diffusion against the analytic decay of a
sinusoid, and the Dirichlet/Neumann thermal boundary conditions.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.core.boundary_conditions import DirichletTemperature, NeumannTemperature
from jax_lab.core.shanchen.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.core.models import BGKSim
from jax_lab.core.shanchen.multiphase import MultiphaseBGK
from jax_lab.core.shanchen.thermal import MultiphaseThermal
from jax_lab.core.thermal import Thermal

PRECISION = "f64/f64"
NX = NY = NZ = 32


@pytest.fixture(autouse=True)
def enable_x64():
    import jax

    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def make_thermal(dim, K=0.05, c_v=1.0):
    lattice = LatticeD2Q9(PRECISION) if dim == 2 else LatticeD3Q19(PRECISION)
    fluid = BGKSim(
        lattice=lattice,
        omega=1.0,
        nx=NX,
        ny=NY,
        nz=0 if dim == 2 else NZ,
        precision=PRECISION,
        io_rate=0,
        print_info_rate=0,
    )
    return Thermal(fluid_solver=fluid, specific_heat=c_v, thermal_conductivity=K)


def sinusoid(dim):
    k = 2.0 * np.pi / NX
    x = np.arange(NX)
    if dim == 2:
        field = np.sin(k * x)[:, None] * np.ones((1, NY))
        return jnp.array(field[..., None]), k
    field = np.sin(k * x)[:, None, None] * np.ones((1, NY, NZ))
    return jnp.array(field[..., None]), k


@pytest.mark.parametrize("dim", [2, 3])
def test_grad_x_of_sinusoid(dim):
    solver = make_thermal(dim)
    T, k = sinusoid(dim)
    grad = solver.grad_x(T)

    x = np.arange(NX)
    expected_x = k * np.cos(k * x)
    # The lattice stencil applied to sin(kx) yields sin(k) / k * expected (second order accurate)
    scale = np.sin(k) / k
    got_x = np.array(grad[..., 0]).mean(axis=tuple(range(1, dim)))
    assert np.allclose(got_x, scale * expected_x, atol=1e-8)
    # No variation along the other axes
    for axis in range(1, dim):
        assert np.allclose(np.array(grad[..., axis]), 0.0, atol=1e-8)


@pytest.mark.parametrize("dim", [2, 3])
def test_laplacian_x_of_sinusoid(dim):
    solver = make_thermal(dim)
    T, k = sinusoid(dim)
    lap = solver.laplacian_x(T)

    # The discrete laplacian of sin(kx) is -k_eff^2 sin(kx) with
    # k_eff^2 = 2(1 - cos k) (central difference symbol)
    k_eff2 = 2.0 * (1.0 - np.cos(k))
    assert np.allclose(np.array(lap), -k_eff2 * np.array(T), atol=1e-8)


def test_rk4_diffusion_decay_2d():
    K = 0.05
    solver = make_thermal(2, K=K)
    T0, k = sinusoid(2)
    T_mean = 1.0
    T = T_mean + T0

    rho = jnp.ones((NX, NY, 1))
    u = jnp.zeros((NX, NY, 2))

    n_steps = 200
    for _ in range(n_steps):
        k1 = solver.RHS(T, rho, u)
        k2 = solver.RHS(T + 0.5 * k1, rho, u)
        k3 = solver.RHS(T + 0.5 * k2, rho, u)
        k4 = solver.RHS(T + k3, rho, u)
        T = T + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # Analytic decay with the discrete wavenumber of the stencil
    k_eff2 = 2.0 * (1.0 - np.cos(k))
    expected = T_mean + np.exp(-K * k_eff2 * n_steps) * np.array(T0)
    assert np.allclose(np.array(T), expected, rtol=1e-4, atol=1e-6)


def test_rk4_applies_dirichlet_bc_at_intermediate_stages():
    solver = make_thermal(2, K=0.2)
    left = solver.fluid_solver.bounding_box_indices["left"]
    right = solver.fluid_solver.bounding_box_indices["right"]
    solver.thermal_BCs = [
        DirichletTemperature(tuple(left.T), prescribed=2.0),
        DirichletTemperature(tuple(right.T), prescribed=1.0),
    ]
    T = jnp.ones((NX, NY, 1))
    rho = jnp.ones((NX, NY, 1))
    u = jnp.zeros((NX, NY, 2))

    advanced = solver._advance_temperature(T, 0, rho, u)

    assert np.allclose(np.array(advanced[0, :, 0]), 2.0)
    assert np.all(np.array(advanced[1, :, 0]) > 1.0)
    assert np.allclose(np.array(advanced[-1, :, 0]), 1.0)


def test_step_conserves_uniform_temperature():
    solver = make_thermal(2)
    T = solver.assign_fields_sharded()
    state = solver.fluid_solver.state_from_populations(solver.fluid_solver.assign_fields_sharded())
    T_new, state, _ = solver.step(T, state, 0)
    assert np.allclose(np.array(T_new), 1.0, atol=1e-12)


CRITICAL_TEMPERATURE = 0.5714285714


def make_multiphase_thermal(temperature_field_type="thermal", K=0.05):
    eos_kwargs = {"a": [9.0 / 49.0], "b": [2.0 / 21.0], "R": [1.0], "temperature_field_type": temperature_field_type}
    if temperature_field_type == "isothermal":
        eos_kwargs["T"] = 0.8 * CRITICAL_TEMPERATURE
    fluid = MultiphaseBGK(
        lattice=LatticeD2Q9(PRECISION),
        omega=[1.0],
        nx=NX,
        ny=NY,
        nz=0,
        n_components=1,
        g_kkprime=-np.ones((1, 1)),
        lambda_=[1.0],
        EOS=VanderWaals(**eos_kwargs),
        precision=PRECISION,
        io_rate=0,
        print_info_rate=0,
        checkpoint_rate=0,
    )
    return MultiphaseThermal(fluid_solver=fluid, specific_heat=1.0, thermal_conductivity=K)


def test_multiphase_thermal_requires_thermal_eos():
    with pytest.raises(ValueError, match="thermal"):
        make_multiphase_thermal(temperature_field_type="isothermal")


def test_divergence_x_of_sinusoid():
    solver = make_multiphase_thermal()
    ux, k = sinusoid(2)
    u = jnp.concatenate([ux, jnp.zeros_like(ux)], axis=-1)
    div_u = solver.divergence_x(u)

    x = np.arange(NX)
    # Same second order discrete symbol as the gradient stencil
    expected = (np.sin(k) / k) * k * np.cos(k * x)
    got = np.array(div_u[..., 0]).mean(axis=1)
    assert np.allclose(got, expected, atol=1e-8)


def test_multiphase_thermal_uniform_state_is_stationary():
    solver = make_multiphase_thermal()
    T_uniform = 0.8 * CRITICAL_TEMPERATURE
    solver.initialize_temperature_field = lambda: T_uniform

    T = solver.assign_fields_sharded()
    state_tree = solver.fluid_solver.state_from_populations(solver.fluid_solver.assign_fields_sharded())
    for timestep in range(3):
        T, state_tree, _ = solver.step(T, state_tree, timestep)
    f_tree = solver.fluid_solver.poststreaming(state_tree)

    # A uniform density and temperature state is a mechanical and thermal equilibrium
    assert np.allclose(np.array(T), T_uniform, atol=1e-10)
    rho_tree, _ = solver.fluid_solver.update_macroscopic(f_tree)
    assert np.allclose(np.array(rho_tree[0]), 1.0, atol=1e-10)


def test_multiphase_step_uses_synchronized_macroscopic_fields():
    solver = make_multiphase_thermal(K=0.01)
    x = np.arange(NX)[:, None, None]
    rho = jnp.asarray(0.95 + 0.02 * np.cos(2.0 * np.pi * x / NX)) * jnp.ones((1, NY, 1))
    velocity = jnp.zeros((NX, NY, 2))
    f_tree = solver.fluid_solver.equilibrium([rho], [velocity])
    T = jnp.full((NX, NY, 1), 0.8 * CRITICAL_TEMPERATURE)

    rho_tree, _ = solver.fluid_solver.update_macroscopic(f_tree)
    u_tree = solver.fluid_solver.macroscopic_velocity(f_tree, rho_tree, T=T)
    expected_temperature = solver._advance_temperature(T, 0, rho_tree, u_tree)

    temperature, _, _ = solver.step(T, solver.fluid_solver.state_from_populations(f_tree), 0)

    assert np.allclose(np.array(temperature), np.array(expected_temperature), rtol=1e-12, atol=1e-12)


def test_dirichlet_and_neumann_bc():
    solver = make_thermal(2)
    bottom = solver.fluid_solver.bounding_box_indices["bottom"]
    left = solver.fluid_solver.bounding_box_indices["left"]

    T = jnp.array(np.random.default_rng(0).uniform(size=(NX, NY, 1)))

    dirichlet = DirichletTemperature(tuple(bottom.T), prescribed=2.5)
    T_d = dirichlet.apply(T, 0)
    assert np.allclose(np.array(T_d[:, 0, 0]), 2.5)

    neumann = NeumannTemperature(tuple(left.T), normal=(-1, 0), prescribed=0.0)
    T_n = neumann.apply(T, 0)
    # Adiabatic: boundary nodes copy their interior neighbor
    assert np.allclose(np.array(T_n[0, :, 0]), np.array(T[1, :, 0]))

    neumann_flux = NeumannTemperature(tuple(left.T), normal=(-1, 0), prescribed=0.1)
    T_q = neumann_flux.apply(T, 0)
    assert np.allclose(np.array(T_q[0, :, 0]), np.array(T[1, :, 0]) + 0.1)
