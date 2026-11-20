"""Collision-model tests: Taylor-Green vortex decay (single-phase and multiphase BGK/MRT), the single-phase cascaded
collision, the fused multiphase cascaded force evaluation, the fused single-phase MRT collision with forcing, and
single-phase BGK/MRT exact-difference forcing."""

import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree import map as tree_map

from jax_lab.core.shanchen.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.core.models import BGKSim, CLBMSim, MRTSim
from jax_lab.core.shanchen.multiphase import MultiphaseBGK, MultiphaseCascade, MultiphaseMRT


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable x64 for each test (the f64 tests need it; other test modules may switch it off after import)."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


# Moment basis of the cascaded collision (D2Q9), shared by the single- and multiphase cascaded tests
CASCADE_MATRIX = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1],
    [0, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 1, -1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 1, -1],
    [0, 0, 0, 0, 0, 1, 1, -1, -1],
    [0, 0, 0, 0, 0, 1, -1, -1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
])


# ---------------------------------------------------------------------------------------------------------------------
# Taylor-Green vortex decay
# ---------------------------------------------------------------------------------------------------------------------
DOMAIN_CASES = (
    pytest.param(LatticeD2Q9, (16, 16, 0), 0.08, id="2d"),
    pytest.param(LatticeD3Q19, (10, 10, 10), 0.12, id="3d"),
)
SINGLE_PHASE_MODELS = (
    pytest.param(BGKSim, id="bgk"),
    pytest.param(MRTSim, id="mrt"),
)
MULTIPHASE_MODELS = (
    pytest.param(MultiphaseBGK, id="bgk"),
    pytest.param(MultiphaseMRT, id="mrt"),
)

PRECISION = "f32/f32"
OMEGA = 1.0
VISCOSITY = (1.0 / OMEGA - 0.5) / 3.0
REFERENCE_VELOCITY = 0.02
NUMBER_OF_UPDATES = 10


@pytest.fixture(autouse=True)
def isolated_output_directory(tmp_path, monkeypatch):
    """Run each test in a temporary directory and remove all generated files."""
    original_directory = Path.cwd()
    output_directory = tmp_path / "collision-output"
    output_directory.mkdir()
    monkeypatch.chdir(output_directory)
    try:
        yield
    finally:
        os.chdir(original_directory)
        shutil.rmtree(output_directory, ignore_errors=True)


def _coordinates(domain):
    spatial_shape = domain[:2] if domain[2] == 0 else domain
    return jnp.meshgrid(
        *[2.0 * jnp.pi * jnp.arange(size, dtype=jnp.float32) / size for size in spatial_shape],
        indexing="ij",
    )


def _taylor_green_fields(domain, time):
    coordinates = _coordinates(domain)
    spatial_shape = domain[:2] if domain[2] == 0 else domain
    decay_rate = VISCOSITY * sum((2.0 * jnp.pi / size) ** 2 for size in spatial_shape)
    decay = jnp.exp(-decay_rate * time)

    x, y = coordinates[:2]
    z_factor = jnp.cos(coordinates[2]) if len(coordinates) == 3 else 1.0
    velocity_x = REFERENCE_VELOCITY * jnp.sin(x) * jnp.cos(y) * z_factor * decay
    velocity_y = -REFERENCE_VELOCITY * jnp.cos(x) * jnp.sin(y) * z_factor * decay
    velocity_z = jnp.zeros_like(velocity_x)
    velocity = jnp.stack((velocity_x, velocity_y), axis=-1) if len(coordinates) == 2 else jnp.stack((velocity_x, velocity_y, velocity_z), axis=-1)

    density_decay = jnp.exp(-2.0 * decay_rate * time)
    density = 1.0 - (REFERENCE_VELOCITY**2 / 12.0) * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * density_decay
    return density[..., None], velocity


class TaylorGreenInitialFields:
    """Provide periodic Taylor-Green initial conditions to a solver."""

    def initialize_macroscopic_fields(self):
        domain = (self.nx, self.ny, self.nz)
        density, velocity = _taylor_green_fields(domain, time=0.0)
        return (
            self.precision_policy.cast_to_output(density),
            self.precision_policy.cast_to_output(velocity),
        )


class MultiphaseTaylorGreenInitialFields:
    """Provide one-component Taylor-Green fields and zero force."""

    def initialize_macroscopic_fields(self):
        domain = (self.nx, self.ny, self.nz)
        density, velocity = _taylor_green_fields(domain, time=0.0)
        return [self.precision_policy.cast_to_output(density)], [self.precision_policy.cast_to_output(velocity)]

    def compute_force(self, rho_tree, T=None):
        return [jnp.zeros((*density.shape[:-1], self.dim), dtype=self.precision_policy.compute_dtype) for density in rho_tree]

    def adjust_surface_tension(self, psi_tree):
        return [jnp.zeros((*psi.shape[:-1], self.q), dtype=self.precision_policy.compute_dtype) for psi in psi_tree]


def _mrt_matrix(lattice):
    e = np.asarray(lattice.c).T
    en = np.linalg.norm(e, axis=1)

    if lattice.d == 2:
        matrix = np.zeros((9, 9))
        matrix[0, :] = en**0
        matrix[1, :] = -4 * en**0 + 3 * en**2
        matrix[2, :] = 4 * en**0 - 10.5 * en**2 + 4.5 * en**4
        matrix[3, :] = e[:, 0]
        matrix[4, :] = (-5 * en**0 + 3 * en**2) * e[:, 0]
        matrix[5, :] = e[:, 1]
        matrix[6, :] = (-5 * en**0 + 3 * en**2) * e[:, 1]
        matrix[7, :] = e[:, 0] ** 2 - e[:, 1] ** 2
        matrix[8, :] = e[:, 0] * e[:, 1]
        return matrix

    matrix = np.zeros((19, 19))
    x, y, z = e.T
    matrix[0, :] = en**0
    matrix[1, :] = 19 * en**2 - 30
    matrix[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    matrix[3, :] = x
    matrix[4, :] = (5 * en**2 - 9) * x
    matrix[5, :] = y
    matrix[6, :] = (5 * en**2 - 9) * y
    matrix[7, :] = z
    matrix[8, :] = (5 * en**2 - 9) * z
    matrix[9, :] = 3 * x**2 - en**2
    matrix[10, :] = (3 * en**2 - 5) * (3 * x**2 - en**2)
    matrix[11, :] = y**2 - z**2
    matrix[12, :] = (3 * en**2 - 5) * (y**2 - z**2)
    matrix[13, :] = x * y
    matrix[14, :] = y * z
    matrix[15, :] = x * z
    matrix[16, :] = (y**2 - z**2) * x
    matrix[17, :] = (z**2 - x**2) * y
    matrix[18, :] = (x**2 - y**2) * z
    return matrix


def _base_parameters(lattice_class, domain):
    lattice = lattice_class(PRECISION)
    nx, ny, nz = domain
    return {
        "lattice": lattice,
        "omega": OMEGA,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
    }


def _mrt_relaxation_parameters(lattice, multiphase):
    value = [OMEGA] if multiphase else OMEGA
    parameters = {
        "M": [_mrt_matrix(lattice)] if multiphase else _mrt_matrix(lattice),
        "s_rho": value,
        "s_e": value,
        "s_eta": value,
        "s_j": value,
        "s_q": value,
        "s_v": value,
    }
    if lattice.d == 3:
        parameters |= {"s_pi": value, "s_m": value}
    if multiphase:
        parameters["kappa"] = [0.0]
    return parameters


def _relative_velocity_error(actual, expected):
    numerator = jnp.sum(jnp.square(actual - expected))
    denominator = jnp.sum(jnp.square(expected))
    return float(jnp.sqrt(numerator / denominator))


@pytest.mark.parametrize("model_class", SINGLE_PHASE_MODELS)
@pytest.mark.parametrize("lattice_class, domain, maximum_error", DOMAIN_CASES)
def test_single_phase_collision_against_taylor_green(model_class, lattice_class, domain, maximum_error):
    """Verify single-phase BGK and equivalent-rate MRT against TGV decay."""
    parameters = _base_parameters(lattice_class, domain)
    if model_class is MRTSim:
        parameters |= _mrt_relaxation_parameters(parameters["lattice"], multiphase=False)
    solver_class = type("SinglePhaseTaylorGreen", (TaylorGreenInitialFields, model_class), {})
    solver = solver_class(**parameters)

    populations = solver.run(NUMBER_OF_UPDATES - 1)
    _, velocity = solver.update_macroscopic(populations)
    _, analytical_velocity = _taylor_green_fields(domain, time=NUMBER_OF_UPDATES)

    assert _relative_velocity_error(velocity, analytical_velocity) < maximum_error


@pytest.mark.parametrize("model_class", MULTIPHASE_MODELS)
@pytest.mark.parametrize("lattice_class, domain, maximum_error", DOMAIN_CASES)
def test_multiphase_collision_against_taylor_green(model_class, lattice_class, domain, maximum_error):
    """Verify multiphase BGK and equivalent-rate MRT against TGV decay."""
    parameters = _base_parameters(lattice_class, domain)
    parameters |= {
        "n_components": 1,
        "omega": [OMEGA],
        "g_kkprime": -np.ones((1, 1)),
        "lambda_": [1.0],
        "EOS": VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
    }
    if model_class is MultiphaseMRT:
        parameters |= _mrt_relaxation_parameters(parameters["lattice"], multiphase=True)
    solver_class = type("MultiphaseTaylorGreen", (MultiphaseTaylorGreenInitialFields, model_class), {})
    solver = solver_class(**parameters)

    populations = solver.run(NUMBER_OF_UPDATES - 1)
    _, velocity_tree = solver.update_macroscopic(populations)
    _, analytical_velocity = _taylor_green_fields(domain, time=NUMBER_OF_UPDATES)

    assert _relative_velocity_error(velocity_tree[0], analytical_velocity) < maximum_error


# ---------------------------------------------------------------------------------------------------------------------
# Single-phase cascaded collision
# ---------------------------------------------------------------------------------------------------------------------


def test_cascaded_collision_uses_relaxed_moments():
    sim = CLBMSim(
        lattice=LatticeD2Q9("f32/f32"),
        omega=1.0,
        nx=4,
        ny=4,
        nz=0,
        precision="f32/f32",
        io_rate=0,
        print_info_rate=0,
        checkpoint_rate=0,
        M=CASCADE_MATRIX,
        s_0=1.0,
        s_1=1.0,
        s_b=1.0,
        s_2=1.0,
        s_3=1.0,
        s_4=1.0,
    )
    fin = jnp.broadcast_to(sim.w, (4, 4, sim.q)).at[..., 1].add(0.01)

    fout = sim.collision(fin)

    assert np.max(np.abs(np.asarray(fout - fin))) > 1e-4
    np.testing.assert_allclose(np.asarray(jnp.sum(fout, axis=-1)), np.asarray(jnp.sum(fin, axis=-1)), atol=1e-5)


def _d3q19_cascade_matrix():
    e = LatticeD3Q19().c.T
    ex, ey, ez = e[:, 0], e[:, 1], e[:, 2]
    rows = [ex**0, ex, ey, ez, ex * ey, ex * ez, ey * ez, ex * ex, ey * ey, ez * ez, ex * ey * ey, ex * ez * ez, ey * ex * ex,
            ez * ex * ex, ey * ez * ez, ez * ey * ey, ex * ex * ey * ey, ex * ex * ez * ez, ey * ey * ez * ez]  # fmt: skip
    return np.stack(rows).astype(np.float64)


@pytest.mark.parametrize("s_v", [None, 1.2])
def test_cascaded_d3q19_shear_wave_decays_with_set_viscosity(s_v):
    """A shear wave u_x(y) is damped only by the off-diagonal (s_v) stress: it must decay at nu = cs2 (1 / s_v - 1/2), with s_v
    defaulting to s_2 (isotropic viscosity). JAX-LaB used to fix s_v at 1.0 whatever was passed."""
    n, s_2, amplitude = 16, 1.6, 1e-3
    kwargs = dict(s_v=s_v) if s_v is not None else {}
    sim = CLBMSim(
        lattice=LatticeD3Q19("f64/f64"), nx=n, ny=n, nz=n, precision="f64/f64", io_rate=0, print_info_rate=0, checkpoint_rate=0,
        M=_d3q19_cascade_matrix(), s_0=1.0, s_1=1.0, s_b=1.0, s_2=s_2, s_3=1.0, s_4=1.0, **kwargs,
    )  # fmt: skip
    rate = s_2 if s_v is None else s_v
    assert sim.s_v == rate and sim.omega == rate
    k = 2.0 * np.pi / n
    profile = np.sin(k * np.arange(n))[None, :, None]
    u0 = np.zeros((n, n, n, 3))
    u0[..., 0] = amplitude * profile
    f = sim.initialize_populations(jnp.ones((n, n, n, 1)), jnp.asarray(u0))

    def mode(f):
        rho, u = sim.update_macroscopic(f)
        return float(np.mean(np.asarray(u[..., 0]) * profile) * 2.0)

    t0, t1 = 50, 250
    for t in range(t1):
        if t == t0:
            a0 = mode(f)
        f = sim.streaming(sim.collision(f))
    nu_measured = np.log(a0 / mode(f)) / (k**2 * (t1 - t0))
    nu_expected = (1.0 / rate - 0.5) / 3.0
    np.testing.assert_allclose(nu_measured, nu_expected, rtol=0.01)


# ---------------------------------------------------------------------------------------------------------------------
# Multiphase cascaded collision: fused force/potential evaluation
# ---------------------------------------------------------------------------------------------------------------------

CASCADE_FORCE_DOMAIN = (12, 10)


def _build_multiphase_cascade():
    return MultiphaseCascade(
        n_components=1,
        lattice=LatticeD2Q9("f32/f32"),
        nx=CASCADE_FORCE_DOMAIN[0],
        ny=CASCADE_FORCE_DOMAIN[1],
        nz=0,
        g_kkprime=-np.ones((1, 1)),
        body_force=[1e-6, -2e-6],
        lambda_=[1.0],
        EOS=VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
        M=[CASCADE_MATRIX],
        s_0=[1.0],
        s_1=[1.0],
        s_b=[0.8],
        s_2=[0.8],
        s_3=[1.0],
        s_4=[1.0],
        sigma=[0.05],
        precision="f32/f32",
        io_rate=0,
        print_info_rate=0,
        checkpoint_rate=0,
    )


def _reference_cascade_collision(sim, fin_tree):
    """Previous collision dataflow, which evaluated force once for velocity and again in apply_force."""
    fin_tree = tree_map(lambda f: sim.precision_policy.cast_to_compute(f), fin_tree)
    rho_tree, _ = sim.update_macroscopic(fin_tree)
    u_tree = sim.macroscopic_velocity(fin_tree, rho_tree)
    moment_tree = tree_map(lambda f, M: jnp.dot(f, M), fin_tree, sim.M)
    central_tree = sim.compute_central_moment(moment_tree, u_tree)
    equilibrium_tree = sim.compute_eq_central_moments(rho_tree)
    relaxed_tree = tree_map(
        lambda central, equilibrium, S: jnp.dot(central, jnp.eye(sim.q) - S) + jnp.dot(equilibrium, S),
        central_tree,
        equilibrium_tree,
        sim.S,
    )
    forced_tree = sim.apply_force(relaxed_tree, rho_tree, u_tree)
    shifted_tree = sim.compute_central_moment_inverse(forced_tree, u_tree)
    return tree_map(lambda shifted, inverse: jnp.dot(shifted, inverse), shifted_tree, sim.M_inv)


def test_cascade_collision_matches_previous_force_dataflow():
    sim = _build_multiphase_cascade()
    rng = np.random.default_rng(0)
    fin_tree = [sim.assign_fields_sharded()[0] + jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(*CASCADE_FORCE_DOMAIN, sim.q)), dtype=jnp.float32)]
    # Both at full float32 matrix-product precision: at the default precision GPUs may evaluate jnp.dot in TF32 (~1e-3
    # relative), which is coarser than this tolerance (the collision's moment sums are exact, but update_macroscopic uses dot).
    with jax.default_matmul_precision("float32"):
        actual = sim.collision(fin_tree)
        expected = _reference_cascade_collision(sim, fin_tree)
    assert np.max(np.abs(np.asarray(actual[0]) - np.asarray(expected[0]))) < 2e-6


# ---------------------------------------------------------------------------------------------------------------------
# Fused single-phase MRT collision with forcing, against an unfused reference
# ---------------------------------------------------------------------------------------------------------------------

FUSED_MRT_DOMAIN = (10, 10, 10)
FUSED_MRT_FORCE = (1e-4, -2e-4, 5e-5)


class ForcedMRT3D(MRTSim):
    def get_force(self):
        return jnp.array(FUSED_MRT_FORCE, dtype=jnp.float64)


def _build_forced_mrt_3d():
    nx, ny, nz = FUSED_MRT_DOMAIN
    lattice = LatticeD3Q19("f64/f64")
    s = 1.0
    return ForcedMRT3D(
        lattice=lattice,
        nx=nx,
        ny=ny,
        nz=nz,
        precision="f64/f64",
        M=_mrt_matrix(lattice),
        s_rho=s,
        s_e=s,
        s_eta=s,
        s_j=s,
        s_q=s,
        s_v=s,
        s_pi=s,
        s_m=s,
    )


def _reference_mrt_collision(sim, f):
    """Independent unfused reference: m = f @ M; mout = -(m - meq) @ S + delta_meq; fout = f + mout @ M_inv."""
    f = sim.precision_policy.cast_to_compute(f)
    m = jnp.dot(f, sim.M)
    rho, u = sim.update_macroscopic(f)
    feq = sim.equilibrium(rho, u)
    meq = jnp.dot(feq, sim.M)
    mout = -jnp.dot(m - meq, sim.S)
    mout = sim.apply_force(mout, meq, rho, u)
    return sim.precision_policy.cast_to_output(f + jnp.dot(mout, sim.M_inv))


def test_fused_symbolic_collision_matches_reference():
    sim = _build_forced_mrt_3d()
    f = sim.assign_fields_sharded()
    rng = np.random.default_rng(0)
    f = f + jnp.asarray(rng.uniform(-0.01, 0.01, size=f.shape))

    actual = sim.collision(f)
    expected = _reference_mrt_collision(sim, f)

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-9


# ---------------------------------------------------------------------------------------------------------------------
# Single-phase BGK/MRT exact-difference (EDM) forcing, against two-equilibrium references
# ---------------------------------------------------------------------------------------------------------------------

EDM_DOMAIN = (8, 8, 0)
EDM_FORCE = (0.001, -0.0005)


class ForcedBGK2D(BGKSim):
    def get_force(self):
        return jnp.array(EDM_FORCE, dtype=jnp.float64)


class ForcedMRT2D(MRTSim):
    def get_force(self):
        return jnp.array(EDM_FORCE, dtype=jnp.float64)


def _random_edm_fields():
    nx, ny = EDM_DOMAIN[0], EDM_DOMAIN[1]
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.uniform(0.5, 2.0, size=(nx, ny, 1)))
    u = jnp.asarray(rng.uniform(-0.02, 0.02, size=(nx, ny, 2)))
    return rho, u


def test_bgk_apply_force_matches_population_space_reference():
    sim = ForcedBGK2D(lattice=LatticeD2Q9("f64/f64"), omega=1.0, nx=EDM_DOMAIN[0], ny=EDM_DOMAIN[1], nz=EDM_DOMAIN[2], precision="f64/f64")
    rho, u = _random_edm_fields()
    feq = sim.equilibrium(rho, u, cast_output=False)
    rng = np.random.default_rng(1)
    fout = feq + jnp.asarray(rng.uniform(-0.01, 0.01, size=feq.shape))

    actual = sim.apply_force(fout, feq, rho, u)

    du = sim.get_force()
    feq_force = sim.equilibrium(rho, u + du, cast_output=False)
    expected = fout + feq_force - feq

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-12


def test_mrt_apply_force_matches_moment_space_reference():
    lattice = LatticeD2Q9("f64/f64")
    s = 1.0
    sim = ForcedMRT2D(
        lattice=lattice,
        nx=EDM_DOMAIN[0],
        ny=EDM_DOMAIN[1],
        nz=EDM_DOMAIN[2],
        precision="f64/f64",
        M=_mrt_matrix(lattice),
        s_rho=s,
        s_e=s,
        s_eta=s,
        s_j=s,
        s_q=s,
        s_v=s,
    )
    rho, u = _random_edm_fields()
    feq = sim.equilibrium(rho, u, cast_output=False)
    meq = jnp.dot(feq, sim.M)
    rng = np.random.default_rng(2)
    m = meq + jnp.asarray(rng.uniform(-0.01, 0.01, size=meq.shape))

    actual = sim.apply_force(m, meq, rho, u)

    du = sim.get_force()
    feq_force = sim.equilibrium(rho, u + du, cast_output=False)
    # Correctly space-matched reference: transform both equilibria into moment space before subtracting,
    # instead of subtracting a moment-space meq from a population-space feq_force.
    expected = m + jnp.dot(feq_force, sim.M) - jnp.dot(feq, sim.M)

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-10
