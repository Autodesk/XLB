"""Viscosity of the single-phase collisions: a decaying shear wave u_x = U sin(k y) must follow exp(-nu k^2 t)."""

import jax
import numpy as np
import pytest

from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q27
from jax_lab.core.models import BGKSim, KBCSim

PRECISION = "f64/f64"
L = 32
U = 0.01
STEPS = 500


@pytest.fixture(autouse=True)
def _enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _shear_wave(base, lattice, dim, omega):
    nz = 0 if dim == 2 else 4

    class ShearWave(base):
        def set_boundary_conditions(self):
            pass

        def initialize_macroscopic_fields(self):
            shape = (L, L) if dim == 2 else (L, L, nz)
            profile = U * np.sin(2 * np.pi * np.arange(L) / L)
            u = np.zeros(shape + (dim,))
            u[..., 0] = profile[None, :] if dim == 2 else profile[None, :, None]
            rho = np.ones(shape + (1,))
            return self.precision_policy.cast_to_output(rho), self.precision_policy.cast_to_output(u)

    kwargs = {"lattice": lattice(PRECISION), "omega": omega, "nx": L, "ny": L, "nz": nz, "precision": PRECISION}
    kwargs.update({"io_rate": 0, "print_info_rate": 0, "checkpoint_rate": 0})
    return ShearWave(**kwargs)


@pytest.mark.parametrize("omega", [1.0, 1.6, 1.9])
@pytest.mark.parametrize(
    ("base", "lattice", "dim"),
    [(BGKSim, LatticeD2Q9, 2), (KBCSim, LatticeD2Q9, 2), (BGKSim, LatticeD3Q27, 3), (KBCSim, LatticeD3Q27, 3)],
    ids=["BGK-D2Q9", "KBC-D2Q9", "BGK-D3Q27", "KBC-D3Q27"],
)
def test_shear_wave_decays_with_lattice_viscosity(base, lattice, dim, omega):
    sim = _shear_wave(base, lattice, dim, omega)
    state = sim.state_from_populations(sim.assign_fields_sharded())
    for t in range(STEPS):
        state, _ = sim.step(state, t)
    f = np.asarray(sim.poststreaming(state))
    ux = (f @ np.asarray(sim.lattice.c).T.astype(float))[..., 0] / f.sum(axis=-1)
    profile = ux.mean(axis=tuple(axis for axis in range(dim) if axis != 1))
    amplitude = 2.0 * np.mean(profile * np.sin(2 * np.pi * np.arange(L) / L)) / U
    nu = (1.0 / omega - 0.5) / 3.0
    np.testing.assert_allclose(amplitude, np.exp(-nu * (2 * np.pi / L) ** 2 * STEPS), rtol=2e-2)
