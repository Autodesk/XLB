"""Momentum exchange force of the full-way bounce-back conditions, checked against exact steady-state momentum balances."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.core.boundary_conditions import BounceBack, BounceBackMoving
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.models import BGKSim

PRECISION = "f64/f64"
NX, NY = 4, 23  # periodic in x; full-way bounce-back walls at y = 0 and y = NY - 1
OMEGA = 1.0
STEPS = 20000  # several viscous times H^2 / nu for H = NY - 2, nu = 1/6


@pytest.fixture(autouse=True)
def _enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _kwargs():
    return {
        "lattice": LatticeD2Q9(PRECISION),
        "omega": OMEGA,
        "nx": NX,
        "ny": NY,
        "nz": 0,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
    }


def _wall_indices(y):
    return (np.arange(NX), np.full(NX, y))


def _run(sim, steps=STEPS):
    """Advance to steady state; return the populations the way run() hands them to output_data."""
    state = sim.state_from_populations(sim.assign_fields_sharded())
    for t in range(steps - 1):
        state, _ = sim.step(state, t)
    state, f_postcollision = sim.step(state, steps - 1, return_fpost=True)
    return np.asarray(sim.poststreaming(state)), np.asarray(f_postcollision)


class ForcedChannel(BGKSim):
    """Poiseuille flow driven by a uniform body force (velocity increment DU per step) between two walls."""

    DU = 1e-6

    def set_boundary_conditions(self):
        self.BCs.append(BounceBack(_wall_indices(0), self.grid_info, self.precision_policy))
        self.BCs.append(BounceBack(_wall_indices(NY - 1), self.grid_info, self.precision_policy))

    def get_force(self):
        force = np.zeros((self.nx, self.ny, 2))
        force[..., 0] = self.DU
        return self.precision_policy.cast_to_output(force)


class CouetteChannel(BGKSim):
    """Couette flow: the top wall moves with LID_VELOCITY in x, the bottom wall is at rest."""

    LID_VELOCITY = 0.01

    def set_boundary_conditions(self):
        self.BCs.append(BounceBack(_wall_indices(0), self.grid_info, self.precision_policy))
        top = tuple(jnp.asarray(i) for i in _wall_indices(NY - 1))
        velocity = jnp.array([self.LID_VELOCITY, 0.0], dtype=self.precision_policy.compute_dtype)
        self.BCs.append(BounceBackMoving(self.grid_info, self.precision_policy, update_function=lambda time: (top, velocity)))


def test_fullway_bounceback_force_balances_body_force():
    sim = ForcedChannel(**_kwargs())
    f_poststreaming, f_postcollision = _run(sim)
    wall_force = sum(np.asarray(bc.momentum_exchange_force(f_poststreaming, f_postcollision)).sum(axis=0) for bc in sim.BCs)
    # Steady state: the walls take up all the momentum the body force puts into the fluid nodes each step
    rho = f_poststreaming.sum(axis=-1)
    body_force = ForcedChannel.DU * rho[:, 1:-1].sum()
    np.testing.assert_allclose(wall_force[0], body_force, rtol=1e-3)
    assert abs(wall_force[1]) < 1e-6 * body_force
    # By symmetry each wall takes half
    bottom = np.asarray(sim.BCs[0].momentum_exchange_force(f_poststreaming, f_postcollision)).sum(axis=0)
    np.testing.assert_allclose(bottom[0], 0.5 * body_force, rtol=1e-3)


def test_moving_bounceback_force_matches_couette_shear():
    sim = CouetteChannel(**_kwargs())
    f_poststreaming, f_postcollision = _run(sim)
    bottom = np.asarray(sim.BCs[0].momentum_exchange_force(f_poststreaming, f_postcollision)).sum(axis=0)
    lid = np.asarray(sim.BCs[1].momentum_exchange_force(f_poststreaming, f_postcollision, timestep=STEPS - 1)).sum(axis=0)
    # Viscous shear rho nu U / H per unit length; full-way walls sit half a cell inside the wall nodes, so H = NY - 2
    nu = (1.0 / OMEGA - 0.5) / 3.0
    shear = nu * CouetteChannel.LID_VELOCITY / (NY - 2) * NX
    np.testing.assert_allclose(bottom[0], shear, rtol=1e-2)
    np.testing.assert_allclose(lid[0], -bottom[0], rtol=1e-3)
    with pytest.raises(ValueError):
        sim.BCs[1].momentum_exchange_force(f_poststreaming, f_postcollision)
