"""Checks of the volume-of-fluid (VoF) two-phase solver: mass conservation, hydrostatic rest state, evaporation, and collision models."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.vof import VoFBGK, VoFMRT, VoFOpenBoundary
from jax_lab.core.vof.multiphase import INTERFACE, LIQUID

PRECISION = "f32/f32"
NX, NY = 48, 48


def _d2q9_mrt_matrix():
    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)
    M = np.zeros((9, 9))
    M[0] = 1.0
    M[1] = -4.0 + 3.0 * en**2
    M[2] = 4.0 - 10.5 * en**2 + 4.5 * en**4
    M[3] = e[:, 0]
    M[4] = (-5.0 + 3.0 * en**2) * e[:, 0]
    M[5] = e[:, 1]
    M[6] = (-5.0 + 3.0 * en**2) * e[:, 1]
    M[7] = e[:, 0] ** 2 - e[:, 1] ** 2
    M[8] = e[:, 0] * e[:, 1]
    return M


def _box_walls(sim):
    return np.concatenate([sim.bounding_box_indices[key] for key in ("bottom", "top", "left", "right")])


class DropletInBox(VoFBGK):
    def initialize_volume_fraction(self):
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        return np.clip(10.5 - np.sqrt((x - 24.0) ** 2 + (y - 22.0) ** 2), 0.0, 1.0)

    def set_boundary_conditions(self):
        walls = _box_walls(self)
        for BCs in self.BCs:
            BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy, theta=np.deg2rad(60.0)))


class PoolInBox(VoFBGK):
    def initialize_volume_fraction(self):
        eps = np.zeros((self.nx, self.ny))
        eps[:, :20] = 1.0
        eps[:, 20] = 0.5
        return eps

    def initialize_macroscopic_fields(self):
        # Hydrostatic in both phases (each in its own density units), with the free surface at y = 20
        g = 0.0 if self.body_force is None else -self.body_force[0, 1]
        rho = 1.0 + g * (20.0 - np.arange(self.ny)) / self.lattice.cs2
        rho = np.broadcast_to(rho[None, :, None], (self.nx, self.ny, 1))
        u = np.zeros((self.nx, self.ny, 2))
        return [rho, rho], [u, u]

    def set_boundary_conditions(self):
        walls = _box_walls(self)
        for BCs in self.BCs:
            BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy))


class EvaporatingPool(PoolInBox):
    def set_boundary_conditions(self):
        walls = np.concatenate([self.bounding_box_indices[key] for key in ("bottom", "left", "right")])
        for BCs in self.BCs:
            BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy))
        top = self.bounding_box_indices["top"][1:-1]
        self.BCs[1].append(VoFOpenBoundary(tuple(top.T), self.grid_info, self.precision_policy, evaporation_flux=1.0e-3, evaporation_start=0))


def _kwargs(**extra):
    kwargs = {
        "lattice": LatticeD2Q9(PRECISION),
        "nx": NX,
        "ny": NY,
        "nz": 0,
        "precision": PRECISION,
        "omega": [1.0, 1.2],
        "density_ratio": 0.1,
        "surface_tension": 5.0e-3,
        "io_rate": 0,
        "print_info_rate": 0,
    }
    kwargs.update(extra)
    return kwargs


def _liquid_mass(sim, f_tree, state):
    fields = sim.compute_macroscopic_fields(f_tree, state)
    flags = np.asarray(state["flags"])
    rho_l = np.asarray(fields["rho_tree"][0])
    mass = np.asarray(state["mass"])
    return float(np.sum(np.where(flags == LIQUID, rho_l, np.where(flags == INTERFACE, mass, 0.0)))), fields


def _advance(sim, steps):
    f_tree, state = sim.assign_fields_sharded()
    initial, _ = _liquid_mass(sim, f_tree, state)
    for timestep in range(1, steps + 1):
        f_tree, state = sim.step(f_tree, state, timestep)
    final, fields = _liquid_mass(sim, f_tree, state)
    return initial, final, fields, state


def test_droplet_conserves_liquid_mass():
    sim = DropletInBox(**_kwargs(body_force=[0.0, -1.0e-5]))
    initial, final, fields, _ = _advance(sim, 400)
    assert abs(final - initial) / initial < 1.0e-4
    assert np.all(np.isfinite(np.asarray(fields["u"])))


def test_flat_pool_stays_at_rest():
    sim = PoolInBox(**_kwargs(body_force=[0.0, -1.0e-5]))
    _, _, fields, state = _advance(sim, 400)
    assert np.max(np.abs(np.asarray(fields["u"]))) < 1.0e-5
    assert np.array_equal(np.unique(np.asarray(state["flags"])[1:-1, 20]), [INTERFACE])


def test_evaporation_requires_open_boundary_and_removes_liquid():
    assert not PoolInBox(**_kwargs())._has_evaporation
    sim = EvaporatingPool(**_kwargs())
    assert sim._has_evaporation
    initial, final, _, _ = _advance(sim, 200)
    # One interface row of NX - 2 cells losing 1e-3 per unit area per step
    expected = 200 * 1.0e-3 * (NX - 2)
    assert initial - final == pytest.approx(expected, rel=0.1)


def test_mrt_with_uniform_rates_matches_bgk():
    omega = 1.25
    rates = {key: [omega, omega] for key in ("s_rho", "s_e", "s_eta", "s_j", "s_q", "s_v")}
    mrt = type("PoolMRT", (VoFMRT,), dict(PoolInBox.__dict__))(**_kwargs(M=[_d2q9_mrt_matrix()] * 2, **rates))
    bgk = PoolInBox(**_kwargs(omega=[omega, omega]))
    f_tree, _ = bgk.assign_fields_sharded()
    rng = np.random.default_rng(0)
    f_tree = [tuple(p * (1.0 + 0.01 * rng.standard_normal(p.shape)).astype(p.dtype) for p in planes) for planes in f_tree]
    for f_mrt, f_bgk in zip(mrt.collision(f_tree), bgk.collision(f_tree), strict=True):
        np.testing.assert_allclose(np.stack(f_mrt), np.stack(f_bgk), rtol=1e-5, atol=1e-6)


def test_viscous_stress_jump_option_runs():
    sim = DropletInBox(**_kwargs(body_force=[0.0, -1.0e-5], viscous_stress_jump=True))
    initial, final, fields, _ = _advance(sim, 100)
    assert abs(final - initial) / initial < 1.0e-4
    assert np.all(np.isfinite(np.asarray(fields["u"])))


def test_compaction_matches_nonzero():
    rng = np.random.default_rng(1)
    mask = rng.random(5000) < 0.07
    index, rows = VoFBGK._compact(jnp.asarray(mask), 512, tile=128)
    expected = np.nonzero(mask)[0]
    np.testing.assert_array_equal(np.asarray(index)[: len(expected)], expected)
    assert np.all(np.asarray(index)[len(expected) :] == mask.size)
    np.testing.assert_array_equal(np.asarray(rows)[expected], np.arange(len(expected)))


class TranslatingDroplet(VoFBGK):
    """Periodic droplet (radius 8) carried by a uniform flow of both phases along x."""

    speed = 2.0e-3

    def initialize_volume_fraction(self):
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        return np.clip(8.5 - np.sqrt((x - 24.0) ** 2 + (y - 24.0) ** 2), 0.0, 1.0)

    def initialize_macroscopic_fields(self):
        rho = np.ones((self.nx, self.ny, 1))
        u = np.zeros((self.nx, self.ny, 2))
        u[..., 0] = self.speed
        return [rho, rho], [u, u]


class TwoDroplets(DropletInBox):
    def initialize_volume_fraction(self):
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        return np.maximum(*(np.clip(6.5 - np.sqrt((x - xc) ** 2 + (y - 24.0) ** 2), 0.0, 1.0) for xc in (12.0, 36.0)))

    def set_boundary_conditions(self):
        pass  # periodic


def _droplet_x(state):
    eps = np.asarray(state["eps"])[..., 0]
    return float(np.sum(eps * np.arange(eps.shape[0])[:, None]) / np.sum(eps))


def test_translating_droplet_keeps_pace_with_the_flow():
    # Galilean invariance with the net capillary force correction (without it, a slow 3D droplet is pinned to the lattice)
    sim = TranslatingDroplet(**_kwargs(omega=[1.2, 1.2], zero_net_capillary_force=True, consistent_link_stress=True))
    f_tree, state = sim.assign_fields_sharded()
    start = _droplet_x(state)
    steps = 1500
    for timestep in range(1, steps + 1):
        f_tree, state = sim.step(f_tree, state, timestep)
    moved = _droplet_x(state) - start
    expected = TranslatingDroplet.speed * steps
    assert moved == pytest.approx(expected, rel=0.05)


def test_droplets_are_labeled_separately():
    sim = TwoDroplets(**_kwargs(zero_net_capillary_force=True))
    _, state = sim.assign_fields_sharded()
    labels = np.asarray(state["label"])[..., 0]
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[12, 24] != labels[36, 24]


def test_checkpoint_restores_droplet_labels_and_shifts(tmp_path):
    # A run restored from a checkpoint (populations, mass, flags, volume fraction, droplet labels and Laplace shifts) continues exactly
    # like an uninterrupted one
    kwargs = _kwargs(omega=[1.2, 1.2], zero_net_capillary_force=True, consistent_link_stress=True)
    directory = str(tmp_path / "checkpoints")
    TranslatingDroplet(**kwargs, checkpoint_rate=100, checkpoint_dir=directory).run(200)
    _, restored = TranslatingDroplet(**kwargs, checkpoint_rate=100, checkpoint_dir=directory, restore_checkpoint=True).run(300)
    _, continuous = TranslatingDroplet(**kwargs).run(300)
    np.testing.assert_array_equal(np.asarray(restored["label"]), np.asarray(continuous["label"]))
    np.testing.assert_allclose(np.asarray(restored["capillary_shift"]), np.asarray(continuous["capillary_shift"]), rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(restored["eps"]), np.asarray(continuous["eps"]), atol=1e-6)
