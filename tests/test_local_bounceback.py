"""Verify the local (per-shard, int32) BounceBack path matches global bounce-back exactly."""

import jax.numpy as jnp
import numpy as np
from _boundary_pass import boundary_pass

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.shanchen.eos import VanderWaals
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.models import BGKSim
from jax_lab.core.shanchen.multiphase import MultiphaseBGK

DOMAIN = (32, 16, 16)
PRECISION = "f32/f32"
OMEGA = 1.0
SEED = 0


def _random_solid_indices(domain, fraction=0.1, seed=SEED):
    """Random solid-node coordinates on the interior of the domain (fixed numpy seed)."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = domain
    count = int(fraction * nx * ny * nz)
    x = rng.integers(1, nx - 1, size=count)
    y = rng.integers(1, ny - 1, size=count)
    z = rng.integers(1, nz - 1, size=count)
    return np.unique(np.stack([x, y, z], axis=-1), axis=0)


class RandomSolidSinglePhase(BGKSim):
    def set_boundary_conditions(self):
        indices = _random_solid_indices(DOMAIN)
        self.BCs.append(BounceBack(tuple(indices.T), self.grid_info, self.precision_policy))


class RandomSolidSinglePhaseScatter(RandomSolidSinglePhase):
    """Bounce-back applied by apply_bc (the local scatter path) instead of inside the collision."""

    def _collision_applies_bounceback(self):
        return False


class RandomSolidMultiphase(MultiphaseBGK):
    def set_boundary_conditions(self):
        indices = _random_solid_indices(DOMAIN)
        self.BCs[0].append(BounceBack(tuple(indices.T), self.grid_info, self.precision_policy))


class RandomSolidMultiphaseScatter(RandomSolidMultiphase):
    """Bounce-back applied by apply_bc (the local scatter path) instead of inside the collision."""

    def _collision_applies_bounceback(self):
        return False


def _base_kwargs():
    nx, ny, nz = DOMAIN
    return {
        "lattice": LatticeD3Q19(PRECISION),
        "omega": OMEGA,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
    }


def _expected_bounceback(fin, indices, opp_indices):
    """Ground-truth global bounce-back: reflect fin at the solid indices."""
    return fin[tuple(indices.T)][..., opp_indices]


def test_local_bounceback_matches_global_single_phase():
    sim = RandomSolidSinglePhaseScatter(**_base_kwargs())
    fin = sim.assign_fields_sharded()
    rng = np.random.default_rng(SEED + 1)
    fin = fin + jnp.asarray(rng.uniform(-0.05, 0.05, size=fin.shape), dtype=fin.dtype)

    fout, _ = boundary_pass(sim, jnp.array(fin), fin)

    indices = np.asarray(sim.BCs[0].indices).T
    expected = _expected_bounceback(np.asarray(fin), indices, np.asarray(sim.lattice.opp_indices))
    actual = np.asarray(fout)[tuple(indices.T)]

    assert np.max(np.abs(actual - expected)) == 0.0


def test_local_bounceback_matches_global_multiphase():
    kwargs = _base_kwargs()
    kwargs |= {
        "n_components": 1,
        "omega": [OMEGA],
        "g_kkprime": -np.ones((1, 1)),
        "lambda_": [1.0],
        "EOS": VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
    }
    sim = RandomSolidMultiphaseScatter(**kwargs)
    fin_tree = sim.assign_fields_sharded()
    rng = np.random.default_rng(SEED + 1)
    fin_tree = [f + jnp.asarray(rng.uniform(-0.05, 0.05, size=f.shape), dtype=f.dtype) for f in fin_tree]

    fout_tree = [boundary_pass(sim, jnp.array(fin_tree[0]), fin_tree[0])[0]]

    indices = np.asarray(sim.BCs[0][0].indices).T
    expected = _expected_bounceback(np.asarray(fin_tree[0]), indices, np.asarray(sim.lattice.opp_indices))
    actual = np.asarray(fout_tree[0])[tuple(indices.T)]

    assert np.max(np.abs(actual - expected)) == 0.0


def _multiphase_kwargs():
    kwargs = _base_kwargs()
    # lambda_ = 0.5 keeps psi = sqrt(2 (lambda^2 p - cs2 rho) / g) real for the perturbed populations (no NaN to compare)
    kwargs |= {
        "n_components": 1,
        "omega": [OMEGA],
        "g_kkprime": -np.ones((1, 1)),
        "lambda_": [0.5],
        "EOS": VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
    }
    return kwargs


def test_fused_bounceback_matches_scatter_path():
    """The bounce-back applied inside the collision (fused) gives bit-identical steps to the apply_bc scatter path."""
    for fused_cls, scatter_cls, kwargs in (
        (RandomSolidSinglePhase, RandomSolidSinglePhaseScatter, _base_kwargs()),
        (RandomSolidMultiphase, RandomSolidMultiphaseScatter, _multiphase_kwargs()),
    ):
        fused, scatter = fused_cls(**kwargs), scatter_cls(**kwargs)
        assert fused._collision_applies_bounceback() and fused.bounceback_mask is not None
        f = fused.assign_fields_sharded()
        rng = np.random.default_rng(SEED + 2)
        multiphase = isinstance(f, list)
        perturb = lambda x: x + jnp.asarray(rng.uniform(-0.05, 0.05, size=x.shape), dtype=x.dtype)  # noqa: E731
        f = [perturb(x) for x in f] if multiphase else perturb(f)
        copy = lambda x: [jnp.array(v) for v in x] if multiphase else jnp.array(x)  # noqa: E731
        # Collision output at the solid nodes is the reflected input
        fout = fused.collision(copy(f))
        f0, fout0 = (f[0], fout[0]) if multiphase else (f, fout)
        bcs = fused.BCs[0] if multiphase else fused.BCs
        indices = np.asarray(bcs[0].indices).T
        expected = _expected_bounceback(np.asarray(f0), indices, np.asarray(fused.lattice.opp_indices))
        assert np.max(np.abs(np.asarray(fout0)[tuple(indices.T)] - expected)) == 0.0
        # Full steps agree bit for bit
        a, b = fused.state_from_populations(copy(f)), scatter.state_from_populations(copy(f))
        for t in range(3):
            a = fused.step(a, t)[0]
            b = scatter.step(b, t)[0]
        a, b = fused.poststreaming(a), scatter.poststreaming(b)
        for x, y in zip(a if multiphase else [a], b if multiphase else [b], strict=True):
            assert np.isfinite(np.asarray(x)).all(), fused_cls.__name__
            assert np.array_equal(np.asarray(x), np.asarray(y)), fused_cls.__name__
