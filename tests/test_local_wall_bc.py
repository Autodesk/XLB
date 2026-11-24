"""Verify the local (per-shard, int32) wall boundary condition path (BounceBackHalfway,
InterpolatedBounceBackBouzidi, InterpolatedBounceBackDifferentiable) matches each boundary condition's own
apply()/prepare_populations() (global indices) exactly, for both LBMBase and Multiphase.
"""

import jax.numpy as jnp
import numpy as np
from _boundary_pass import boundary_pass, reference_poststreaming

from jax_lab.core.boundary_conditions import (
    BounceBackHalfway,
    InterpolatedBounceBackBouzidi,
    InterpolatedBounceBackDifferentiable,
)
from jax_lab.core.shanchen.eos import VanderWaals
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.models import BGKSim
from jax_lab.core.shanchen.multiphase import MultiphaseBGK

DOMAIN = (16, 16, 16)
PRECISION = "f32/f32"
OMEGA = 1.0
SEED = 0


def _sphere_mask(domain, center, radius):
    x, y, z = np.meshgrid(*(np.arange(n) for n in domain), indexing="ij")
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= radius**2


def _sphere_sdf(domain, center, radius):
    """Signed distance to a spherical solid surface: positive in fluid, negative inside the solid."""
    x, y, z = np.meshgrid(*(np.arange(n) for n in domain), indexing="ij")
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    return distance - radius


CENTER = (8, 8, 8)
RADIUS = 4.0
SOLID_MASK = _sphere_mask(DOMAIN, CENTER, RADIUS)
SDF = _sphere_sdf(DOMAIN, CENTER, RADIUS)
SOLID_INDICES = np.array(np.where(SOLID_MASK)).T


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


VEL = jnp.array([0.001, 0.0, -0.0005], dtype=jnp.float32)


def _make_single_phase_sim(bc_class, with_vel):
    vel = VEL if with_vel else None

    class Sim(BGKSim):
        def set_boundary_conditions(self):
            if bc_class is BounceBackHalfway:
                bc = bc_class(tuple(SOLID_INDICES.T), self.grid_info, self.precision_policy, vel=vel)
            else:
                bc = bc_class(tuple(SOLID_INDICES.T), SDF, self.grid_info, self.precision_policy, vel=vel)
            self.BCs.append(bc)

    return Sim(**_base_kwargs())


def _make_multiphase_sim(bc_class, with_vel):
    vel = VEL if with_vel else None
    kwargs = _base_kwargs()
    kwargs |= {
        "n_components": 1,
        "omega": [OMEGA],
        "g_kkprime": -np.ones((1, 1)),
        "lambda_": [1.0],
        "EOS": VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
    }

    class Sim(MultiphaseBGK):
        def set_boundary_conditions(self):
            if bc_class is BounceBackHalfway:
                bc = bc_class(tuple(SOLID_INDICES.T), self.grid_info, self.precision_policy, vel=vel)
            else:
                bc = bc_class(tuple(SOLID_INDICES.T), SDF, self.grid_info, self.precision_policy, vel=vel)
            self.BCs[0].append(bc)

    return Sim(**kwargs)


def _random_fin(shape, dtype):
    rng = np.random.default_rng(SEED)
    return jnp.asarray(rng.uniform(0.1, 0.3, size=shape), dtype=dtype)


def _expected_and_actual_single_phase(bc_class, with_vel):
    sim = _make_single_phase_sim(bc_class, with_vel)
    fin = _random_fin(sim.assign_fields_sharded().shape, sim.precision_policy.output_dtype)
    bc = sim.BCs[0]

    # Ground truth: the boundary condition's own (global-index) prepare_populations + apply on the streamed populations
    expected_full = reference_poststreaming(sim, bc, fin)
    _, actual_full = boundary_pass(sim, fin, fin)

    expected_fluid = np.asarray(expected_full)[bc.indices]
    actual_fluid = np.asarray(actual_full)[bc.indices]
    expected_solid = np.asarray(expected_full)[tuple(bc.solid_indices)]
    actual_solid = np.asarray(actual_full)[tuple(bc.solid_indices)]
    return expected_fluid, actual_fluid, expected_solid, actual_solid


def _check_single_phase(bc_class, with_vel, tolerance=0.0):
    expected_fluid, actual_fluid, expected_solid, actual_solid = _expected_and_actual_single_phase(bc_class, with_vel)
    assert np.max(np.abs(actual_fluid - expected_fluid)) <= tolerance
    assert np.max(np.abs(actual_solid - expected_solid)) <= tolerance


def test_halfway_bounceback_matches_global_single_phase():
    # Pure gather/scatter (no arithmetic), like plain BounceBack: exact match.
    _check_single_phase(BounceBackHalfway, with_vel=False)


def test_halfway_bounceback_with_velocity_matches_global_single_phase():
    _check_single_phase(BounceBackHalfway, with_vel=True, tolerance=1e-7)


def test_bouzidi_bounceback_matches_global_single_phase():
    # Involves interpolation (division), so the gathered-block and global-array computation graphs are not
    # bit-identical (1-ULP-level float32 noise) even though they compute the same formula on the same inputs -
    # same tolerance used for the other arithmetic-based optimizations (see test_wetting_denominator.py,
    # test_non_equilibrium_extrapolation.py).
    _check_single_phase(InterpolatedBounceBackBouzidi, with_vel=False, tolerance=1e-6)


def test_differentiable_bounceback_matches_global_single_phase():
    _check_single_phase(InterpolatedBounceBackDifferentiable, with_vel=False, tolerance=1e-6)


def _check_multiphase(bc_class, with_vel, tolerance=0.0):
    sim = _make_multiphase_sim(bc_class, with_vel)
    fin_tree = sim.assign_fields_sharded()
    fin = _random_fin(fin_tree[0].shape, sim.precision_policy.output_dtype)
    bc = sim.BCs[0][0]

    expected_full = reference_poststreaming(sim, bc, fin)
    _, actual_full = boundary_pass(sim, fin, fin)

    fluid_diff = np.abs(np.asarray(actual_full)[bc.indices] - np.asarray(expected_full)[bc.indices])
    solid_diff = np.abs(np.asarray(actual_full)[tuple(bc.solid_indices)] - np.asarray(expected_full)[tuple(bc.solid_indices)])
    assert np.max(fluid_diff) <= tolerance
    assert np.max(solid_diff) <= tolerance


def test_halfway_bounceback_matches_global_multiphase():
    _check_multiphase(BounceBackHalfway, with_vel=False)


def test_bouzidi_bounceback_matches_global_multiphase():
    _check_multiphase(InterpolatedBounceBackBouzidi, with_vel=False, tolerance=1e-6)
