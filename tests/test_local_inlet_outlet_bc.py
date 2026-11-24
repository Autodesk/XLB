"""Verify the local (per-shard, int32) inlet/outlet boundary condition path (ZouHe, Regularized,
EquilibriumBC) matches each boundary condition's own apply() (global indices) exactly, for both LBMBase and
Multiphase.
"""

import jax.numpy as jnp
import numpy as np
from _boundary_pass import boundary_pass, reference_poststreaming

from jax_lab.core.boundary_conditions import EquilibriumBC, Regularized, ZouHe
from jax_lab.core.shanchen.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.models import BGKSim
from jax_lab.core.shanchen.multiphase import MultiphaseBGK

DOMAIN = (24, 24, 0)
PRECISION = "f32/f32"
OMEGA = 1.0
SEED = 0


def _base_kwargs():
    nx, ny, nz = DOMAIN
    return {
        "lattice": LatticeD2Q9(PRECISION),
        "omega": OMEGA,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
    }


def _make_bc(bc_class, ttype, indices, precision_policy, grid_info):
    n = indices.shape[0]
    if bc_class is EquilibriumBC:
        rho = np.ones((n, 1), dtype=precision_policy.compute_dtype)
        vel = np.zeros((n, 2), dtype=precision_policy.compute_dtype)
        vel[:, 0] = 0.01
        return EquilibriumBC(tuple(indices.T), grid_info, precision_policy, rho, vel)
    if ttype == "velocity":
        prescribed = np.zeros((n, 2), dtype=precision_policy.compute_dtype)
        prescribed[:, 0] = 0.01
    else:
        prescribed = np.full((n, 1), 1.02, dtype=precision_policy.compute_dtype)
    return bc_class(tuple(indices.T), grid_info, precision_policy, ttype, prescribed)


def _make_single_phase_sim(bc_class, ttype, face):
    class Sim(BGKSim):
        def set_boundary_conditions(self):
            indices = self.bounding_box_indices[face]
            bc = _make_bc(bc_class, ttype, indices, self.precision_policy, self.grid_info)
            self.BCs.append(bc)

    return Sim(**_base_kwargs())


def _make_multiphase_sim(bc_class, ttype, face):
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
            indices = self.bounding_box_indices[face]
            bc = _make_bc(bc_class, ttype, indices, self.precision_policy, self.grid_info)
            self.BCs[0].append(bc)

    return Sim(**kwargs)


def _random_fin(shape, dtype):
    rng = np.random.default_rng(SEED)
    return jnp.asarray(rng.uniform(0.1, 0.3, size=shape), dtype=dtype)


def _check_single_phase(bc_class, ttype, face, tolerance=1e-6):
    sim = _make_single_phase_sim(bc_class, ttype, face)
    fin = _random_fin(sim.assign_fields_sharded().shape, sim.precision_policy.output_dtype)
    bc = sim.BCs[0]

    # Ground truth: the boundary condition's own (global-index) prepare_populations + apply on the streamed populations
    expected_full = reference_poststreaming(sim, bc, fin)
    _, actual_full = boundary_pass(sim, fin, fin)

    expected_bd = np.asarray(expected_full)[bc.indices]
    actual_bd = np.asarray(actual_full)[bc.indices]
    assert np.max(np.abs(actual_bd - expected_bd)) <= tolerance


def test_equilibrium_bc_matches_global_single_phase():
    # Pure scatter of a precomputed constant (no arithmetic in apply()): exact match.
    _check_single_phase(EquilibriumBC, None, "top", tolerance=0.0)


def test_zouhe_velocity_matches_global_single_phase():
    _check_single_phase(ZouHe, "velocity", "bottom")


def test_zouhe_pressure_matches_global_single_phase():
    _check_single_phase(ZouHe, "pressure", "left")


def test_regularized_velocity_matches_global_single_phase():
    _check_single_phase(Regularized, "velocity", "bottom")


def test_regularized_pressure_matches_global_single_phase():
    _check_single_phase(Regularized, "pressure", "left")


def _check_multiphase(bc_class, ttype, face, tolerance=1e-6):
    sim = _make_multiphase_sim(bc_class, ttype, face)
    fin_tree = sim.assign_fields_sharded()
    fin = _random_fin(fin_tree[0].shape, sim.precision_policy.output_dtype)
    bc = sim.BCs[0][0]

    expected_full = reference_poststreaming(sim, bc, fin)
    _, actual_full = boundary_pass(sim, fin, fin)

    diff = np.abs(np.asarray(actual_full)[bc.indices] - np.asarray(expected_full)[bc.indices])
    assert np.max(diff) <= tolerance


def test_equilibrium_bc_matches_global_multiphase():
    _check_multiphase(EquilibriumBC, None, "top", tolerance=0.0)


def test_zouhe_velocity_matches_global_multiphase():
    _check_multiphase(ZouHe, "velocity", "bottom")


def test_regularized_velocity_matches_global_multiphase():
    _check_multiphase(Regularized, "velocity", "bottom")
