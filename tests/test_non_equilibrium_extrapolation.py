"""Verify the time step's extrapolation boundary conditions against their global-index computations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _boundary_pass import boundary_pass, reference_poststreaming

from jax_lab.core.boundary_conditions import BounceBack, ExactNonEquilibriumExtrapolation, NonEquilibriumExtrapolation
from jax_lab.core.shanchen.eos import VanderWaals
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.models import BGKSim
from jax_lab.core.shanchen.multiphase import MultiphaseBGK

DOMAIN = (16, 16, 16)
PRECISION = "f32/f32"
OMEGA = 1.0
SEED = 0


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


def _make_sim(bc_class):
    class Sim(BGKSim):
        def set_boundary_conditions(self):
            bottom = self.bounding_box_indices["bottom"]
            self.BCs.append(BounceBack(tuple(bottom.T), self.grid_info, self.precision_policy))
            top = self.bounding_box_indices["top"]
            prescribed_rho = 0.9 * jnp.ones((top.shape[0], 1))
            if bc_class is ExactNonEquilibriumExtrapolation:
                self.BCs.append(bc_class(tuple(top.T), self.grid_info, self.precision_policy, prescribed_rho, "density"))
            else:
                self.BCs.append(bc_class(tuple(top.T), self.grid_info, self.precision_policy, prescribed_rho))

    return Sim(**_base_kwargs())


def _random_fin(sim):
    fin = sim.assign_fields_sharded()
    rng = np.random.default_rng(SEED)
    return fin + jnp.asarray(rng.uniform(-0.05, 0.05, size=fin.shape), dtype=fin.dtype)


def test_local_non_equilibrium_extrapolation_matches_global_application():
    sim = _make_sim(NonEquilibriumExtrapolation)
    fin = _random_fin(sim)
    bc = sim.BCs[1]

    expected = reference_poststreaming(sim, bc, fin)
    _, fout = boundary_pass(sim, fin, fin)

    np.testing.assert_allclose(np.asarray(fout)[bc.indices], np.asarray(expected)[bc.indices], rtol=0.0, atol=1e-6)


def test_exact_non_equilibrium_extrapolation_matches_prescribed_density():
    sim = _make_sim(ExactNonEquilibriumExtrapolation)
    fin = _random_fin(sim)

    _, fout = boundary_pass(sim, fin, fin)

    top = sim.bounding_box_indices["top"]
    boundary_rho = np.asarray(jnp.sum(fout, axis=-1, keepdims=True))[tuple(top.T)]
    assert np.allclose(boundary_rho, 0.9, atol=1e-5)


@pytest.mark.skipif(jax.device_count() < 2, reason="Multiple devices required for local extrapolation-index coverage")
def test_local_exact_extrapolation_supports_multiple_components():
    class Sim(MultiphaseBGK):
        def set_boundary_conditions(self):
            for component, face, density in ((0, "top", 0.9), (1, "bottom", 1.1)):
                indices = self.bounding_box_indices[face]
                prescribed = density * jnp.ones((indices.shape[0], 1))
                self.BCs[component].append(
                    ExactNonEquilibriumExtrapolation(tuple(indices.T), self.grid_info, self.precision_policy, prescribed, "density")
                )

    nx, ny, nz = DOMAIN
    sim = Sim(
        lattice=LatticeD3Q19(PRECISION),
        omega=[OMEGA, OMEGA],
        nx=nx,
        ny=ny,
        nz=nz,
        precision=PRECISION,
        io_rate=0,
        print_info_rate=0,
        checkpoint_rate=0,
        n_components=2,
        g_kkprime=np.array([[-1.0, -0.2], [-0.2, -1.0]]),
        lambda_=[1.0, 1.0],
        EOS=VanderWaals(a=[9.0 / 49.0] * 2, b=[2.0 / 21.0] * 2, R=[1.0, 1.0], T=0.8 * 0.5714285714),
    )
    fin_tree = sim.assign_fields_sharded()
    expected_tree = [reference_poststreaming(sim, sim.BCs[component][0], fin) for component, fin in enumerate(fin_tree)]
    fout_tree = [boundary_pass(sim, fin, fin, component)[1] for component, fin in enumerate(fin_tree)]
    for component, face in ((0, "top"), (1, "bottom")):
        indices = tuple(sim.bounding_box_indices[face].T)
        assert np.allclose(np.asarray(fout_tree[component])[indices], np.asarray(expected_tree[component])[indices], atol=1e-6)
