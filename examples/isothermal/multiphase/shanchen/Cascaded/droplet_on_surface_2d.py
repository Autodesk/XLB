"""
Multi-component droplet on wall example useful for tuning contact angle parameters. The gas phase densities are changed to account for different air
mass fraction. The densities are changed by same amounts to keep total pressure constant.

The collision matrix is based on:
1. Fei, L., Derome, D. & Carmeliet, J. Pore-scale study on the effect of heterogeneity on evaporation in porous media.
Journal of Fluid Mechanics 983, A6 (2024).
"""

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, config
from jax.tree import map as tree_map

from jax_lab.core.boundary_conditions import BounceBack, ExactNonEquilibriumExtrapolation
from jax_lab.core.shanchen.eos import PengRobinson
from jax_lab.core.base import stack_components
from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.core.shanchen.multiphase import MultiphaseCascade
from jax_lab.core.utils import save_fields_vtk

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# config.update("jax_default_matmul_precision", "float32")


class DropletOnWall2D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2 - 100) ** 2)
        rho = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (rho_w_l - rho_w_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((nx, ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree = [rho]
        rho = 0.5 * (rho_a_l + rho_a_g) - 0.5 * (rho_a_l - rho_a_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((nx, ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree, T=None):
        rho_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)
        # Shan-Chen potential using the scaled pressure lambda_^2 * p (the library's convention; lambda_ replaces the old k = lambda_^2)
        psi_tree = tree_map(
            lambda lam, p, rho, G: jnp.sqrt(2 * (lam**2 * p - self.lattice.cs2 * rho) / G),
            self.lambda_,
            p_tree,
            rho_tree,
            self.g_kkprime.diagonal().tolist(),
        )
        psi_tree[1] = rho_tree[1]
        # Zhang-Chen potential
        return psi_tree

    @partial(jit, static_argnums=(0, 2))
    def compute_force(self, rho_tree, inter_component_interaction=True, T=None):
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree = self.compute_potential(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree)
        if inter_component_interaction:
            # -g_AB rho_0 sum_i G_ff_i c_i rho_1(x - c_i), as one fused neighbour stencil
            F_1 = -g_AB * rho_tree[0] * self.neighbor_stencil(rho_tree[1], np.asarray(self.G_ff)[None, :] * np.asarray(self.lattice.c))
            F_2 = F_1
            F_tree = [F_1, F_2]
            fluid_fluid_force = tree_map(lambda f1, f2: f1 + f2, fluid_fluid_force, F_tree)
        if self.body_force is not None:
            return tree_map(lambda ff, rho: ff + self.body_force * rho, fluid_fluid_force, rho_tree)
        else:
            return fluid_fluid_force

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F_tree, F_intra_tree, psi_tree):
        def f(F, F_intra, sigma, psi, s_b):
            Fx = F_intra[..., 0]
            Fy = F_intra[..., 1]
            eta = 4 * sigma * (Fx**2 + Fy**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))  # For mechanical stability
            Fx = F[..., 0]
            Fy = F[..., 1]
            # One fused pass (a chain of .at[..., k].set would be one pass per component, see stack_components)
            return stack_components(
                {1: Fx, 2: Fy, 3: eta, 6: Fy * self.lattice.cs2, 7: Fx * self.lattice.cs2, 8: eta * self.lattice.cs2},
                jax.ShapeDtypeStruct((self.nx, self.ny, self.lattice.q), self.precision_policy.compute_dtype),
            )

        return tree_map(lambda F, F_intra, sigma, psi, s_b: f(F, F_intra, sigma, psi, s_b), F_tree, F_intra_tree, self.sigma, psi_tree, self.s_b)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash_tree, rho_tree, u_tree, T=None):
        F_tree = self.compute_force(rho_tree)
        F_intra_tree = self.compute_force(rho_tree, inter_component_interaction=False)
        psi_tree = self.compute_potential(rho_tree)
        C_tree = self.compute_force_central_moments(F_tree, F_intra_tree, psi_tree)
        Tf_tree = tree_map(lambda S, C: jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * S), self.S, C_tree)
        return tree_map(lambda Tdash, Tf: Tdash + Tf, Tdash_tree, Tf_tree)

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None, T=None):
        p_tree = self.eos.EOS(rho_tree)
        p_tree[1] = self.lattice.cs2 * rho_tree[1]
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree):
        return p_tree[0] + p_tree[1] + g_AB * rho_tree[0] * rho_tree[1]

    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(
            BounceBack(tuple(ind.T), self.grid_info, self.precision_policy, theta_w[tuple(ind.T)], phi_w[tuple(ind.T)], delta_rho_w[tuple(ind.T)])
        )
        self.BCs[1].append(
            BounceBack(tuple(ind.T), self.grid_info, self.precision_policy, theta_a[tuple(ind.T)], phi_a[tuple(ind.T)], delta_rho_a[tuple(ind.T)])
        )

    def output_data(self, **kwargs):
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_tree"][1][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho_water": rho_water[..., 0], "rho_air": rho_air[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        save_fields_vtk(timestep, fields, "output_temp", "data")


class DropletOnWall2DGeometric(DropletOnWall2D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(
            BounceBack(
                tuple(ind.T),
                self.grid_info,
                self.precision_policy,
                theta_w[tuple(ind.T)],
            )
        )
        self.BCs[1].append(
            BounceBack(
                tuple(ind.T),
                self.grid_info,
                self.precision_policy,
                theta_a[tuple(ind.T)],
            )
        )


if __name__ == "__main__":
    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

    # Cascaded LBM collision matrix
    M = np.array([
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

    rho_w_l = 6.499210784
    rho_w_g = 0.379598891

    a = [3 / 49, 1]
    b = [2 / 21, 1]
    R = [1, 1]
    Tc = 0.1093785558
    T = 0.86 * Tc

    kwargs = {"a": a, "b": b, "pr_omega": [0.344, 1], "R": R, "T": T}
    eos = PengRobinson(**kwargs)

    s2 = 0.8  # This sets the kinematic viscosity
    s_0 = [1.0, 1.0]  # Mass conservation
    s_1 = [1.0, 1.0]  # Fixed
    s_2 = [s2, s2]
    s_b = [0.8, 0.8]
    s_3 = [0.8, 0.8]  # [(16 - 8 * s2) / (8 - s2), (16 - 8 * s2) / (8 - s2)]  # No slip
    s_4 = [1.0, 1.0]  # No slip

    g_kkprime = -1 * np.ones((2, 2))
    g_kkprime[1, 1] = 1
    g_kkprime[0, 1] = 0
    g_kkprime[1, 0] = 0
    g_AB = -0.15

    precision = "f32/f32"

    r = 50
    width = 5
    nx = 300
    ny = 350
    theta_w = 80 * np.pi / 180 * np.ones((nx, ny, 1))
    phi_w = 1.2 * np.ones((nx, ny, 1))
    delta_rho_w = 0.0 * np.ones((nx, ny, 1))
    theta_a = 100 * np.pi / 180 * np.ones((nx, ny, 1))
    phi_a = 0.0 * np.ones((nx, ny, 1))
    delta_rho_a = 0.2 * np.ones((nx, ny, 1))

    # Water
    rho_w_l = 6.499210784
    rho_w_g = 0.379598891
    # Air
    rho_a_l = 0.0019
    rho_a_g = 0.0019
    # Modify gas phase densities for different vapor fraction
    rho_w_g = rho_w_g - 0.0003
    rho_a_g = rho_a_g + 0.0003

    # Circular wall
    R = 70
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    x, y = np.meshgrid(x, y)
    x = x.T
    y = y.T
    circ = (x - nx / 2) ** 2 + (y - ny / 2 + 20) ** 2 - R**2
    ind = np.array(np.where(circ <= 0), dtype=int).T
    kwargs = {
        "n_components": 2,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": g_kkprime,
        "body_force": [0.0, 0.0],
        "EOS": eos,
        "lambda_": [1.0, 1.0],
        "M": [M, M],
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "sigma": [0.099, 0.0],
        "wetting_formulation": "geometric",
        "precision": precision,
        "io_rate": 10000,
        "print_info_rate": 10000,
        "compute_MLUPS": False,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    # os.system("rm -rf output*/ *.vtk")
    sim = DropletOnWall2DGeometric(**kwargs)
    sim.run(40000)
