"""
Single component 3D bubble rise example. A light vapor bubble is immersed near the bottom of a tall liquid column and rises under buoyancy.
The coexistence densities are computed using Maxwell's Construction (Van der Waals EOS). Gravity is applied as a Boussinesq-style buoyancy
force based on the density deviation from the domain average, and is held off for an initial relaxation period so the sharp initial interface
can settle before buoyancy starts driving the flow. Boundary conditions are no-slip (bounce-back) at the top and bottom walls and periodic
along x and y.

The buoyancy formulation and staged-gravity relaxation follow the same pattern validated in examples/thermal/multiphase/shanchen/MRT/pool_boiling_3d.py.

The collision matrix is based on:
1. Coveney, P. V. et al. Multiple-relaxation-time lattice Boltzmann models in three dimensions. Philosophical Transactions of the Royal Society of
London. Series A: Mathematical, Physical and Engineering Sciences 360, 437-451 (2002).
"""

import os
import subprocess
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree import map as tree_map

from jax_lab.core.base import _lattice_dot
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.shanchen.multiphase import MultiphaseMRT
from jax_lab.core.shanchen.eos import VanderWaals
from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.utils import save_fields_hdf5_xdmf

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# from jax_lab.core.utils import save_fields_vtk


class BubbleRise3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.arange(self.nx, dtype=np.float32)[:, None, None]
        y = np.arange(self.ny, dtype=np.float32)[None, :, None]
        z = np.arange(self.nz, dtype=np.float32)[None, None, :]

        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - bubble_z_center) ** 2)
        # Vapor (rho_g) inside the bubble, liquid (rho_l) filling the rest of the column.
        rho = 0.5 * (rho_g + rho_l) - 0.5 * (rho_g - rho_l) * np.tanh(2 * (dist - bubble_radius) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree = [rho]

        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        walls = np.concatenate((self.bounding_box_indices["bottom"], self.bounding_box_indices["top"]))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self.grid_info, self.precision_policy))

    @partial(jit, static_argnums=(0,))
    def compute_buoyancy_force(self, rho_tree, timestep):
        """
        Boussinesq-style buoyancy: only the density deviation from the domain average drives motion, so the bulk
        liquid column does not itself get hydrostatically compressed against the walls. Held off for the first
        `gravity_relaxation_steps` so the sharp initial interface can relax before gravity is applied.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field.

        timestep (int): Current timestep.

        Returns
        -------
        (pytree of jax.numpy.ndarray): Buoyancy force field.
        """
        rho_average = self.compute_total_density(rho_tree).mean()
        gravity_vector = jnp.array([0.0, 0.0, -gravity], dtype=self.precision_policy.compute_dtype)
        gravity_active = jnp.asarray(timestep > gravity_relaxation_steps, dtype=self.precision_policy.compute_dtype)
        return tree_map(lambda rho: gravity_active * (rho - rho_average) * gravity_vector, rho_tree)

    @partial(jit, static_argnums=(0,))
    def macroscopic_velocity(self, f_tree, rho_tree, T=None, timestep=None):
        u_tree = super().macroscopic_velocity(f_tree, rho_tree, T=T)
        if timestep is None:
            timestep = gravity_relaxation_steps + 1
        buoyancy_tree = self.compute_buoyancy_force(rho_tree, timestep)
        return tree_map(lambda u, force, rho: u + 0.5 * force / rho, u_tree, buoyancy_tree, rho_tree)

    # post_collision returns stack_populations stacks (see LBMBase.post_collision)
    post_collision_per_direction = True

    def post_collision(self, f_postcollision_tree, f_poststreaming_tree, timestep):
        """
        Buoyancy by exact-difference forcing, f* += feq(rho, u + F / rho) - feq(rho, u) with F from compute_buoyancy_force,
        then the rest population restores the density, f*_0 = rho - sum_(i>0) f*_i. Evaluated per lattice direction and
        stacked with stack_populations, so XLA fuses the update into one kernel.
        """
        directions, weights = self._lattice_directions, self._lattice_weights
        rho_tree, u_tree = self.update_macroscopic(f_poststreaming_tree)
        buoyancy_tree = self.compute_buoyancy_force(rho_tree, timestep)
        out_tree = []
        for f_post, rho, u, force in zip(f_postcollision_tree, rho_tree, u_tree, buoyancy_tree, strict=True):
            rho = rho[..., 0]
            u = [u[..., axis] for axis in range(self.dim)]
            u_forced = [v + force[..., axis] / rho for axis, v in enumerate(u)]
            usqr = 1.5 * sum(v * v for v in u)
            usqr_forced = 1.5 * sum(v * v for v in u_forced)
            out = []
            for i, (direction, w) in enumerate(zip(directions, weights, strict=True)):
                cu = 3.0 * _lattice_dot(direction, u)
                cu_forced = 3.0 * _lattice_dot(direction, u_forced)
                feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
                feq_forced = rho * w * (1.0 + cu_forced * (1.0 + 0.5 * cu_forced) - usqr_forced)
                out.append(self.precision_policy.cast_to_compute(f_post[..., i]) + feq_forced - feq)
            out[0] = rho - sum(out[1:])
            out_tree.append(self.stack_populations([self.precision_policy.cast_to_output(o) for o in out]))
        return out_tree

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}

        vapor_mask = rho[..., 0] < 0.5 * (rho_l + rho_g)
        vapor_fraction = float(np.mean(vapor_mask))
        column_counts = vapor_mask.sum(axis=(0, 1))
        centroid_z = float(np.average(np.arange(self.nz), weights=column_counts)) if column_counts.sum() > 0 else float("nan")
        logger.info(f"timestep {timestep}: vapor fraction = {vapor_fraction:.4f}, bubble centroid z = {centroid_z:.2f}")

        save_fields_hdf5_xdmf(timestep, fields, "output", "data")
        # save_fields_vtk(timestep, fields, "output", "data")


if __name__ == "__main__":
    nx = 64
    ny = 64
    nz = 128

    width = 4  # Initial liquid-vapor interface thickness
    bubble_radius = 20
    bubble_z_center = nz // 4  # Start near the bottom, leaving room in the column for it to rise.
    gravity_relaxation_steps = 1000

    e = LatticeD3Q19().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((19, 19))
    M[0, :] = en**0
    M[1, :] = 19 * en**2 - 30
    M[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    M[3, :] = e[:, 0]
    M[4, :] = (5 * en**2 - 9) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (5 * en**2 - 9) * e[:, 1]
    M[7, :] = e[:, 2]
    M[8, :] = (5 * en**2 - 9) * e[:, 2]
    M[9, :] = 3 * e[:, 0] ** 2 - en**2
    M[10, :] = (3 * en**2 - 5) * (3 * e[:, 0] ** 2 - en**2)
    M[11, :] = e[:, 1] ** 2 - e[:, 2] ** 2
    M[12, :] = (3 * en**2 - 5) * (e[:, 1] ** 2 - e[:, 2] ** 2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1] ** 2 - e[:, 2] ** 2) * e[:, 0]
    M[17, :] = (e[:, 2] ** 2 - e[:, 0] ** 2) * e[:, 1]
    M[18, :] = (e[:, 0] ** 2 - e[:, 1] ** 2) * e[:, 2]

    # Same Van der Waals coexistence point validated for compact interfaces in droplet_3d.py; the gravity magnitude
    # is the one validated for buoyancy-driven flow in pool_boiling_3d.py.
    a = 9 / 49
    b = 2 / 21
    R = 1.0
    Tc = 0.5714285714
    T = 0.8 * Tc
    rho_l = 6.764470400
    rho_g = 0.838834226

    gravity = 3e-5
    shear_relaxation = 1.25

    eos = VanderWaals(a=a, b=b, R=R, T=T)

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "body_force": None,
        "lambda_": [0.27**0.5],  # k = lambda_**2 = 0.27
        "s_rho": [0.0],
        "s_e": [0.8],
        "s_eta": [1.0],
        "s_j": [0.0],
        "s_q": [1.0],
        "s_pi": [1.0],
        "s_m": [1.0],
        "s_v": [shear_relaxation],
        "M": [M],
        "kappa": [0.0],
        "precision": precision,
        "io_rate": 1000,
        "compute_MLUPS": False,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    subprocess.run("rm -rf output*/ *.vtk *.h5 *.xmf", shell=True, check=True)
    sim = BubbleRise3D(**kwargs)
    sim.run(13000)
