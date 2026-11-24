"""
Evaporating hemispherical droplet on a bottom wall with geometric wetting.

The top surface is open to the environment using the multiphase extrapolation
outflow boundary condition. Contact angle and liquid volume are written as the
droplet evaporates.

The collision matrix is based on:
1. Fei, L., Derome, D. & Carmeliet, J. Pore-scale study on the effect of heterogeneity on evaporation in porous media.
Journal of Fluid Mechanics 983, A6 (2024).
"""

import csv
import os
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import config, jit
from jax.tree import map as tree_map
from skimage.measure import find_contours

from jax_lab.core.boundary_conditions import BounceBack, ExtrapolationOutflowMultiphase
from jax_lab.core.shanchen.eos import PengRobinson
from jax_lab.core.base import stack_components
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.shanchen.multiphase import MultiphaseCascade
from jax_lab.core.utils import save_fields_hdf5_xdmf

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

config.update("jax_default_matmul_precision", "float32")


class DropletEvaporationHysteresis(MultiphaseCascade):
    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value=None):
        self._k = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value=None):
        self._A = value

    def initialize_macroscopic_fields(self):
        dist_sq = (x_coord - droplet_center_x) ** 2 + (y_coord - droplet_center_y) ** 2 + (z_coord - droplet_center_z) ** 2
        rho = np.empty((self.nx, self.ny, self.nz, 1), dtype=np.float32)
        rho[..., 0] = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (rho_w_l - rho_w_g) * np.tanh(2 * (np.sqrt(dist_sq) - droplet_radius) / interface_width)
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.precision_policy.cast_to_output(u)
        return [rho], [u]

    def set_boundary_conditions(self):
        top = self.bounding_box_indices["top"]
        bottom = self.bounding_box_indices["bottom"]
        self.BCs[0].append(ExtrapolationOutflowMultiphase(tuple(top.T), self.grid_info, self.precision_policy))
        self.BCs[0].append(BounceBack(tuple(bottom.T), self.grid_info, self.precision_policy, theta_wall[tuple(bottom.T)]))

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree):
        # F_k = G_kk psi_k sum_i G_ff_i c_i psi_k(x - c_i), as one fused neighbour stencil
        weights = np.asarray(self.G_ff)[None, :] * np.asarray(self.lattice.c)
        return [G * psi * self.neighbor_stencil(psi, weights) for G, psi in zip(self.g_kkprime.diagonal().tolist(), psi_tree, strict=True)]

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree, T=None):
        rho_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)
        psi_tree = tree_map(lambda p, rho, G: jnp.sqrt(2 * (p - self.lattice.cs2 * rho) / G), p_tree, rho_tree, self.g_kkprime.diagonal().tolist())
        return psi_tree

    @partial(jit, static_argnums=(0, 2))
    def compute_force(self, rho_tree, inter_component_interaction=True, T=None):
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree = self.compute_potential(rho_tree)
        force_tree = self.compute_fluid_fluid_force(psi_tree)
        if self.body_force is not None:
            force_tree = tree_map(lambda force, rho: force + self.body_force * rho, force_tree, rho_tree)
        if self.wetting_formulation == "geometric":
            force_tree = tree_map(lambda force, fluid_mask: force * fluid_mask, force_tree, self.geometric_fluid_mask)
        return force_tree

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F_tree, psi_tree):
        def f(F, sigma, psi, s_b):
            Fx = F[..., 0]
            Fy = F[..., 1]
            Fz = F[..., 2]
            eta = 2 * sigma * (Fx**2 + Fy**2 + Fz**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))  # For mechanical stability
            # One fused pass (a chain of .at[..., k].set would be one pass per component, see stack_components)
            return stack_components(
                {
                    1: Fx,
                    2: Fy,
                    3: Fz,
                    7: eta,
                    8: eta,
                    9: eta,
                    10: Fx * self.lattice.cs2,
                    11: Fx * self.lattice.cs2,
                    12: Fy * self.lattice.cs2,
                    13: Fz * self.lattice.cs2,
                    14: Fy * self.lattice.cs2,
                    15: Fz * self.lattice.cs2,
                },
                jax.ShapeDtypeStruct((self.nx, self.ny, self.nz, self.lattice.q), self.precision_policy.compute_dtype),
            )

        return tree_map(f, F_tree, self.sigma, psi_tree, self.s_b)

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        rho_scalar = rho[..., 0]

        fields = {"rho_water": rho_scalar[...], "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}
        mask = np.array(self.solid_mask_streamed[0][..., 0])
        save_fields_hdf5_xdmf(timestep, fields, output_dir, "data", static_fields={"mask": mask})

        contact_angle = measure_contact_angle(rho_scalar)
        liquid_volume = np.sum((rho_scalar >= interface_level) & (~mask.astype(bool)))
        liquid_mass = np.sum(rho_scalar[~mask.astype(bool)], dtype=np.float64)
        contact_angle_writer.writerow([timestep, contact_angle, liquid_volume, liquid_mass])
        contact_angle_file.flush()

        logger.info(f"Contact angle: {contact_angle}")
        logger.info(f"Liquid volume: {liquid_volume}")
        logger.info(f"Liquid mass: {liquid_mass}")


def measure_contact_angle(rho_scalar):
    section = rho_scalar[:, droplet_center_y, :]
    contours = find_contours(section, level=interface_level)
    if not contours:
        return np.nan

    contour = max(contours, key=len)
    points = np.column_stack((contour[:, 0], contour[:, 1]))
    fit_points = points[points[:, 1] >= contact_fit_zmin]
    if fit_points.shape[0] < min_fit_points:
        return np.nan

    circle = fit_circle(fit_points)
    if circle is None:
        return np.nan
    center, radius = circle
    return float(np.degrees(np.arccos(np.clip(-center[1] / radius, -1.0, 1.0))))


def fit_circle(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return None
    A = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(points.shape[0])))
    b = np.sum(points * points, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    radius_sq = sol[2] + np.dot(sol[:2], sol[:2])
    if radius_sq <= 0.0:
        return None
    return sol[:2], np.sqrt(radius_sq)


if __name__ == "__main__":
    e = LatticeD3Q19().c.T
    ex = e[:, 0]
    ey = e[:, 1]
    ez = e[:, 2]
    M = np.zeros((19, 19))
    M[0, :] = ex**0
    M[1, :] = ex
    M[2, :] = ey
    M[3, :] = ez
    M[4, :] = ex * ey
    M[5, :] = ex * ez
    M[6, :] = ey * ez
    M[7, :] = ex * ex
    M[8, :] = ey * ey
    M[9, :] = ez * ez
    M[10, :] = ex * ey * ey
    M[11, :] = ex * ez * ez
    M[12, :] = ey * ex * ex
    M[13, :] = ez * ex * ex
    M[14, :] = ey * ez * ez
    M[15, :] = ez * ey * ey
    M[16, :] = ex * ex * ey * ey
    M[17, :] = ex * ex * ez * ez
    M[18, :] = ey * ey * ez * ez

    precision = "f32/f32"
    nx = 250
    ny = 200
    nz = 200
    droplet_radius = 45
    prescribed_contact_angle = 60.0 * np.pi / 180.0
    interface_width = 4
    output_dir = "droplet_evaporation_hysteresis"

    rho_w_l = 6.499210784
    rho_w_g = 0.379598891
    interface_level = 0.5 * (rho_w_l + rho_w_g)
    contact_fit_zmin = 3.0
    min_fit_points = 30

    droplet_center_x = nx // 2
    droplet_center_y = ny // 2
    droplet_center_z = -droplet_radius * np.cos(prescribed_contact_angle)
    droplet_base_radius = droplet_radius * np.sin(prescribed_contact_angle)
    x_coord = np.arange(nx, dtype=np.float32)[:, None, None]
    y_coord = np.arange(ny, dtype=np.float32)[None, :, None]
    z_coord = np.arange(nz, dtype=np.float32)[None, None, :]

    s2 = 0.8
    s_0 = [1.0]
    s_1 = [1.0]
    s_2 = [s2]
    s_b = [0.8]
    s_3 = [(16 - 8 * s2) / (8 - s2)]
    s_4 = [1.0]

    g_kkprime = -1 * np.ones((1, 1))
    eos = PengRobinson(a=[3 / 49], b=[2 / 21], pr_omega=[0.344], R=[1.0], T=0.86 * 0.1093785558)
    theta_wall = prescribed_contact_angle * np.ones((nx, ny, nz, 1))

    subprocess.run(["rm", "-rf", output_dir], check=True)
    Path(output_dir).mkdir(exist_ok=True)
    contact_angle_file = (Path(output_dir) / "contact_angles.csv").open("w", newline="", encoding="utf-8")
    contact_angle_writer = csv.writer(contact_angle_file)
    contact_angle_writer.writerow(["timestep", "contact_angle_deg", "liquid_volume_cells", "liquid_mass"])
    contact_angle_file.flush()

    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": g_kkprime,
        # Scale factor of the pseudopotential (psi = sqrt(2 (lambda_^2 p - cs2 rho) / g)); 1.0 matches this example's
        # compute_potential, which applies no pressure scaling
        "lambda_": [1.0],
        "EOS": eos,
        "M": [M],
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "body_force": [0, 0, -2.5e-5],
        "sigma": [0.102],
        "precision": precision,
        "io_rate": 100,
        "print_info_rate": 100,
        "compute_MLUPS": False,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
        "wetting_formulation": "geometric",
    }

    try:
        sim = DropletEvaporationHysteresis(**kwargs)
        sim.run(4000)
    finally:
        contact_angle_file.close()
