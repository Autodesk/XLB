"""
This script performs a 2D simulation of Couette flow using the lattice Boltzmann method (LBM). 
"""

import os
import jax.numpy as jnp
import numpy as np
from src.utils import *
from jax.config import config


from src.models import BGKSim
from src.boundary_conditions import *
from src.lattice import LatticeD2Q9

# config.update('jax_disable_jit', True)
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

class Couette(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        outlet = self.boundingBoxIndices["right"]
        inlet = self.boundingBoxIndices["left"]

        rho_wall = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

        self.BCs.append(DoNothing(tuple(outlet.T), self.gridInfo, self.precisionPolicy))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][..., 1:-1, :])
        u = np.array(kwargs["u"][..., 1:-1, :])
        timestep = kwargs["timestep"]
        u_prev = kwargs["u_prev"][..., 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)
        err = np.sum(np.abs(u_old - u_new))
        print("error= {:07.6f}".format(err))
        save_image(timestep, u)
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
        save_fields_vtk(timestep, fields)

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)
    nx = 501
    ny = 101

    Re = 100.0
    prescribed_vel = 0.1
    clength = nx - 1

    visc = prescribed_vel * clength / Re

    omega = 1.0 / (3.0 * visc + 0.5)
    assert omega < 1.98, "omega must be less than 2.0"
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 100
        }
    sim = Couette(**kwargs)
    sim.run(20000)
