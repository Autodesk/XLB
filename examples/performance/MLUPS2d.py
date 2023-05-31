"""
This script computes the MLUPS (Million Lattice Updates per Second) in 2D by simulating fluid flow inside a 2D cavity.
"""

import os

from src.models import BGKSim
from src.lattice import LatticeD2Q9

#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax.numpy as jnp
import numpy as np
from src.utils import *
from jax.config import config
from time import time
import argparse
from src.boundary_conditions import *

#config.update('jax_disable_jit', True)
# Use 8 CPU devices

precision = 'f32/f32'


class Cavity(BGKSim):

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices['left'], self.boundingBoxIndices['right'], self.boundingBoxIndices['bottom']))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices['top']

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = u_wall
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))


if __name__ == '__main__':

    lattice = LatticeD2Q9(precision)

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("N", help="The total number of voxels will be NxN", type=int)
    parser.add_argument("timestep", help="Number of timesteps", type=int)
    args = parser.parse_args()

    N = args.N
    max_iter = args.timestep
    Re = 100.0
    u_wall = 0.1
    clength = N - 1

    visc = u_wall * clength / Re
    omega = 1.0 / (3. * visc + 0.5)
    print('omega = ', omega)
    assert omega < 2.0, "omega must be less than 2.0"
    os.system('rm -rf ./*.vtk && rm -rf ./*.png')
    sim = Cavity(lattice, omega, N, N, precision=precision)
    sim.run(max_iter, MLUPS=True)
