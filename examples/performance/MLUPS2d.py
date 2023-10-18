"""
This script computes the MLUPS (Million Lattice Updates per Second) in 2D by simulating fluid flow inside a 2D cavity.
"""

import os
import argparse
import jax.numpy as jnp
import numpy as np
from jax.config import config
from time import time

from src.utils import *
from src.boundary_conditions import *
from src.lattice import LatticeD2Q9
from src.models import BGKSim

class Cavity(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    precision = 'f32/f32'
    lattice = LatticeD2Q9(precision)

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("N", help="The total number of voxels will be NxN", type=int)
    parser.add_argument("timestep", help="Number of timesteps", type=int)
    args = parser.parse_args()

    n = args.N
    max_iter = args.timestep
    Re = 100.0
    u_wall = 0.1
    clength = n - 1

    visc = u_wall * clength / Re
    omega = 1.0 / (3. * visc + 0.5)
    print('omega = ', omega)

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': n,
        'ny': n,
        'nz': 0,
        'precision': precision,
        'compute_MLUPS': True
    }
    
    os.system('rm -rf ./*.vtk && rm -rf ./*.png')
    sim = Cavity(**kwargs)
    sim.run(max_iter)
