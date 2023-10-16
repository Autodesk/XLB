"""
This example implements a 3D Lid-Driven Cavity Flow simulation using the lattice Boltzmann method (LBM). 
The Lid-Driven Cavity Flow is a standard test case for numerical schemes applied to fluid dynamics, which involves fluid in a square cavity with a moving lid (top boundary).

In this example you'll be introduced to the following concepts:

1. Lattice: The simulation employs a D2Q9 lattice. It's a 2D lattice model with nine discrete velocity directions, which is typically used for 2D simulations.

2. Boundary Conditions: The code implements two types of boundary conditions:

    BounceBack: This condition is applied to the stationary walls, except the top wall. It models a no-slip boundary where the velocity of fluid at the wall is zero.
    EquilibriumBC: This condition is used for the moving lid (top boundary). It defines a boundary with a set velocity, simulating the "driving" of the cavity by the lid.

4. Visualization: The simulation outputs data in VTK format for visualization. The data can be visualized using software like Paraview.

"""

import os

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import numpy as np
from src.utils import *
from jax.config import config

from src.models import BGKSim, KBCSim
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.boundary_conditions import *

class Cavity(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate(
            (self.boundingBoxIndices['left'], self.boundingBoxIndices['right'],
             self.boundingBoxIndices['front'], self.boundingBoxIndices['back'],
             self.boundingBoxIndices['bottom']))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices['top']

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

    def output_data(self, **kwargs):
        # 1: -1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs['rho'][1:-1, 1:-1, 1:-1])
        u = np.array(kwargs['u'][1:-1, 1:-1, 1:-1, :])
        timestep = kwargs['timestep']
        u_prev = kwargs['u_prev'][1:-1, 1:-1, 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}'.format(err))
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(timestep, fields)
        # Calculate the velocity magnitude
        u_mag = np.linalg.norm(u, axis=2)
        # live_volume_randering(timestep, u_mag)

if __name__ == '__main__':
    nx = 101
    ny = 101
    nz = 101

    Re = 50000.0
    prescribed_vel = 0.1
    clength = nx - 1

    precision = 'f32/f32'
    lattice = LatticeD3Q27(precision)

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3. * visc + 0.5)
    
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 100,
        'downsampling_factor': 2
    }
    sim = Cavity(**kwargs)
    sim.run(2000)