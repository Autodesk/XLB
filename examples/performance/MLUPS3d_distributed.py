"""
This script computes the MLUPS (Million Lattice Updates per Second) in 3D by simulating fluid flow inside a 2D cavity. 
This script is equivalent to MLUPS3d.py, but uses JAX distributed to run the simulation on distributed systems (multi-host, multi-GPUs).
Please refer to https://jax.readthedocs.io/en/latest/multi_process.html for more information on JAX distributed.
"""


# Standard Libraries
import argparse
import os
import jax
# Initialize JAX distributed. The IP, number of processes and process id must be updated.
# Currently set on local host for testing purposes. 
# Can be tested on a two GPU system as follows: 
# (export PYTHONPATH=.; CUDA_VISIBLE_DEVICES=0 python3 examples/performance/MLUPS3d_distributed.py 100 100 & CUDA_VISIBLE_DEVICES=1 python3 examples/performance/MLUPS3d_distributed.py 100 100 &)
#IMPORTANT: jax distributed must be initialized before any jax computation is performed
jax.distributed.initialize(f'127.0.0.1:1234', 2, process_id=int(os.environ['CUDA_VISIBLE_DEVICES']))

print('Process id: ', jax.process_index())
print('Number of total devices (over all processes): ', jax.device_count())
print('Number of local devices:', jax.local_device_count())


import jax.numpy as jnp
import numpy as np

from jax.config import config

from src.boundary_conditions import *
from src.models import BGKSim
from src.lattice import LatticeD3Q19
from src.utils import *

#config.update('jax_disable_jit', True)
# Use 8 CPU devices
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
#config.update("jax_enable_x64", True)

class Cavity(BGKSim):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices['left'], self.boundingBoxIndices['right'], self.boundingBoxIndices['bottom'], self.boundingBoxIndices['front'], self.boundingBoxIndices['back']))
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
    lattice = LatticeD3Q19(precision)

    # Create a parser that will read the command line arguments
    parser = argparse.ArgumentParser("Calculate MLUPS for a 3D cavity flow simulation")
    parser.add_argument("N", help="The total number of voxels in one direction. The final dimension will be N*NxN", default=100, type=int)
    parser.add_argument("N_ITERS", help="Number of timesteps", default=10000, type=int)    

    args = parser.parse_args()
    n = args.N
    n_iters = args.N_ITERS

    # Store the Reynolds number in the variable Re
    Re = 100.0
    # Store the velocity of the lid in the variable u_wall
    u_wall = 0.1
    # Store the length of the cavity in the variable clength
    clength = n - 1

    # Compute the viscosity from the Reynolds number, the lid velocity, and the length of the cavity
    visc = u_wall * clength / Re
    # Compute the relaxation parameter from the viscosity
    omega = 1.0 / (3. * visc + 0.5)
    
    # Create a new instance of the Cavity class
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': n,
        'ny': n,
        'nz': n,
        'precision': precision,
        'compute_MLUPS': True
    }

    sim = Cavity(**kwargs)    # Run the simulation
    sim.run(n_iters)