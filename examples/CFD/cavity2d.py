"""
This example implements a 2D Lid-Driven Cavity Flow simulation using the lattice Boltzmann method (LBM). 
The Lid-Driven Cavity Flow is a standard test case for numerical schemes applied to fluid dynamics, which involves fluid in a square cavity with a moving lid (top boundary).

In this example you'll be introduced to the following concepts:

1. Lattice: The simulation employs a D2Q9 lattice. It's a 2D lattice model with nine discrete velocity directions, which is typically used for 2D simulations.

2. Boundary Conditions: The code implements two types of boundary conditions:

    BounceBackHalfway: This condition is applied to the stationary walls (left, right, and bottom). It models a no-slip boundary where the velocity of fluid at the wall is zero.
    EquilibriumBC: This condition is used for the moving lid (top boundary). It defines a boundary with a set velocity, simulating the "driving" of the cavity by the lid.

3. Checkpointing: The simulation supports checkpointing. Checkpoints are saved periodically (determined by the 'checkpoint_rate'), allowing the simulation to be stopped and restarted from the last checkpoint. This can be beneficial for long simulations or in case of unexpected interruptions.

4. Visualization: The simulation outputs data in VTK format for visualization. It also provides images of the velocity field and saves the boundary conditions at each time step. The data can be visualized using software like Paraview.

"""
from jax.config import config
import numpy as np
import jax.numpy as jnp
import os

from src.boundary_conditions import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9
from src.utils import *

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

class Cavity(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices["top"]

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][1:-1, 1:-1])
        u = np.array(kwargs["u"][1:-1, 1:-1, :])
        timestep = kwargs["timestep"]

        save_image(timestep, u)
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
        save_fields_vtk(timestep, fields)
        save_BCs_vtk(timestep, self.BCs, self.gridInfo)

if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    nx = 200
    ny = 200

    Re = 200.0
    prescribed_vel = 0.1
    clength = nx - 1

    checkpoint_rate = 1000
    checkpoint_dir = os.path.abspath("./checkpoints")

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 100,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir,
        'restore_checkpoint': True,
    }

    sim = Cavity(**kwargs)
    sim.run(5000)
