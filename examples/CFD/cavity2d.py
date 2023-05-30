from src.boundary_conditions import *
from jax.config import config
from src.utils import *
import numpy as np
from src.lattice import LatticeD2Q9
from src.models import BGKSim, KBCSim
import jax.numpy as jnp
import os

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
jax.config.update("jax_array", True)
# disable JIt compilation
# jax.config.update('jax_disable_jit', True)


precision = "f32/f32"

class Cavity(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.downsampling_factor = 2
        self.error_report_rate=1000
        self.io_rate=1000

    def set_boundary_conditions(self):

        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices["top"]

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = u_wall
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
    lattice = LatticeD2Q9(precision)

    nx = 301
    ny = 301

    Re = 2000.0
    u_wall = 0.1
    clength = nx - 1

    checkpoint_rate = 100
    checkpoint_dir = "./checkpoints"

    visc = u_wall * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    print("omega = ", omega)
    assert omega < 2.0, "omega must be less than 2.0"
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': 301,
        'ny': 301,
        'nz': 0,
        'precision': precision,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir
    }

    sim = Cavity(**kwargs)
    sim.run(3000, restore_checkpoint=True)
