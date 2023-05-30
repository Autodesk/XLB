from src.models import BGKSim
from src.boundary_conditions import *
from src.lattice import LatticeD2Q9
import jax.numpy as jnp
import numpy as np
from src.utils import *
from jax.config import config
import os

# config.update('jax_disable_jit', True)
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

precision = "f32/f32"

class ChannelFlow2D(BGKSim):

    def set_boundary_conditions(self):
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))

        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        left_wall = self.boundingBoxIndices["left"]
        right_wall = self.boundingBoxIndices["right"]

        # Set the input boundary to 1/4 of the left wall
        inlet_quarter = int(left_wall.shape[0] / 4)
        inlet = left_wall[:inlet_quarter]

        # Set the output boundary to 1/4 of the right wall
        outlet_quarter = int(3 * right_wall.shape[0] / 4)
        outlet = right_wall[outlet_quarter:]
        left_wall = left_wall[inlet_quarter:]
        right_wall = right_wall[:outlet_quarter]

        rho_wall = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = u_wall
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))

        self.BCs.append(DoNothing(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        # Append the rest of the left and right walls to the bounceback
        walls = np.concatenate((walls, left_wall, right_wall))
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))


    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][1:-1, 1:-1, :])
        u = np.array(kwargs["u"][1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        u_prev = kwargs["u_prev"][1:-1, 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)
        err = np.sum(np.abs(u_old - u_new))
        print("error= {:07.6f}".format(err))
        save_image(timestep, u)
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
        save_fields_vtk(timestep, fields)

    def initialize_macroscopic_fields(self):
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.output_dtype, initVal=1.0, sharding=self.sharding)
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.output_dtype, initVal=0.0, sharding=self.sharding)
        return rho, u
    
    def get_force(self):
        force = np.zeros((self.nx, self.ny, 2), dtype=self.precisionPolicy.compute_dtype)
        values = np.linspace(-0.0002, -0.0002, self.nx - 2)
        force[1:-1, 1:-1, 1] = values[:, np.newaxis]
        return force


if __name__ == "__main__":
    lattice = LatticeD2Q9(precision)

    nx = 502
    ny = 101

    Re = 100.0
    u_wall = 0.1
    clength = nx - 1

    visc = u_wall * clength / Re

    omega = 1.0 / (3.0 * visc + 0.5)
    print("omega = ", omega)
    assert omega < 1.98, "omega must be less than 2.0"
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    sim = ChannelFlow2D(lattice, omega, nx, ny, precision=precision)
    sim.run(100000, error_report_rate=1000, io_rate=1000)
