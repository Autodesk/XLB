"""
The given script sets up a simulation for the Taylor-Green vortex flow.
The Taylor-Green vortex is a type of two-dimensional, incompressible fluid flow with a known analytical solution, making it an ideal test case for fluid dynamics simulations.
The flow is characterized by a pair of counter-rotating vortices. In this script, the initial fields for the Taylor-Green vortex are set using a known function.
"""


from src.boundary_conditions import *
from jax.config import config
from src.utils import *
import numpy as np
from src.lattice import LatticeD2Q9
from src.models import BGKSim, KBCSim, AdvectionDiffusionBGK
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import os
import matplotlib.pyplot as plt

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
# disable JIt compilation

jax.config.update('jax_enable_x64', True)

def taylor_green_initial_fields(xx, yy, u0, rho0, nu, time):
    ux = -u0 * np.cos(xx) * np.sin(yy) * np.exp(-2 * nu * time)
    uy = u0 * np.sin(xx) * np.cos(yy) * np.exp(-2 * nu * time)
    rho = 1.0 - rho0 * u0 ** 2 / 12. * (np.cos(2. * xx) + np.cos(2. * yy)) * np.exp(-4 * nu * time)
    return ux, uy, np.expand_dims(rho, axis=-1)

class TaylorGreenVortex(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # no boundary conditions implying periodic BC in all directions
        return

    def initialize_macroscopic_fields(self):
        ux, uy, rho = taylor_green_initial_fields(xx, yy, vel_ref, 1, 0., 0.)
        rho = self.distributed_array_init(rho.shape, self.precisionPolicy.output_dtype, init_val=1.0, sharding=self.sharding)
        u = np.stack([ux, uy], axis=-1)
        u = self.distributed_array_init(u.shape, self.precisionPolicy.output_dtype, init_val=u, sharding=self.sharding)
        return rho, u

    def initialize_populations(self, rho, u):
        omegaADE = 1.0
        kwargs = {'lattice': lattice, 'nx': self.nx, 'ny': self.ny, 'nz': self.nz,  'precision': precision, 'omega': omegaADE, 'vel': u, 'print_info_rate': 0, 'io_rate': 0}
        ADE = AdvectionDiffusionBGK(**kwargs)
        ADE.initialize_macroscopic_fields = self.initialize_macroscopic_fields
        print("Initializing the distribution functions using the specified macroscopic fields....")
        f = ADE.run(20000)
        return f

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"])
        u = np.array(kwargs["u"])
        timestep = kwargs["timestep"]

        # theoretical results
        time = timestep * (kx**2 + ky**2)/2.
        ux_th, uy_th, rho_th = taylor_green_initial_fields(xx, yy, vel_ref, 1, visc, time)
        vel_err_L2 = np.sqrt(np.sum((u[..., 0]-ux_th)**2 + (u[..., 1]-uy_th)**2) / np.sum(ux_th**2 + uy_th**2))
        print("error= {:07.6f}".format(vel_err_L2))
        if timestep == endTime:
            ErrL2ResList.append(vel_err_L2)
        # save_image(timestep, u)


if __name__ == "__main__":
    precision = "f64/f64"
    lattice = LatticeD2Q9(precision)

    resList = [32, 64, 128, 256, 512]
    ErrL2ResList = []

    for nx in resList:
        print("Running at nx = ny = {:07.6f}".format(nx))
        ny = nx
        twopi = 2.0 * np.pi
        coord = np.array([(i, j) for i in range(nx) for j in range(ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        kx, ky = twopi / nx, twopi / ny
        xx = xx.reshape((nx, ny)) * kx
        yy = yy.reshape((nx, ny)) * ky

        Re = 1000.0
        vel_ref = 0.04*32/nx

        visc = vel_ref * nx / Re
        omega = 1.0 / (3.0 * visc + 0.5)
        print("omega = ", omega)
        os.system("rm -rf ./*.vtk && rm -rf ./*.png")
        kwargs = {
            'lattice': lattice,
            'omega': omega,
            'nx': nx,
            'ny': ny,
            'nz': 0,
            'precision': precision,
            'io_rate': 500,
            'print_info_rate': 500
        }
        sim = TaylorGreenVortex(**kwargs)
        endTime = int(20000*nx/32.0)
        sim.run(endTime)
    plt.loglog(resList, ErrL2ResList, '-o')
    plt.loglog(resList, 1e-3*(np.array(resList)/128)**(-2), '--')
    plt.savefig('Error.png'); plt.savefig('Error.pdf', format='pdf')
