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
jax.config.update("jax_array", True)
# disable JIt compilation
# jax.config.update('jax_disable_jit', True)
jax.config.update('jax_enable_x64', True)


precision = "f64/f64"
resList = [32, 64, 128, 256, 512]
ErrL2ResList = []


def taylor_green_initial_fields(xx, yy, u0, rho0, nu, time):
    ux = -u0 * np.cos(xx) * np.sin(yy) * np.exp(-2 * nu * time)
    uy = u0 * np.sin(xx) * np.cos(yy) * np.exp(-2 * nu * time)
    rho = 1.0 - rho0 * u0 ** 2 / 12. * (np.cos(2. * xx) + np.cos(2. * yy)) * np.exp(-4 * nu * time)
    return ux, uy, rho

class TaylorGreenVortex(KBCSim):

    def set_boundary_conditions(self):
        # no boundary conditions implying periodic BC in all directions
        return

    def initialize_macroscopic_fields(self):
        ux, uy, rho = taylor_green_initial_fields(xx, yy, vel_ref, 1, 0., 0.)
        rho_sharding = PositionalSharding(mesh_utils.create_device_mesh((self.n_devices, 1, 1)))
        u_sharding = PositionalSharding(mesh_utils.create_device_mesh((self.n_devices, 1, 1, 1)))
        rho = self.distributed_array_init(rho.shape, self.precision_policy.output_dtype, initVal=1.0, sharding=rho_sharding)
        u = np.stack([ux, uy], axis=-1)
        u = self.distributed_array_init(u.shape, self.precision_policy.output_dtype, initVal=u, sharding=u_sharding)
        return rho, u

    def initialize_populations(self, rho, u):
        omegaADE = 1.0
        ADE = AdvectionDiffusionBGK(u, lattice, omegaADE, self.nx, self.ny, precision=precision)
        ADE.initialize_macroscopic_fields = self.initialize_macroscopic_fields
        print("Initializing the distribution functions using the specified macroscopic fields....")
        f = ADE.run(20000, print_iter=0, io_iter=0)
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
    lattice = LatticeD2Q9(precision)

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
        assert omega < 2.0, "omega must be less than 2.0"
        os.system("rm -rf ./*.vtk && rm -rf ./*.png")
        sim = TaylorGreenVortex(lattice, omega, nx, ny, precision=precision, optimize=False)
        endTime = int(20000*nx/32.0)
        sim.run(endTime, print_iter=5000, io_iter=5000)
    plt.loglog(resList, ErrL2ResList, '-o')
    plt.loglog(resList, 1e-3*(np.array(resList)/128)**(-2), '--')
    plt.savefig('Error.png'); plt.savefig('Error.pdf', format='pdf')
