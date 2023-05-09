from time import time
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

# disable JIt compilation
# jax.config.update('jax_disable_jit', True)
jax.config.update('jax_array', True)
jax.config.update('jax_enable_x64', True)

precision = 'f64/f64'
u_inlet = 0.005
diam = 80

class Cylinder(KBCSim):

    def set_boundary_conditions(self):

        # define the cylinder surface
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = 2.*diam, 2.*diam
        cylinder = (xx - cx)**2 + (yy-cy)**2 <= (diam/2.)**2
        cylinder = coord[cylinder]
        self.BCs.append(BounceBackHalfway(tuple(cylinder.T), self.grid_info, self.precision_policy))
        # wall = np.concatenate([cylinder, self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        # self.BCs.append(BounceBack(tuple(wall.T), self.grid_info, self.precision_policy))

        outlet = self.boundingBoxIndices['right']
        rho_outlet = np.ones(outlet.shape[0], dtype=self.precision_policy.compute_dtype)
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.grid_info, self.precision_policy))
        # self.BCs.append(Regularized(tuple(outlet.T), self.grid_info, self.precision_policy, 'pressure', rho_outlet))

        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones(inlet.shape[0], dtype=self.precision_policy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet,
                                             yy_inlet.min(),
                                             yy_inlet.max()-yy_inlet.min(), 3.0 / 2.0 * u_inlet)
        # self.BCs.append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel_inlet))
        self.BCs.append(Regularized(tuple(inlet.T), self.grid_info, self.precision_policy, 'velocity', vel_inlet))

        wall = np.concatenate([self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        self.BCs.append(BounceBack(tuple(wall.T), self.grid_info, self.precision_policy))


    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][..., 1:-1])
        u = np.array(kwargs["u"][..., 1:-1, :])
        timestep = kwargs["timestep"]
        u_prev = kwargs["u_prev"][..., 1:-1, :]

        # compute lift and drag over the cyliner
        cylinder = self.BCs[0]
        boundary_force = cylinder.momentum_exchange_force(kwargs['f_poststreaming'], kwargs['f_postcollision'])
        boundary_force = np.sum(boundary_force, axis=0)
        drag = boundary_force[0]
        lift = boundary_force[1]
        cd = 2. * drag / (u_inlet ** 2 * diam)
        cl = 2. * lift / (u_inlet ** 2 * diam)

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)
        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}, CL = {:07.6f}, CD = {:07.6f}'.format(err, cl, cd))
        save_image(timestep, u)

# Helper function to specify a parabolic poiseuille profile
poiseuille_profile  = lambda x,x0,d,umax: np.maximum(0.,4.*umax/(d**2)*((x-x0)*d-(x-x0)**2))

if __name__ == '__main__':

    lattice = LatticeD2Q9(precision)

    nx = int(22*diam)
    ny = int(4.1*diam)

    Re = 100.0
    visc = u_inlet * diam / Re
    omega = 1.0 / (3. * visc + 0.5)

    print('omega = ', omega)
    print("Mesh size: ", nx, ny)
    print("Number of voxels: ", nx * ny)

    assert omega < 2.0, "omega must be less than 2.0"
    os.system('rm -rf ./*.vtk && rm -rf ./*.png')
    sim = Cylinder(lattice, omega, nx, ny, optimize=False, precision=precision)

    # need to retain fpost-collision for computation of lift and drag
    sim.ret_fpost = True
    sim.run(1000000, print_iter=500, io_iter=500)
