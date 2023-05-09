import numpy as np
# from IPython import display
import matplotlib.pylab as plt
from src.models import BGKSim, KBCSim
from src.boundary_conditions import *
from src.lattice import *
import jax.numpy as jnp
import numpy as np
from src.utils import *
from jax.config import config
import os
#config.update('jax_disable_jit', True)
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
import scipy

precision = 'f32/f32'

def makeNacaAirfoil(length, thickness=30, angle=0):
    def nacaAirfoil(x, thickness, chordLength):
        coeffs = [0.2969, -0.1260, -0.3516, 0.2843, -0.1015]
        exponents = [0.5, 1, 2, 3, 4]
        af = [coeff * (x / chordLength) ** exp for coeff, exp in zip(coeffs, exponents)]
        return 5. * thickness / 100 * chordLength * np.sum(af)

    x = np.arange(length)
    y = np.arange(-int(length * thickness / 200), int(length * thickness / 200))
    xx, yy = np.meshgrid(x, y)
    domain = np.where(np.abs(yy) < nacaAirfoil(xx, thickness, length), 1, 0).T

    domain = scipy.ndimage.rotate(np.rot90(domain), -angle)
    domain = np.where(domain > 0.5, 1, 0)

    return domain

class Airfoil(KBCSim):

    def set_boundary_conditions(self):
        tx, ty = np.array([self.nx, self.ny], dtype=int) - airfoil.shape

        airfoil_mask = np.pad(airfoil, ((tx // 3, tx - tx // 3), (ty // 2, ty - ty // 2)), 'constant', constant_values=False)
        airfoil_mask = np.repeat(airfoil_mask[:, :, np.newaxis], self.nz, axis=2)
        
        airfoil_indices = np.argwhere(airfoil_mask)
        wall = np.concatenate((airfoil_indices,
                               self.boundingBoxIndices['bottom'], self.boundingBoxIndices['top']))
        self.BCs.append(BounceBack(tuple(wall.T), self.grid_info, self.precision_policy))

        doNothing = self.boundingBoxIndices['right']
        self.BCs.append(DoNothing(tuple(doNothing.T), self.grid_info, self.precision_policy))

        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones(inlet.shape[0], dtype=self.precision_policy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)

        vel_inlet[:, 0] = inlet_vel
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel_inlet))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs['rho'][..., 1:-1])
        u = np.array(kwargs['u'][..., 1:-1, :])
        timestep = kwargs['timestep']
        u_prev = kwargs['u_prev'][..., 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}'.format(err))
        # save_image(timestep, rho, u)
        fields = {"rho": rho, "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(timestep, fields)

if __name__ == '__main__':

    airfoil_length = 101
    airfoil_thickness = 30
    airfoil_angle = 20
    airfoil = makeNacaAirfoil(length=airfoil_length, thickness=airfoil_thickness, angle=airfoil_angle).T

    lattice = LatticeD3Q27(precision=precision)

    nx = airfoil.shape[0]
    ny = airfoil.shape[1]

    print("airfoil shape: ", airfoil.shape)

    ny = 3 * ny
    nx = 4 * nx
    nz = 101

    Re = 10000.0
    inlet_vel = 0.1
    clength = airfoil_length

    visc = inlet_vel * clength / Re
    omega = 1.0 / (3. * visc + 0.5)
    print('omega = ', omega)
    assert omega < 2.0, "omega must be less than 2.0"
    os.system('rm -rf ./*.vtk && rm -rf ./*.png')
    sim = Airfoil(lattice, omega, nx, ny, nz, precision=precision)
    print('Domain size: ', sim.nx, sim.ny, sim.nz)

    sim.run(20000, print_iter=200, io_iter=1000)