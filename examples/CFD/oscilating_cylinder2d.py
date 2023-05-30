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
diam = 20

class Cylinder(KBCSim):

    def set_boundary_conditions(self):

        wall = np.concatenate([self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        coord = np.array([np.unravel_index(i, (self.nx, self.ny)) for i in range(self.nx*self.ny)]) 
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = 2.*diam, 2.*diam
        cyl = ((xx) - cx)**2 + (yy-cy)**2 <= (diam/2.)**2
        cyl = jnp.array(coord[cyl])
    
        # Define update rules for boundary conditions
        def update_function(time: int):
            # Move the cylinder up and down sinusoidally with time
            
            # Define the scale for the sinusoidal motion
            scale = 10000
            
            # Amplitude of the motion, a quarter of the y-dimension of the grid
            A = ny // 4
            
            # Calculate the new y-coordinates of the cylinder. The cylinder moves up and down, 
            # its motion dictated by the sinusoidal function. We use `astype(int)` to ensure 
            # the indices are integers, as they will be used for array indexing.
            new_y_coords = cyl[:, 1] + jnp.array((jnp.sin(time/scale)*A).astype(int))
            
            # Define the indices of the grid points occupied by the cylinder
            indices = (cyl[:, 0], new_y_coords)
            
            # Calculate the velocity of the cylinder. The x-component is always 0 (the cylinder 
            # doesn't move horizontally), and the y-component is the derivative of the sinusoidal 
            # function governing the cylinder's motion, scaled by the amplitude and the scale factor.
            velocity = jnp.array([0., jnp.cos(time/scale)* A / scale], dtype=self.precisionPolicy.compute_dtype)
            
            return indices, velocity

        self.BCs.append(BounceBackMoving(self.gridInfo, self.precisionPolicy, update_function=update_function))


        outlet = self.boundingBoxIndices['right']
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))

        inlet = self.boundingBoxIndices['left']
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet,
                                             yy_inlet.min(),
                                             yy_inlet.max()-yy_inlet.min(), 3.0 / 2.0 * u_inlet)
        self.BCs.append(Regularized(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))


    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][..., 1:-1, :])
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
        # u magnitude
        fields = {'rho': rho[..., 0], 'u': np.linalg.norm(u, axis=2)}
        save_fields_vtk(timestep, fields)
        save_BCs_vtk(timestep, self.BCs, self.gridInfo)

# Helper function to specify a parabolic poiseuille profile
poiseuille_profile  = lambda x,x0,d,umax: np.maximum(0.,4.*umax/(d**2)*((x-x0)*d-(x-x0)**2))

if __name__ == '__main__':

    lattice = LatticeD2Q9(precision)

    nx = int(22*diam)
    ny = int(4.1*diam)

    Re = 10.0
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
    sim.run(1000000, error_report_rate=500, io_rate=500)
