"""
This script conducts a 2D simulation of flow around a cylinder using the lattice Boltzmann method (LBM). This is a classic problem in fluid dynamics and is often used to examine the behavior of fluid flow over a bluff body.

In this example you'll be introduced to the following concepts:

1. Lattice: A D2Q9 lattice is used, which is a two-dimensional lattice model with nine discrete velocity directions. This type of lattice allows for a precise representation of fluid flow in two dimensions.

2. Boundary Conditions: The script implements several types of boundary conditions:

    BounceBackMoving: This condition is applied to the cylinder surface. Unlike the usual BounceBack condition, this one takes into account the motion of the cylinder.
    ExtrapolationOutflow: This condition is applied at the outlet (right boundary), where the fluid is allowed to exit the simulation domain freely.
    Regularized: This condition is applied at the inlet (left boundary) and models the inflow of fluid into the domain with a specified velocity profile. Another Regularized condition is used for the stationary top and bottom walls.
3. Velocity Profile: The script uses a Poiseuille flow profile for the inlet velocity. This is a parabolic profile commonly seen in pipe flow.

4. Drag and lift calculation: The script computes the lift and drag on the cylinder, which are important quantities in fluid dynamics and aerodynamics.

5. Visualization: The simulation outputs data in VTK format for visualization. It also generates images of the velocity field. The data can be visualized using software like ParaView.

"""


import os
import jax
from time import time
from jax.config import config
import numpy as np
import jax.numpy as jnp

from src.utils import *
from src.boundary_conditions import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_enable_x64', True)

class Cylinder(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                                             yy_inlet.max()-yy_inlet.min(), 3.0 / 2.0 * prescribed_vel)
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
        cd = 2. * drag / (prescribed_vel ** 2 * diam)
        cl = 2. * lift / (prescribed_vel ** 2 * diam)

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
    precision = 'f64/f64'
    lattice = LatticeD2Q9(precision)
    prescribed_vel = 0.005
    diam = 20
    nx = int(22*diam)
    ny = int(4.1*diam)

    Re = 10.0
    visc = prescribed_vel * diam / Re
    omega = 1.0 / (3. * visc + 0.5)

    os.system('rm -rf ./*.vtk && rm -rf ./*.png')
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': 500,
        'print_info_rate': 500,
        'return_fpost': True,    # Need to retain fpost-collision for computation of lift and drag
    }
    sim = Cylinder(**kwargs)

    sim.run(1000000)
