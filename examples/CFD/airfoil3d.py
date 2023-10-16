"""
This is a example for simulating fluid flow around a NACA airfoil using the lattice Boltzmann method (LBM). 
The LBM is a computational fluid dynamics method for simulating fluid flow and is particularly effective 
for complex geometries and multiphase flow. 

In this example you'll be introduced to the following concepts:

1. Lattice: The example uses a D3Q27 lattice, which is a three-dimensional lattice model that considers 
    27 discrete velocity directions. This allows for a more accurate representation of the fluid flow 
    in three dimensions.

2. NACA Airfoil Generation: The example includes a function to generate a NACA airfoil shape, which is 
    common in aerodynamics. The function allows for customization of the length, thickness, and angle 
    of the airfoil.

3. Boundary Conditions: The example includes several boundary conditions. These include a "bounce back" 
    condition on the airfoil surface and the top and bottom of the domain, a "do nothing" condition 
    at the outlet (right side of the domain), and an "equilibrium" condition at the inlet 
    (left side of the domain) to simulate a uniform flow.

4. Simulation Parameters: The example allows for the setting of various simulation parameters, 
    including the Reynolds number, inlet velocity, and characteristic length. 

5. Visualization: The example outputs data in VTK format, which can be visualized using software such 
    as Paraview. The error between the old and new velocity fields is also printed out at each time step 
    to monitor the convergence of the solution.
"""


import numpy as np
# from IPython import display
import matplotlib.pylab as plt
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.boundary_conditions import *
import numpy as np
from src.utils import *
from jax.config import config
import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
import scipy

# Function to create a NACA airfoil shape given its length, thickness, and angle of attack
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        tx, ty = np.array([self.nx, self.ny], dtype=int) - airfoil.shape

        airfoil_mask = np.pad(airfoil, ((tx // 3, tx - tx // 3), (ty // 2, ty - ty // 2)), 'constant', constant_values=False)
        airfoil_mask = np.repeat(airfoil_mask[:, :, np.newaxis], self.nz, axis=2)
        
        airfoil_indices = np.argwhere(airfoil_mask)
        wall = np.concatenate((airfoil_indices,
                               self.boundingBoxIndices['bottom'], self.boundingBoxIndices['top']))
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        doNothing = self.boundingBoxIndices['right']
        self.BCs.append(DoNothing(tuple(doNothing.T), self.gridInfo, self.precisionPolicy))

        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros((inlet.shape), dtype=self.precisionPolicy.compute_dtype)

        vel_inlet[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel_inlet))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs['rho'][..., 1:-1, :])
        u = np.array(kwargs['u'][..., 1:-1, :])
        timestep = kwargs['timestep']
        u_prev = kwargs['u_prev'][..., 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}'.format(err))
        # save_image(timestep, rho, u)
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(timestep, fields)

if __name__ == '__main__':
    airfoil_length = 101
    airfoil_thickness = 30
    airfoil_angle = 20
    airfoil = makeNacaAirfoil(length=airfoil_length, thickness=airfoil_thickness, angle=airfoil_angle).T
    precision = 'f32/f32'

    lattice = LatticeD3Q27(precision)

    nx = airfoil.shape[0]
    ny = airfoil.shape[1]

    ny = 3 * ny
    nx = 4 * nx
    nz = 101

    Re = 10000.0
    prescribed_vel = 0.1
    clength = airfoil_length

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3. * visc + 0.5)
    
    os.system('rm -rf ./*.vtk && rm -rf ./*.png')

    # Set the parameters for the simulation
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 100,
    }

    sim = Airfoil(**kwargs)
    sim.run(20000)