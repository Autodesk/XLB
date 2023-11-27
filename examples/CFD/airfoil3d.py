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

5. In-situ visualization: The example outputs rendering images of the q-criterion using
     PhantomGaze library (https://github.com/loliverhennigh/PhantomGaze) without any I/O overhead 
     while the data is still on the GPU. 
"""


import numpy as np
# from IPython import display
import matplotlib.pylab as plt
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.boundary_conditions import *
import numpy as np
from src.utils import *
from jax import config
import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
import scipy

# PhantomGaze for in-situ rendering
import phantomgaze as pg

def makeNacaAirfoil(length, thickness=30, angle=0):
    def nacaAirfoil(x, thickness, chordLength):
        coeffs = [0.2969, -0.1260, -0.3516, 0.2843, -0.1015]
        exponents = [0.5, 1, 2, 3, 4]
        yt = [coeff * (x / chordLength) ** exp for coeff, exp in zip(coeffs, exponents)]
        yt = 5. * thickness / 100 * chordLength * np.sum(yt)

        return yt

    x = np.linspace(0, length, num=length)
    yt = np.array([nacaAirfoil(xi, thickness, length) for xi in x])

    y_max = int(np.max(yt)) + 1
    domain = np.zeros((2 * y_max, len(x)), dtype=int)

    for i, xi in enumerate(x):
        upper_bound = int(y_max + yt[i])
        lower_bound = int(y_max - yt[i])
        domain[lower_bound:upper_bound, i] = 1

    domain = scipy.ndimage.rotate(domain, angle, reshape=True)
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

        # Store airfoil boundary for visualization
        self.visualization_bc = jnp.zeros((self.nx, self.ny, self.nz), dtype=jnp.float32)
        self.visualization_bc = self.visualization_bc.at[tuple(airfoil_indices.T)].set(1.0)

        doNothing = self.boundingBoxIndices['right']
        self.BCs.append(DoNothing(tuple(doNothing.T), self.gridInfo, self.precisionPolicy))

        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros((inlet.shape), dtype=self.precisionPolicy.compute_dtype)

        vel_inlet[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel_inlet))

    def output_data(self, **kwargs):
        # Compute q-criterion and vorticity using finite differences
        # Get velocity field  
        u = kwargs['u'][..., 1:-1, :]
        # vorticity and q-criterion
        norm_mu, q = q_criterion(u)

        # Make phantomgaze volume
        dx = 0.01
        origin = (0.0, 0.0, 0.0)
        upper_bound = (self.visualization_bc.shape[0] * dx, self.visualization_bc.shape[1] * dx, self.visualization_bc.shape[2] * dx)
        q_volume = pg.objects.Volume(
            q,
            spacing=(dx, dx, dx),
            origin=origin,
        )
        norm_mu_volume = pg.objects.Volume(
            norm_mu,
            spacing=(dx, dx, dx),
            origin=origin,
        )
        boundary_volume = pg.objects.Volume(
            self.visualization_bc,
            spacing=(dx, dx, dx),
            origin=origin,
        )

        # Make colormap for norm_mu
        colormap = pg.Colormap("jet", vmin=0.0, vmax=0.05)

        # Get camera parameters
        focal_point = (self.visualization_bc.shape[0] * dx / 2, self.visualization_bc.shape[1] * dx / 2, self.visualization_bc.shape[2] * dx / 2)
        radius = 5.0
        angle = kwargs['timestep'] * 0.0001
        camera_position = (focal_point[0] + radius * np.sin(angle), focal_point[1], focal_point[2] + radius * np.cos(angle))

        # Rotate camera 
        camera = pg.Camera(position=camera_position, focal_point=focal_point, view_up=(0.0, 1.0, 0.0), max_depth=30.0, height=1080, width=1920, background=pg.SolidBackground(color=(0.0, 0.0, 0.0)))

        # Make wireframe
        screen_buffer = pg.render.wireframe(lower_bound=origin, upper_bound=upper_bound, thickness=0.01, camera=camera)

        # Render axes
        screen_buffer = pg.render.axes(size=0.1, center=(0.0, 0.0, 1.1), camera=camera, screen_buffer=screen_buffer)

        # Render q-criterion
        screen_buffer = pg.render.contour(q_volume, threshold=0.00003, color=norm_mu_volume, colormap=colormap, camera=camera, screen_buffer=screen_buffer)

        # Render boundary
        boundary_colormap = pg.Colormap("bone_r", vmin=0.0, vmax=3.0, opacity=np.linspace(0.0, 6.0, 256))
        screen_buffer = pg.render.volume(boundary_volume, camera=camera, colormap=boundary_colormap, screen_buffer=screen_buffer)

        # Show the rendered image
        plt.imsave('q_criterion_' + str(kwargs['timestep']).zfill(7) + '.png', np.minimum(screen_buffer.image.get(), 1.0))


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
    nx = 5 * nx
    nz = 101

    Re = 30000.0
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
