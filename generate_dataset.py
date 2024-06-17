import os
import json
import threading
import jax
from time import time
from jax import config
import numpy as np
import jax.numpy as jnp
from termcolor import colored
# from flax import linen as nn

from src.utils import *
from src.boundary_conditions import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_enable_x64', True)


class Cylinder(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.saved_data = []

    def set_boundary_conditions(self):
        # Define the cylinder surface
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = 2. * _diam, 2. * _diam
        cylinder = (xx - cx) ** 2 + (yy - cy) ** 2 <= (_diam / 2.) ** 2
        cylinder = coord[cylinder]
        implicit_distance = np.reshape((xx - cx) ** 2 + (yy - cy) ** 2 - (_diam / 2.) ** 2, (self.nx, self.ny))
        self.BCs.append(
            InterpolatedBounceBackBouzidi(tuple(cylinder.T), implicit_distance, self.gridInfo, self.precisionPolicy))

        # Outflow BC
        outlet = self.boundingBoxIndices['right']
        rho_outlet = np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        # self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))

        # Inlet BC
        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet,
                                             yy_inlet.min(),
                                             yy_inlet.max() - yy_inlet.min(), 3.0 / 2.0 * _prescribed_vel)
        self.BCs.append(Regularized(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))

        # No-slip BC for top and bottom
        wall = np.concatenate([self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        vel_wall = np.zeros(wall.shape, dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(Regularized(tuple(wall.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_wall))

    def output_data(self, **kwargs):
        self.saved_data.append(kwargs['u'])
        if kwargs['timestep'] % 200 == 0:
            save_image(kwargs['timestep'], kwargs['u'])
        # 1:-1 to remove boundary voxels (not needed for visualization when using bounce-back)

    def get_force(self):
        pass


# Helper function to specify a parabolic poiseuille profile
poiseuille_profile = lambda x, x0, d, umax: np.maximum(0., 4. * umax / (d ** 2) * ((x - x0) * d - (x - x0) ** 2))


def generate_sim_dataaset(diam, t_start, t_end, output_stride, output_offset):
    global _diam
    global _prescribed_vel
    _diam = diam
    precision = 'f64/f64'
    # diam_list = [10, 20, 30, 40, 60, 80]
    scale_factor = 80 / diam
    prescribed_vel = 0.003 * scale_factor
    _prescribed_vel = prescribed_vel
    lattice = LatticeD2Q9(precision)

    nx = int(22 * diam)
    ny = int(4.1 * diam)

    Re = 100.0
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
        'return_fpost': True  # Need to retain fpost-collision for computation of lift and drag
    }
    # characteristic time
    tc = prescribed_vel / diam
    if t_end < int(100 // tc):
        print(colored("WARNING: timestep_end is too small, Karman flow may not appear", "red"))
    sim = Cylinder(**kwargs)
    for data in sim.run_batch_generator(t_end, t_start, output_stride=output_stride, output_offset=output_offset, generator_size=500):
         yield data
    return

def save_data_batch(np_data, seq_number):
    np.save("./data/ref_data_diam_80_seq_{}".format(seq_number), np_data)

generated_data = generate_sim_dataaset(10, 0, 2000, 1, output_offset=1000)
seq_number = 0
for data in generated_data:
    np_data = np.stack(data, axis=0)
    print("Saving ... ")
    threading.Thread(target=save_data_batch, args=(np_data, seq_number)).start()
    seq_number += 1