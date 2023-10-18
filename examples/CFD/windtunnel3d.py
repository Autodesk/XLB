"""
This script performs a Lattice Boltzmann Method (LBM) simulation of fluid flow over a car model. Here are the main concepts and steps in the simulation:

Here are the main concepts introduced simulation:

1. Lattice: Given the usually high Reynolds number required for these simulations, a D3Q27 lattice is used, which is a three-dimensional lattice model with 27 discrete velocity directions.

2. Loading geometry and voxelization: The geometry of the car is loaded from a STL file. 
This is a file format commonly used for 3D models. The model is then voxelized to a binary matrix which represents the presence or absence of the object in the lattice. We use the DrivAer model, which is a common car model used for aerodynamic simulations.

3. Output: After each specified number of iterations, the script outputs the state of the simulation. This includes the error (difference between consecutive velocity fields), lift and drag coefficients, and visualization files in the VTK format.
"""


import os
import jax
import trimesh
from time import time
import numpy as np
import jax.numpy as jnp
from jax.config import config

from src.utils import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.boundary_conditions import *

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

# disable JIt compilation

jax.config.update('jax_array', True)

class Car(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def voxelize_stl(self, stl_filename, length_lbm_unit):
        mesh = trimesh.load_mesh(stl_filename, process=False)
        length_phys_unit = mesh.extents.max()
        pitch = length_phys_unit/length_lbm_unit
        mesh_voxelized = mesh.voxelized(pitch=pitch)
        mesh_matrix = mesh_voxelized.matrix
        return mesh_matrix, pitch

    def set_boundary_conditions(self):
        print('Voxelizing mesh...')
        time_start = time()
        stl_filename = 'stl-files/DrivAer-Notchback.stl'
        car_length_lbm_unit = self.nx / 4
        car_voxelized, pitch = voxelize_stl(stl_filename, car_length_lbm_unit)
        car_matrix = car_voxelized.matrix
        print('Voxelization time for pitch={}: {} seconds'.format(pitch, time() - time_start))
        print("Car matrix shape: ", car_matrix.shape)

        self.car_area = np.prod(car_matrix.shape[1:])
        tx, ty, tz = np.array([nx, ny, nz]) - car_matrix.shape
        shift = [tx//4, ty//2, 0]
        car_indices = np.argwhere(car_matrix) + shift
        self.BCs.append(BounceBackHalfway(tuple(car_indices.T), self.gridInfo, self.precisionPolicy))

        wall = np.concatenate((self.boundingBoxIndices['bottom'], self.boundingBoxIndices['top'],
                               self.boundingBoxIndices['front'], self.boundingBoxIndices['back']))
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        doNothing = self.boundingBoxIndices['right']
        self.BCs.append(DoNothing(tuple(doNothing.T), self.gridInfo, self.precisionPolicy))
        self.BCs[-1].implementationStep = 'PostCollision'
        # rho_outlet = np.ones(doNothing.shape[0], dtype=self.precisionPolicy.compute_dtype)
        # self.BCs.append(ZouHe(tuple(doNothing.T),
        #                                          self.gridInfo,
        #                                          self.precisionPolicy,
        #                                          'pressure', rho_outlet))

        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)

        vel_inlet[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel_inlet))
        # self.BCs.append(ZouHe(tuple(inlet.T),
        #                                          self.gridInfo,
        #                                          self.precisionPolicy,
        #                                          'velocity', vel_inlet))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs['rho'][..., 1:-1, 1:-1, :])
        u = np.array(kwargs['u'][..., 1:-1, 1:-1, :])
        timestep = kwargs['timestep']
        u_prev = kwargs['u_prev'][..., 1:-1, 1:-1, :]

        # compute lift and drag over the car
        car = self.BCs[0]
        boundary_force = car.momentum_exchange_force(kwargs['f_poststreaming'], kwargs['f_postcollision'])
        boundary_force = np.sum(boundary_force, axis=0)
        drag = np.sqrt(boundary_force[0]**2 + boundary_force[1]**2)     #xy-plane
        lift = boundary_force[2]                                        #z-direction
        cd = 2. * drag / (prescribed_vel ** 2 * self.car_area)
        cl = 2. * lift / (prescribed_vel ** 2 * self.car_area)

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}, CL = {:07.6f}, CD = {:07.6f}'.format(err, cl, cd))
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(timestep, fields)

if __name__ == '__main__':
    precision = 'f32/f32'
    lattice = LatticeD3Q27(precision)

    nx = 601
    ny = 351
    nz = 251

    Re = 50000.0
    prescribed_vel = 0.05
    clength = nx - 1

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3. * visc + 0.5)

    os.system('rm -rf ./*.vtk && rm -rf ./*.png')

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 100,
        'print_info_rate': 100,
        'return_fpost': True  # Need to retain fpost-collision for computation of lift and drag
    }
    sim = Car(**kwargs)
    sim.run(200000)
