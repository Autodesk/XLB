import os
import jax
import trimesh
from time import time
import numpy as np
import jax.numpy as jnp
from jax import config

from xlb.solver import IncompressibleNavierStokesSolver
from xlb.velocity_set import D3Q27, D3Q19
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid_backend import GridBackend
from xlb.operator.boundary_condition import BounceBack, BounceBackHalfway, DoNothing, EquilibriumBC


class WindTunnel(IncompressibleNavierStokesSolver):
    """
    This class extends the IncompressibleNavierStokesSolver class to define the boundary conditions for the wind tunnel simulation.
    Units are in meters, seconds, and kilograms.
    """

    def __init__(
        self, 
        stl_filename: str
        stl_center: tuple[float, float, float] = (0.0, 0.0, 0.0), # m
        inlet_velocity: float = 27.78 # m/s
        lower_bounds: tuple[float, float, float] = (0.0, 0.0, 0.0), # m
        upper_bounds: tuple[float, float, float] = (1.0, 0.5, 0.5), # m
        dx: float = 0.01, # m
        viscosity: float = 1.42e-5, # air at 20 degrees Celsius
        density: float = 1.2754, # kg/m^3
        collision="BGK",
        equilibrium="Quadratic",
        velocity_set=D3Q27(),
        precision_policy=PrecisionPolicy.FP32FP32,
        compute_backend=ComputeBackend.JAX,
        grid_backend=GridBackend.JAX,
        grid_configs={},
    ):

        # Set parameters
        self.inlet_velocity = inlet_velocity
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.dx = dx
        self.viscosity = viscosity
        self.density = density

        # Get fluid properties needed for the simulation
        self.velocity_conversion = 0.05 / inlet_velocity
        self.dt = self.dx * self.velocity_conversion
        self.lbm_viscosity = self.viscosity * self.dt / (self.dx ** 2)
        self.tau = 0.5 + self.lbm_viscosity
        self.lbm_density = 1.0
        self.mass_conversion = self.dx ** 3 * (self.density / self.lbm_density)

        # Make boundary conditions


        # Initialize the IncompressibleNavierStokesSolver
        super().__init__(
            omega=self.tau,
            shape=shape,
            collision=collision,
            equilibrium=equilibrium,
            boundary_conditions=boundary_conditions,
            initializer=initializer,
            forcing=forcing,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
            grid_backend=grid_backend,
            grid_configs=grid_configs,
        )

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

    sim = Car(**kwargs)
    sim.run(200000)
