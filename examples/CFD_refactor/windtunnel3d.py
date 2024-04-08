# Wind tunnel simulation using the XLB library

from typing import Any
import os
import jax
import trimesh
from time import time
import numpy as np
import warp as wp
import pyvista as pv
import tqdm
import matplotlib.pyplot as plt

wp.init()

import xlb
from xlb.operator import Operator

class UniformInitializer(Operator):

    def _construct_warp(self):
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            vel: float,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Set the velocity
            u[0, i, j, k] = vel
            u[1, i, j, k] = 0.0
            u[2, i, j, k] = 0.0

            # Set the density
            rho[0, i, j, k] = 1.0

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, rho, u, vel):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                vel,
            ],
            dim=rho.shape[1:],
        )
        return rho, u

class MomentumTransfer(Operator):

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.wp_c
        _opp_indices = self.velocity_set.wp_opp_indices
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(
            self.velocity_set.q, dtype=wp.uint8
        )  # TODO fix vec bool

        # Find velocity index for 0, 0, 0
        for l in range(self.velocity_set.q):
            if _c[0, l] == 0 and _c[1, l] == 0 and _c[2, l] == 0:
                zero_index = l
        _zero_index = wp.int32(zero_index)
        print(f"Zero index: {_zero_index}")

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            boundary_id: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            momentum: wp.array(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the boundary id
            _boundary_id = boundary_id[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Determin if boundary is an edge by checking if center is missing
            is_edge = wp.bool(False)
            if _boundary_id == wp.uint8(xlb.operator.boundary_condition.HalfwayBounceBackBC.id):
                if _missing_mask[_zero_index] != wp.uint8(1):
                    is_edge = wp.bool(True)

            # If the boundary is an edge then add the momentum transfer
            m = wp.vec3()
            if is_edge:
                for l in range(self.velocity_set.q):
                    if _missing_mask[l] == wp.uint8(1):
                        phi = 2.0 * f[_opp_indices[l], index[0], index[1], index[2]]

                        # Compute the momentum transfer
                        for d in range(self.velocity_set.d):
                            m[d] += phi * wp.float32(_c[d, _opp_indices[l]])

            wp.atomic_add(momentum, 0, m)

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, f, boundary_id, missing_mask):

        # Allocate the momentum field
        momentum = wp.zeros((1), dtype=wp.vec3)

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f, boundary_id, missing_mask, momentum],
            dim=f.shape[1:],
        )
        return momentum.numpy()


class WindTunnel:
    """
    Wind tunnel simulation using the XLB library
    """

    def __init__(
        self, 
        stl_filename: str,
        inlet_velocity: float = 27.78, # m/s
        lower_bounds: tuple[float, float, float] = (0.0, 0.0, 0.0), # m
        upper_bounds: tuple[float, float, float] = (1.0, 0.5, 0.5), # m
        dx: float = 0.01, # m
        viscosity: float = 1.42e-5, # air at 20 degrees Celsius
        density: float = 1.2754, # kg/m^3
        solve_time: float = 1.0, # s
        #collision="BGK",
        collision="KBC",
        equilibrium="Quadratic",
        velocity_set="D3Q27",
        precision_policy=xlb.PrecisionPolicy.FP32FP32,
        compute_backend=xlb.ComputeBackend.WARP,
        grid_configs={},
        save_state_frequency=1024,
        monitor_frequency=32,
    ):

        # Set parameters
        self.stl_filename = stl_filename
        self.inlet_velocity = inlet_velocity
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.dx = dx
        self.solve_time = solve_time
        self.viscosity = viscosity
        self.density = density
        self.save_state_frequency = save_state_frequency
        self.monitor_frequency = monitor_frequency

        # Get fluid properties needed for the simulation
        self.base_velocity = 0.05 # LBM units
        self.velocity_conversion = self.base_velocity / inlet_velocity
        self.dt = self.dx * self.velocity_conversion
        self.lbm_viscosity = self.viscosity * self.dt / (self.dx ** 2)
        self.tau = 0.5 + self.lbm_viscosity
        self.omega = 1.0 / self.tau
        print(f"tau: {self.tau}")
        print(f"omega: {self.omega}")
        self.lbm_density = 1.0
        self.mass_conversion = self.dx ** 3 * (self.density / self.lbm_density)
        self.nr_steps = int(solve_time / self.dt)

        # Get the grid shape
        self.nx = int((upper_bounds[0] - lower_bounds[0]) / dx)
        self.ny = int((upper_bounds[1] - lower_bounds[1]) / dx)
        self.nz = int((upper_bounds[2] - lower_bounds[2]) / dx)
        self.shape = (self.nx, self.ny, self.nz)

        # Set the compute backend
        self.compute_backend = xlb.ComputeBackend.WARP

        # Set the precision policy
        self.precision_policy = xlb.PrecisionPolicy.FP32FP32

        # Set the velocity set
        if velocity_set == "D3Q27":
            self.velocity_set = xlb.velocity_set.D3Q27()
        elif velocity_set == "D3Q19":
            self.velocity_set = xlb.velocity_set.D3Q19()
        else:
            raise ValueError("Invalid velocity set")

        # Make grid
        self.grid = xlb.grid.WarpGrid(shape=self.shape)

        # Make feilds
        self.rho = self.grid.create_field(cardinality=1, precision=xlb.Precision.FP32)
        self.u = self.grid.create_field(cardinality=self.velocity_set.d, precision=xlb.Precision.FP32)
        self.f0 = self.grid.create_field(cardinality=self.velocity_set.q, precision=xlb.Precision.FP32)
        self.f1 = self.grid.create_field(cardinality=self.velocity_set.q, precision=xlb.Precision.FP32)
        self.boundary_id = self.grid.create_field(cardinality=1, precision=xlb.Precision.UINT8)
        self.missing_mask = self.grid.create_field(cardinality=self.velocity_set.q, precision=xlb.Precision.BOOL)

        # Make operators
        self.initializer = UniformInitializer(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.momentum_transfer = MomentumTransfer(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        if collision == "BGK":
            self.collision = xlb.operator.collision.BGK(
                omega=self.omega,
                velocity_set=self.velocity_set,
                precision_policy=self.precision_policy,
                compute_backend=self.compute_backend,
            )
        elif collision == "KBC":
            self.collision = xlb.operator.collision.KBC(
                omega=self.omega,
                velocity_set=self.velocity_set,
                precision_policy=self.precision_policy,
                compute_backend=self.compute_backend,
            )
        self.equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.macroscopic = xlb.operator.macroscopic.Macroscopic(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.stream = xlb.operator.stream.Stream(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.equilibrium_bc = xlb.operator.boundary_condition.EquilibriumBC(
            rho=self.lbm_density,
            u=(self.base_velocity, 0.0, 0.0),
            equilibrium_operator=self.equilibrium,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.half_way_bc = xlb.operator.boundary_condition.HalfwayBounceBackBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.full_way_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.do_nothing_bc = xlb.operator.boundary_condition.DoNothingBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
            collision=self.collision,
            equilibrium=self.equilibrium,
            macroscopic=self.macroscopic,
            stream=self.stream,
            boundary_conditions=[
                self.half_way_bc,
                self.full_way_bc,
                self.equilibrium_bc,
                self.do_nothing_bc
            ],
        )
        self.planar_boundary_masker = xlb.operator.boundary_masker.PlanarBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.stl_boundary_masker = xlb.operator.boundary_masker.STLBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

        # Make list to store drag coefficients
        self.drag_coefficients = []

    def initialize_flow(self):
        """
        Initialize the flow field
        """

        # Set initial conditions
        self.rho, self.u = self.initializer(self.rho, self.u, self.base_velocity)
        self.f0 = self.equilibrium(self.rho, self.u, self.f0)

    def initialize_boundary_conditions(self):
        """
        Initialize the boundary conditions
        """

        # Set inlet bc (bottom x face)
        lower_bound = (0, 1, 1) # no edges
        upper_bound = (0, self.ny-1, self.nz-1)
        direction = (1, 0, 0)
        self.boundary_id, self.missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.equilibrium_bc.id,
            self.boundary_id,
            self.missing_mask,
            (0, 0, 0)
        )
    
        # Set outlet bc (top x face)
        lower_bound = (self.nx-1, 1, 1)
        upper_bound = (self.nx-1, self.ny-1, self.nz-1)
        direction = (-1, 0, 0)
        self.boundary_id, self.missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.do_nothing_bc.id,
            self.boundary_id,
            self.missing_mask,
            (0, 0, 0)
        )
    
        # Set full way bc (bottom y face)
        lower_bound = (0, 0, 0)
        upper_bound = (self.nx, 0, self.nz)
        direction = (0, 1, 0)
        self.boundary_id, self.missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            self.boundary_id,
            self.missing_mask,
            (0, 0, 0)
        )
    
        # Set full way bc (top y face)
        lower_bound = (0, self.ny-1, 0)
        upper_bound = (self.nx, self.ny-1, self.nz)
        direction = (0, -1, 0)
        self.boundary_id, self.missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            self.boundary_id,
            self.missing_mask,
            (0, 0, 0)
        )

        # Set full way bc (bottom z face)
        lower_bound = (0, 0, 0)
        upper_bound = (self.nx, self.ny, 0)
        direction = (0, 0, 1)
        self.boundary_id, self.missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            self.boundary_id,
            self.missing_mask,
            (0, 0, 0)
        )

        # Set full way bc (top z face)
        lower_bound = (0, 0, self.nz-1)
        upper_bound = (self.nx, self.ny, self.nz-1)
        direction = (0, 0, -1)
        self.boundary_id, self.missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            self.boundary_id,
            self.missing_mask,
            (0, 0, 0)
        )

        # Set stl half way bc
        self.boundary_id, self.missing_mask = self.stl_boundary_masker(
            self.stl_filename,
            self.lower_bounds,
            (self.dx, self.dx, self.dx),
            self.half_way_bc.id,
            self.boundary_id,
            self.missing_mask,
            (0, 0, 0)
        )

    def save_state(
        self,
        postfix: str,
        save_velocity_distribution: bool = False,
    ):
        """
        Save the solid id array.
        """

        # Create grid
        grid = pv.RectilinearGrid(
            np.linspace(self.lower_bounds[0], self.upper_bounds[0], self.nx, endpoint=False),
            np.linspace(self.lower_bounds[1], self.upper_bounds[1], self.ny, endpoint=False),
            np.linspace(self.lower_bounds[2], self.upper_bounds[2], self.nz, endpoint=False),
        ) # TODO off by one?
        grid["boundary_id"] = self.boundary_id.numpy().flatten("F")
        grid["u"] = self.u.numpy().transpose(1, 2, 3, 0).reshape(-1, 3, order="F")
        grid["rho"] = self.rho.numpy().flatten("F")
        if save_velocity_distribution:
            grid["f0"] = self.f0.numpy().transpose(1, 2, 3, 0).reshape(-1, self.velocity_set.q, order="F")
        grid.save(f"state_{postfix}.vtk")

    def step(self):
        self.f1 = self.stepper(self.f0, self.f1, self.boundary_id, self.missing_mask, 0)
        self.f0, self.f1 = self.f1, self.f0

    def compute_rho_u(self):
        self.rho, self.u = self.macroscopic(self.f0, self.rho, self.u)

    def monitor(self):
        # Compute the momentum transfer
        momentum = self.momentum_transfer(self.f0, self.boundary_id, self.missing_mask)[0]
        drag = momentum[0]
        lift = momentum[2]
        c_d = 2.0 * drag / (self.base_velocity ** 2 * self.cross_section)
        c_l = 2.0 * lift / (self.base_velocity ** 2 * self.cross_section)
        self.drag_coefficients.append(c_d)

    def plot_drag_coefficient(self):
        plt.plot(self.drag_coefficients[-30:])
        plt.xlabel("Time step")
        plt.ylabel("Drag coefficient")
        plt.savefig("drag_coefficient.png")
        plt.close()

    def run(self):

        # Initialize the flow field
        self.initialize_flow()

        # Initialize the boundary conditions
        self.initialize_boundary_conditions()

        # Compute cross section
        np_boundary_id = self.boundary_id.numpy()
        cross_section = np.sum(np_boundary_id == self.half_way_bc.id, axis=(0, 1))
        self.cross_section = np.sum(cross_section > 0)

        # Run the simulation
        for i in tqdm.tqdm(range(self.nr_steps)):

            # Step
            self.step()

            # Monitor
            if i % self.monitor_frequency == 0:
                self.monitor()

            # Save monitor plot
            if i % (self.monitor_frequency * 10) == 0:
                self.plot_drag_coefficient()

            # Save state
            if i % self.save_state_frequency == 0:
                self.compute_rho_u()
                self.save_state(str(i).zfill(8))

if __name__ == '__main__':

    # Parameters
    inlet_velocity = 0.01 # m/s
    stl_filename = "fastback_baseline.stl"
    lower_bounds = (-4.0, -2.5, -1.5)
    upper_bounds = (12.0, 2.5, 2.5)
    dx = 0.03
    solve_time = 10000.0

    # Make wind tunnel
    wind_tunnel = WindTunnel(
        stl_filename=stl_filename,
        inlet_velocity=inlet_velocity,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solve_time=solve_time,
        dx=dx,
    )

    # Run the simulation
    wind_tunnel.run()
    wind_tunnel.save_state("final", save_velocity_distribution=True)

