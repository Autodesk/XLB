import xlb
import trimesh
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    FullwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


class WindTunnel3D:
    def __init__(self, omega, wind_speed, grid_shape, velocity_set, backend, precision_policy):
        # initialize backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.backend = backend
        self.precision_policy = precision_policy
        self.omega = omega
        self.boundary_conditions = []
        self.wind_speed = wind_speed

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=backend)

        # Setup the simulation BC and stepper
        self._setup()

        # Make list to store drag coefficients
        self.time_steps = []
        self.drag_coefficients = []
        self.lift_coefficients = []

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_stepper()
        # Initialize fields using the stepper
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

    def voxelize_stl(self, stl_filename, length_lbm_unit):
        mesh = trimesh.load_mesh(stl_filename, process=False)
        length_phys_unit = mesh.extents.max()
        pitch = length_phys_unit / length_lbm_unit
        mesh_voxelized = mesh.voxelized(pitch=pitch)
        mesh_matrix = mesh_voxelized.matrix
        return mesh_matrix, pitch

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        inlet = box_no_edge["left"]
        outlet = box_no_edge["right"]
        walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()

        # Load the mesh (replace with your own mesh)
        stl_filename = "../stl-files/DrivAer-Notchback.stl"
        mesh = trimesh.load_mesh(stl_filename, process=False)
        mesh_vertices = mesh.vertices

        # Transform the mesh points to be located in the right position in the wind tunnel
        mesh_vertices -= mesh_vertices.min(axis=0)
        mesh_extents = mesh_vertices.max(axis=0)
        length_phys_unit = mesh_extents.max()
        length_lbm_unit = self.grid_shape[0] / 4
        dx = length_phys_unit / length_lbm_unit
        mesh_vertices = mesh_vertices / dx
        shift = np.array([self.grid_shape[0] / 4, (self.grid_shape[1] - mesh_extents[1] / dx) / 2, 0.0])
        car = mesh_vertices + shift
        self.car_cross_section = np.prod(mesh_extents[1:]) / dx**2

        return inlet, outlet, walls, car

    def setup_boundary_conditions(self):
        inlet, outlet, walls, car = self.define_boundary_indices()
        bc_left = RegularizedBC("velocity", prescribed_value=(self.wind_speed, 0.0, 0.0), indices=inlet)
        bc_walls = FullwayBounceBackBC(indices=walls)
        bc_do_nothing = ExtrapolationOutflowBC(indices=outlet)
        bc_car = HalfwayBounceBackBC(mesh_vertices=car)
        self.boundary_conditions = [bc_walls, bc_left, bc_do_nothing, bc_car]

    def setup_stepper(self):
        self.stepper = IncompressibleNavierStokesStepper(
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="KBC",
        )

    def run(self, num_steps, print_interval, post_process_interval=100):
        # Setup the operator for computing surface forces at the interface of the specified BC
        bc_car = self.boundary_conditions[-1]
        self.momentum_transfer = MomentumTransfer(bc_car)

        start_time = time.time()
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if (i + 1) % print_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration: {i + 1}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def post_process(self, i):
        # Write the results. We'll use JAX backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            f_0 = wp.to_jax(self.f_0)
        else:
            f_0 = self.f_0

        macro = Macroscopic(
            compute_backend=ComputeBackend.JAX,
            precision_policy=self.precision_policy,
            velocity_set=xlb.velocity_set.D3Q27(precision_policy=self.precision_policy, backend=ComputeBackend.JAX),
        )

        rho, u = macro(f_0)

        # remove boundary cells
        u = u[:, 1:-1, 1:-1, 1:-1]
        u_magnitude = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5

        fields = {"u_magnitude": u_magnitude}

        save_fields_vtk(fields, timestep=i)
        save_image(fields["u_magnitude"][:, self.grid_shape[1] // 2, :], timestep=i)

        # Compute lift and drag
        boundary_force = self.momentum_transfer(self.f_0, self.f_1, self.bc_mask, self.missing_mask)
        drag = np.sqrt(boundary_force[0] ** 2 + boundary_force[1] ** 2)  # xy-plane
        lift = boundary_force[2]
        c_d = 2.0 * drag / (self.wind_speed**2 * self.car_cross_section)
        c_l = 2.0 * lift / (self.wind_speed**2 * self.car_cross_section)
        self.drag_coefficients.append(c_d)
        self.lift_coefficients.append(c_l)
        self.time_steps.append(i)

        # Save monitor plot
        self.plot_drag_coefficient()

    def plot_drag_coefficient(self):
        # Compute moving average of drag coefficient, 100, 1000, 10000
        drag_coefficients = np.array(self.drag_coefficients)
        self.drag_coefficients_ma_10 = np.convolve(drag_coefficients, np.ones(10) / 10, mode="valid")
        self.drag_coefficients_ma_100 = np.convolve(drag_coefficients, np.ones(100) / 100, mode="valid")
        self.drag_coefficients_ma_1000 = np.convolve(drag_coefficients, np.ones(1000) / 1000, mode="valid")
        self.drag_coefficients_ma_10000 = np.convolve(drag_coefficients, np.ones(10000) / 10000, mode="valid")
        self.drag_coefficients_ma_100000 = np.convolve(drag_coefficients, np.ones(100000) / 100000, mode="valid")

        # Plot drag coefficient
        plt.plot(self.time_steps, drag_coefficients, label="Raw")
        if len(self.time_steps) > 10:
            plt.plot(self.time_steps[9:], self.drag_coefficients_ma_10, label="MA 10")
        if len(self.time_steps) > 100:
            plt.plot(self.time_steps[99:], self.drag_coefficients_ma_100, label="MA 100")
        if len(self.time_steps) > 1000:
            plt.plot(self.time_steps[999:], self.drag_coefficients_ma_1000, label="MA 1,000")
        if len(self.time_steps) > 10000:
            plt.plot(self.time_steps[9999:], self.drag_coefficients_ma_10000, label="MA 10,000")
        if len(self.time_steps) > 100000:
            plt.plot(self.time_steps[99999:], self.drag_coefficients_ma_100000, label="MA 100,000")

        plt.ylim(-1.0, 1.0)
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Drag coefficient")
        plt.savefig("drag_coefficient_ma.png")
        plt.close()


if __name__ == "__main__":
    # Grid parameters
    grid_size_x, grid_size_y, grid_size_z = 512, 128, 128
    grid_shape = (grid_size_x, grid_size_y, grid_size_z)

    # Configuration
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, backend=backend)
    wind_speed = 0.02
    num_steps = 100000
    print_interval = 1000

    # Set up Reynolds number and deduce relaxation time (omega)
    Re = 50000.0
    clength = grid_size_x - 1
    visc = wind_speed * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    # Print simulation info
    print("\n" + "=" * 50 + "\n")
    print("Simulation Configuration:")
    print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
    print(f"Backend: {backend}")
    print(f"Velocity set: {velocity_set}")
    print(f"Precision policy: {precision_policy}")
    print(f"Prescribed velocity: {wind_speed}")
    print(f"Reynolds number: {Re}")
    print(f"Max iterations: {num_steps}")
    print("\n" + "=" * 50 + "\n")

    simulation = WindTunnel3D(omega, wind_speed, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps, print_interval, post_process_interval=1000)
