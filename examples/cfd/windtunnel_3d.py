import xlb
import trimesh
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    EquilibriumBC,
    DoNothingBC,
    RegularizedBC,
    HalfwayBounceBackBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp


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
        self.grid, self.f_0, self.f_1, self.missing_mask, self.boundary_map = create_nse_fields(grid_shape)
        self.stepper = None
        self.boundary_conditions = []

        # Setup the simulation BC, its initial conditions, and the stepper
        self._setup(omega, wind_speed)

    def _setup(self, omega, wind_speed):
        self.setup_boundary_conditions(wind_speed)
        self.setup_boundary_masker()
        self.initialize_fields()
        self.setup_stepper(omega)

    def voxelize_stl(self, stl_filename, length_lbm_unit):
        mesh = trimesh.load_mesh(stl_filename, process=False)
        length_phys_unit = mesh.extents.max()
        pitch = length_phys_unit / length_lbm_unit
        mesh_voxelized = mesh.voxelized(pitch=pitch)
        mesh_matrix = mesh_voxelized.matrix
        return mesh_matrix, pitch

    def define_boundary_indices(self):
        inlet = self.grid.boundingBoxIndices["left"]
        outlet = self.grid.boundingBoxIndices["right"]
        walls = [
            self.grid.boundingBoxIndices["bottom"][i]
            + self.grid.boundingBoxIndices["top"][i]
            + self.grid.boundingBoxIndices["front"][i]
            + self.grid.boundingBoxIndices["back"][i]
            for i in range(self.velocity_set.d)
        ]

        # Load the mesh
        stl_filename = "examples/cfd/stl-files/DrivAer-Notchback.stl"
        mesh = trimesh.load_mesh(stl_filename, process=False)
        mesh_points = mesh.vertices

        # Transform the mesh points to be located in the right position in the wind tunnel
        mesh_points -= mesh_points.min(axis=0)
        mesh_extents = mesh_points.max(axis=0)
        length_phys_unit = mesh_extents.max()
        length_lbm_unit = self.grid_shape[0] / 4
        dx = length_phys_unit / length_lbm_unit
        shift = np.array([self.grid_shape[0] * dx / 4, (self.grid_shape[1] * dx - mesh_extents[1]) / 2, 0.0])
        car = mesh_points + shift
        self.grid_spacing = dx

        return inlet, outlet, walls, car

    def setup_boundary_conditions(self, wind_speed):
        inlet, outlet, walls, car = self.define_boundary_indices()
        bc_left = EquilibriumBC(rho=1.0, u=(wind_speed, 0.0, 0.0), indices=inlet)
        # bc_left = RegularizedBC('velocity', (wind_speed, 0.0, 0.0), indices=inlet)
        bc_walls = FullwayBounceBackBC(indices=walls)
        bc_do_nothing = ExtrapolationOutflowBC(indices=outlet)
        bc_car = HalfwayBounceBackBC(mesh_points=car)
        # bc_car = FullwayBounceBackBC(mesh_points=car)
        self.boundary_conditions = [bc_left, bc_do_nothing, bc_walls, bc_car]

    def setup_boundary_masker(self):
        indices_boundary_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )
        mesh_boundary_masker = MeshBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )
        bclist_other = self.boundary_conditions[:-1]
        bc_mesh = self.boundary_conditions[-1]
        dx = self.grid_spacing
        origin, spacing = (0, 0, 0), (dx, dx, dx)
        self.boundary_map, self.missing_mask = indices_boundary_masker(bclist_other, self.boundary_map, self.missing_mask)
        self.boundary_map, self.missing_mask = mesh_boundary_masker(bc_mesh, origin, spacing, self.boundary_map, self.missing_mask)

    def initialize_fields(self):
        self.f_0 = initialize_eq(self.f_0, self.grid, self.velocity_set, self.backend)

    def setup_stepper(self, omega):
        self.stepper = IncompressibleNavierStokesStepper(omega, boundary_conditions=self.boundary_conditions, collision_type="KBC")

    def run(self, num_steps, print_interval, post_process_interval=100):
        start_time = time.time()
        for i in range(num_steps):
            self.f_1 = self.stepper(self.f_0, self.f_1, self.boundary_map, self.missing_mask, i)
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

        macro = Macroscopic(compute_backend=ComputeBackend.JAX)

        rho, u = macro(f_0)

        # remove boundary cells
        u = u[:, 1:-1, 1:-1, 1:-1]
        u_magnitude = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5

        fields = {"u_magnitude": u_magnitude}

        save_fields_vtk(fields, timestep=i)
        save_image(fields["u_magnitude"][:, grid_size_y // 2, :], timestep=i)


if __name__ == "__main__":
    # Grid parameters
    grid_size_x, grid_size_y, grid_size_z = 512, 128, 128
    grid_shape = (grid_size_x, grid_size_y, grid_size_z)

    # Configuration
    backend = ComputeBackend.WARP
    velocity_set = xlb.velocity_set.D3Q27()
    precision_policy = PrecisionPolicy.FP32FP32
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
