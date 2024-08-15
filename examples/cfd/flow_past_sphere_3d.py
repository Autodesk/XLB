import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    ZouHeBC,
    EquilibriumBC,
    DoNothingBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp


class FlowOverSphere:
    def __init__(self, omega, grid_shape, velocity_set, backend, precision_policy):
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
        self.grid, self.f_0, self.f_1, self.missing_mask, self.boundary_mask = create_nse_fields(grid_shape)
        self.stepper = None
        self.boundary_conditions = []

        # Setup the simulation BC, its initial conditions, and the stepper
        self._setup(omega)

    def _setup(self, omega):
        self.setup_boundary_conditions()
        self.setup_boundary_masks()
        self.initialize_fields()
        self.setup_stepper(omega)

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

        sphere_radius = self.grid_shape[1] // 12
        x = np.arange(self.grid_shape[0])
        y = np.arange(self.grid_shape[1])
        z = np.arange(self.grid_shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        indices = np.where(
            (X - self.grid_shape[0] // 6) ** 2 + (Y - self.grid_shape[1] // 2) ** 2 + (Z - self.grid_shape[2] // 2) ** 2 < sphere_radius**2
        )
        sphere = [tuple(indices[i]) for i in range(self.velocity_set.d)]

        return inlet, outlet, walls, sphere

    def setup_boundary_conditions(self):
        inlet, outlet, walls, sphere = self.define_boundary_indices()
        bc_left = ZouHeBC("velocity", (0.04, 0.0, 0.0), indices=inlet)
        bc_walls = FullwayBounceBackBC(indices=walls)
        bc_outlet = ZouHeBC("pressure", 1.0, indices=outlet)
        bc_sphere = FullwayBounceBackBC(indices=sphere)
        self.boundary_conditions = [bc_left, bc_outlet, bc_sphere, bc_walls]
        # Note: it is important to add bc_walls to be after bc_outlet/bc_inlet because
        # of the corner nodes. This way the corners are treated as wall and not inlet/outlet.
        # TODO: how to ensure about this behind in the src code?

    def setup_boundary_masks(self):
        indices_boundary_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )
        self.boundary_mask, self.missing_mask = indices_boundary_masker(self.boundary_conditions, self.boundary_mask, self.missing_mask, (0, 0, 0))

    def initialize_fields(self):
        self.f_0 = initialize_eq(self.f_0, self.grid, self.velocity_set, self.backend)

    def setup_stepper(self, omega):
        self.stepper = IncompressibleNavierStokesStepper(omega, boundary_conditions=self.boundary_conditions)

    def run(self, num_steps, post_process_interval=100):
        for i in range(num_steps):
            self.f_1 = self.stepper(self.f_0, self.f_1, self.boundary_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

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
        rho = rho[:, 1:-1, 1:-1, 1:-1][0]
        u_magnitude = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5

        fields = {"u_magnitude": u_magnitude,
                  "u_x": u[0],
                  "u_y": u[1],
                  "u_z": u[2],
                  "rho": rho}

        save_fields_vtk(fields, timestep=i)
        save_image(fields["u_magnitude"][:, self.grid_shape[1] // 2, :], timestep=i)


if __name__ == "__main__":
    # Running the simulation
    grid_shape = (512//2, 128//2, 128//2)
    velocity_set = xlb.velocity_set.D3Q19()
    backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP32FP32
    omega = 1.6

    simulation = FlowOverSphere(omega, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps=10000, post_process_interval=1000)
