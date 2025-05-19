import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
import jax.numpy as jnp
import numpy as np


class LidDrivenCavity2D:
    def __init__(self, omega, prescribed_vel, grid_shape, velocity_set, compute_backend, precision_policy):
        # initialize compute_backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=compute_backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.compute_backend = compute_backend
        self.precision_policy = precision_policy
        self.omega = omega
        self.boundary_conditions = []
        self.prescribed_vel = prescribed_vel

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)

        # Setup the simulation BC and stepper
        self._setup()

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_stepper()
        # Initialize fields using the stepper
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        lid = box_no_edge["top"]
        walls = [box["bottom"][i] + box["left"][i] + box["right"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return lid, walls

    def setup_boundary_conditions(self):
        lid, walls = self.define_boundary_indices()
        bc_top = EquilibriumBC(rho=1.0, u=(self.prescribed_vel, 0.0), indices=lid)
        bc_walls = HalfwayBounceBackBC(indices=walls)
        self.boundary_conditions = [bc_walls, bc_top]

    def setup_stepper(self):
        self.stepper = IncompressibleNavierStokesStepper(
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="BGK",
        )

    def run(self, num_steps, post_process_interval=100):
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def post_process(self, i):
        # Write the results. We'll use JAX compute_backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            # If the compute_backend is warp, we need to drop the last dimension added by warp for 2D simulations
            f_0 = wp.to_jax(self.f_0)[..., 0]
        else:
            f_0 = self.f_0

        macro = Macroscopic(
            compute_backend=ComputeBackend.JAX,
            precision_policy=self.precision_policy,
            velocity_set=xlb.velocity_set.D2Q9(precision_policy=self.precision_policy, compute_backend=ComputeBackend.JAX),
        )

        rho, u = macro(f_0)

        # remove boundary cells
        rho = rho[:, 1:-1, 1:-1]
        u = u[:, 1:-1, 1:-1]
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5

        fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude}

        save_fields_vtk(fields, timestep=i, prefix="lid_driven_cavity")
        save_image(fields["u_magnitude"], timestep=i, prefix="lid_driven_cavity")


if __name__ == "__main__":
    # Running the simulation
    grid_size = 500
    grid_shape = (grid_size, grid_size)
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

    # Setting fluid viscosity and relaxation parameter.
    Re = 200.0
    prescribed_vel = 0.05
    clength = grid_shape[0] - 1
    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    simulation = LidDrivenCavity2D(omega, prescribed_vel, grid_shape, velocity_set, compute_backend, precision_policy)
    simulation.run(num_steps=50000, post_process_interval=1000)
