import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, check_bc_overlaps
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
import jax.numpy as jnp
import numpy as np


class LidDrivenCavity2D:
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
        self.grid, self.f_0, self.f_1, self.missing_mask, self.bc_mask = create_nse_fields(grid_shape)
        self.stepper = None
        self.boundary_conditions = []

        # Setup the simulation BC, its initial conditions, and the stepper
        self._setup(omega)

    def _setup(self, omega):
        self.setup_boundary_conditions()
        self.setup_boundary_masker()
        self.initialize_fields()
        self.setup_stepper(omega)

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        lid = box_no_edge["top"]
        walls = [box["bottom"][i] + box["left"][i] + box["right"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return lid, walls

    def setup_boundary_conditions(self):
        lid, walls = self.define_boundary_indices()
        bc_top = EquilibriumBC(rho=1.0, u=(0.02, 0.0), indices=lid)
        bc_walls = HalfwayBounceBackBC(indices=walls)
        self.boundary_conditions = [bc_walls, bc_top]

    def setup_boundary_masker(self):
        # check boundary condition list for duplicate indices before creating bc mask
        check_bc_overlaps(self.boundary_conditions, self.velocity_set.d, self.backend)
        indices_boundary_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )
        self.bc_mask, self.missing_mask = indices_boundary_masker(self.boundary_conditions, self.bc_mask, self.missing_mask)

    def initialize_fields(self):
        self.f_0 = initialize_eq(self.f_0, self.grid, self.velocity_set, self.precision_policy, self.backend)

    def setup_stepper(self, omega):
        self.stepper = IncompressibleNavierStokesStepper(omega, boundary_conditions=self.boundary_conditions)

    def run(self, num_steps, post_process_interval=100):
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def post_process(self, i):
        # Write the results. We'll use JAX backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            # If the backend is warp, we need to drop the last dimension added by warp for 2D simulations
            f_0 = wp.to_jax(self.f_0)[..., 0]
        else:
            f_0 = self.f_0

        macro = Macroscopic(
            compute_backend=ComputeBackend.JAX,
            precision_policy=self.precision_policy,
            velocity_set=xlb.velocity_set.D2Q9(precision_policy=self.precision_policy, backend=ComputeBackend.JAX),
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
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=backend)
    omega = 1.6

    simulation = LidDrivenCavity2D(omega, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps=5000, post_process_interval=1000)
