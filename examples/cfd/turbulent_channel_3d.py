import xlb
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import RegularizedBC
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json


# helper functions for this benchmark example
def vonKarman_loglaw_wall(yplus):
    vonKarmanConst = 0.41
    cplus = 5.5
    uplus = np.log(yplus) / vonKarmanConst + cplus
    return uplus


def get_dns_data():
    """
    Reference: DNS of Turbulent Channel Flow up to Re_tau=590, 1999,
    Physics of Fluids, vol 11, 943-945.
    https://turbulence.oden.utexas.edu/data/MKM/chan180/profiles/chan180.means
    """
    file_name = "examples/cfd/data/turbulent_channel_dns_data.json"
    with open(file_name, "r") as file:
        return json.load(file)


class TurbulentChannel3D:
    def __init__(self, channel_half_width, Re_tau, u_tau, grid_shape, velocity_set, backend, precision_policy):
        # initialize backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

        self.channel_half_width = channel_half_width
        self.Re_tau = Re_tau
        self.u_tau = u_tau
        self.visc = u_tau * channel_half_width / Re_tau
        self.omega = 1.0 / (3.0 * self.visc + 0.5)
        # DeltaPlus = Re_tau / channel_half_width
        # DeltaPlus = u_tau / nu * Delta where u_tau / nu = Re_tau / channel_half_width

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.backend = backend
        self.precision_policy = precision_policy
        self.grid, self.f_0, self.f_1, self.missing_mask, self.bc_mask = create_nse_fields(grid_shape)
        self.stepper = None
        self.boundary_conditions = []

        # Setup the simulation BC, its initial conditions, and the stepper
        self._setup()

    def get_force(self):
        # define the external force
        shape = (self.velocity_set.d,)
        force = np.zeros(shape)
        force[0] = self.Re_tau**2 * self.visc**2 / self.channel_half_width**3
        return force

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_boundary_masker()
        self.initialize_fields()
        self.setup_stepper()

    def define_boundary_indices(self):
        # top and bottom sides of the channel are no-slip and the other directions are periodic
        boundingBoxIndices = self.grid.bounding_box_indices(remove_edges=True)
        walls = [boundingBoxIndices["bottom"][i] + boundingBoxIndices["top"][i] for i in range(self.velocity_set.d)]
        return walls

    def setup_boundary_conditions(self):
        walls = self.define_boundary_indices()
        bc_walls = RegularizedBC("velocity", (0.0, 0.0, 0.0), indices=walls)
        self.boundary_conditions = [bc_walls]

    def setup_boundary_masker(self):
        indices_boundary_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )
        self.bc_mask, self.missing_mask = indices_boundary_masker(self.boundary_conditions, self.bc_mask, self.missing_mask)

    def initialize_fields(self):
        shape = (self.velocity_set.d,) + (self.grid_shape)
        np.random.seed(0)
        u_init = np.random.random(shape)
        if self.backend == ComputeBackend.JAX:
            u_init = jnp.full(shape=shape, fill_value=1e-2 * u_init)
        else:
            u_init = wp.array(1e-2 * u_init, dtype=self.precision_policy.compute_precision.wp_dtype)
        self.f_0 = initialize_eq(self.f_0, self.grid, self.velocity_set, self.precision_policy, self.backend, u=u_init)

    def setup_stepper(self):
        force = self.get_force()
        self.stepper = IncompressibleNavierStokesStepper(
            self.omega, boundary_conditions=self.boundary_conditions, collision_type="KBC", forcing_scheme="exact_difference", force_vector=force
        )

    def run(self, num_steps, print_interval, post_process_interval=100):
        start_time = time.time()
        for i in range(num_steps):
            self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
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

        # compute velocity magnitude
        u_magnitude = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5
        fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_z": u[2], "u_magnitude": u_magnitude}
        save_fields_vtk(fields, timestep=i)
        save_image(fields["u_magnitude"][:, grid_size_y // 2, :], timestep=i)

        # Save monitor plot
        self.plot_uplus(u, i)
        return

    def plot_uplus(self, u, timestep):
        # Compute moving average of drag coefficient, 100, 1000, 10000
        # mean streamwise velocity in wall units u^+(z)
        # Wall distance in wall units to be used inside output_data
        zz = np.arange(self.grid_shape[-1])
        zz = np.minimum(zz, zz.max() - zz)
        yplus = zz * self.u_tau / self.visc
        uplus = np.mean(u[0], axis=(0, 1)) / self.u_tau
        uplus_loglaw = vonKarman_loglaw_wall(yplus)
        dns_dic = get_dns_data()
        plt.clf()
        plt.semilogx(yplus, uplus, "r.", yplus, uplus_loglaw, "k:", dns_dic["y+"], dns_dic["Umean"], "b-")
        ax = plt.gca()
        ax.set_xlim([0.1, 300])
        ax.set_ylim([0, 20])
        fname = "uplus_" + str(timestep // 10000).zfill(5) + ".png"
        plt.savefig(fname, format="png")


if __name__ == "__main__":
    # Problem Configuration
    # h: channel half-width
    channel_half_width = 50

    # Define channel geometry based on h
    grid_size_x = 6 * channel_half_width
    grid_size_y = 3 * channel_half_width
    grid_size_z = 2 * channel_half_width

    # Grid parameters
    grid_shape = (grid_size_x, grid_size_y, grid_size_z)

    # Define flow regime
    # Set up Reynolds number and deduce relaxation time (omega)
    Re_tau = 180
    u_tau = 0.001

    # Runtime & backend configurations
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP64FP64
    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, backend=backend)
    num_steps = 10000000
    print_interval = 100000

    # Print simulation info
    print("\n" + "=" * 50 + "\n")
    print("Simulation Configuration:")
    print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
    print(f"Backend: {backend}")
    print(f"Velocity set: {velocity_set}")
    print(f"Precision policy: {precision_policy}")
    print(f"Reynolds number: {Re_tau}")
    print(f"Max iterations: {num_steps}")
    print("\n" + "=" * 50 + "\n")

    simulation = TurbulentChannel3D(channel_half_width, Re_tau, u_tau, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps, print_interval, post_process_interval=100000)
