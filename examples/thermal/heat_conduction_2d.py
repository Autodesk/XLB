"""
Steady heat conduction in a 2D slab with internal heat generation.

Solves the first stage of the thermal model, the pure heat equation

    rho * cp * dT/dt = k * laplacian(T) + q'''

on a slab that is cooled on the top and bottom faces, insulated on the sides, and heated
uniformly throughout its volume. At steady state the temperature profile across the slab is
the classic parabola

    T(y) = T_wall + (q''' * L^2) / (2 * k) * (y / L) * (1 - y / L)

which the example compares against the simulation.
"""

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.helper import omega_from_diffusivity
from xlb.operator.stepper import ThermalStepper
from xlb.operator.boundary_condition import ScalarDirichletBC, ScalarNeumannBC
from xlb.operator.macroscopic import ZeroMoment
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import matplotlib.pyplot as plt
import numpy as np


class HeatConduction2D:
    def __init__(self, diffusivity, source, wall_temperature, grid_shape, velocity_set, compute_backend, precision_policy):
        xlb.init(
            velocity_set=velocity_set,
            default_backend=compute_backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.compute_backend = compute_backend
        self.precision_policy = precision_policy
        self.wall_temperature = wall_temperature
        self.source_value = source

        # The scalar relaxation rate follows from the lattice diffusivity alpha = k / (rho * cp)
        self.omega = omega_from_diffusivity(diffusivity)
        self.diffusivity = diffusivity

        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)

        self._setup()

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_stepper()

        # The scalar populations and the boundary masks are owned by the thermal stepper
        self.g_0, self.g_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields(phi_init=self.wall_temperature)

        # A zero velocity field reduces the transport equation to pure conduction
        self.u, self.source = self.stepper.prepare_aux_fields()
        self.source = self.source.at[0].set(self.source_value)

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        cooled = [box["bottom"][i] + box["top"][i] for i in range(self.velocity_set.d)]
        # The corner nodes are left to the cooled walls, so the side walls drop their edges
        insulated = [box_no_edge["left"][i] + box_no_edge["right"][i] for i in range(self.velocity_set.d)]
        return cooled, insulated

    def setup_boundary_conditions(self):
        cooled, insulated = self.define_boundary_indices()
        bc_cooled = ScalarDirichletBC(value=self.wall_temperature, indices=cooled)
        bc_insulated = ScalarNeumannBC(flux=0.0, indices=insulated)
        self.boundary_conditions = [bc_cooled, bc_insulated]

    def setup_stepper(self):
        self.stepper = ThermalStepper(
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
        )

    def run(self, num_steps, post_process_interval=1000):
        for i in range(num_steps):
            self.g_0, self.g_1 = self.stepper(self.g_0, self.g_1, self.u, self.source, self.bc_mask, self.missing_mask, self.omega, i)
            self.g_0, self.g_1 = self.g_1, self.g_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def temperature(self):
        """Recover the temperature field as the zeroth moment of the scalar populations."""
        return np.asarray(ZeroMoment()(self.g_0))[0]

    def analytical_profile(self):
        """Steady parabolic profile across the slab, evaluated at the cell centres."""
        ny = self.grid_shape[1]
        # Halfway walls sit at y = 0 and y = ny, so cell j has its centre at j + 0.5
        y = (np.arange(ny) + 0.5) / ny
        peak = self.source_value * ny**2 / (2.0 * self.diffusivity)
        return self.wall_temperature + peak * y * (1.0 - y)

    def post_process(self, i):
        temperature = self.temperature()

        save_fields_vtk({"T": temperature}, timestep=i, prefix="heat_conduction")
        save_image(temperature, timestep=i, prefix="heat_conduction")

        # Compare the mid-slab profile against the analytical parabola. The mid-column is
        # used because the corners of the domain mix the Dirichlet and Neumann walls.
        profile = temperature[self.grid_shape[0] // 2]
        expected = self.analytical_profile()
        error = np.abs(profile - expected).max() / expected.max()
        print(f"step {i:>7d}  T_max = {profile.max():.6f}  (analytical {expected.max():.6f})  rel. error = {error:.2e}")

    def plot_profile(self):
        profile = self.temperature()[self.grid_shape[0] // 2]
        y = (np.arange(self.grid_shape[1]) + 0.5) / self.grid_shape[1]

        plt.figure(figsize=(6, 4))
        plt.plot(y, self.analytical_profile(), "k-", label="analytical")
        plt.plot(y, profile, "o", markerfacecolor="none", label="XLB")
        plt.xlabel("y / L")
        plt.ylabel("T")
        plt.title("Steady conduction with uniform volumetric heating")
        plt.legend()
        plt.tight_layout()
        plt.savefig("heat_conduction_profile.png", dpi=150)
        print("wrote heat_conduction_profile.png")


if __name__ == "__main__":
    grid_shape = (64, 64)
    compute_backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP64FP64

    # D2Q5 is the natural stencil for a scalar in 2D: it is cheaper than D2Q9 and the
    # diagonal directions contribute nothing to isotropic diffusion at this order
    velocity_set = xlb.velocity_set.D2Q5(precision_policy=precision_policy, compute_backend=compute_backend)

    # Lattice diffusivity alpha = k / (rho * cp). Values near 1/6 give omega = 1 and the
    # best-conditioned relaxation; staying within [0.01, 0.5] keeps the scheme accurate.
    diffusivity = 1.0 / 6.0

    # Volumetric source in lattice units, i.e. q''' * dt / (rho * cp) per step
    source = 1e-6

    simulation = HeatConduction2D(
        diffusivity=diffusivity,
        source=source,
        wall_temperature=0.0,
        grid_shape=grid_shape,
        velocity_set=velocity_set,
        compute_backend=compute_backend,
        precision_policy=precision_policy,
    )

    # Conduction relaxes on the diffusive timescale L^2 / alpha, so the slab needs a few
    # tens of thousands of steps to reach steady state at this resolution
    simulation.run(num_steps=60000, post_process_interval=10000)
    simulation.plot_profile()
