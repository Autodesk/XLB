"""
Passive scalar transport in a lid-driven cavity.

Solves the second stage of the thermal model, advection-diffusion of a passive scalar,

    rho * cp * (dT/dt + u . grad(T)) = k * laplacian(T) + q'''

The flow is the standard lid-driven cavity: a moving top wall drives a recirculation inside
a closed box. The temperature is carried by that recirculation while being conducted between
a hot left wall and a cold right wall, with insulated top and bottom faces.

The coupling is one-way and explicit. Each step the velocity is recovered from the flow
populations and handed to the thermal stepper, which owns a separate set of populations on
its own (D2Q5) stencil. The scalar exerts no force back on the flow; that is the Boussinesq
stage, which builds on this example.
"""

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.helper import omega_from_diffusivity
from xlb.operator.stepper import IncompressibleNavierStokesStepper, ThermalStepper
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    EquilibriumBC,
    ScalarDirichletBC,
    ScalarNeumannBC,
)
from xlb.operator.macroscopic import Macroscopic, ZeroMoment
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import matplotlib.pyplot as plt
import numpy as np


class ThermalLidDrivenCavity2D:
    def __init__(self, reynolds, peclet, prescribed_vel, grid_shape, compute_backend, precision_policy):
        self.flow_velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

        xlb.init(
            velocity_set=self.flow_velocity_set,
            default_backend=compute_backend,
            default_precision_policy=precision_policy,
        )

        # A scalar in 2D needs only the five-velocity stencil
        self.thermal_velocity_set = xlb.velocity_set.D2Q5(precision_policy=precision_policy, compute_backend=compute_backend)

        self.grid_shape = grid_shape
        self.compute_backend = compute_backend
        self.precision_policy = precision_policy
        self.prescribed_vel = prescribed_vel

        clength = grid_shape[0] - 1

        # Reynolds number sets the momentum diffusivity, Peclet number the thermal one
        viscosity = prescribed_vel * clength / reynolds
        self.omega = 1.0 / (3.0 * viscosity + 0.5)

        self.diffusivity = prescribed_vel * clength / peclet
        self.omega_thermal = omega_from_diffusivity(self.diffusivity)

        print(f"Re = {reynolds:g}, Pe = {peclet:g}, Pr = {viscosity / self.diffusivity:.3g}")
        print(f"omega_flow = {self.omega:.4f}, omega_thermal = {self.omega_thermal:.4f}")

        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)

        self._setup()

    def _setup(self):
        self.setup_stepper()
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.flow_stepper.prepare_fields()

        # The thermal stepper carries its own populations and its own boundary masks, so the
        # thermal and flow boundary conditions are completely independent
        self.g_0, self.g_1, self.bc_mask_thermal, self.missing_mask_thermal = self.thermal_stepper.prepare_fields(phi_init=0.0)
        self.u, self.source = self.thermal_stepper.prepare_aux_fields()

        self.macroscopic = Macroscopic()
        self.zero_moment = ZeroMoment(
            velocity_set=self.thermal_velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        lid = box_no_edge["top"]
        walls = [box["bottom"][i] + box["left"][i] + box["right"][i] for i in range(self.flow_velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return lid, walls

    def define_thermal_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        hot = box["left"]
        cold = box["right"]
        # The corners are left to the hot and cold walls, so the insulated faces drop them
        insulated = [box_no_edge["bottom"][i] + box_no_edge["top"][i] for i in range(self.thermal_velocity_set.d)]
        return hot, cold, insulated

    def setup_stepper(self):
        lid, walls = self.define_boundary_indices()
        flow_bcs = [
            HalfwayBounceBackBC(indices=walls),
            EquilibriumBC(rho=1.0, u=(self.prescribed_vel, 0.0), indices=lid),
        ]
        self.flow_stepper = IncompressibleNavierStokesStepper(
            grid=self.grid,
            boundary_conditions=flow_bcs,
            collision_type="BGK",
        )

        hot, cold, insulated = self.define_thermal_boundary_indices()
        thermal_bcs = [
            ScalarDirichletBC(value=1.0, indices=hot),
            ScalarDirichletBC(value=0.0, indices=cold),
            ScalarNeumannBC(flux=0.0, indices=insulated),
        ]
        self.thermal_stepper = ThermalStepper(
            grid=self.grid,
            boundary_conditions=thermal_bcs,
            velocity_set=self.thermal_velocity_set,
        )

    def run(self, num_steps, post_process_interval=1000):
        for i in range(num_steps):
            # Advance the flow
            self.f_0, self.f_1 = self.flow_stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            # Hand the freshly computed velocity to the scalar, then advance the scalar
            _, self.u = self.macroscopic(self.f_0)
            self.g_0, self.g_1 = self.thermal_stepper(
                self.g_0, self.g_1, self.u, self.source, self.bc_mask_thermal, self.missing_mask_thermal, self.omega_thermal, i
            )
            self.g_0, self.g_1 = self.g_1, self.g_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def fields(self):
        rho, u = self.macroscopic(self.f_0)
        temperature = self.zero_moment(self.g_0)
        return np.asarray(rho)[0], np.asarray(u), np.asarray(temperature)[0]

    def post_process(self, i):
        rho, u, temperature = self.fields()
        u_magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2)

        fields = {"rho": rho, "u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude, "T": temperature}
        save_fields_vtk(fields, timestep=i, prefix="thermal_lid_driven_cavity")
        save_image(temperature, timestep=i, prefix="thermal_lid_driven_cavity_T")

        # The wall-to-wall heat flow is the useful integral quantity. It is reported as a
        # Nusselt number, i.e. the total flux normalized by the pure-conduction flux.
        nusselt = self.nusselt_number(temperature)
        print(f"step {i:>7d}  T in [{temperature.min():.4f}, {temperature.max():.4f}]  u_max = {u_magnitude.max():.5f}  Nu = {nusselt:.4f}")

    def nusselt_number(self, temperature):
        """Average Nusselt number on the hot wall, normalized by the conduction solution.

        The halfway wall sits half a cell outside the first fluid column, so the wall
        gradient is estimated from that half-cell distance.
        """
        nx = self.grid_shape[0]
        wall_gradient = (1.0 - temperature[0]) / 0.5
        conduction_gradient = 1.0 / nx
        return float(np.mean(wall_gradient) / conduction_gradient)

    def plot(self):
        _, u, temperature = self.fields()
        u_magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        im = axes[0].imshow(temperature.T, origin="lower", cmap="RdBu_r", vmin=0.0, vmax=1.0)
        axes[0].contour(temperature.T, levels=12, colors="k", linewidths=0.4)
        axes[0].set_title("Temperature (hot left, cold right)")
        fig.colorbar(im, ax=axes[0], label="T")

        im = axes[1].imshow(u_magnitude.T, origin="lower", cmap="viridis")
        axes[1].set_title("Velocity magnitude")
        fig.colorbar(im, ax=axes[1], label="|u|")

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        plt.savefig("thermal_lid_driven_cavity.png", dpi=150)
        print("wrote thermal_lid_driven_cavity.png")


if __name__ == "__main__":
    grid_shape = (128, 128)
    compute_backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP32FP32

    simulation = ThermalLidDrivenCavity2D(
        reynolds=200.0,
        peclet=200.0,
        prescribed_vel=0.05,
        grid_shape=grid_shape,
        compute_backend=compute_backend,
        precision_policy=precision_policy,
    )

    simulation.run(num_steps=20000, post_process_interval=4000)
    simulation.plot()
