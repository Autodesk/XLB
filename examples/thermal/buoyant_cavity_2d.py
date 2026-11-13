"""
Mixed convection in a 2D cavity with Boussinesq buoyancy (two-way coupling).

Script version of examples/tutorials/06_buoyant_cavity.ipynb that runs on both the JAX and
the Warp backends. The left wall is hot, the right wall cold, the top and bottom walls are
insulated; both side walls move upwards, and the temperature feeds back on the flow through
the Boussinesq body force

    F = b * (T - T_ref)

with b the buoyancy vector (pointing up). Each step:

  1. the flow advances under the buoyancy force of the previous step,
  2. its velocity is recovered and handed to the thermal solver,
  3. the temperature advances, advected by that velocity,
  4. the temperature is turned into the next buoyancy force with BoussinesqForce.

On Warp every field stays on the device: the velocity is recovered straight into the
thermal stepper's advecting-velocity field, and BoussinesqForce writes the force into a
preallocated field that the flow stepper reads per node.
"""

import xlb
import xlb.velocity_set
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.helper import omega_from_diffusivity
from xlb.operator.stepper import IncompressibleNavierStokesStepper, ThermalStepper
from xlb.operator.boundary_condition import EquilibriumBC, HalfwayBounceBackBC, ScalarDirichletBC, ScalarNeumannBC
from xlb.operator.force import BoussinesqForce
from xlb.operator.macroscopic import Macroscopic, ZeroMoment
from xlb.utils import save_fields_vtk, save_image
import matplotlib.pyplot as plt
import numpy as np


class BuoyantCavity2D:
    def __init__(self, reynolds, prandtl, richardson, wall_vel, grid_shape, compute_backend, precision_policy):
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
        self.wall_vel = wall_vel
        self.richardson = richardson
        self.reynolds = reynolds
        self.t_hot, self.t_cold = 1.0, 0.0

        clength = grid_shape[0] - 1

        # Re sets the momentum diffusivity, Pe = Re * Pr the thermal one
        viscosity = wall_vel * clength / reynolds
        self.omega_flow = 1.0 / (3.0 * viscosity + 0.5)
        self.diffusivity = wall_vel * clength / (reynolds * prandtl)
        self.omega_thermal = omega_from_diffusivity(self.diffusivity, cs2=self.thermal_velocity_set)

        # Ri = |b| * dT * L / U^2 with dT = 1, so |b| = Ri * U^2 / L
        self.buoyancy = np.array([0.0, richardson * wall_vel**2 / clength])

        print(f"Re = {reynolds:g}, Pr = {prandtl:g}, Ri = {richardson:g}")
        print(f"omega_flow = {self.omega_flow:.4f}, omega_thermal = {self.omega_thermal:.4f}")

        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)
        self._setup()

    def _setup(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)

        # Flow: moving side walls, no-slip top and bottom (which own the corners)
        noslip = [box["bottom"][i] + box["top"][i] for i in range(self.flow_velocity_set.d)]
        flow_bcs = [
            HalfwayBounceBackBC(indices=noslip),
            EquilibriumBC(rho=1.0, u=(0.0, self.wall_vel), indices=box_no_edge["left"]),
            EquilibriumBC(rho=1.0, u=(0.0, self.wall_vel), indices=box_no_edge["right"]),
        ]
        # A zero uniform force switches the collision to ForcedCollision; the buoyancy
        # force itself is rebuilt from the temperature and passed on every call
        self.flow_stepper = IncompressibleNavierStokesStepper(
            grid=self.grid,
            boundary_conditions=flow_bcs,
            collision_type="BGK",
            force_vector=np.zeros(self.flow_velocity_set.d),
        )
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.flow_stepper.prepare_fields()

        # Scalar: hot left, cold right, insulated top and bottom (the side walls own the corners)
        insulated = [box_no_edge["bottom"][i] + box_no_edge["top"][i] for i in range(self.thermal_velocity_set.d)]
        thermal_bcs = [
            ScalarDirichletBC(value=self.t_hot, indices=box["left"]),
            ScalarDirichletBC(value=self.t_cold, indices=box["right"]),
            ScalarNeumannBC(flux=0.0, indices=insulated),
        ]
        self.thermal_stepper = ThermalStepper(
            grid=self.grid,
            boundary_conditions=thermal_bcs,
            velocity_set=self.thermal_velocity_set,
        )
        self.g_0, self.g_1, self.bc_mask_thermal, self.missing_mask_thermal = self.thermal_stepper.prepare_fields(phi_init=self.t_cold)
        self.u, self.source = self.thermal_stepper.prepare_aux_fields()

        self.macroscopic = Macroscopic()
        self.zero_moment = ZeroMoment(velocity_set=self.thermal_velocity_set)
        self.boussinesq = BoussinesqForce(buoyancy=self.buoyancy, reference_value=self.t_cold)

        # Warp operators write into preallocated outputs, so every field stays on the device
        d = self.flow_velocity_set.d
        if self.compute_backend == ComputeBackend.WARP:
            self.rho = self.grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
            self.phi = self.grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
            self.force = self.grid.create_field(cardinality=d, dtype=self.precision_policy.store_precision)
        else:
            self.force = np.zeros((d, *self.grid_shape))

    def run(self, num_steps, post_process_interval=1000):
        for i in range(num_steps):
            # 1. Advance the flow under the current buoyancy force
            self.f_0, self.f_1 = self.flow_stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega_flow, i, self.force)
            self.f_0, self.f_1 = self.f_1, self.f_0

            # 2. Recover the velocity and hand it to the scalar
            if self.compute_backend == ComputeBackend.WARP:
                self.rho, self.u = self.macroscopic(self.f_0, self.rho, self.u)
            else:
                _, self.u = self.macroscopic(self.f_0)

            # 3. Advance the scalar, advected by that velocity
            self.g_0, self.g_1 = self.thermal_stepper(
                self.g_0, self.g_1, self.u, self.source, self.bc_mask_thermal, self.missing_mask_thermal, self.omega_thermal, i
            )
            self.g_0, self.g_1 = self.g_1, self.g_0

            # 4. Turn the new temperature into the buoyancy force of the next step
            if self.compute_backend == ComputeBackend.WARP:
                self.phi = self.zero_moment(self.g_0, self.phi)
                self.force = self.boussinesq(self.phi, self.force)
            else:
                self.force = self.boussinesq(self.zero_moment(self.g_0))

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def fields(self):
        if self.compute_backend == ComputeBackend.WARP:
            # Warp fields carry a trailing z axis of size 1 in 2D
            self.rho, self.u = self.macroscopic(self.f_0, self.rho, self.u)
            self.phi = self.zero_moment(self.g_0, self.phi)
            return self.u.numpy()[..., 0], self.phi.numpy()[0, ..., 0]
        _, u = self.macroscopic(self.f_0)
        return np.asarray(u), np.asarray(self.zero_moment(self.g_0))[0]

    def nusselt_number(self, temperature):
        """Average Nusselt number on the hot wall, normalized by the conduction solution.

        The halfway wall sits half a cell outside the first fluid column, so the wall
        gradient is estimated from that half-cell distance.
        """
        nx = self.grid_shape[0]
        wall_gradient = (self.t_hot - temperature[0]) / 0.5
        conduction_gradient = (self.t_hot - self.t_cold) / nx
        return float(np.mean(wall_gradient) / conduction_gradient)

    def post_process(self, i):
        u, temperature = self.fields()
        u_magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2)

        fields = {"u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude, "T": temperature}
        save_fields_vtk(fields, timestep=i, prefix="buoyant_cavity")
        save_image(temperature, timestep=i, prefix="buoyant_cavity_T")

        nusselt = self.nusselt_number(temperature)
        print(f"step {i:>7d}  T in [{temperature.min():.4f}, {temperature.max():.4f}]  u_max = {u_magnitude.max():.5f}  Nu = {nusselt:.4f}")

    def plot(self):
        u, temperature = self.fields()
        u_magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2)
        x, y = np.arange(self.grid_shape[0]), np.arange(self.grid_shape[1])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        im = axes[0].imshow(temperature.T, origin="lower", cmap="RdBu_r", vmin=self.t_cold, vmax=self.t_hot)
        axes[0].contour(temperature.T, levels=10, colors="k", linewidths=0.4)
        axes[0].set_title("Temperature (hot left, cold right)")
        fig.colorbar(im, ax=axes[0], label="T")

        im = axes[1].imshow(u_magnitude.T, origin="lower", cmap="viridis")
        axes[1].streamplot(x, y, u[0].T, u[1].T, color="w", density=1.0, linewidth=0.5)
        axes[1].set_title("Velocity magnitude")
        fig.colorbar(im, ax=axes[1], label="|u|")

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.suptitle(f"Re = {self.reynolds:g}, Ri = {self.richardson:g}")
        plt.tight_layout()
        plt.savefig("buoyant_cavity.png", dpi=150)
        print("wrote buoyant_cavity.png")


if __name__ == "__main__":
    grid_shape = (128, 128)
    compute_backend = ComputeBackend.JAX  # or ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    simulation = BuoyantCavity2D(
        reynolds=150.0,
        prandtl=0.71,
        richardson=5.0,
        wall_vel=0.05,
        grid_shape=grid_shape,
        compute_backend=compute_backend,
        precision_policy=precision_policy,
    )

    simulation.run(num_steps=50000, post_process_interval=5000)
    simulation.plot()
