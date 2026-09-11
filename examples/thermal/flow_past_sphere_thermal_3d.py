"""
Flow past a cold sphere in 3D, with the temperature advected by the flow (passive scalar).

This is the 3D counterpart of the tutorials/05_flow_past_square_thermal_2d.ipynb notebook,
combining the flow set-up of examples/cfd/flow_past_sphere_3d.py with a D3Q7 thermal
solver: warm fluid enters a channel at a parabolic inlet profile, meets a sphere held at a
fixed cold temperature, and carries a cold wake downstream.

Two solvers, two independent stencils. The flow keeps the default D3Q19 (or D3Q27); the
scalar uses D3Q7, the minimal 3D stencil that recovers the advection-diffusion equation --
see xlb/velocity_set/d3q7.py for why it has cs^2 = 1/4 rather than the 1/3 used everywhere
else in XLB, and why omega_from_diffusivity must be told that explicitly.

The coupling is one-way and explicit: at every step the flow advances, the velocity is
recovered from its populations, and the thermal solver advances using that velocity. The
scalar exerts no force back on the flow; that is the Boussinesq stage
(examples/tutorials/06_buoyant_cavity.ipynb), not implemented here.

Boundary condition choices, and why they matter
------------------------------------------------
This case has an inlet AND an outlet, unlike a closed cavity, and getting the outlet and
the corner/edge assignment wrong is the single most common way to make this kind of
simulation diverge silently. Three rules, established and validated on the 2D version of
this exact case:

  1. The outlet must use ScalarOutflowBC, not ScalarNeumannBC. Bounce-back
     (ScalarNeumannBC) nulls the *total* normal flux, advective plus diffusive, which is
     the right adiabatic condition at a no-slip wall (u = 0) but reflects the outgoing
     scalar straight back into the domain wherever u . n != 0 -- exactly the situation at
     an outlet. See bc_scalar_outflow.py for the full derivation and the exact-answer test
     that demonstrates the difference (T staying at 1 vs. diverging to O(10-100)).
  2. Adiabatic walls must not own an edge/corner shared with the inlet or outlet. XLB
     flags a lattice direction as missing wherever nothing streams in from outside the
     domain; on the domain's end faces the streamwise direction is legitimately missing
     for that reason. A wall that also owns that edge bounces the streamwise direction
     back too, reflecting the domain's own through-flow. Walls are therefore built with
     `remove_edges=True`; the inlet and outlet take the full face.
  3. The flow's ExtrapolationOutflowBC does not leave a physically meaningful macroscopic
     velocity on the outlet face -- recovering u there gives a value close to zero
     regardless of the real flow speed one cell upstream, because those populations are
     extrapolation bookkeeping rather than a converged hydrodynamic state. The flow
     solver never reads that value back, so it never mattered before; the scalar solver
     does read it, and a near-zero velocity face looks to it like a stagnation wall
     sitting across the exit. The velocity is repaired (copied from the interior
     neighbour plane) before being handed to the thermal stepper.

Run directly to execute a short simulation and write periodic VTK/PNG snapshots.
"""

import numpy as np
import jax.numpy as jnp

import xlb
import xlb.velocity_set
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.helper import omega_from_diffusivity
from xlb.operator.stepper import IncompressibleNavierStokesStepper, ThermalStepper
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    ScalarDirichletBC,
    ScalarOutflowBC,
    ScalarNeumannBC,
)
from xlb.operator.macroscopic import Macroscopic, ZeroMoment
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.utils import save_fields_vtk, save_image


class FlowPastSphereThermal3D:
    def __init__(
        self,
        reynolds,
        peclet,
        u_inlet,
        grid_shape,
        sphere_radius=None,
        compute_backend=ComputeBackend.JAX,
        precision_policy=PrecisionPolicy.FP64FP64,
    ):
        self.flow_velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend)
        xlb.init(
            velocity_set=self.flow_velocity_set,
            default_backend=compute_backend,
            default_precision_policy=precision_policy,
        )

        # The minimal 3D scalar stencil: seven velocities, cs^2 = 1/4 (not 1/3)
        self.thermal_velocity_set = xlb.velocity_set.D3Q7(precision_policy=precision_policy, compute_backend=compute_backend)

        self.grid_shape = grid_shape
        self.compute_backend = compute_backend
        self.precision_policy = precision_policy
        self.u_inlet = u_inlet
        self.sphere_radius = sphere_radius or grid_shape[1] // 12
        self.sphere_center = (grid_shape[0] // 6, grid_shape[1] // 2, grid_shape[2] // 2)

        clength = 2.0 * self.sphere_radius

        # Reynolds number sets the momentum diffusivity, Peclet number the thermal one
        viscosity = u_inlet * clength / reynolds
        self.omega_flow = 1.0 / (3.0 * viscosity + 0.5)

        self.diffusivity = u_inlet * clength / peclet
        # D3Q7 has cs^2 = 1/4: passing the velocity set (rather than relying on the
        # default 1/3) is what keeps this correct on the minimal 3D scalar stencil.
        self.omega_thermal = omega_from_diffusivity(self.diffusivity, cs2=self.thermal_velocity_set)

        print(f"Domain:      {grid_shape}")
        print(f"Sphere:      radius={self.sphere_radius}, center={self.sphere_center}")
        print(f"Re = {reynolds:g}, Pe = {peclet:g}, Pr = {viscosity / self.diffusivity:.3g}")
        print(f"omega_flow = {self.omega_flow:.4f}, omega_thermal = {self.omega_thermal:.4f}")

        cell_peclet = u_inlet / self.diffusivity
        print(f"Cell Peclet u*dx/alpha = {cell_peclet:.2f}", end="")
        if cell_peclet > 2.0:
            print(" (> ~2: expect some over/undershoot at the wake edges; this is a "
                  "resolution limit of the scheme, not a boundary-condition defect)")
        else:
            print(" (below the monotonicity limit)")

        self.grid = grid_factory(grid_shape, compute_backend=compute_backend)
        self._setup()

    # ------------------------------------------------------------------
    # Geometry and boundary indices
    # ------------------------------------------------------------------

    def _sphere_indices(self):
        cx, cy, cz = self.sphere_center
        x = np.arange(self.grid_shape[0])
        y = np.arange(self.grid_shape[1])
        z = np.arange(self.grid_shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 < self.sphere_radius**2
        idx = np.where(mask)
        return [tuple(idx[i].tolist()) for i in range(self.flow_velocity_set.d)]

    def define_flow_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        inlet = box_no_edge["left"]
        outlet = box_no_edge["right"]
        walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(self.flow_velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return inlet, outlet, walls

    def define_thermal_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)

        # Inlet and outlet take the FULL face (including its edges), and the adiabatic
        # walls use remove_edges=True, so a wall never owns a node where the streamwise
        # direction is the one legitimately missing (see module docstring, rule 2).
        thermal_inlet = box["left"]
        thermal_outlet = box["right"]
        thermal_walls = [
            box_no_edge["bottom"][i] + box_no_edge["top"][i] + box_no_edge["front"][i] + box_no_edge["back"][i]
            for i in range(self.thermal_velocity_set.d)
        ]
        thermal_walls = np.unique(np.array(thermal_walls), axis=-1).tolist()
        return thermal_inlet, thermal_outlet, thermal_walls

    # ------------------------------------------------------------------
    # Steppers
    # ------------------------------------------------------------------

    def _inlet_profile(self):
        ny, nz = self.grid_shape[1], self.grid_shape[2]
        H_y = float(ny - 1)
        H_z = float(nz - 1)
        u_max = self.u_inlet

        def profile_jax():
            y = jnp.arange(ny)
            z = jnp.arange(nz)
            Y, Z = jnp.meshgrid(y, z, indexing="ij")
            y_center = Y - H_y / 2.0
            z_center = Z - H_z / 2.0
            r_squared = (2.0 * y_center / H_y) ** 2 + (2.0 * z_center / H_z) ** 2
            u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
            return jnp.stack([u_x, jnp.zeros_like(u_x), jnp.zeros_like(u_x)])

        return profile_jax

    def _setup(self):
        inlet, outlet, walls = self.define_flow_boundary_indices()
        self.sphere = self._sphere_indices()

        flow_bcs = [
            HalfwayBounceBackBC(indices=walls),
            RegularizedBC("velocity", profile=self._inlet_profile(), indices=inlet),
            ExtrapolationOutflowBC(indices=outlet),
            HalfwayBounceBackBC(indices=self.sphere),
        ]
        self.bc_sphere = flow_bcs[-1]
        self.flow_stepper = IncompressibleNavierStokesStepper(
            grid=self.grid,
            boundary_conditions=flow_bcs,
            collision_type="BGK",
        )
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.flow_stepper.prepare_fields()

        thermal_inlet, thermal_outlet, thermal_walls = self.define_thermal_boundary_indices()
        thermal_bcs = [
            ScalarNeumannBC(flux=0.0, indices=thermal_walls),
            ScalarOutflowBC(indices=thermal_outlet),
            ScalarDirichletBC(value=1.0, indices=thermal_inlet),
            ScalarDirichletBC(value=0.0, indices=self.sphere),
        ]
        self.thermal_stepper = ThermalStepper(
            grid=self.grid,
            boundary_conditions=thermal_bcs,
            velocity_set=self.thermal_velocity_set,
        )

        # The initial field must agree with the boundary conditions: the bulk starts at
        # the inlet temperature, but the sphere's own nodes start at ITS Dirichlet value
        # (0), or anti-bounce-back computes g = -g_pre[opp] on the very first step and
        # gives exactly T = -1 on every sphere node, a spike that then advects downstream.
        phi_init = np.ones((1, *self.grid_shape))
        phi_init[0, self.sphere[0], self.sphere[1], self.sphere[2]] = 0.0

        self.g_0, self.g_1, self.bc_mask_thermal, self.missing_mask_thermal = self.thermal_stepper.prepare_fields(phi_init=phi_init)
        self.u_thermal, self.source_thermal = self.thermal_stepper.prepare_aux_fields()

        self.macroscopic = Macroscopic(
            compute_backend=self.compute_backend,
            precision_policy=self.precision_policy,
            velocity_set=self.flow_velocity_set,
        )
        self.zero_moment = ZeroMoment(
            velocity_set=self.thermal_velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.momentum_transfer = MomentumTransfer(self.bc_sphere, compute_backend=self.compute_backend)
        self.sphere_cross_section = np.pi * self.sphere_radius**2

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def sanitize_outlet_velocity(self, u):
        """Repair the velocity on the outlet face before handing it to the scalar solver.

        See the module docstring (rule 3): ExtrapolationOutflowBC leaves a macroscopic
        velocity close to zero on its own face, which the flow solver never reads back but
        which would otherwise look like a stagnation wall to the thermal solver.
        """
        return u.at[:, -1, :, :].set(u[:, -2, :, :])

    def drag_lift_coefficients(self):
        force = self.momentum_transfer(self.f_0, self.f_1, self.bc_mask, self.missing_mask)
        cd = 2.0 * force[0] / (self.u_inlet**2 * self.sphere_cross_section)
        cl = 2.0 * force[2] / (self.u_inlet**2 * self.sphere_cross_section)
        return float(cd), float(cl)

    def run(self, num_steps, post_process_interval=500):
        for step in range(num_steps):
            # 1. Advance the flow
            self.f_0, self.f_1 = self.flow_stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega_flow, step)
            self.f_0, self.f_1 = self.f_1, self.f_0

            # 2. Recover the velocity, repair the outlet face, hand it to the scalar
            _, u_flow = self.macroscopic(self.f_0)
            self.u_thermal = self.sanitize_outlet_velocity(jnp.asarray(u_flow))

            # 3. Advance the scalar, advected by that velocity
            self.g_0, self.g_1 = self.thermal_stepper(
                self.g_0, self.g_1, self.u_thermal, self.source_thermal,
                self.bc_mask_thermal, self.missing_mask_thermal,
                self.omega_thermal, step,
            )
            self.g_0, self.g_1 = self.g_1, self.g_0

            if step % post_process_interval == 0 or step == num_steps - 1:
                self.post_process(step)

    def fields(self):
        rho, u = self.macroscopic(self.f_0)
        temperature = self.zero_moment(self.g_0)
        return np.asarray(rho)[0], np.asarray(u), np.asarray(temperature)[0]

    def post_process(self, step):
        rho, u, temperature = self.fields()
        u_magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        cd, cl = self.drag_lift_coefficients()

        fields = {
            "rho": rho,
            "u_x": u[0],
            "u_y": u[1],
            "u_z": u[2],
            "u_magnitude": u_magnitude,
            "T": temperature,
        }
        save_fields_vtk(fields, timestep=step, prefix="flow_past_sphere_thermal_3d")

        # save_image only accepts 2D (or 3D-vector-over-2D) fields, so slice at the
        # sphere's mid-plane rather than passing the full 3D volume
        mid_z = self.grid_shape[2] // 2
        save_image(temperature[:, :, mid_z], timestep=step, prefix="flow_past_sphere_thermal_3d_T")

        print(
            f"step {step:>7d}  Cd={cd:8.4f} Cl={cl:8.4f}  "
            f"T in [{temperature.min():.4f}, {temperature.max():.4f}]  u_max = {u_magnitude.max():.5f}"
        )


if __name__ == "__main__":
    grid_shape = (192, 64, 64)
    compute_backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP64FP64

    simulation = FlowPastSphereThermal3D(
        reynolds=300.0,
        peclet=300.0,
        u_inlet=0.04,
        grid_shape=grid_shape,
        compute_backend=compute_backend,
        precision_policy=precision_policy,
    )

    simulation.run(num_steps=10000, post_process_interval=1000)
