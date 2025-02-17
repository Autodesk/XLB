import xlb
import trimesh
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, check_bc_overlaps
from xlb.operator.stepper import IBMStepper, IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    EquilibriumBC,
    DoNothingBC,
    RegularizedBC,
    HalfwayBounceBackBC,
    ExtrapolationOutflowBC,
    GradsApproximationBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker, MeshDistanceBoundaryMasker
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from xlb.helper.ibm_helper import prepare_immersed_boundary


class FlowOverSphereIBM:
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
        self.u_max = 0.01

        # Setup the simulation BC, its initial conditions, and the stepper
        self._setup(omega)

        # Add these attributes for IBM
        sphere_mesh = trimesh.creation.icosphere(radius=1.0 / np.sqrt(3))
        self.vertices_wp, self.vertex_areas_wp = prepare_immersed_boundary(
            sphere_mesh,
            max_lbm_length=20,  # Size of sphere in lattice units
            translation=[grid_shape[0] // 4, grid_shape[1] // 2, grid_shape[2] // 2],  # Place sphere at 1/4 of domain
        )
        self.velocities_wp = wp.zeros(shape=self.vertices_wp.shape[0], dtype=wp.vec3)

    def _setup(self, omega):
        self.setup_boundary_conditions()
        self.setup_boundary_masker()
        self.initialize_fields()
        self.setup_stepper(omega)

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        inlet = box_no_edge["left"]
        outlet = box_no_edge["right"]
        walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()

        return inlet, outlet, walls

    def setup_boundary_conditions(self):
        inlet, outlet, walls = self.define_boundary_indices()
        bc_left = RegularizedBC("velocity", indices=inlet)
        # bc_left = EquilibriumBC(rho = 1, u=(0.04, 0.0, 0.0), indices=inlet)
        bc_walls = FullwayBounceBackBC(indices=walls)
        # bc_outlet = RegularizedBC("pressure", 1.0, indices=outlet)
        # bc_outlet = DoNothingBC(indices=outlet)
        bc_outlet = ExtrapolationOutflowBC(indices=outlet)
        self.boundary_conditions = [bc_walls, bc_left, bc_outlet]

    def setup_boundary_masker(self):
        # check boundary condition list for duplicate indices before creating bc mask
        check_bc_overlaps(self.boundary_conditions, self.velocity_set.d, self.backend)

        indices_boundary_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )
        self.bc_mask, self.missing_mask = indices_boundary_masker(self.boundary_conditions, self.bc_mask, self.missing_mask, (0, 0, 0))

    def initialize_fields(self):
        self.f_0 = initialize_eq(self.f_0, self.grid, self.velocity_set, self.precision_policy, self.backend)

    def bc_profile(self):
        # Get the grid dimensions for the y and z directions
        _dtype = self.precision_policy.store_precision.wp_dtype
        u_max = _dtype(self.u_max)  # u_max = 0.04
        H_y = _dtype(self.grid_shape[1] - 1)  # Height in y direction
        H_z = _dtype(self.grid_shape[2] - 1)  # Height in z direction

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            # Poiseuille flow profile: parabolic velocity distribution
            y = _dtype(index[1])
            z = _dtype(index[2])

            # Calculate normalized distance from center
            y_center = y - (H_y / self.precision_policy.store_precision.wp_dtype(2.0))
            z_center = z - (H_z / _dtype(2.0))
            r_squared = (_dtype(2.0) * y_center / H_y) ** _dtype(2.0) + (_dtype(2.0) * z_center / H_z) ** _dtype(2.0)

            # Parabolic profile: u = u_max * (1 - r²)
            return wp.vec(u_max * wp.max(_dtype(0.0), _dtype(1.0) - r_squared), _dtype(0.0), _dtype(0.0), _dtype(0.0), _dtype(0.0), length=5)

        return bc_profile_warp

    def initialize_bc_aux_data(self):
        for bc in self.boundary_conditions:
            if bc.needs_aux_init:
                self.f_0, self.f_1 = bc.aux_data_init(self.bc_profile(), self.f_0, self.f_1, self.bc_mask, self.missing_mask)

    def setup_stepper(self, omega):
        # Add oscillation parameters
        _oscillation_amplitude = 10.0  # Amplitude in lattice units
        _oscillation_period = 10000.0  # Period in timesteps
        y_shape = self.grid_shape[1]
        # Define the update kernel for sinusoidal motion
        @wp.kernel
        def lagr_update_kernel(timestep: int, forces: wp.array(dtype=wp.vec3), vertices: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3)):
            idx = wp.tid()

            # Calculate vertical displacement and velocity based on time
            t = float(timestep)
            omega = 2.0 * 3.14159 / _oscillation_period

            # Update position - only modify y component for vertical motion
            # Keep original x,z coordinates
            current_y = vertices[idx][1]
            base_y = float(y_shape // 2)  # Center position

            # Calculate new y position and velocity
            new_y = base_y + _oscillation_amplitude * wp.sin(omega * t)
            new_vy = _oscillation_amplitude * omega * wp.cos(omega * t)

            # Update vertex position
            vertices[idx] = wp.vec3(vertices[idx][0], new_y, vertices[idx][2])

            # Update velocity (only y component changes)
            velocities[idx] = wp.vec3(0.0, new_vy, 0.0)

        self.initialize_bc_aux_data()
        self.stepper = IBMStepper(omega, self.grid_shape, lagr_update_kernel, self.boundary_conditions, collision_type="BGK")

    def run(self, num_steps, print_interval, post_process_interval=100):
        start_time = time.time()
        for i in range(num_steps):
            # Update the stepper call to include IBM parameters
            self.f_0, self.f_1 = self.stepper(
                self.f_0,
                self.f_1,
                self.vertices_wp,  # Add vertices for IBM
                self.vertex_areas_wp,  # Add vertex areas for IBM
                self.velocities_wp,  # Add velocities for IBM
                self.bc_mask,
                self.missing_mask,
                i,
            )

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
            velocity_set=xlb.velocity_set.D3Q19(precision_policy=self.precision_policy, backend=ComputeBackend.JAX),
        )

        rho, u = macro(f_0)

        # remove boundary cells
        u = u[:, 1:-1, 1:-1, 1:-1]
        u_magnitude = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5

        fields = {"u_magnitude": u_magnitude}

        # save_fields_vtk(fields, timestep=i)
        save_image(fields["u_magnitude"][:, self.grid_shape[1] // 2, :], timestep=i)

        return


if __name__ == "__main__":
    # Grid parameters
    grid_shape = (512 // 2, 128 // 2, 128 // 2)

    # Configuration
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, backend=backend)

    u_max = 0.03
    num_steps = 10000
    print_interval = 1000

    Re = 500.0
    clength = grid_shape[0] - 1
    visc = u_max * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    # Print simulation info
    print("\n" + "=" * 50 + "\n")
    print("Simulation Configuration:")
    print(f"Grid size: {grid_shape[0]} x {grid_shape[1]} x {grid_shape[2]}")
    print(f"Backend: {backend}")
    print(f"Velocity set: {velocity_set}")
    print(f"Precision policy: {precision_policy}")
    print(f"Prescribed velocity: {u_max}")
    print(f"Reynolds number: {Re}")
    print(f"Max iterations: {num_steps}")
    print("\n" + "=" * 50 + "\n")

    omega = 1.7

    simulation = FlowOverSphereIBM(omega, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps=80000, print_interval=1000, post_process_interval=2000)
