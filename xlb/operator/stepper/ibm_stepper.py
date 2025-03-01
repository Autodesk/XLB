from functools import partial
from posixpath import dirname
from jax import jit
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker
from xlb.helper import check_bc_overlaps
from xlb.helper.nse_solver import create_nse_fields
from xlb.operator.stepper.nse_stepper import IncompressibleNavierStokesStepper
from warp.utils import ScopedTimer


class IBMStepper(IncompressibleNavierStokesStepper):
    def __init__(
        self,
        grid,
        boundary_conditions=[],
        collision_type="BGK",
    ):
        super().__init__(grid, boundary_conditions, collision_type)

        self.grid_dim = grid.shape
        dim_x, dim_y, dim_z = self.grid_dim

        # Initialize Eulerian points array
        self.f_eulerian_points = wp.zeros(shape=(dim_x * dim_y * dim_z), dtype=wp.vec3)
        self.f_eulerian_forces = wp.zeros(shape=(dim_x * dim_y * dim_z), dtype=wp.vec3)
        self.f_eulerian_velocities = wp.zeros(shape=(dim_x * dim_y * dim_z), dtype=wp.vec3)
        self.f_eulerian_weights = wp.zeros(shape=(dim_x * dim_y * dim_z), dtype=self.compute_dtype)

        @wp.func
        def hash_to_grid_idx(hash_idx: int, dim_x: int, dim_y: int) -> wp.vec3i:
            """Convert hash grid index to 3D grid coordinates"""
            k = hash_idx // (dim_x * dim_y)
            j = (hash_idx % (dim_x * dim_y)) // dim_x
            i = hash_idx % dim_x
            return wp.vec3i(i, j, k)

        @wp.func
        def grid_to_hash_idx(i: int, j: int, k: int, dim_x: int, dim_y: int) -> int:
            """Convert 3D grid coordinates to hash grid index"""
            return k * (dim_x * dim_y) + j * dim_x + i

        @wp.kernel
        def init_eulerian_points(points: wp.array(dtype=wp.vec3), dim_x: int, dim_y: int, dim_z: int):
            idx = wp.tid()
            grid_pos = hash_to_grid_idx(idx, dim_x, dim_y)
            points[idx] = wp.vec3(float(grid_pos[0]) + 0.5, float(grid_pos[1]) + 0.5, float(grid_pos[2]) + 0.5)

        # Launch kernel to initialize points
        wp.launch(kernel=init_eulerian_points, dim=dim_x * dim_y * dim_z, inputs=[self.f_eulerian_points, dim_x, dim_y, dim_z])

        self.hash_grid = wp.HashGrid(dim_x=dim_x, dim_y=dim_y, dim_z=dim_z)
        self.hash_grid.build(self.f_eulerian_points, 2.0)  # 2.0 is the radius

        self.s_lagr_forces_initialized = False

        self._construct_ibm_warp()

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_0, f_1, bc_mask, missing_mask, timestep):
        raise NotImplementedError("IBM stepper is not implemented in JAX backend. Please use WARP backend.")

    def _construct_ibm_warp(self):
        # Set local constants
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _opp_indices = self.velocity_set.opp_indices
        _weights = self.velocity_set.w
        _c = self.velocity_set.c
        _dim_x = self.grid_dim[0]
        _dim_y = self.grid_dim[1]

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id

        # Gather IDs of ExtrapolationOutflowBC boundary conditions
        extrapolation_outflow_bc_ids = []
        for bc_name, bc_id in bc_to_id.items():
            if bc_name.startswith("ExtrapolationOutflowBC"):
                extrapolation_outflow_bc_ids.append(bc_id)
        # Group active boundary conditions
        active_bcs = set(boundary_condition_registry.id_to_bc[bc.id] for bc in self.boundary_conditions)

        _opp_indices = self.velocity_set.opp_indices

        @wp.func
        def hash_to_grid_idx(hash_idx: int, dim_x: int, dim_y: int) -> wp.vec3i:
            """Convert hash grid index to 3D grid coordinates"""
            k = hash_idx // (dim_x * dim_y)
            j = (hash_idx % (dim_x * dim_y)) // dim_x
            i = hash_idx % dim_x
            return wp.vec3i(i, j, k)

        @wp.func
        def grid_to_hash_idx(i: int, j: int, k: int, dim_x: int, dim_y: int) -> int:
            """Convert 3D grid coordinates to hash grid index"""
            return k * (dim_x * dim_y) + j * dim_x + i

        # Smoothing function as proposed by Peskin
        @wp.func
        def peskin_weight(r: float):
            abs_r = wp.abs(r)
            if abs_r <= 1.0:
                return self.compute_dtype(0.125) * (
                    self.compute_dtype(3.0)
                    - 2.0 * abs_r
                    + wp.sqrt(self.compute_dtype(1.0) + self.compute_dtype(4.0) * abs_r - self.compute_dtype(4.0) * abs_r * abs_r)
                )
            elif abs_r <= 2.0:
                return self.compute_dtype(0.125) * (
                    self.compute_dtype(5.0)
                    - 2.0 * abs_r
                    - wp.sqrt(self.compute_dtype(-7.0) + self.compute_dtype(12.0) * abs_r - self.compute_dtype(4.0) * abs_r * abs_r)
                )
            else:
                return self.compute_dtype(0.0)

        @wp.func
        def weight(x: wp.vec3, Xk: wp.vec3):
            r = x - Xk
            return peskin_weight(r[0]) * peskin_weight(r[1]) * peskin_weight(r[2])

        # Kernel to initialize the force on Lagrangian points (Step 1)
        @wp.kernel
        def initialize_lagr_force(
            solid_lagr_velocities: wp.array(dtype=wp.vec3),
            fluid_lagr_velocities: wp.array(dtype=wp.vec3),
            lag_forces: wp.array(dtype=wp.vec3),
        ):
            tid = wp.tid()
            vk = solid_lagr_velocities[tid]
            u_Xk = fluid_lagr_velocities[tid]

            # Initialize force
            lag_forces[tid] = vk - u_Xk

        # Kernel to interpolate force from Lagrangian to Eulerian grid (Step 2)
        @wp.kernel
        def interpolate_force_to_eulerian_atomic(
            lag_positions: wp.array(dtype=wp.vec3),
            lag_forces: wp.array(dtype=wp.vec3),
            lag_areas: wp.array(dtype=Any),
            eul_positions: wp.array(dtype=wp.vec3),
            eul_forces: wp.array(dtype=wp.vec3),
            eul_weights: wp.array(dtype=Any),  # Accumulator for weights
            grid: wp.uint64,
        ):
            tid = wp.tid()
            Xk = lag_positions[tid]
            Fk = lag_forces[tid]
            Ak = lag_areas[tid]

            # Query neighboring Eulerian points
            query = wp.hash_grid_query(grid, Xk, 2.0)
            index = int(0)

            while wp.hash_grid_query_next(query, index):
                x_pos = eul_positions[index]
                w = weight(x_pos, Xk)
                # First accumulate the weight
                wp.atomic_add(eul_weights, index, w)
                # Then accumulate the weighted force
                delta_f = Fk * w * Ak
                wp.atomic_add(eul_forces, index, delta_f)

        @wp.kernel
        def normalize_eulerian_forces(
            eul_forces: wp.array(dtype=wp.vec3),
            eul_weights: wp.array(dtype=Any),
        ):
            tid = wp.tid()
            weight_sum = eul_weights[tid]
            if weight_sum > self.compute_dtype(0.0):
                eul_forces[tid] = eul_forces[tid] / weight_sum

        # Kernel to correct the fluid velocity at Eulerian grid points (Step 3)
        @wp.kernel
        def correct_eulerian_velocity(eul_velocities: wp.array(dtype=wp.vec3), eul_forces: wp.array(dtype=wp.vec3)):
            tid = wp.tid()
            eul_velocities[tid] = eul_velocities[tid] + eul_forces[tid]

        # Kernel to interpolate corrected velocities back to Lagrangian points (Step 4)
        @wp.kernel
        def interpolate_velocity_to_lagrangian(
            lag_positions: wp.array(dtype=wp.vec3),
            eul_positions: wp.array(dtype=wp.vec3),
            eul_velocities: wp.array(dtype=wp.vec3),
            lag_fluid_velocities: wp.array(dtype=wp.vec3),
            grid: wp.uint64,
        ):
            tid = wp.tid()
            Xk = lag_positions[tid]

            # Initialize numerator and denominator for interpolation
            numerator = wp.vec3(self.compute_dtype(0.0), self.compute_dtype(0.0), self.compute_dtype(0.0))
            denominator = self.compute_dtype(0.0)

            # Query neighboring Eulerian points
            query = wp.hash_grid_query(grid, Xk, 2.0)
            index = int(0)

            while wp.hash_grid_query_next(query, index):
                x_pos = eul_positions[index]
                u = eul_velocities[index]
                w = weight(x_pos, Xk)
                numerator += u * w
                denominator += w

            if denominator > self.compute_dtype(0.0):
                u_interp = numerator / denominator
            else:
                u_interp = wp.vec3(0.0, 0.0, 0.0)

            lag_fluid_velocities[tid] = u_interp

        # Kernel to update the force at Lagrangian points (Step 5)
        @wp.kernel
        def update_lagr_force(
            solid_lagr_velocities: wp.array(dtype=wp.vec3),
            fluid_lagr_velocities: wp.array(dtype=wp.vec3),
            lag_forces: wp.array(dtype=wp.vec3),
        ):
            tid = wp.tid()
            vk = solid_lagr_velocities[tid]
            uk = fluid_lagr_velocities[tid]
            delta_F = vk - uk
            lag_forces[tid] += delta_F

        @wp.kernel
        def compute_eulerian_velocity_from_f_1(f_1: wp.array4d(dtype=Any), eul_velocities: wp.array(dtype=wp.vec3)):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            # Read from thread local memory
            _f_1_thread = _f_vec()

            for l in range(self.velocity_set.q):
                _f_1_thread[l] = self.compute_dtype(f_1[l, index[0], index[1], index[2]])

            _rho, _u = self.macroscopic.warp_functional(_f_1_thread)

            eul_velocities[grid_to_hash_idx(i, j, k, _dim_x, _dim_y)] = _u

        @wp.kernel
        def correct_population_ibm(f_1: wp.array4d(dtype=Any), eul_forces: wp.array(dtype=wp.vec3)):
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Initialize thread-local storage for populations
            _f1_thread = _f_vec()

            # Retrieve f_1 values for the current grid point
            for l in range(self.velocity_set.q):
                _f1_thread[l] = self.compute_dtype(f_1[l, index[0], index[1], index[2]])

            # Compute macroscopic quantities (rho, u) from f_1
            _rho, _u = self.macroscopic.warp_functional(_f1_thread)

            # Retrieve the force at the current grid point
            force = eul_forces[grid_to_hash_idx(i, j, k, _dim_x, _dim_y)]

            # Compute equilibrium with force applied
            feq_force = self.equilibrium.warp_functional(_rho, _u + force)
            feq = self.equilibrium.warp_functional(_rho, _u)

            # Update f_1 with the new post-collision population
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1], index[2]] += self.store_dtype(feq_force[l] - feq[l])

        # Add a new kernel for conservation-aware force normalization
        @wp.kernel
        def conservation_aware_normalize_forces(
            eul_forces: wp.array(dtype=wp.vec3),
            eul_weights: wp.array(dtype=Any),
            eul_velocities: wp.array(dtype=wp.vec3),
        ):
            tid = wp.tid()
            weight_sum = eul_weights[tid]

            if weight_sum > self.compute_dtype(0.0):
                # The accumulated forces contain the weighted sum of desired velocity changes
                # We normalize by weight_sum to get the average desired velocity change
                # This ensures that we're applying just the right amount of force to achieve
                # the collective boundary condition, not over-enforcing it
                eul_forces[tid] = eul_forces[tid] / weight_sum

                # Note: In the IBM method, eul_forces now represents the target velocity,
                # not the force directly. The actual force application happens in correct_eulerian_velocity
                # and correct_population_ibm kernels.

        # Add a new kernel that combines force interpolation and conservation in one step
        @wp.kernel
        def improved_interpolate_force_to_eulerian(
            lag_positions: wp.array(dtype=wp.vec3),
            lag_forces: wp.array(dtype=wp.vec3),
            lag_areas: wp.array(dtype=Any),
            eul_positions: wp.array(dtype=wp.vec3),
            eul_velocities: wp.array(dtype=wp.vec3),  # Current Eulerian velocities
            eul_forces: wp.array(dtype=wp.vec3),  # Will store desired velocity, not force directly
            eul_weights: wp.array(dtype=Any),  # For normalization
            grid: wp.uint64,
        ):
            tid = wp.tid()
            Xk = lag_positions[tid]
            Fk = lag_forces[tid]  # Fk here represents the desired velocity change at the Lagrangian point
            Ak = lag_areas[tid]

            # Query neighboring Eulerian points
            query = wp.hash_grid_query(grid, Xk, 2.0)
            index = int(0)

            while wp.hash_grid_query_next(query, index):
                x_pos = eul_positions[index]
                w = weight(x_pos, Xk)

                # The weight represents how much this Lagrangian point influences this Eulerian point
                wp.atomic_add(eul_weights, index, w)

                # We accumulate the weighted desired velocity from each Lagrangian point
                # Each Lagrangian point contributes according to its weight and area
                target_velocity = Fk * w * Ak
                wp.atomic_add(eul_forces, index, target_velocity)

        # Add this to the constructor
        self.improved_interpolate_force_to_eulerian = improved_interpolate_force_to_eulerian

        @wp.kernel
        def physics_based_normalize_and_correct(
            eul_forces: wp.array(dtype=wp.vec3),  # Contains accumulated weighted velocities
            eul_weights: wp.array(dtype=Any),  # Contains sum of weights
            eul_velocities: wp.array(dtype=wp.vec3),  # Current velocities
        ):
            tid = wp.tid()
            weight_sum = eul_weights[tid]

            if weight_sum > self.compute_dtype(0.0):
                # Calculate the physically correct target velocity at this Eulerian point
                # by taking the weighted average of all desired velocities from influencing Lagrangian points
                target_velocity = eul_forces[tid] / weight_sum

                # The force we need to apply is the difference between the target and current velocity
                # This is the minimal force needed to satisfy the boundary conditions collectively
                correction_force = target_velocity - eul_velocities[tid]

                # Store the correction force back in eul_forces for use in later steps
                eul_forces[tid] = correction_force

                # Apply the correction directly to the velocity field
                # TODO: This step can be kept or moved to correct_eulerian_velocity
                eul_velocities[tid] += correction_force

        # Add this to the constructor
        self.physics_based_normalize_and_correct = physics_based_normalize_and_correct

        self.initialize_lagr_force = initialize_lagr_force
        self.compute_eulerian_velocity_from_f_1 = compute_eulerian_velocity_from_f_1
        self.interpolate_force_to_eulerian_atomic = interpolate_force_to_eulerian_atomic
        self.correct_eulerian_velocity = correct_eulerian_velocity
        self.interpolate_velocity_to_lagrangian = interpolate_velocity_to_lagrangian
        self.update_lagr_force = update_lagr_force
        self.correct_population_ibm = correct_population_ibm
        self.normalize_eulerian_forces = conservation_aware_normalize_forces

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        f_0,
        f_1,
        s_lagr_vertices_wp,
        lagr_solid_vertex_areas_wp,
        lagr_solid_velocities_wp,
        lagr_update_kernel,
        bc_mask,
        missing_mask,
        omega,
        timestep,
    ):
        # Create Warp arrays for IBM method if not already created
        self.s_lagr_forces = wp.zeros(shape=(s_lagr_vertices_wp.shape[0]), dtype=wp.vec3)
        self.f_lagr_velocities = wp.zeros(shape=(s_lagr_vertices_wp.shape[0]), dtype=wp.vec3)

        # Single ScopedTimer with synchronization for the entire method
        with ScopedTimer("IBM_Stepper", use_nvtx=True, synchronize=True, cuda_filter=wp.TIMING_ALL):
            # Step 1: Perform LBM step
            wp.launch(kernel=self.warp_kernel, dim=f_0.shape[1:], inputs=[f_0, f_1, bc_mask, missing_mask, omega, timestep])

            num_iterations = 1
            for _ in range(num_iterations):
                self.f_eulerian_forces.zero_()
                self.f_eulerian_weights.zero_()
                wp.launch(kernel=self.compute_eulerian_velocity_from_f_1, dim=f_1.shape[1:], inputs=[f_1, self.f_eulerian_velocities])

                # Step 2: Use our improved interpolation that accumulates forces with the proper weighting
                wp.launch(
                    kernel=self.improved_interpolate_force_to_eulerian,
                    dim=s_lagr_vertices_wp.shape[0],
                    inputs=[
                        s_lagr_vertices_wp,
                        self.s_lagr_forces,
                        lagr_solid_vertex_areas_wp,
                        self.f_eulerian_points,
                        self.f_eulerian_velocities,
                        self.f_eulerian_forces,
                        self.f_eulerian_weights,
                        wp.uint64(self.hash_grid.id),
                    ],
                )

                # Step 3: Use our physics-based normalization and correction
                wp.launch(
                    kernel=self.physics_based_normalize_and_correct,
                    dim=self.f_eulerian_forces.shape[0],
                    inputs=[self.f_eulerian_forces, self.f_eulerian_weights, self.f_eulerian_velocities],
                )

                # Step 4: Interpolate corrected velocities back to Lagrangian points
                wp.launch(
                    kernel=self.interpolate_velocity_to_lagrangian,
                    dim=s_lagr_vertices_wp.shape[0],
                    inputs=[s_lagr_vertices_wp, self.f_eulerian_points, self.f_eulerian_velocities, self.f_lagr_velocities, wp.uint64(self.hash_grid.id)],
                )

                # Step 5: Update Lagrangian forces
                wp.launch(
                    kernel=self.update_lagr_force,
                    dim=s_lagr_vertices_wp.shape[0],
                    inputs=[lagr_solid_velocities_wp, self.f_lagr_velocities, self.s_lagr_forces],
                )

            # Step 6: Correct populations
            wp.launch(kernel=self.correct_population_ibm, dim=f_1.shape[1:], inputs=[f_1, self.f_eulerian_forces])

            # Step 7: Update solid velocities and positions
            wp.launch(
                kernel=lagr_update_kernel,
                dim=s_lagr_vertices_wp.shape[0],
                inputs=[
                    timestep,
                    self.s_lagr_forces,
                    s_lagr_vertices_wp,
                    lagr_solid_velocities_wp,
                ],
            )

        return f_0, f_1
