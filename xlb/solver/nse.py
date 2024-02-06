# Base class for all stepper operators

from functools import partial
from jax import jit

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_condition import ImplementationStep
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.collision import BGK, KBC
from xlb.operator.stream import Stream
from xlb.operator.macroscopic import Macroscopic
from xlb.solver.solver import Solver
from xlb.operator import Operator


class IncompressibleNavierStokes(Solver):
    def __init__(
        self,
        grid,
        omega,
        velocity_set: VelocitySet = None,
        compute_backend=None,
        precision_policy=None,
        boundary_conditions=[],
        collision_kernel="BGK",
    ):
        self.grid = grid
        self.omega = omega
        self.collision_kernel = collision_kernel
        super().__init__(velocity_set=velocity_set, compute_backend=compute_backend, precision_policy=precision_policy, boundary_conditions=boundary_conditions)
        self.create_operators()

    # Operators
    def create_operators(self):
        self.macroscopic = Macroscopic(
            velocity_set=self.velocity_set, compute_backend=self.compute_backend
        )
        self.equilibrium = QuadraticEquilibrium(
            velocity_set=self.velocity_set, compute_backend=self.compute_backend
        )
        self.collision = (
            KBC(
                omega=self.omega,
                velocity_set=self.velocity_set,
                compute_backend=self.compute_backend,
            )
            if self.collision_kernel == "KBC"
            else BGK(
                omega=self.omega,
                velocity_set=self.velocity_set,
                compute_backend=self.compute_backend,
            )
        )
        self.stream = Stream(self.grid,
            velocity_set=self.velocity_set, compute_backend=self.compute_backend
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def step(self, f, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """

        # Cast to compute precision
        f = self.precision_policy.cast_to_compute(f)

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(
            f,
            feq,
        )

        # # Apply collision type boundary conditions
        # for id_number, bc in self.collision_boundary_conditions.items():
        #     f_post_collision = bc(
        #         f_pre_collision,
        #         f_post_collision,
        #         boundary_id == id_number,
        #         mask,
        #     )
        f_pre_streaming = f_post_collision

        ## Apply forcing
        # if self.forcing_op is not None:
        #    f = self.forcing_op.apply_jax(f, timestep)

        # Apply streaming
        f_post_streaming = self.stream(f_pre_streaming)

        # Apply boundary conditions
        # for id_number, bc in self.stream_boundary_conditions.items():
        #     f_post_streaming = bc(
        #         f_pre_streaming,
        #         f_post_streaming,
        #         boundary_id == id_number,
        #         mask,
        #     )

        # Copy back to store precision
        f = self.precision_policy.cast_to_store(f_post_streaming)

        return f
