# Base class for all stepper operators

from functools import partial
from jax import jit

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator.stepper.stepper import Stepper
from xlb.operator.boundary_condition import ImplementationStep


class NSE(Stepper):
    """
    Class that handles the construction of lattice boltzmann stepping operator for the Navier-Stokes equations

    TODO: Check that the given operators (collision, stream, equilibrium, macroscopic, ...) are compatible
    with the Navier-Stokes equations
    """

    def __init__(
        self,
        collision,
        stream,
        equilibrium,
        macroscopic,
        boundary_conditions=[],
        forcing=None,
        precision_policy=None,
    ):
        super().__init__(
            collision,
            stream,
            equilibrium,
            macroscopic,
            boundary_conditions,
            forcing,
            precision_policy,
        )

    @partial(jit, static_argnums=(0,))
    def apply_jax(self, f, boundary_id, mask, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """

        # Cast to compute precision
        f_pre_collision = self.precision_policy.cast_to_compute_jax(f)

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_pre_collision)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(
            f,
            feq,
            rho,
            u,
        )

        # Apply collision type boundary conditions
        for id_number, bc in self.collision_boundary_conditions.items():
            f_post_collision = bc(
                f_pre_collision,
                f_post_collision,
                boundary_id == id_number,
                mask,
            )
        f_pre_streaming = f_post_collision

        ## Apply forcing
        # if self.forcing_op is not None:
        #    f = self.forcing_op.apply_jax(f, timestep)

        # Apply streaming
        f_post_streaming = self.stream(f_pre_streaming)

        # Apply boundary conditions
        for id_number, bc in self.stream_boundary_conditions.items():
            f_post_streaming = bc(
                f_pre_streaming,
                f_post_streaming,
                boundary_id == id_number,
                mask,
            )

        # Copy back to store precision
        f = self.precision_policy.cast_to_store_jax(f_post_streaming)

        return f