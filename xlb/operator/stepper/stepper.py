# Base class for all stepper operators

from ast import Raise
from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp

from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.precision_caster import PrecisionCaster
from xlb.operator.equilibrium import Equilibrium
from xlb import DefaultConfig


class Stepper(Operator):
    """
    Class that handles the construction of lattice boltzmann stepping operator
    """

    def __init__(self, operators, boundary_conditions):
        self.operators = operators
        self.boundary_conditions = boundary_conditions
        # Get velocity set, precision policy, and compute backend
        velocity_sets = set(
            [op.velocity_set for op in self.operators if op is not None]
        )
        assert (
            len(velocity_sets) < 2
        ), "All velocity sets must be the same. Got {}".format(velocity_sets)
        velocity_set = (
            DefaultConfig.velocity_set if not velocity_sets else velocity_sets.pop()
        )

        precision_policies = set(
            [op.precision_policy for op in self.operators if op is not None]
        )
        assert (
            len(precision_policies) < 2
        ), "All precision policies must be the same. Got {}".format(precision_policies)
        precision_policy = (
            DefaultConfig.default_precision_policy
            if not precision_policies
            else precision_policies.pop()
        )

        compute_backends = set(
            [op.compute_backend for op in self.operators if op is not None]
        )
        assert (
            len(compute_backends) < 2
        ), "All compute backends must be the same. Got {}".format(compute_backends)
        compute_backend = (
            DefaultConfig.default_backend
            if not compute_backends
            else compute_backends.pop()
        )

        # Add boundary conditions
        # Warp cannot handle lists of functions currently
        # Because of this we manually unpack the boundary conditions
        ############################################
        # TODO: Fix this later
        ############################################
        from xlb.operator.boundary_condition.bc_equilibrium import EquilibriumBC
        from xlb.operator.boundary_condition.bc_do_nothing import DoNothingBC
        from xlb.operator.boundary_condition.bc_halfway_bounce_back import (
            HalfwayBounceBackBC,
        )
        from xlb.operator.boundary_condition.bc_fullway_bounce_back import (
            FullwayBounceBackBC,
        )

        self.equilibrium_bc = None
        self.do_nothing_bc = None
        self.halfway_bounce_back_bc = None
        self.fullway_bounce_back_bc = None

        for bc in boundary_conditions:
            if isinstance(bc, EquilibriumBC):
                self.equilibrium_bc = bc
            elif isinstance(bc, DoNothingBC):
                self.do_nothing_bc = bc
            elif isinstance(bc, HalfwayBounceBackBC):
                self.halfway_bounce_back_bc = bc
            elif isinstance(bc, FullwayBounceBackBC):
                self.fullway_bounce_back_bc = bc

        if self.equilibrium_bc is None:
            # Select the equilibrium operator based on its type
            self.equilibrium_bc = EquilibriumBC(
                rho=1.0,
                u=(0.0, 0.0, 0.0),
                equilibrium_operator=next(
                    (op for op in self.operators if isinstance(op, Equilibrium)), None
                ),
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend,
            )
        if self.do_nothing_bc is None:
            self.do_nothing_bc = DoNothingBC(
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend,
            )
        if self.halfway_bounce_back_bc is None:
            self.halfway_bounce_back_bc = HalfwayBounceBackBC(
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend,
            )
        if self.fullway_bounce_back_bc is None:
            self.fullway_bounce_back_bc = FullwayBounceBackBC(
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend,
            )
        ############################################

        # Initialize operator
        super().__init__(velocity_set, precision_policy, compute_backend)
