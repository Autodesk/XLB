# Base class for all stepper operators

from functools import partial
from jax import jit
import jax

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.quadratic_equilibrium import QuadraticEquilibrium
from xlb.operator.collision.bgk import BGK
from xlb.operator.collision.kbc import KBC
from xlb.operator.stream import Stream
from xlb.operator.macroscopic import Macroscopic
from xlb.solver.solver import Solver
from xlb.operator import Operator


class IncompressibleNavierStokesSolver(Solver):

    _equilibrium_registry = {
        "Quadratic": QuadraticEquilibrium,
    }
    _collision_registry = {
        "BGK": BGK,
        "KBC": KBC,
    }

    def __init__(
        self,
        omega: float,
        domain_shape: tuple[int, int, int],
        collision="BGK",
        equilibrium="Quadratic",
        boundary_conditions=[],
        velocity_set = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(
            domain_shape=domain_shape,
            boundary_conditions=boundary_conditions,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
            precision_policy=precision_policy,
        )

        # Set omega
        self.omega = omega

        # Create operators
        self.collision = self._get_collision(collision)(
            omega=self.omega,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.stream = Stream(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.equilibrium = self._get_equilibrium(equilibrium)(
            velocity_set=self.velocity_set, precision_policy=self.precision_policy, compute_backend=self.compute_backend
        )
        self.macroscopic = Macroscopic(
            velocity_set=self.velocity_set, precision_policy=self.precision_policy, compute_backend=self.compute_backend
        )

        # Create stepper operator
        self.stepper = IncompressibleNavierStokesStepper(
            collision=self.collision,
            stream=self.stream,
            equilibrium=self.equilibrium,
            macroscopic=self.macroscopic,
            boundary_conditions=self.boundary_conditions,
            forcing=None,
        )

    def monitor(self):
        pass

    def run(self, steps: int, monitor_frequency: int = 1, compute_mlups: bool = False):

        # Run steps
        for _ in range(steps):
            # Run step
            self.stepper(
                f0=self.grid.get_field("f0"),
                f1=self.grid.get_field("f1")
            )
            self.grid.swap_fields("f0", "f1")

    def checkpoint(self):
        raise NotImplementedError("Checkpointing not yet implemented")

    def _get_collision(self, collision: str):
        if isinstance(collision, str):
            try:
                return self._collision_registry[collision]
            except KeyError:
                raise ValueError(f"Collision {collision} not recognized for incompressible Navier-Stokes solver")
        elif issubclass(collision, Operator):
            return collision
        else:
            raise ValueError(f"Collision {collision} not recognized for incompressible Navier-Stokes solver")

    def _get_equilibrium(self, equilibrium: str):
        if isinstance(equilibrium, str):
            try:
                return self._equilibrium_registry[equilibrium]
            except KeyError:
                raise ValueError(f"Equilibrium {equilibrium} not recognized for incompressible Navier-Stokes solver")
        elif issubclass(equilibrium, Operator):
            return equilibrium
        else:
            raise ValueError(f"Equilibrium {equilibrium} not recognized for incompressible Navier-Stokes solver")
