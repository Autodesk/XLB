# Base class for all stepper operators

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.boundary_condition import ImplementationStep
from xlb.operator.precision_caster import PrecisionCaster


class Stepper(Operator):
    """
    Class that handles the construction of lattice boltzmann stepping operator
    """

    def __init__(
        self,
        collision,
        stream,
        equilibrium,
        macroscopic,
        boundary_conditions=[],
        forcing=None,
    ):
        # Set parameters
        self.collision = collision
        self.stream = stream
        self.equilibrium = equilibrium
        self.macroscopic = macroscopic
        self.boundary_conditions = boundary_conditions
        self.forcing = forcing

        # Get all operators for checking
        self.operators = [
            collision,
            stream,
            equilibrium,
            macroscopic,
            *boundary_conditions,
        ]

        # Get velocity set, precision policy, and compute backend
        velocity_sets = set([op.velocity_set for op in self.operators])
        assert len(velocity_sets) == 1, "All velocity sets must be the same"
        velocity_set = velocity_sets.pop()
        precision_policies = set([op.precision_policy for op in self.operators])
        assert len(precision_policies) == 1, "All precision policies must be the same"
        precision_policy = precision_policies.pop()
        compute_backends = set([op.compute_backend for op in self.operators])
        assert len(compute_backends) == 1, "All compute backends must be the same"
        compute_backend = compute_backends.pop()

        # Get collision and stream boundary conditions
        self.collision_boundary_conditions = {}
        self.stream_boundary_conditions = {}
        for id_number, bc in enumerate(self.boundary_conditions):
            bc_id = id_number + 1
            if bc.implementation_step == ImplementationStep.COLLISION:
                self.collision_boundary_conditions[bc_id] = bc
            elif bc.implementation_step == ImplementationStep.STREAMING:
                self.stream_boundary_conditions[bc_id] = bc
            else:
                raise ValueError("Boundary condition step not recognized")

        # Make operators for converting the precisions
        #self.cast_to_compute = PrecisionCaster(

        # Make operator for setting boundary condition arrays
        self.set_boundary = SetBoundary(
            self.collision_boundary_conditions,
            self.stream_boundary_conditions,
            velocity_set,
            precision_policy,
            compute_backend,
        )
        self.operators.append(self.set_boundary)

        # Initialize operator
        super().__init__(velocity_set, precision_policy, compute_backend)

    ######################################################
    # TODO: This is a hacky way to do this. Need to refactor
    ######################################################
    """
    def _construct_warp_bc_functional(self):
        # identity collision boundary condition
        @wp.func
        def identity(
            f_pre: self._warp_lattice_vec,
            f_post: self._warp_lattice_vec,
            mask: self._warp_bool_lattice_vec,
        ):
            return f_post
        def get_bc_functional(id_number, self.collision_boundary_conditions):
            if id_number in self.collision_boundary_conditions.keys():
                return self.collision_boundary_conditions[id_number].warp_functional
            else:
                return identity

        # Manually set the boundary conditions TODO: Extremely hacky
        collision_bc_functional_0 = get_bc_functional(0, self.collision_boundary_conditions)
        collision_bc_functional_1 = get_bc_functional(1, self.collision_boundary_conditions)
        collision_bc_functional_2 = get_bc_functional(2, self.collision_boundary_conditions)
        collision_bc_functional_3 = get_bc_functional(3, self.collision_boundary_conditions)
        collision_bc_functional_4 = get_bc_functional(4, self.collision_boundary_conditions)
        collision_bc_functional_5 = get_bc_functional(5, self.collision_boundary_conditions)
        collision_bc_functional_6 = get_bc_functional(6, self.collision_boundary_conditions)
        collision_bc_functional_7 = get_bc_functional(7, self.collision_boundary_conditions)
        collision_bc_functional_8 = get_bc_functional(8, self.collision_boundary_conditions)

        # Make the warp boundary condition functional
        @wp.func
        def warp_bc(
            f_pre: self._warp_lattice_vec,
            f_post: self._warp_lattice_vec,
            mask: self._warp_bool_lattice_vec,
            boundary_id: wp.uint8,
        ):
            if boundary_id == 0:
                f_post = collision_bc_functional_0(f_pre, f_post, mask)
            elif boundary_id == 1:
                f_post = collision_bc_functional_1(f_pre, f_post, mask)
            elif boundary_id == 2:
                f_post = collision_bc_functional_2(f_pre, f_post, mask)
            elif boundary_id == 3:
                f_post = collision_bc_functional_3(f_pre, f_post, mask)
            elif boundary_id == 4:
                f_post = collision_bc_functional_4(f_pre, f_post, mask)
            elif boundary_id == 5:
                f_post = collision_bc_functional_5(f_pre, f_post, mask)
            elif boundary_id == 6:
                f_post = collision_bc_functional_6(f_pre, f_post, mask)
            elif boundary_id == 7:
                f_post = collision_bc_functional_7(f_pre, f_post, mask)
            elif boundary_id == 8:
                f_post = collision_bc_functional_8(f_pre, f_post, mask)

            return f_post




    ######################################################
    """



        



class SetBoundary(Operator):
    """
    Class that handles the construction of lattice boltzmann boundary condition operator
    This will probably never be used directly and it might be better to refactor it
    """

    def __init__(
        self,
        collision_boundary_conditions,
        stream_boundary_conditions,
        velocity_set,
        precision_policy,
        compute_backend,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set parameters
        self.collision_boundary_conditions = collision_boundary_conditions
        self.stream_boundary_conditions = stream_boundary_conditions

    def _apply_all_bc(self, ijk, boundary_id, mask, bc):
        """
        Apply all boundary conditions
        """
        for id_number, bc in self.collision_boundary_conditions.items():
            boundary_id, mask = bc.boundary_masker(ijk, boundary_id, mask, id_number)
        for id_number, bc in self.stream_boundary_conditions.items():
            boundary_id, mask = bc.boundary_masker(ijk, boundary_id, mask, id_number)
        return boundary_id, mask

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, ijk):
        """
        Set boundary condition arrays
        These store the boundary condition information for each boundary
        """
        boundary_id = jnp.zeros(ijk.shape[:-1], dtype=jnp.uint8)
        mask = jnp.zeros(ijk.shape[:-1] + (self.velocity_set.q,), dtype=jnp.bool_)
        return self._apply_all_bc(ijk, boundary_id, mask, bc)

    @Operator.register_backend(ComputeBackend.PALLAS)
    def pallas_implementation(self, ijk):
        """
        Set boundary condition arrays
        These store the boundary condition information for each boundary
        """
        raise NotImplementedError("Pallas implementation not available")

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, ijk):
        """
        Set boundary condition arrays
        These store the boundary condition information for each boundary
        """
        boundary_id = wp.zeros(ijk.shape[:-1], dtype=wp.uint8)
        mask = wp.zeros(ijk.shape[:-1] + (self.velocity_set.q,), dtype=wp.bool)
        return self._apply_all_bc(ijk, boundary_id, mask, bc)
