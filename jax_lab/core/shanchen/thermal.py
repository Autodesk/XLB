"""
Shan-Chen (pseudopotential) multiphase coupling of the hybrid thermal LBM solver, with phase change driven by the
temperature-dependent EOS of the fluid solver. See jax_lab.core.thermal for the shared finite difference solver and references.
"""

import operator
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.tree import map as tree_map
from jax.tree import reduce

from ..thermal import Thermal
from .multiphase import Multiphase, MultiphaseBGK, MultiphaseMRT


class MultiphaseThermal(Thermal):
    """
    Multiphase implementation of the hybrid thermal LBM solver with phase
    change support. The fluid is advanced by the wrapped Multiphase solver
    (pytrees of distribution functions, one per component) while a single
    shared temperature field is advanced by the finite difference solver.

    The temperature equation solved is (reference 1 in the module docstring):
    \\frac{\\partial T}{\\partial t} = -\\mathbf{u} \\cdot \\nabla T + \\frac{1}{\\rho c_v}(\\nabla \\cdot (K \\nabla T)
    - T \\frac{\\partial p_{EOS}}{\\partial T} \\nabla \\cdot \\mathbf{u} + S_T)

    where the T (dp_EOS/dT) div(u) term accounts for the latent heat released or absorbed during phase change, rho and u are the mixture density
    and velocity, and S_T is a user defined source term (see source()).

    The temperature field is passed into the fluid solver every step, so the EOS of the fluid solver must be constructed with
    temperature_field_type="thermal"; the pressure (and hence the pseudopotential) then follows the local temperature, which is what drives
    evaporation and condensation.

    Assumes all components share a single temperature field (thermal equilibrium between components at every node).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.fluid_solver.eos.temperature_field_type != "thermal":
            raise ValueError('The EOS of the fluid solver must be constructed with temperature_field_type="thermal" for MultiphaseThermal.')
        if self.apply_buoyancy:
            raise ValueError("apply_buoyancy is not supported for MultiphaseThermal; use the body_force parameter of the fluid solver instead.")

    def _shares_fluid_force(self):
        """
        Whether step() can compute the interaction force once and share it with the fluid collision: only when every part
        involved is the library default (so the shared force is exactly what the collision would compute) and the
        collision is a fused one that takes it (MultiphaseMRT, or MultiphaseBGK on its per-direction path), without the
        kappa surface-tension term (whose fused path computes the force through its own stencil).
        """
        fluid = self.fluid_solver
        cls = type(fluid)
        if not (
            cls.step is Multiphase.step
            and cls.compute_force is Multiphase.compute_force
            and cls.macroscopic_velocity is Multiphase.macroscopic_velocity
            and cls.update_macroscopic is Multiphase.update_macroscopic
            and not getattr(fluid, "_has_surface_tension", False)
        ):
            return False
        if cls.collision is MultiphaseMRT.collision:
            return True
        return cls.collision is MultiphaseBGK.collision and fluid._uses_default_bgk_methods()

    @partial(jit, static_argnums=(0,), inline=True)
    def divergence_x(self, u):
        """
        Compute the divergence of a vector field using the isotropic lattice
        difference stencil, by summing the diagonal entries of the gradient of
        each velocity component.

        Parameters
        ----------
        u (jax.numpy.ndarray): Vector field of shape (nx, ny, 2) in 2D or (nx, ny, nz, 3) in 3D.

        Returns
        -------
        jax.numpy.ndarray: Divergence of the field, of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.
        """
        if type(self).grad_x is Thermal.grad_x:
            # One stencil pass over the vector field (instead of dim gradients of its components)
            return self.lattice_stencil(u, ("div",))[0]
        div_u = jnp.zeros_like(u[..., :1])
        for alpha in range(self.dim):
            div_u = div_u + self.grad_x(u[..., alpha : alpha + 1])[..., alpha : alpha + 1]
        return div_u

    @partial(jit, static_argnums=(0,), inline=True)
    def RHS(self, T, rho_tree, u_tree):
        """
        Right hand side of the thermal equation for multiphase flow:
        -\\mathbf{u} \\cdot \\nabla T + \\frac{1}{\\rho c_v}(K \\nabla^2 T + \\nabla K \\cdot \\nabla T
        - T \\frac{\\partial p_{EOS}}{\\partial T} \\nabla \\cdot \\mathbf{u} + S_T)

        The mixture (total) density and velocity are used; the phase change
        term uses the sum of dp_EOS/dT over all components.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.

        rho_tree (pytree of jax.numpy.ndarray): Density field of all components.

        u_tree (pytree of jax.numpy.ndarray): Velocity field of all components.

        Returns
        -------
        jax.numpy.ndarray: RHS of the thermal equation, of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.
        """
        return self.RHS_prepared(T, self.prepare_RHS(rho_tree, u_tree))

    def prepare_RHS(self, rho_tree, u_tree):
        """
        The parts of RHS that do not depend on T (frozen over the Runge-Kutta substeps; see Thermal.prepare_RHS): the
        mixture velocity, its divergence and rho * c_v of the mixture density, plus the component densities for the
        temperature-dependent dp_EOS/dT.
        """
        # Mixture (total) density and velocity
        rho = self.fluid_solver.compute_total_density(rho_tree)
        u = self.fluid_solver.compute_total_velocity(rho_tree, u_tree)
        # Regrouped coefficients (see Thermal.prepare_RHS): RHS = v . grad T + a lap T - T dp_EOS/dT w (+ S / (rho c_v))
        rho_c_v = rho * self.c_v
        velocity = -u + self.grad_K / rho_c_v
        diffusivity = self.K / rho_c_v
        expansion = self.divergence_x(u) / rho_c_v
        return rho_tree, velocity, diffusivity, expansion, None if self._uses_default_source() else rho_c_v

    def RHS_prepared(self, T, prepared):
        """RHS from the output of prepare_RHS (same formula as RHS, terms regrouped)."""
        rho_tree, velocity, diffusivity, expansion, rho_c_v = prepared
        grad_T, lap_T = self.grad_and_laplacian(T)
        dp_eos_dT = reduce(operator.add, self.fluid_solver.eos.dp_eos_dT(rho_tree, T))
        rhs = jnp.sum(velocity * grad_T, axis=-1, keepdims=True) + diffusivity * lap_T - T * dp_eos_dT * expansion
        if rho_c_v is not None:
            rhs = rhs + self.source(T) / rho_c_v
        return rhs

    @partial(jit, static_argnums=(0, 4))
    def step(self, T_prev, state_tree, timestep, return_fpost=False):
        """
        This function performs a single step of the multiphase hybrid thermal LBM simulation.

        It evaluates the component fields and temperature right hand side at
        the current time level, then advances the fluid and temperature fields
        together by one time step. The current temperature is passed to the
        thermal EOS used by the fluid collision.

        Parameters
        ----------
        T_prev (jax.numpy.ndarray): The temperature field from the previous timestep.

        state_tree (list of PopulationState): The fluid states (see Multiphase.step).

        timestep (int): The current timestep of the simulation.

        return_fpost (bool, optional): If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        T (jax.numpy.ndarray): The temperature field after the simulation step.

        state_tree (list of PopulationState): The fluid states after the simulation step.

        f_postcollision_tree (pytree of jax.numpy.ndarray or None): The post-collision distribution functions after the
        simulation step, or None if return_fpost is False.
        """
        T = self.precision_policy.cast_to_compute(T_prev)
        T = self.apply_bc(T, timestep)
        fluid = self.fluid_solver
        f_compute_tree = [self.precision_policy.cast_to_compute(fluid.stream(state)) for state in state_tree]
        rho_tree, _ = fluid.update_macroscopic(f_compute_tree)
        if self._shares_fluid_force():
            # The interaction force is computed once and shared with the collision of this step (same populations and
            # temperature, so identical values), instead of once for the velocity here and again in the collision
            force_tree = fluid.compute_force(rho_tree, T=T)
            # The fluid step and its collision are traced inline, so the shared force belongs to this trace. The collision
            # also hands back the density and force-corrected velocity it computes (Multiphase.macroscopic_velocity with that
            # force, same operations), so the populations are not read again here for the temperature equation
            fluid_step = type(fluid).step.__wrapped__
            collision = type(fluid).collision.__wrapped__
            exported = []
            fluid._shared_force = force_tree
            fluid._export_macroscopic = exported
            fluid.collision = lambda fin_tree, T=None: collision(fluid, fin_tree, T=T)
            try:
                state_tree, f_postcollision_tree = fluid_step(fluid, state_tree, timestep, return_fpost, T)
            finally:
                fluid._shared_force = None
                fluid._export_macroscopic = None
                del fluid.collision
            if len(exported) != 1:
                raise RuntimeError("Expected the fused collision to export its macroscopic fields exactly once per step.")
            _, u_tree = exported[0]
        else:
            u_tree = fluid.macroscopic_velocity(f_compute_tree, rho_tree, T=T)
            state_tree, f_postcollision_tree = fluid.step(state_tree, timestep, return_fpost, T)
        T = self._advance_temperature(T, timestep, rho_tree, u_tree)

        T = self.apply_bc(T, timestep)

        return self.precision_policy.cast_to_output(T), state_tree, f_postcollision_tree

    def update_macroscopic_output(self, f_tree, T):
        """
        Compute the component densities and force-corrected velocities for I/O.

        Parameters
        ----------
        f_tree (pytree of jax.numpy.ndarray): Post-streaming distribution functions.

        T (jax.numpy.ndarray): Temperature field.

        Returns
        -------
        rho_tree (pytree of jax.numpy.ndarray): Density field of all components.

        u_tree (pytree of jax.numpy.ndarray): Velocity field of all components.
        """
        f_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), f_tree)
        rho_tree, _ = self.fluid_solver.update_macroscopic(f_tree)
        u_tree = self.fluid_solver.macroscopic_velocity(f_tree, rho_tree, T=self.precision_policy.cast_to_compute(T))
        return rho_tree, u_tree
