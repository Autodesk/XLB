"""
Three-dimensional version of the stable MRT pool-boiling example.
"""

import operator
import os
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import config, jit
from jax.tree import map as tree_map
from jax.tree import reduce

from jax_lab.core.base import _lattice_dot, _lattice_sum
from jax_lab.core.boundary_conditions import BounceBack, DirichletTemperature
from jax_lab.core.shanchen.eos import PengRobinson
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.shanchen.multiphase import MultiphaseMRT
from jax_lab.core.shanchen.thermal import MultiphaseThermal
from jax_lab.core.utils import save_fields_hdf5_xdmf

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_pool_boiling_3d")


class PoolFluid3D(MultiphaseMRT):
    def set_boundary_conditions(self):
        bottom = tuple(self.bounding_box_indices["bottom"].T)
        top = tuple(self.bounding_box_indices["top"].T)
        for component in range(self.n_components):
            # self.BCs[component].append(BounceBack(bottom, self.grid_info, self.precision_policy, theta=contact_angle))
            self.BCs[component].append(BounceBack(bottom, self.grid_info, self.precision_policy))
            self.BCs[component].append(BounceBack(top, self.grid_info, self.precision_policy))

    def initialize_macroscopic_fields(self):
        z = np.arange(self.nz)[None, None, :]
        rho = np.where(z < h_liquid, rho_l, rho_g)
        rho = np.broadcast_to(rho[..., None], (self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho_tree = [self.precision_policy.cast_to_output(rho)]

        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype)
        u_tree = [self.precision_policy.cast_to_output(u)]
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_buoyancy_force(self, rho_tree, timestep):
        """Compute density-relative buoyancy after interface relaxation.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Component density fields.
        timestep (int): Current timestep.

        Returns
        -------
        pytree of jax.numpy.ndarray: Component buoyancy-force fields.
        """
        rho_average = self.compute_total_density(rho_tree).mean()
        gravity_vector = jnp.array([0.0, 0.0, -gravity], dtype=self.precision_policy.compute_dtype)
        gravity_active = jnp.asarray(timestep > gravity_relaxation_steps, dtype=self.precision_policy.compute_dtype)
        return tree_map(lambda rho: gravity_active * (rho - rho_average) * gravity_vector, rho_tree)

    # Fluid-fluid force of the current time step, shared by the coupled step with the collision (see the thermal step)
    _shared_force = None

    def compute_force(self, rho_tree, T=None):
        """Fluid-fluid force. Within the coupled step it returns the force already computed for this time step from the
        same populations and temperature (identical values) instead of computing it a second time in the collision.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Component density fields.
        T (jax.numpy.ndarray, optional): Temperature field.

        Returns
        -------
        pytree of jax.numpy.ndarray: Fluid-fluid force fields.
        """
        if self._shared_force is not None:
            return self._shared_force
        return super().compute_force(rho_tree, T=T)

    def force_corrected_fields(self, f_tree, T, timestep):
        """Density and force-corrected velocity (as macroscopic_velocity), with the density and momentum summed per lattice
        direction instead of matrix products of the populations (exact in single precision, and fused into one kernel).

        Parameters
        ----------
        f_tree (pytree of jax.numpy.ndarray): Component populations.
        T (jax.numpy.ndarray): Temperature field.
        timestep (int): Current timestep.

        Returns
        -------
        tuple: Component density, velocity and fluid-fluid force fields.
        """
        directions = self._lattice_directions
        channels_tree = [[self.precision_policy.cast_to_compute(f[..., i]) for i in range(self.q)] for f in f_tree]
        rho_tree = [sum(channels)[..., None] for channels in channels_tree]
        momentum_tree = [jnp.stack([_lattice_sum(directions, channels, axis) for axis in range(self.dim)], axis=-1) for channels in channels_tree]
        force_tree = self.compute_force(rho_tree, T=T)
        buoyancy_tree = self.compute_buoyancy_force(rho_tree, timestep)
        u_tree = [
            (momentum + 0.5 * force + 0.5 * buoyancy) / rho
            for momentum, force, buoyancy, rho in zip(momentum_tree, force_tree, buoyancy_tree, rho_tree, strict=True)
        ]
        return rho_tree, u_tree, force_tree

    @partial(jit, static_argnums=(0,))
    def macroscopic_velocity(self, f_tree, rho_tree, T=None, timestep=None):
        """Compute force-corrected velocity with delayed buoyancy.

        Parameters
        ----------
        f_tree (pytree of jax.numpy.ndarray): Component populations.
        rho_tree (pytree of jax.numpy.ndarray): Component density fields.
        T (jax.numpy.ndarray, optional): Temperature field.
        timestep (int, optional): Current timestep.

        Returns
        -------
        pytree of jax.numpy.ndarray: Component velocity fields.
        """
        u_tree = super().macroscopic_velocity(f_tree, rho_tree, T=T)
        if timestep is None:
            timestep = gravity_relaxation_steps + 1
        buoyancy_tree = self.compute_buoyancy_force(rho_tree, timestep)
        return tree_map(lambda u, force, rho: u + 0.5 * force / rho, u_tree, buoyancy_tree, rho_tree)

    def add_buoyancy(self, f_poststreaming_tree, f_postcollision_tree, timestep):
        """Add the delayed density-relative buoyancy to the post-collision populations.

        Exact-difference forcing f* += feq(rho, u + F / rho) - feq(rho, u) with F = (rho - mean(rho)) g (after
        gravity_relaxation_steps), then the rest population is reset to f*_0 = rho - sum_(i>0) f*_i. Same formulas as
        compute_buoyancy_force and equilibrium, evaluated per lattice direction so XLA fuses the whole update into one kernel
        instead of building two full equilibrium arrays.

        Parameters
        ----------
        f_poststreaming_tree (pytree of jax.numpy.ndarray): Pre-collision populations (rho and u are taken from them).
        f_postcollision_tree (pytree of jax.numpy.ndarray): Post-collision populations.
        timestep (int): Current timestep.

        Returns
        -------
        pytree of jax.numpy.ndarray: Post-collision populations with the buoyancy applied.
        """
        dtype = self.precision_policy.compute_dtype
        directions, weights = self._lattice_directions, self._lattice_weights
        channels_tree = [[self.precision_policy.cast_to_compute(f[..., i]) for i in range(self.q)] for f in f_poststreaming_tree]
        rho_tree = [sum(channels) for channels in channels_tree]
        rho_average = reduce(operator.add, rho_tree).mean()
        gravity_active = jnp.asarray(timestep > gravity_relaxation_steps, dtype=dtype)
        out_tree = []
        for channels, rho, f_post in zip(channels_tree, rho_tree, f_postcollision_tree, strict=True):
            u = [_lattice_sum(directions, channels, axis) / rho for axis in range(self.dim)]
            u_forced = [v + gravity_active * (rho - rho_average) * g / rho for v, g in zip(u, (0.0, 0.0, -gravity), strict=True)]
            usqr = 1.5 * sum(v * v for v in u)
            usqr_forced = 1.5 * sum(v * v for v in u_forced)
            out = []
            for i, (direction, w) in enumerate(zip(directions, weights, strict=True)):
                cu = 3.0 * _lattice_dot(direction, u)
                cu_forced = 3.0 * _lattice_dot(direction, u_forced)
                feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
                feq_forced = rho * w * (1.0 + cu_forced * (1.0 + 0.5 * cu_forced) - usqr_forced)
                out.append(self.precision_policy.cast_to_compute(f_post[..., i]) + feq_forced - feq)
            out[0] = rho - sum(out[1:])
            out_tree.append(self.stack_populations([self.precision_policy.cast_to_output(o) for o in out]))
        return out_tree

    # post_collision returns stack_populations stacks (see LBMBase.post_collision)
    post_collision_per_direction = True

    def post_collision(self, f_postcollision_tree, f_poststreaming_tree, timestep):
        """Apply the delayed buoyancy to the post-collision populations (see add_buoyancy)."""
        return self.add_buoyancy(f_poststreaming_tree, f_postcollision_tree, timestep)


class PoolBoiling3D(MultiphaseThermal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thermal_diffusivity_factor = kwargs.get("thermal_diffusivity_factor")
        self.initial_mass = None

    def initialize_temperature_field(self):
        temperature = T_sat * np.ones((nx, ny, nz, 1))
        rng = np.random.default_rng(temperature_disturbance_seed)
        temperature[:, :, 1, 0] += rng.normal(0.0, temperature_disturbance_std * T_sat, size=(nx, ny))
        return temperature

    def set_thermal_boundary_conditions(self):
        self.thermal_BCs = []
        bbox = self.fluid_solver.bounding_box_indices
        self.thermal_BCs.append(DirichletTemperature(tuple(bbox["bottom"].T), prescribed=T_bottom))
        self.thermal_BCs.append(DirichletTemperature(tuple(bbox["top"].T), prescribed=T_top))

    @partial(jit, static_argnums=(0, 4))
    def step(self, T_prev, state_tree, timestep, return_fpost=False):
        """Advance the coupled fields with timestep-aware buoyancy.

        Parameters
        ----------
        T_prev (jax.numpy.ndarray): Previous temperature field.
        state_tree (list of PopulationState): Fluid states (see Multiphase.step).
        timestep (int): Current timestep.
        return_fpost (bool, optional): Whether to return post-collision populations.

        Returns
        -------
        tuple: Temperature, fluid states, and optional post-collision populations.
        """
        T = self.precision_policy.cast_to_compute(T_prev)
        T = self.apply_bc(T, timestep)
        fluid = self.fluid_solver
        f_poststreaming_tree = [fluid.stream(state) for state in state_tree]
        rho_tree, u_tree, force_tree = fluid.force_corrected_fields(f_poststreaming_tree, T, timestep)

        # The collision reuses force_tree (same populations and temperature): the fluid step and its collision are traced
        # inline here, so the shared force belongs to this trace
        fluid_step = type(fluid).step.__wrapped__
        collision = type(fluid).collision.__wrapped__
        fluid._shared_force = force_tree
        fluid.collision = lambda fin_tree, T=None: collision(fluid, fin_tree, T=T)
        try:
            state_tree, f_postcollision_tree = fluid_step(fluid, state_tree, timestep, return_fpost, T)
        finally:
            fluid._shared_force = None
            del fluid.collision
        T = self._advance_temperature(T, timestep, rho_tree, u_tree)
        T = self.apply_bc(T, timestep)
        return self.precision_policy.cast_to_output(T), state_tree, f_postcollision_tree

    @partial(jit, static_argnums=(0,), inline=True)
    def RHS(self, T, rho_tree, u_tree):
        """Evaluate the phase-change energy equation.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.
        rho_tree (pytree of jax.numpy.ndarray): Component density fields.
        u_tree (pytree of jax.numpy.ndarray): Component velocity fields.

        Returns
        -------
        jax.numpy.ndarray: Temperature time derivative.
        """
        return self.RHS_prepared(T, self.prepare_RHS(rho_tree, u_tree))

    def prepare_RHS(self, rho_tree, u_tree):
        """T-independent parts of RHS, evaluated once per time step (they are frozen over the Runge-Kutta substeps)."""
        rho = self.fluid_solver.compute_total_density(rho_tree)
        u = self.fluid_solver.compute_total_velocity(rho_tree, u_tree)
        u = u.at[:, :, (0, -1), :].set(0.0)
        conductivity = rho * self.thermal_diffusivity_factor
        # Regrouped coefficients, so each Runge-Kutta stage reads as few fields as possible (see Thermal.prepare_RHS):
        # RHS = v . grad T + a lap T - T dp_EOS/dT w (+ S / (rho c_v))
        rho_c_v = rho * self.c_v
        velocity = -u + self.grad_x(conductivity) / rho_c_v
        diffusivity = conductivity / rho_c_v
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

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        rho = np.array(kwargs["rho"][0][0, ..., 0])
        u = np.array(kwargs["u"][0][0, ...])
        temperature = np.array(kwargs["T"][0, ..., 0])
        u[:, :, (0, -1), :] = 0.0
        if not np.isfinite(rho).all() or not np.isfinite(temperature).all():
            logger.info(f"Simulation diverged at timestep {timestep}.")
            self.stop_simulation = True
            return

        mass = float(rho.sum(dtype=np.float64))
        if self.initial_mass is None:
            self.initial_mass = mass
        relative_mass_error = abs(mass - self.initial_mass) / self.initial_mass
        vapor_fraction = float(np.mean(rho < 0.5 * (rho_l + rho_g)))
        fields = {
            "rho": rho,
            "u_x": u[..., 0],
            "u_y": u[..., 1],
            "u_z": u[..., 2],
            "T": temperature,
        }
        save_fields_hdf5_xdmf(timestep, fields, output_dir, prefix="pool_boiling")
        logger.info(
            f"timestep {timestep}: vapor fraction = {vapor_fraction:.4f}, "
            f"T/Tc = {temperature.min() / Tc:.4f}/{temperature.max() / Tc:.4f}, "
            f"mass error = {relative_mass_error:.2e}"
        )


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 256
    ny = 256
    nz = 128

    h_liquid = int(0.6 * nz)
    temperature_disturbance_seed = 0
    temperature_disturbance_std = 0.07
    gravity_relaxation_steps = 1000
    contact_angle = np.deg2rad(60.0)

    a = 2 / 49
    b = 2 / 21
    R = 1.0
    pr_omega = 0.344
    Tc = 0.0729190375
    T_sat = 0.86 * Tc
    latent_heat = 0.3813
    specific_heat = 6.0
    jacob_number = 0.22
    T_bottom = T_sat + jacob_number * latent_heat / specific_heat
    T_top = T_sat

    rho_l = 6.499210784
    rho_g = 0.379598891

    gravity = 3e-5
    thermal_diffusivity_factor = 0.028
    shear_relaxation = 1.25

    e = LatticeD3Q19().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((19, 19))
    M[0, :] = en**0
    M[1, :] = 19 * en**2 - 30
    M[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    M[3, :] = e[:, 0]
    M[4, :] = (5 * en**2 - 9) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (5 * en**2 - 9) * e[:, 1]
    M[7, :] = e[:, 2]
    M[8, :] = (5 * en**2 - 9) * e[:, 2]
    M[9, :] = 3 * e[:, 0] ** 2 - en**2
    M[10, :] = (3 * en**2 - 5) * (3 * e[:, 0] ** 2 - en**2)
    M[11, :] = e[:, 1] ** 2 - e[:, 2] ** 2
    M[12, :] = (3 * en**2 - 5) * (e[:, 1] ** 2 - e[:, 2] ** 2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1] ** 2 - e[:, 2] ** 2) * e[:, 0]
    M[17, :] = (e[:, 2] ** 2 - e[:, 0] ** 2) * e[:, 1]
    M[18, :] = (e[:, 0] ** 2 - e[:, 1] ** 2) * e[:, 2]

    eos = PengRobinson(a=[a], b=[b], R=[R], pr_omega=[pr_omega], temperature_field_type="thermal")
    fluid = PoolFluid3D(
        n_components=1,
        lattice=LatticeD3Q19(precision),
        nx=nx,
        ny=ny,
        nz=nz,
        g_kkprime=-1.0 * np.ones((1, 1)),
        EOS=eos,
        body_force=None,
        lambda_=[0.27**0.5],  # k = lambda_**2 = 0.27
        M=[M],
        s_rho=[0.0],
        s_e=[0.8],
        s_eta=[1.0],
        s_j=[0.0],
        s_q=[1.0],
        s_pi=[1.0],
        s_m=[1.0],
        s_v=[shear_relaxation],
        kappa=[0.0],
        precision=precision,
        io_rate=1000,
        print_info_rate=1000,
        checkpoint_rate=0,
        # wetting_formulation="geometric",
    )
    sim = PoolBoiling3D(
        fluid_solver=fluid,
        specific_heat=specific_heat,
        thermal_conductivity=1.0,
        thermal_diffusivity_factor=thermal_diffusivity_factor,
        rk_substeps=2,
    )
    sim.run(800000)
