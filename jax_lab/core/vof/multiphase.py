"""
Definition of the VoF class for simulating two-phase flows with the volume-of-fluid (VoF) lattice Boltzmann method.
"""

import itertools
import logging
import time
from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import jit, shard_map
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding, PartitionSpec

from ..base import LBMBase
from ..boundary_conditions import INLET_OUTLET_BC_TYPES, WALL_BC_TYPES, BounceBack, BounceBackHalfway, NonEquilibriumExtrapolation, link_fullway_walls
from ..lattice import LatticeD2Q9, LatticeD3Q19
from ..utils import colored, downsample_field
from .boundary_conditions import VoFOpenBoundary

logger = logging.getLogger(__name__)

# Cell types
GAS = 0
INTERFACE = 1
LIQUID = 2
SOLID = 3


def _lattice_dot(direction, vector):
    """c_i . v for an integer lattice direction and per-axis fields, as additions only (0.0 for the rest direction)."""
    total = None
    for component, value in zip(direction, vector, strict=True):
        if component == 0:
            continue
        term = value * float(component) if abs(component) != 1 else (value if component > 0 else -value)
        total = term if total is None else total + term
    return 0.0 if total is None else total


class VoF(LBMBase):
    """
    Two-phase (liquid-gas or liquid-liquid) model based on the volume-of-fluid (free-surface) lattice Boltzmann method.

    Each phase is evolved on its own lattice (pytree of two distribution arrays, liquid first), non-dimensionalized by its own density, so both
    lattices carry densities close to one regardless of the physical density ratio. A liquid mass field m and a cell type field (gas, interface,
    liquid, solid) track the interface; the liquid volume fraction is eps = m / rho_liquid. The lattices interact only through interface cells:

    1. Velocity: both lattices share one interface velocity, the momentum-conserving average u_I = (rho_l u_l + r rho_g u_g) / (rho_l + r rho_g)
       (paper Eq. 13), imposed on the momentum moment only, with r the physical density ratio rho_gas / rho_liquid.

    2. Pressure: populations that stream into a liquid interface cell from a gas cell are rebuilt from the gas-side pressure plus the capillary jump,
       p_l = p_g + sigma * kappa (+ optional viscous normal-stress jump, paper Eq. 12), so rho_l = 1 + r (rho_g - 1) + (sigma kappa) / cs2.
       Populations that stream into a gas interface cell from a liquid cell are rebuilt with the local gas density, so the liquid acts on the
       gas as a moving wall while the gas sets the liquid pressure (Dirichlet-Neumann coupling, stable at large density ratios).
       Both use f_i = feq_i(rho, u_I) + feq_opp(i)(rho, u_I) - f_opp(i) + 2 g_opp(i) (Koerner et al. 2005, plus twice the post-collision
       non-equilibrium part g of the cell, which continues each phase's viscous stress across the interface link). With consistent_link_stress,
       g is that of this step's reconstructed populations, the same that the shear-stress return below uses, so that the link momentum is
       returned exactly.

    3. Shear stress: the momentum these interface links exchange with each lattice is returned through u_I of the next step, so the viscous
       stress of each phase reaches the other one and the total momentum is conserved. The body force of interface cells is weighted by the
       volume fraction the same way.

    4. Mass: interface cells exchange liquid mass with liquid and interface neighbors through their links, interface-interface exchange weighted
       by the mean volume fraction. Cells with eps > 1 + alpha (or without gas neighbors) fill, cells with eps < -alpha (or without liquid
       neighbors) empty, their excess mass is spread evenly over the neighboring interface cells, and the interface layer is kept closed.

    The curvature is kappa = div(n) with n = -grad(eps) / |grad(eps)| the outward liquid normal (paper Eq. 11), evaluated with isotropic lattice
    stencils on a smoothed volume fraction. At walls with a contact angle theta, n = cos(theta) n_wall + sin(theta) n_tangent (paper Eq. 10).
    Evaporation (paper Eqs. 16-22) is only active when a VoFOpenBoundary with a nonzero evaporation_flux is defined.

    Performance layout: the populations of each lattice are stored direction-major, as one array per lattice direction (a tuple of q planes),
    in their post-collision state. Every step is one fused pass per lattice (pull streaming, collision, full-way bounce-back and the reset of
    the inactive cells), plus the interface geometry over scalar fields, while all interface work runs on a compacted band of interface cells
    and their neighbors (gathered rows, scattered back in place), so its cost scales with the interface size.

    Parameters
    ----------
    density_ratio (float): Physical density ratio rho_gas / rho_liquid.

    surface_tension (float, optional): Surface tension in liquid lattice units, sigma dt^2 / (rho_liquid dx^3). Defaults to 0.0.

    body_force (array-like, optional): Body force per unit mass (acceleration) in lattice units, either one vector applied to both phases (e.g.
        gravity) or a list of two vectors [liquid, gas]. Defaults to None.

    alpha (float, optional): Stabilizing threshold of the cell type transitions (paper Eq. 14). Defaults to 1e-3.

    viscous_stress_jump (bool, optional): Add the viscous normal-stress jump 2 (mu_l e_l - mu_g e_g), e = n . E . n, to the interface pressure
        jump (paper Eq. 12). The strain rates are evaluated locally from the non-equilibrium populations. Defaults to False.

    curvature_smoothing (int, optional): Passes of lattice-weight smoothing applied to the volume fraction before the normals and curvature
        are computed. Each pass reduces the cell-to-cell curvature noise of the sharp volume fraction field and the contact-angle hysteresis
        of the cell-based contact line. Defaults to 3.

    ghost_contact_angle (bool, optional): Extend the volume fraction into solid wall cells with a contact angle so that the interface meets
        the wall at theta (ghost cells, see _interface_geometry_m), instead of the neutral mean of the fluid neighbors. Defaults to True.

    zero_net_capillary_force (bool, optional): Remove the net capillary force of every droplet (see _balanced_curvature). The capillary
        pressure sigma kappa enters the liquid through the reconstructed populations of its interface links, and the curvature of a sharp
        volume fraction on a fixed grid is not exact: summed over a closed surface, its error leaves a net force that depends on where the
        droplet sits within its cells. Nothing in the gas takes up its reaction, so it pins slow droplets to the lattice and breaks the
        conservation of momentum (a droplet translating with the gas at 1e-4 oscillates in place). With this option the curvature of each
        droplet is corrected by the smallest change (least squares) that makes its net capillary force vanish. Droplets are labeled by
        connected liquid and interface cells at initialization, and labels follow the interface (merged droplets take the smaller label;
        the parts of a droplet that breaks up keep one label and are balanced together). Only for free droplets: a droplet on a wall carries
        a real net capillary force (taken up by the wall at the contact line). Defaults to False.

    max_droplets (int, optional): Largest number of droplet labels with zero_net_capillary_force. Defaults to 64.

    consistent_link_stress (bool, optional): Rebuild the missing interface populations with the same viscous part that is returned through
        the shared interface velocity (see _band_branch), so that the interface conserves momentum when it moves or deforms. Without it a
        droplet with surface tension moving through the grid gains or loses momentum (25 % in 1e4 steps for a 40-cell droplet carried by
        the gas). The rebuild feeds back with 1 - s_nu, so it is unstable when the shear relaxation rate s_nu of either phase approaches 2
        (low viscosity: a 3D droplet impact at nu_l = 5.6e-4 diverges); tested stable up to s_nu = 1.43. Use it with
        zero_net_capillary_force for droplets moving through the grid at moderate viscosity. Defaults to False.

    capillary_curvature_degree (int, optional): With zero_net_capillary_force, each droplet's curvature is replaced by its least-squares
        fit (weighted by the gas-link area of the rows) by a polynomial of this degree in the interface normal components, see
        _fitted_curvature. On a near-spherical droplet the normal is the radial direction, so the fit keeps the spherical harmonics up to
        l = degree: the mean curvature (Laplace pressure) and the low shape modes (restoring force, oscillations). It drops the fine part of
        the gradient curvature, an error pattern tied to the grid: on a droplet that moves through the grid this pattern shakes the liquid
        and grows until the droplet blows up (after ~0.7 cells of travel for a 40-cell droplet), while the fitted curvature keeps it clean.
        0 keeps the local curvature (fine for droplets at rest). Defaults to 3.

    capillary_shift_relaxation (float, optional): The fit is shifted per droplet by a uniform amount so that the projected-area mean of
        the curvature, which sets the pressure inside, stays that of the local curvature (the link-area weighted fit alone lowers the
        Laplace pressure by ~2 %). The shift is kept in the state and relaxed towards its instantaneous value at this rate per step, so that
        it does not jump when cells change type. Defaults to 0.02.


    References
    ----------
    1. Mohammadrezaei, S., Farajzadeh, R. & Niasar, V. Modeling the drying process of CO2-water in porous media with a novel
    volume-of-fluid lattice Boltzmann model. Water Resources Research 62, e2025WR040829 (2026).

    2. Koerner, C., Thies, M., Hofmann, T., Thuerey, N. & Ruede, U. Lattice Boltzmann model for free surface flow for modeling foaming.
    Journal of Statistical Physics 121, 179-196 (2005).

    3. Thuerey, N. Physically based animation of free surface flows with the lattice Boltzmann method. PhD thesis, University of
    Erlangen-Nuremberg (2007).

    4. Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann equation. In Proceedings of the
    5th International EHD Workshop (pp. 241-246). University of Poitiers, Poitiers, France.

    Notes
    -----
    1. Boundary conditions are defined per lattice: self.BCs[0] for the liquid lattice and self.BCs[1] for the gas lattice. Walls are usually
       added to both. Supported: full-way BounceBack walls (the contact angle, in radians, is read from their theta) and
       NonEquilibriumExtrapolation/VoFOpenBoundary open boundaries; other boundary conditions raise NotImplementedError.
    2. initialize_volume_fraction must return the initial liquid volume fraction. initialize_macroscopic_fields may return
       ([rho_l, rho_g], [u_l, u_g]); it defaults to rest at unit density.
    3. Pressures are gauge values in liquid units: p = cs2 (rho_l - 1) in the liquid and r cs2 (rho_g - 1) in the gas.
    4. The populations passed to output_data ("f_postcollision_tree") are post-collision, one array per lattice direction.
    """

    def __init__(self, **kwargs):
        self.n_components = 2
        super().__init__(**kwargs)
        self.density_ratio = kwargs.get("density_ratio")
        self.surface_tension = kwargs.get("surface_tension", 0.0)
        self.body_force = kwargs.get("body_force", None)
        self.alpha = kwargs.get("alpha", 1.0e-3)
        self.viscous_stress_jump = kwargs.get("viscous_stress_jump", False)
        self.curvature_smoothing = kwargs.get("curvature_smoothing", 3)
        if not isinstance(self.curvature_smoothing, int) or self.curvature_smoothing < 0:
            raise ValueError("curvature_smoothing must be a non-negative integer")
        self.ghost_contact_angle = kwargs.get("ghost_contact_angle", True)
        self.zero_net_capillary_force = bool(kwargs.get("zero_net_capillary_force", False))
        self.consistent_link_stress = bool(kwargs.get("consistent_link_stress", False))
        self.max_droplets = kwargs.get("max_droplets", 64)
        self.capillary_curvature_degree = kwargs.get("capillary_curvature_degree", 3)
        if not isinstance(self.capillary_curvature_degree, int) or self.capillary_curvature_degree < 0:
            raise ValueError("capillary_curvature_degree must be a non-negative integer")
        self.capillary_shift_relaxation = float(kwargs.get("capillary_shift_relaxation", 0.02))
        if not 0.0 < self.capillary_shift_relaxation <= 1.0:
            raise ValueError("capillary_shift_relaxation must be in (0, 1]")
        if not isinstance(self.max_droplets, int) or self.max_droplets < 1:
            raise ValueError("max_droplets must be a positive integer")

        self._lattice_directions = tuple(tuple(int(v) for v in np.asarray(self.lattice.c)[:, i]) for i in range(self.q))
        self._lattice_weights = tuple(float(v) for v in np.asarray(self.lattice.w))
        self._opposite = tuple(int(v) for v in np.asarray(self.lattice.opp_indices))
        self._neighbor_indices = tuple(i for i, direction in enumerate(self._lattice_directions) if any(direction))

        for component_bcs in self.BCs:
            for bc in component_bcs:
                if not isinstance(bc, (BounceBack, NonEquilibriumExtrapolation)):
                    raise NotImplementedError(
                        f"{type(bc).__name__} is not supported by VoF: use BounceBack walls and NonEquilibriumExtrapolation/VoFOpenBoundary "
                        "open boundaries (applied within the fused direction-major step)."
                    )
        open_bcs = [bc for component_bcs in self.BCs for bc in component_bcs if isinstance(bc, VoFOpenBoundary)]
        fluxes = {bc.evaporation_flux for bc in open_bcs if bc.evaporation_flux != 0.0}
        if len(fluxes) > 1:
            raise ValueError("All VoFOpenBoundary instances with evaporation must use the same evaporation_flux.")
        self._evaporation_flux = fluxes.pop() if fluxes else 0.0
        self._evaporation_start = min((bc.evaporation_start for bc in open_bcs if bc.evaporation_flux != 0.0), default=0)
        self._has_evaporation = self._evaporation_flux != 0.0

        (
            self.solid_mask,
            self.contact_mask,
            self.wall_normal,
            self.wall_cos,
            self.wall_sin,
            self.boundary_cells,
            self.boundary_normal,
            self.boundary_cot,
            self.boundary_fluid_dirs,
        ) = self._create_wall_geometry()

        P = PartitionSpec
        field_spec = P("x", None, None) if self.dim == 2 else P("x", None, None, None)
        self.interface_geometry = jit(
            shard_map(
                self._interface_geometry_m,
                mesh=self.mesh,
                in_specs=(field_spec,) * 6 + (P("x", None, None),) * 4,
                out_specs=(field_spec,) * self.dim,
                check_vma=False,
            )
        )
        self.fluid_field = (~self.solid_mask).astype(self.precision_policy.compute_dtype)
        self._open_bc_phases, self._open_bc_data = self._create_open_boundary_data()
        plane_spec = self._plane_spec()
        self.dense_kernel = jit(
            shard_map(
                self._dense_m,
                mesh=self.mesh,
                in_specs=(plane_spec, plane_spec, field_spec, field_spec, tuple((P("x", None, None),) * 4 for _ in self._open_bc_data)),
                out_specs=(plane_spec, plane_spec, field_spec, P()),
                check_vma=False,
            )
        )
        # Band kernels per static band capacity (see _band_m), built on first use, and the capacity queued for the next step
        self._band_kernels = {}
        self._next_capacity = None

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("omega must be a list [omega_liquid, omega_gas]")
        self._omega = [float(v) for v in value]

    @property
    def density_ratio(self):
        return self._density_ratio

    @density_ratio.setter
    def density_ratio(self, value):
        if value is None:
            raise ValueError("density_ratio (rho_gas / rho_liquid) must be provided")
        if not isinstance(value, (int, float)) or value <= 0.0:
            raise ValueError("density_ratio must be a positive number")
        self._density_ratio = float(value)

    @property
    def surface_tension(self):
        return self._surface_tension

    @surface_tension.setter
    def surface_tension(self, value):
        if not isinstance(value, (int, float)) or value < 0.0:
            raise ValueError("surface_tension must be a non-negative number")
        self._surface_tension = float(value)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, (int, float)) or not 0.0 < value < 0.5:
            raise ValueError("alpha must be a number in (0, 0.5)")
        self._alpha = float(value)

    @property
    def body_force(self):
        return self._body_force

    @body_force.setter
    def body_force(self, value):
        if value is None:
            self._body_force = None
            self._acceleration = (None, None)
            return
        acceleration = np.asarray(value, dtype=np.float64)
        if acceleration.ndim == 1:
            acceleration = np.stack([acceleration, acceleration])
        if acceleration.shape != (2, self.dim):
            raise ValueError("body_force must be one vector of length dim or a list of two such vectors [liquid, gas]")
        self._body_force = acceleration
        # Static per-phase accelerations (None when zero) so that collision() skips the forcing term entirely when unused
        self._acceleration = tuple(tuple(float(v) for v in a) if np.any(a != 0.0) else None for a in acceleration)

    def _half_acceleration(self, phase):
        """a / 2 of a phase as a compute-precision vector, the shift between the bare and the physical (force-corrected) velocity."""
        acceleration = self._acceleration[phase]
        values = acceleration if acceleration is not None else (0.0,) * self.dim
        return 0.5 * jnp.asarray(values, dtype=self.precision_policy.compute_dtype)

    def _shear_relaxation(self):
        """Shear relaxation rate of each phase, [liquid, gas]. Overridden by the collision models."""
        raise NotImplementedError

    def kinematic_viscosity(self):
        """Kinematic viscosity of each phase in lattice units, nu = cs2 (1 / s_nu - 1 / 2)."""
        return [self.lattice.cs2 * (1.0 / s - 0.5) for s in self._shear_relaxation()]

    def _create_boundary_data(self):
        """
        Create boundary data for both lattices by setting boundary conditions, creating grid masks, and preparing local masks and normal arrays.
        self.BCs[0] holds the liquid lattice boundary conditions and self.BCs[1] the gas lattice ones.
        """
        self.BCs = [[], []]
        self.set_boundary_conditions()
        for name, BCs in zip(("liquid", "gas"), self.BCs, strict=True):
            logger.info(f"Lattice: {name}")
            solid_halo_list = [np.array(bc.indices).T for bc in BCs if bc.is_solid]
            solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

            start = time.time()
            grid_mask = self.create_grid_mask(solid_halo_voxels)
            logger.info("Time to create the grid mask: %.6f seconds", time.time() - start)

            start = time.time()
            for bc in BCs:
                assert bc.implementation_step in ["PostStreaming", "PostCollision"]
                bc.create_local_mask_and_normal_arrays(grid_mask)
            link_fullway_walls(BCs)
            logger.info("Time to create the local masks and normal arrays: %.6f seconds", time.time() - start)

    def _make_local_bounceback_indices(self):
        """
        Full-way BounceBack is applied densely, as a node mask selected within the fused step (see _step_m), instead of through index
        scatters. The mask of both lattices (shape (..., 2)) is built here and no local index arrays are returned.
        """
        spatial = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        walls = np.zeros((*spatial, 2), dtype=np.int8)
        for phase, BCs in enumerate(self.BCs):
            for bc in BCs:
                if isinstance(bc, BounceBack):
                    walls[tuple(np.asarray(bc.indices, dtype=np.int64)) + (phase,)] = 1
        self._has_bounceback = tuple(bool(np.any(walls[..., phase])) for phase in range(2))
        self.bounceback_walls = self.distributed_array_init((*spatial, 2), jnp.int8, init_val=walls)
        return [None, None]

    def _create_open_boundary_data(self):
        """
        Local (per-shard) data of the NonEquilibriumExtrapolation/VoFOpenBoundary open boundaries of each lattice: boundary cells, their
        interior neighbors (along the inward normal, in the same shard), the missing-direction mask and the prescribed density.

        Returns
        -------
        (tuple, tuple): The lattices that have open boundaries, and per such lattice a tuple of four distributed arrays.
        """
        phases, data = [], []
        for phase, BCs in enumerate(self.BCs):
            open_bcs = [bc for bc in BCs if isinstance(bc, NonEquilibriumExtrapolation)]
            if not open_bcs:
                continue
            for bc in open_bcs:
                bc.find_neighbors()
            cells = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in open_bcs])
            neighbors = np.vstack([np.asarray(bc.indices_nbr, dtype=np.int32).T for bc in open_bcs])
            missing = np.vstack([np.asarray(bc.imissing_mask, dtype=bool) for bc in open_bcs])
            prescribed = np.vstack([
                np.broadcast_to(np.asarray(bc.prescribed, dtype=np.float64).reshape(-1, 1), (len(bc.indices[0]), 1)) for bc in open_bcs
            ])
            local_cells, (local_neighbors, local_missing, local_prescribed) = self._split_local_indices(cells, neighbors, missing, prescribed)
            local_nx = self.nx // self.n_devices
            for device in range(self.n_devices):
                local_neighbors[device, :, 0] -= device * local_nx
            valid = local_cells[..., 0] < local_nx
            local_neighbors[~valid] = 0
            if np.any(valid & ((local_neighbors[..., 0] < 0) | (local_neighbors[..., 0] >= local_nx))):
                raise NotImplementedError("Open boundary neighbors must lie in the same x-shard as their boundary cells.")
            phases.append(phase)
            data.append((
                self._distribute_local(local_cells, jnp.int32),
                self._distribute_local(local_neighbors, jnp.int32),
                self._distribute_local(local_missing, jnp.bool_),
                self._distribute_local(local_prescribed, self.precision_policy.compute_dtype),
            ))
        return tuple(phases), tuple(data)

    def _make_local_wall_bc_data(self):
        """One wall_bc_data dict and one solid-pin index array (or None) per lattice, see LBMBase._make_local_wall_bc_data."""
        wall_bc_data_by_phase = []
        solid_pin_indices_by_phase = []
        for BCs in self.BCs:
            wall_bc_data = {}
            for bc_type, _, has_weights in WALL_BC_TYPES:
                local_indices, (local_imissing, local_iknown, local_vel, local_weights) = self._collect_wall_bc_data(BCs, bc_type, has_weights)
                wall_bc_data[bc_type] = (
                    self._distribute_local(local_indices, jnp.int32),
                    self._distribute_local(local_imissing, jnp.uint8),
                    self._distribute_local(local_iknown, jnp.uint8),
                    self._distribute_local(local_vel),
                    self._distribute_local(local_weights),
                )
            wall_bc_data_by_phase.append(wall_bc_data)
            solid_pin_indices_by_phase.append(self._distribute_local(self._collect_solid_pin_indices(BCs), jnp.int32))
        return wall_bc_data_by_phase, solid_pin_indices_by_phase

    def _make_local_inlet_outlet_data(self):
        """One inlet_outlet_bc_data dict per lattice, see LBMBase._make_local_inlet_outlet_data."""
        inlet_outlet_bc_data_by_phase = []
        for BCs in self.BCs:
            inlet_outlet_bc_data = {}
            for bc_type, ttype, _ in INLET_OUTLET_BC_TYPES:
                local_indices, (local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed) = (
                    self._collect_inlet_outlet_bc_data(BCs, bc_type, ttype)
                )
                inlet_outlet_bc_data[(bc_type, ttype)] = (
                    self._distribute_local(local_indices, jnp.int32),
                    self._distribute_local(local_normals),
                    self._distribute_local(local_imiddle_mask, jnp.bool_),
                    self._distribute_local(local_iknown_mask, jnp.bool_),
                    self._distribute_local(local_imissing, jnp.uint8),
                    self._distribute_local(local_iknown, jnp.uint8),
                    self._distribute_local(local_prescribed),
                )
            inlet_outlet_bc_data_by_phase.append(inlet_outlet_bc_data)
        return inlet_outlet_bc_data_by_phase

    def _make_local_equilibrium_bc_data(self):
        """One distributed local EquilibriumBC index array and value array (or None) per lattice, see LBMBase._make_local_equilibrium_bc_data."""
        indices_by_phase = []
        out_by_phase = []
        for BCs in self.BCs:
            local_indices, local_out = self._collect_equilibrium_bc_data(BCs)
            indices_by_phase.append(self._distribute_local(local_indices, jnp.int32))
            out_by_phase.append(self._distribute_local(local_out))
        return indices_by_phase, out_by_phase

    def _host_neighbor(self, field, direction):
        """Value of a global host array at x + direction (periodic)."""
        return np.roll(field, tuple(-int(c) for c in direction), axis=tuple(range(self.dim)))

    def _create_wall_geometry(self):
        """
        Build the static solid mask and the contact-angle data from the wall boundary conditions of both lattices.

        Wall cells are fluid cells with a solid neighbor and solid cells with a fluid neighbor. Their wall normal is the normalized isotropic
        gradient of the fluid indicator (pointing into the fluid), and their contact angle is the one of the solid node, or the mean over the
        neighboring solid nodes for fluid cells.

        The solid cells next to the fluid are also listed per shard (local coordinates, wall normal, and cot(theta) clipped to [-2, 2], i.e.
        theta of about 27 to 153 degrees, with a flag marking the ghost cells that carry a contact angle; see _interface_geometry_m).

        Returns
        -------
        (tuple of jax.numpy.ndarray): Distributed solid mask, contact mask, wall normal, cos(theta), sin(theta), and the local ghost cell
        coordinates, wall normals and cot(theta).
        """
        spatial = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        solid = np.zeros(spatial, dtype=bool)
        theta = np.full(spatial, np.nan)
        for bc in (bc for component_bcs in self.BCs for bc in component_bcs):
            if not isinstance(bc, (BounceBack, BounceBackHalfway)):
                continue
            indices = tuple(np.asarray(getattr(bc, "solid_indices", bc.indices), dtype=np.int64))
            solid[indices] = True
            if bc.theta is not None:
                theta[indices] = np.broadcast_to(np.reshape(np.asarray(bc.theta, dtype=np.float64), -1), (len(indices[0]),))

        neighbors = [d for d in self._lattice_directions if any(d)]
        fluid = (~solid).astype(np.float64)
        gradient = np.zeros((*spatial, self.dim))
        any_solid = np.zeros(spatial, dtype=bool)
        any_fluid = np.zeros(spatial, dtype=bool)
        theta_sum = np.zeros(spatial)
        theta_count = np.zeros(spatial)
        for d in neighbors:
            weight = self._lattice_weights[self._lattice_directions.index(d)] / self.lattice.cs2
            neighbor_fluid = self._host_neighbor(fluid, d)
            gradient += weight * neighbor_fluid[..., None] * np.asarray(d, dtype=np.float64)
            any_solid |= neighbor_fluid == 0.0
            any_fluid |= neighbor_fluid == 1.0
            neighbor_theta = self._host_neighbor(theta, d)
            valid = ~np.isnan(neighbor_theta)
            theta_sum += np.where(valid, neighbor_theta, 0.0)
            theta_count += valid

        wall = (~solid & any_solid) | (solid & any_fluid)
        norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
        wall_normal = np.where(wall[..., None] & (norm > 0.0), gradient / np.maximum(norm, 1e-12), 0.0)
        theta = np.where(np.isnan(theta), np.where(theta_count > 0, theta_sum / np.maximum(theta_count, 1), np.nan), theta)
        contact = wall & ~np.isnan(theta) & (norm[..., 0] > 0.0)
        theta = np.where(contact, theta, 0.0)
        self._has_contact_angle = bool(np.any(contact))
        self._solid_host = solid

        dtype = self.precision_policy.compute_dtype
        # Solid cells next to the fluid, whose volume fraction is extended from the fluid every smoothing pass (ghost cells when they carry a
        # contact angle and ghost_contact_angle is set, the mean of their fluid neighbors otherwise)
        boundary = solid & any_fluid
        cells = np.argwhere(boundary).astype(np.int32)
        if len(cells) == 0:
            cells = np.array([[self.nx - 1] + [0] * (self.dim - 1)], dtype=np.int32)
            ghost_normal, ghost_cot, fluid_dirs = np.zeros((1, self.dim)), np.zeros((1, 2)), np.full((1, 1), -1, dtype=np.int32)
            local_cells, (local_normal, local_cot, local_dirs) = self._split_local_indices(cells, ghost_normal, ghost_cot, fluid_dirs)
            local_cells[...] = 0
            local_cells[..., 0] = self.nx // self.n_devices
        else:
            ghost_normal = wall_normal[boundary]
            is_ghost = (contact[boundary] & bool(self.ghost_contact_angle)).astype(np.float64)
            cot = np.clip(1.0 / np.tan(np.clip(theta[boundary], 1e-3, np.pi - 1e-3)), -2.0, 2.0)
            ghost_cot = np.stack((np.where(is_ghost > 0, cot, 0.0), is_ghost), axis=-1)
            # Directions of the fluid neighbors of every boundary cell, padded with -1 to the largest count
            fluid_neighbor = np.stack(
                [self._host_neighbor(fluid, d)[boundary] > 0.0 if any(d) else np.zeros(len(cells), bool) for d in self._lattice_directions], axis=-1
            )
            width = int(fluid_neighbor.sum(axis=1).max())
            order = np.argsort(~fluid_neighbor, axis=1, kind="stable")[:, :width]
            fluid_dirs = np.where(np.take_along_axis(fluid_neighbor, order, axis=1), order, -1).astype(np.int32)
            local_cells, (local_normal, local_cot, local_dirs) = self._split_local_indices(cells, ghost_normal, ghost_cot, fluid_dirs)
            local_dirs[local_cells[..., 0] >= self.nx // self.n_devices] = -1
        return (
            self.distributed_array_init((*spatial, 1), jnp.bool_, init_val=solid[..., None]),
            self.distributed_array_init((*spatial, 1), jnp.bool_, init_val=contact[..., None]),
            self.distributed_array_init((*spatial, self.dim), dtype, init_val=wall_normal),
            self.distributed_array_init((*spatial, 1), dtype, init_val=np.cos(theta)[..., None]),
            self.distributed_array_init((*spatial, 1), dtype, init_val=np.sin(theta)[..., None]),
            self._distribute_local(local_cells, jnp.int32),
            self._distribute_local(local_normal, dtype),
            self._distribute_local(local_cot, dtype),
            self._distribute_local(local_dirs, jnp.int32),
        )

    def _interpolate(self, field, points):
        """Multilinear interpolation of a single-channel field (*shape, 1) at fractional index points (n, dim), clamped to the field."""
        shape = np.asarray(field.shape[: self.dim])
        base = jnp.clip(jnp.floor(points), 0, shape - 2).astype(jnp.int32)
        frac = jnp.clip(points - base, 0.0, 1.0)
        value = 0.0
        for corner in np.ndindex(*(2,) * self.dim):
            weight = 1.0
            for axis, bit in enumerate(corner):
                weight = weight * (frac[:, axis] if bit else 1.0 - frac[:, axis])
            value = value + weight * field[tuple(base[:, axis] + bit for axis, bit in enumerate(corner))][:, 0]
        return value

    def _x_halos(self, field):
        """x-halo planes of a local shard: the plane at x = -1 (left) and x = nx_local (right), periodic across devices."""
        return self.send_right(field[-1:], "x"), self.send_left(field[:1], "x")

    def _rolled(self, field, shift, halos=None):
        """
        field shifted by shift (|shift| <= 1 per axis) over a whole local shard, out(x) = field(x - shift), periodic in y/z and across devices
        in x (halos: the planes at x = -1 and x = nx_local, see _x_halos; derived from the field itself on a single device).

        Every shift is a sum of zero-padded slices: pad and add fuse into the consumer, whereas the concatenation behind jnp.roll is
        materialized as a separate kernel per shifted array.
        """
        zero = jnp.zeros((), field.dtype)
        for axis, s in enumerate(shift):
            if s == 0:
                continue
            n = field.shape[axis]
            if axis == 0 and halos is not None:
                wrapped = halos[0] if s == 1 else halos[1]
            else:
                wrapped = jax.lax.slice_in_dim(field, n - 1, n, axis=axis) if s == 1 else jax.lax.slice_in_dim(field, 0, 1, axis=axis)
            kept = jax.lax.slice_in_dim(field, 0, n - 1, axis=axis) if s == 1 else jax.lax.slice_in_dim(field, 1, n, axis=axis)
            kept_pad = [(0, 0, 0)] * field.ndim
            wrapped_pad = [(0, 0, 0)] * field.ndim
            kept_pad[axis] = (1, 0, 0) if s == 1 else (0, 1, 0)
            wrapped_pad[axis] = (0, n - 1, 0) if s == 1 else (n - 1, 0, 0)
            field = jax.lax.pad(kept, zero, kept_pad) + jax.lax.pad(wrapped, zero, wrapped_pad)
        return field

    def _view(self, field, halos, offset):
        """Value at x + offset (|offset| <= 1 per axis) over a whole local shard (see _rolled)."""
        return self._rolled(field, tuple(-o for o in offset), halos if self.n_devices > 1 else None)

    def _stencil_sum(self, field, weights):
        """sum_i weights_i field(x + c_i) over a whole local shard, for static per-direction weights (zero weights skipped)."""
        halos = self._x_halos(field)
        terms = [w * self._view(field, halos, d) for d, w in zip(self._lattice_directions, weights, strict=True) if w != 0.0]
        return reduce(lambda a, b: a + b, terms)

    def _gradient_full(self, field):
        """Isotropic lattice gradient grad f = sum_i (w_i / cs2) c_i f(x + c_i) of a single-channel field over a whole local shard, one pass."""
        halos = self._x_halos(field) if self.n_devices > 1 else None
        gradient = [0.0] * self.dim
        for direction, weight in zip(self._lattice_directions, self._lattice_weights, strict=True):
            if not any(direction):
                continue
            value = self._view(field, halos, direction)
            for axis, component in enumerate(direction):
                if component != 0:
                    gradient[axis] = gradient[axis] + (weight / self.lattice.cs2 * component) * value
        return gradient

    def _gather_neighbors(self, field, coords):
        """Values of a single-channel field at x + c_i for every lattice direction i at the given local cells (tuple of (n,)), shape (n, q)."""
        halos = self._x_halos(field) if self.n_devices > 1 else None
        nx_local = field.shape[0]
        columns = []
        for direction in self._lattice_directions:
            source_x = coords[0] + direction[0]
            rest = tuple((coords[axis] + direction[axis]) % field.shape[axis] for axis in range(1, self.dim))
            if halos is None:
                columns.append(field[(source_x % nx_local,) + rest][:, 0])
                continue
            zero = jnp.zeros_like(source_x)
            inside = field[(jnp.clip(source_x, 0, nx_local - 1),) + rest][:, 0]
            left, right = halos[0][(zero,) + rest][:, 0], halos[1][(zero,) + rest][:, 0]
            columns.append(jnp.where(source_x < 0, left, jnp.where(source_x >= nx_local, right, inside)))
        return jnp.stack(columns, axis=-1)

    def _interface_geometry_m(
        self, eps, fluid, contact, wall_normal, wall_cos, wall_sin, boundary_cells, boundary_normal, boundary_cot, boundary_dirs
    ):
        """
        Outward liquid normal of a local shard.

        The volume fraction is smoothed curvature_smoothing times with the lattice weights, then n = -grad(eps) / |grad(eps)| (paper Eq. 11
        normal), replaced at wall cells by the contact-angle normal (paper Eq. 10). The curvature kappa = div(n) (positive for a liquid droplet)
        is evaluated at the interface cells from these normals (_band_branch).

        Before and after every smoothing pass, the solid cells next to the fluid (boundary cells) are set from the fluid so that walls do not
        read as gas: the mean of their fluid neighbors, or, at solid cells with a contact angle (ghost cells), the volume fraction across the
        wall shifted along it by cot(theta) towards the liquid, eps_ghost = eps(x + n_wall - cot(theta) t), so that the interface continued
        into the wall meets it at theta and the normals next to the wall carry the contact angle as well. The boundary cells are a static list,
        so this costs a gather and a scatter over them only; the dense passes are the smoothing, the gradient and the divergence.

        Parameters
        ----------
        eps (jax.numpy.ndarray): Liquid volume fraction, shape (..., 1).

        fluid (jax.numpy.ndarray): Fluid indicator (1 - solid), shape (..., 1).

        contact (jax.numpy.ndarray): Contact-angle mask, shape (..., 1).

        wall_normal (jax.numpy.ndarray): Wall normal pointing into the fluid, shape (..., dim).

        wall_cos, wall_sin (jax.numpy.ndarray): cos(theta) and sin(theta) at contact cells, shape (..., 1).

        boundary_cells, boundary_normal, boundary_cot (jax.numpy.ndarray): Local coordinates, wall normals, and cot(theta) with the ghost
            flag of the boundary cells, padded rows pointing outside the shard, shapes (1, n, dim), (1, n, dim) and (1, n, 2).

        boundary_dirs (jax.numpy.ndarray): Directions of the fluid neighbors of every boundary cell, padded with -1, shape (1, n, k).

        Returns
        -------
        (tuple of jax.numpy.ndarray): The dim normal components, each of shape (..., 1) (separate arrays: interleaving them as channels of one
        array costs a full extra pass). The curvature kappa = div(n) is only needed at interface cells and is evaluated there (_band_branch).
        """
        dtype = self.precision_policy.compute_dtype
        barrier = jax.lax.optimization_barrier
        cells, wall = boundary_cells[0], boundary_normal[0].astype(dtype)
        cot, is_ghost = boundary_cot[0][:, :1].astype(dtype), boundary_cot[0][:, 1:] > 0.0
        nx_local = eps.shape[0]
        valid = cells[:, 0] < nx_local
        index = tuple(jnp.where(valid, cells[:, axis], 1 << 30) for axis in range(self.dim))
        safe = tuple(jnp.where(valid, cells[:, axis], 0) for axis in range(self.dim))
        c = np.asarray(self.c, dtype=np.float64)
        w = np.asarray(self.lattice.w, dtype=np.float64)
        use_ghost = self._has_contact_angle and self.ghost_contact_angle

        # Fluid neighbors of the boundary cells: coordinates (n, k) per axis and whether each entry is used
        dirs = boundary_dirs[0]
        used = dirs >= 0
        inv_count = 1.0 / jnp.maximum(jnp.sum(used, axis=-1, keepdims=True), 1).astype(dtype)
        offsets = jnp.asarray(np.asarray(self.c, dtype=np.int32).T)[jnp.maximum(dirs, 0)]
        neighbor_x = safe[0][:, None] + offsets[..., 0]
        neighbor_rest = tuple((safe[axis][:, None] + offsets[..., axis]) % eps.shape[axis] for axis in range(1, self.dim))

        def neutral(field):
            # Mean of the fluid neighbors, with x-halo planes on several devices
            if self.n_devices == 1:
                values = field[(neighbor_x % nx_local,) + neighbor_rest][..., 0]
            else:
                left, right = self._x_halos(field)
                zero = jnp.zeros_like(neighbor_x)
                inside = field[(jnp.clip(neighbor_x, 0, nx_local - 1),) + neighbor_rest][..., 0]
                values = jnp.where(
                    neighbor_x < 0,
                    left[(zero,) + neighbor_rest][..., 0],
                    jnp.where(neighbor_x >= nx_local, right[(zero,) + neighbor_rest][..., 0], inside),
                )
            return jnp.sum(jnp.where(used, values, 0.0), axis=-1, keepdims=True) * inv_count

        def set_boundary(field, points=None):
            values = neutral(field)
            if points is not None:
                values = jnp.where(is_ghost, self._interpolate(field, points)[:, None], values)
            return barrier(field.at[index].set(values, mode="drop"))

        field = eps.astype(dtype)
        points = None
        if use_ghost:
            # Tangent towards the gas along the wall, from the outward normal at the boundary cells with the solid neighbors read as the
            # neutral extension
            fluid_neighbors = self._gather_neighbors(fluid, safe)
            neighbors = jnp.where(fluid_neighbors > 0.0, self._gather_neighbors(field, safe), neutral(field))
            normal_at = -jnp.stack([jnp.sum(neighbors * (w * c[axis]), axis=-1) for axis in range(self.dim)], axis=-1)
            normal_at = normal_at / jnp.maximum(jnp.linalg.norm(normal_at, axis=-1, keepdims=True), 1e-12)
            tangent = normal_at - jnp.sum(normal_at * wall, axis=-1, keepdims=True) * wall
            t_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)
            tangent = jnp.where(t_norm > 1e-6, tangent / jnp.maximum(t_norm, 1e-6), 0.0)
            points = jnp.stack(safe, axis=-1).astype(dtype) + wall - cot * tangent
        field = set_boundary(field, points)

        # Smoothing the volume fraction before taking normals removes most of the cell-to-cell noise of the curvature of a sharp field
        for _ in range(self.curvature_smoothing):
            field = set_boundary(barrier(self._stencil_sum(field, list(self._lattice_weights))), points)

        gradient = self._gradient_full(field)
        norm = jnp.sqrt(sum(g * g for g in gradient))
        inv_norm = jnp.where(norm > 1e-8, 1.0 / jnp.maximum(norm, 1e-8), 0.0)
        normal = [-g * inv_norm for g in gradient]
        if self._has_contact_angle:
            wall_n = [wall_normal[..., axis : axis + 1].astype(dtype) for axis in range(self.dim)]
            n_wall = sum(n * v for n, v in zip(normal, wall_n, strict=True))
            tangent = [n - n_wall * v for n, v in zip(normal, wall_n, strict=True)]
            t_norm = jnp.sqrt(sum(t * t for t in tangent))
            override = contact & (t_norm > 1e-6)
            inv_t = 1.0 / jnp.maximum(t_norm, 1e-6)
            normal = [jnp.where(override, wall_cos * v + wall_sin * t * inv_t, n) for n, v, t in zip(normal, wall_n, tangent, strict=True)]
        return tuple(normal)

    def _equilibrium_q(self, rho, u):
        """Equilibrium of rho (..., 1) and u (..., dim) as one (..., q) array, the lattice velocities entering as compile-time constants."""
        cu = 3.0 * self._project_q(u)
        usqr = 1.5 * jnp.sum(u * u, axis=-1, keepdims=True)
        return rho * np.asarray(self.lattice.w, dtype=np.float64) * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

    def _project_q(self, v):
        """c_i . v for every lattice direction, v of shape (..., dim), as one (..., q) array."""
        c = np.asarray(self.c, dtype=np.float64)
        return sum(v[..., axis : axis + 1] * c[axis] for axis in range(self.dim))

    def _moments_q(self, f):
        """Density (..., 1) and velocity (..., dim) of a (..., q) population array."""
        c = np.asarray(self.c, dtype=np.float64)
        rho = jnp.sum(f, axis=-1, keepdims=True)
        u = jnp.concatenate([jnp.sum(f * c[axis], axis=-1, keepdims=True) for axis in range(self.dim)], axis=-1) / rho
        return rho, u

    def _plane_spec(self):
        """Sharding of one population plane (one lattice direction over the shard), split along x."""
        return PartitionSpec("x", *([None] * (self.dim - 1)))

    def _band_kernel(self, capacity):
        """Jitted shard_map of _band_m for a static band capacity (one compiled kernel per capacity, built on first use)."""
        if capacity not in self._band_kernels:
            P = PartitionSpec
            field_spec = P("x", *([None] * self.dim))
            plane_spec = self._plane_spec()
            # State fields: mass, flags, eps (and the droplet labels with zero_net_capillary_force)
            n_state = 4 if self.zero_net_capillary_force else 3
            shift_spec = (P(),) if self.zero_net_capillary_force else ()
            self._band_kernels[capacity] = jit(
                shard_map(
                    partial(self._band_m, capacity=capacity),
                    mesh=self.mesh,
                    in_specs=(plane_spec,) * 4 + (field_spec,) * n_state + shift_spec + (field_spec,) * 2 + (P(),),
                    out_specs=(plane_spec, plane_spec) + (field_spec,) * n_state + shift_spec,
                    check_vma=False,
                ),
                donate_argnums=(2, 3),
            )
        return self._band_kernels[capacity]

    def _band_capacity_for(self, count):
        """Static row capacity for count rows: the smallest of 1, 1.25, 1.5, 1.75 or 2 times a power of two that fits, at least 1024."""
        size = self.nx * self.ny * (self.nz if self.dim == 3 else 1) // self.n_devices
        base = 2 ** int(np.floor(np.log2(max(count, 1))))
        capacity = next(int(base * factor) for factor in (1.0, 1.25, 1.5, 1.75, 2.0) if base * factor >= count)
        return int(min(size, max(1024, capacity)))

    def _halo_slabs(self, planes):
        """x-halo slabs of every population plane of a shard: the plane at x = -1 (left) and x = nx_local (right), periodic across devices."""
        edges = jnp.stack([p[-1:] for p in planes])
        left = self.send_right(edges, "x")
        edges = jnp.stack([p[:1] for p in planes])
        right = self.send_left(edges, "x")
        return list(left), list(right)

    def _pull_planes(self, planes, left, right):
        """Streamed populations f_i(x - c_i) of every plane of a shard (pull streaming), fused into their consumer (see _rolled)."""
        return [
            self._rolled(plane, direction, (l, r) if self.n_devices > 1 else None)
            for plane, direction, l, r in zip(planes, self._lattice_directions, left, right, strict=True)
        ]

    def _pull_rows(self, planes, left, right, coords):
        """Streamed populations f_i(x - c_i) at the given local cells (tuple of (n,) coordinates), shape (n, q)."""
        nx_local = planes[0].shape[0]
        columns = []
        for plane, direction, l, r in zip(planes, self._lattice_directions, left, right, strict=True):
            source_x = coords[0] - direction[0]
            rest = tuple((coords[axis] - direction[axis]) % plane.shape[axis] for axis in range(1, self.dim))
            inside = plane[(jnp.clip(source_x, 0, nx_local - 1),) + rest]
            zero = jnp.zeros_like(source_x)
            columns.append(jnp.where(source_x < 0, l[(zero,) + rest], jnp.where(source_x >= nx_local, r[(zero,) + rest], inside)))
        return jnp.stack(columns, axis=-1)

    @staticmethod
    def _scatter_rows(planes, coords, rows):
        """Planes with the populations of the given local cells replaced by rows (n, q); out-of-shard coordinates are dropped."""
        return [plane.at[coords].set(rows[:, i].astype(plane.dtype), mode="drop") for i, plane in enumerate(planes)]

    def _collide_rows(self, rows, phase, inactive):
        """Collision of populations given as rows (n, q), reset to rest where inactive (n, 1)."""
        columns = self._collide_planes([rows[:, i] for i in range(self.q)], phase)
        rest = np.asarray(self.lattice.w, dtype=np.float64)
        return jnp.where(inactive, rest, jnp.stack(columns, axis=-1))

    def _dense_m(self, P_l, P_g, flags, walls, open_bcs):
        """
        Dense part of a time step of a local shard, for direction-major populations stored as one plane per lattice direction (post-collision
        state): pull streaming, collision, full-way bounce-back and the reset of the inactive cells of each lattice, fused into one pass over
        each lattice, then the open boundaries (non-equilibrium extrapolation) recomputed row by row. Also marks the band of interface cells
        and their neighbors that _band_m recomputes.

        Parameters
        ----------
        P_l, P_g (tuple of jax.numpy.ndarray): Post-collision populations of the liquid and gas lattices, one plane per direction.

        flags (jax.numpy.ndarray): Cell types, shape (..., 1).

        walls (jax.numpy.ndarray): Full-way bounce-back nodes of the liquid and gas lattices, shape (..., 2).

        open_bcs (tuple): Local data of the open boundaries of each lattice that has some (see _create_open_boundary_data).

        Returns
        -------
        (tuple): New planes of both lattices, the band (..., 1; 1 for interface cells, 2 for their gas and liquid neighbors), and the largest
        numbers of interface and neighbor cells over all shards.
        """
        dtype = self.precision_policy.compute_dtype
        rest = np.asarray(self.lattice.w, dtype=np.float64)
        # The dense pass reads its own (barrier) copy of the cell types: sharing them with the band-size reduction below, XLA fuses both into
        # one kernel emitted as a far slower reduction
        cell, walls = jax.lax.optimization_barrier((flags[..., 0], walls))
        lattices = []
        for phase, planes in enumerate((P_l, P_g)):
            planes = [p.astype(dtype) for p in planes]
            left, right = self._halo_slabs(planes)
            streamed = self._pull_planes(planes, left, right)
            post = self._collide_planes(streamed, phase)
            inactive = cell == (GAS if phase == 0 else LIQUID)
            wall = walls[..., phase] > 0 if self._has_bounceback[phase] else None
            post = [jnp.where(inactive, rest[i], p if wall is None else jnp.where(wall, streamed[self._opposite[i]], p)) for i, p in enumerate(post)]
            # One kernel per lattice (the next lattice reads the cell types through the barrier, so it is not fused as a sibling): fused
            # together, the 2 q outputs exceed the registers and spill
            post, cell, walls = jax.lax.optimization_barrier((post, cell, walls))
            lattices.append({"planes": planes, "left": left, "right": right, "new": post})

        for phase, data in zip(self._open_bc_phases, open_bcs, strict=True):
            lattices[phase]["new"] = self._open_boundary_rows(lattices[phase], phase, cell, *data)

        interface = flags == INTERFACE
        near = self._stencil_sum(interface.astype(dtype), [1.0 if any(d) else 0.0 for d in self._lattice_directions]) > 0.0
        band = jnp.where(interface, 1, jnp.where(near & (flags != SOLID), 2, 0)).astype(jnp.int8)
        count = jax.lax.pmax(jnp.stack((jnp.sum(band == 1, dtype=jnp.int32), jnp.sum(band == 2, dtype=jnp.int32))), "x")
        cast = self.precision_policy.cast_to_output
        return tuple(cast(p) for p in lattices[0]["new"]), tuple(cast(p) for p in lattices[1]["new"]), band, count

    def _band_m(self, P_l, P_g, new_l, new_g, mass, flags, eps, *rest, capacity):
        """
        Band part of a time step of a local shard (see _band_branch), writing the recomputed band rows over the dense result in place.

        Parameters
        ----------
        P_l, P_g (tuple of jax.numpy.ndarray): Post-collision populations of the previous step, one plane per direction.

        new_l, new_g (tuple of jax.numpy.ndarray): Dense result of _dense_m (donated, updated in place).

        mass, flags, eps (jax.numpy.ndarray): Liquid mass, cell types and volume fraction, shape (..., 1).

        rest: The droplet labels (..., 1) with zero_net_capillary_force, then geometry (tuple of the normal components from interface_geometry,
            each of shape (..., 1)), band (band mask from _dense_m, shape (..., 1)) and flux (evaporation flux of this step, scalar).

        capacity (tuple of int): Static numbers of interface and neighbor rows, at least the numbers of every shard.

        Returns
        -------
        (tuple): Updated planes of both lattices, liquid mass, cell types and volume fraction (and droplet labels).
        """
        label, shift, (geometry, band, flux) = (rest[0], rest[1], rest[2:]) if self.zero_net_capillary_force else (None, None, rest)
        dtype = self.precision_policy.compute_dtype
        lattices = []
        for planes, new in ((P_l, new_l), (P_g, new_g)):
            planes = [p.astype(dtype) for p in planes]
            left, right = self._halo_slabs(planes)
            lattices.append({"planes": planes, "left": left, "right": right, "new": list(new)})
        out = self._band_branch(capacity, lattices, mass, flags, eps, geometry, flux, band, label, shift)
        return (tuple(out[0]), tuple(out[1])) + tuple(out[2:])

    def _open_boundary_rows(self, lattice, phase, cell, cells, neighbors, missing, prescribed):
        """
        Non-equilibrium extrapolation (open boundary) rows of one lattice: the missing streamed populations take the equilibrium at the
        prescribed density and the neighbor's velocity plus the neighbor's non-equilibrium part, then the rows collide and overwrite the
        dense result.
        """
        cells, neighbors, missing, prescribed = cells[0], neighbors[0], missing[0], prescribed[0]
        valid = cells[:, 0] < cell.shape[0]
        coords = tuple(jnp.where(valid, cells[:, axis], 1 << 30) for axis in range(self.dim))
        safe = tuple(jnp.where(valid, cells[:, axis], 0) for axis in range(self.dim))
        neighbor = tuple(jnp.where(valid, neighbors[:, axis], 0) for axis in range(self.dim))
        boundary = self._pull_rows(lattice["planes"], lattice["left"], lattice["right"], safe)
        inner = self._pull_rows(lattice["planes"], lattice["left"], lattice["right"], neighbor)
        rho, u = self._moments_q(inner)
        rows = jnp.where(missing, self._equilibrium_q(prescribed.astype(rho.dtype), u) + inner - self._equilibrium_q(rho, u), boundary)
        inactive = (cell[safe] == (GAS if phase == 0 else LIQUID))[:, None]
        return self._scatter_rows(lattice["new"], coords, self._collide_rows(rows, phase, inactive))

    def _band_branch(self, capacity, lattices, mass, flags, eps, geometry, flux, band, label=None, shift=None):
        """
        Interface update of the band: mass exchange, reconstruction of the missing populations, evaporation, cell type conversion with excess
        mass redistribution, initialization of the cells that newly carry a phase, the shared interface velocity, and the collision of the
        recomputed rows, which overwrite the dense result.

        The missing populations are f_i = feq_i(rho, u_I) + feq_opp(i)(rho, u_I) - f_opp(i) + 2 g_opp(i), i.e. the free-surface reconstruction
        (Koerner et al. 2005) plus twice the post-collision non-equilibrium part g of the cell, so that each lattice sees its own viscous stress
        continued across the interface link instead of a stress-free boundary. With consistent_link_stress, the populations are rebuilt a
        second time with g = (1 - s_nu) fneq of the first reconstruction of this step, so that exactly the momentum the links carry is
        returned below. The momentum these links exchange with each lattice (and the
        body force applied beyond the volume-fraction weighted one) is returned to the shared interface velocity (paper Eq. 13), which hands
        the viscous stress of each phase to the other one and conserves the total momentum.

        The band has two kinds of rows (band == 1: interface cells, band == 2: their gas and liquid neighbors), compacted separately into
        capacity = (interface rows, neighbor rows). Only interface rows read, update and write populations; neighbor rows only take part in the
        cell type conversion, and the few of them that become interface cells are initialized, collided and written. Populations are handled
        as q columns of rows (no layout transposes). A map from cells to rows gives the band neighbors of every row, so the values exchanged
        between neighbors (filled and emptied cells, excess mass, moments) are gathers over the rows only; on several devices, the neighbors
        across the x boundary of the shard come from one-plane slabs exchanged with the next device.
        """
        dtype = self.precision_policy.compute_dtype
        spatial = mass.shape[:-1]
        size = int(np.prod(spatial))
        nx_local = spatial[0]
        r = self.density_ratio
        q, dim = self.q, self.dim
        directions, weights, opposite = self._lattice_directions, self._lattice_weights, self._opposite
        neighbors_i = self._neighbor_indices
        half_a_l = [0.5 * a for a in (self._acceleration[0] or (0.0,) * dim)]
        half_a_g = [0.5 * a for a in (self._acceleration[1] or (0.0,) * dim)]
        multi = self.n_devices > 1
        n_int, n_nbr = capacity
        total = n_int + n_nbr

        index, row_map = self._compact_band(band.reshape(-1), n_int, n_nbr)
        row_map = row_map.reshape(spatial)
        valid_row = index < size
        coords = jnp.unravel_index(jnp.minimum(index, size - 1), spatial)
        drop = tuple(jnp.where(valid_row, axis, 1 << 30) for axis in coords)
        coords_int, drop_int = tuple(c[:n_int] for c in coords), tuple(d[:n_int] for d in drop)

        # Source cell x + offset of given rows: coordinates (wrapped in y/z, and in x on a single device) and whether it lies in the shard
        def source(offset, at):
            sx = at[0] + offset[0]
            rest = tuple((at[axis] + offset[axis]) % spatial[axis] for axis in range(1, dim))
            if not multi:
                return (sx % nx_local,) + rest, None
            return (jnp.clip(sx, 0, nx_local - 1),) + rest, (sx < 0, sx >= nx_local, rest)

        def field_at(field, place, halos=None):
            # Values of a dense (..., 1) field at the source cells of one direction, with x-halo planes on several devices
            at, outside = place
            value = field[at][:, 0]
            if outside is None:
                return value
            below, above, rest = outside
            zero = jnp.zeros_like(rest[0])
            return jnp.where(below, halos[0][(zero,) + rest][:, 0], jnp.where(above, halos[1][(zero,) + rest][:, 0], value))

        def x_halos(field):
            return self._x_halos(field) if multi else None

        upstream = [source(tuple(-c for c in d), coords) for d in directions]
        upstream_int = [source(tuple(-c for c in d), coords_int) for d in directions]
        downstream_int = [source(d, coords_int) for d in directions]

        source_rows = [jnp.where(valid_row, row_map[upstream[i][0]], total) for i in range(q)]
        if multi:
            for i in range(q):
                below, above, _ = upstream[i][1]
                source_rows[i] = jnp.where(below | above, total, source_rows[i])

        def from_band_neighbors(values):
            # Values of per-row arrays (total,) at the upstream neighbor of every row (0 outside the band)
            extended = jnp.concatenate((values, jnp.zeros((1,), values.dtype)))
            out = [extended[source_rows[i]] for i in range(q)]
            if multi:
                # Neighbors across the x boundary of the shard: one-plane slabs of the band values exchanged with the next devices
                plane = spatial[1:]
                rest = tuple(coords[axis] for axis in range(1, dim))
                first = jnp.zeros(plane, values.dtype).at[rest].add(jnp.where(valid_row & (coords[0] == 0), values, 0))
                last = jnp.zeros(plane, values.dtype).at[rest].add(jnp.where(valid_row & (coords[0] == nx_local - 1), values, 0))
                from_left, from_right = self.send_right(last[None], "x")[0], self.send_left(first[None], "x")[0]
                for i in range(q):
                    below, above, rest_i = upstream[i][1]
                    out[i] = jnp.where(below, from_left[rest_i], jnp.where(above, from_right[rest_i], out[i]))
            return out

        def rows(field):
            return field.reshape(size, -1).at[index].get(mode="fill", fill_value=0)[:, 0]

        def scatter(field, values):
            return field.reshape(size, -1).at[index].set(values.astype(field.dtype)[:, None], mode="drop").reshape(field.shape)

        def pad_rows(values, fill=0.0):
            # Per-interface-row values extended to all rows
            return jnp.concatenate((values, jnp.full((n_nbr,), fill, values.dtype)))

        # Scalars of all rows; the type of the cell each population streamed from (x - c_i)
        m, e, cell = rows(mass), rows(eps), rows(flags)
        flag_halos = x_halos(flags.astype(dtype))
        source_flags = [field_at(flags.astype(dtype), upstream[i], flag_halos) for i in range(q)]
        from_interface = {i: source_flags[i] == INTERFACE for i in neighbors_i}

        # Interface rows: populations, neighbor volume fractions and the curvature kappa = div(n) = sum_i (w_i / cs2) c_i . n(x + c_i)
        liquid, gas = lattices

        def own(planes):
            return [plane.at[coords_int].get(mode="fill", fill_value=0) for plane in planes]

        def streamed(lattice):
            return [
                field_at(plane[..., None], upstream_int[i], (lattice["left"][i][..., None], lattice["right"][i][..., None]))
                for i, plane in enumerate(lattice["planes"])
            ]

        fp_l, fp_g, fs_l, fs_g = own(liquid["planes"]), own(gas["planes"]), streamed(liquid), streamed(gas)
        m_i, e_i, cell_i = m[:n_int], e[:n_int], cell[:n_int]
        flags_i = [f[:n_int] for f in source_flags]
        eps_halos = x_halos(eps)
        source_eps = [field_at(eps, upstream_int[i], eps_halos) for i in range(q)]
        interface = cell_i == INTERFACE
        from_gas = {i: flags_i[i] == GAS for i in neighbors_i}
        from_interface_i = {i: flags_i[i] == INTERFACE for i in neighbors_i}
        from_liquid = {i: flags_i[i] == LIQUID for i in neighbors_i}
        normal_halos = [x_halos(g) for g in geometry]
        normal = [g.reshape(size).at[index[:n_int]].get(mode="fill", fill_value=0) for g in geometry]
        curvature = 0.0
        for i in neighbors_i:
            n_at = [field_at(g, downstream_int[i], h) for g, h in zip(geometry, normal_halos, strict=True)]
            curvature = curvature + (weights[i] / self.lattice.cs2) * _lattice_dot(directions[i], n_at)

        # Interface velocity before the last collision (shared by both lattices): the post-collision momentum minus the full body force
        rho_in_l, u_post_l = self._macroscopic_planes(fp_l)
        rho_in_g, _ = self._macroscopic_planes(fp_g)
        u_liquid = [v - 2.0 * a for v, a in zip(u_post_l, half_a_l, strict=True)]
        u_interface = [v + a for v, a in zip(u_liquid, half_a_l, strict=True)]
        u_gas = [v - a for v, a in zip(u_interface, half_a_g, strict=True)]
        if self.zero_net_capillary_force:
            label_i = rows(label)[:n_int]
            active_i = interface & (label_i > 0)
            if self.capillary_curvature_degree:
                curvature, shift = self._fitted_curvature(curvature, active_i, from_gas, label_i, normal, shift)
            curvature = self._balanced_curvature(curvature, active_i, from_gas, label_i, normal)
        rho_boundary = 1.0 + r * (rho_in_g - 1.0) + self.surface_tension * curvature / self.lattice.cs2

        # Mass exchange of interface cells: incoming minus outgoing population of every link to a liquid or interface cell
        delta = 0.0
        for i in neighbors_i:
            weight = jnp.where(from_liquid[i], 1.0, jnp.where(from_interface_i[i], 0.5 * (e_i + source_eps[i]), 0.0))
            delta = delta + weight * (fs_l[i] - fp_l[opposite[i]])
        m_i = jnp.where(interface, m_i + delta, m_i)

        # Missing populations with the capillary-corrected liquid density or the local gas density, moving with the interface velocity
        g_l = [f - feq for f, feq in zip(fp_l, self._equilibrium_planes(*self._macroscopic_planes(fp_l)), strict=True)]
        g_g = [f - feq for f, feq in zip(fp_g, self._equilibrium_planes(*self._macroscopic_planes(fp_g)), strict=True)]
        missing_l = {i: interface & from_gas[i] for i in neighbors_i}
        missing_g = {i: interface & from_liquid[i] for i in neighbors_i}

        streamed_l, streamed_g = fs_l, fs_g

        def reconstruct(rho_b, stress):
            feq = self._equilibrium_planes(rho_b, u_liquid)
            return [
                f if i not in missing_l else jnp.where(missing_l[i], feq[i] + feq[opposite[i]] - fp_l[opposite[i]] + 2.0 * stress[opposite[i]], f)
                for i, f in enumerate(streamed_l)
            ]

        def reconstruct_gas(stress):
            return [
                f if i not in missing_g else jnp.where(missing_g[i], feq_g[i] + feq_g[opposite[i]] - fp_g[opposite[i]] + 2.0 * stress[opposite[i]], f)
                for i, f in enumerate(streamed_g)
            ]

        rec_l = reconstruct(rho_boundary, g_l)
        if self.viscous_stress_jump:
            # Normal strain rates of the streamed (pre-collision) populations, the liquid ones after a first reconstruction
            s_l, s_g = self._shear_relaxation()
            nu_l, nu_g = self.kinematic_viscosity()
            strain_l = self._normal_strain_planes(rec_l, normal, s_l)
            strain_g = self._normal_strain_planes(fs_g, normal, s_g)
            rho_boundary = rho_boundary + 2.0 * (rho_in_l * nu_l * strain_l - r * rho_in_g * nu_g * strain_g) / self.lattice.cs2
            rec_l = reconstruct(rho_boundary, g_l)
        fs_l = rec_l
        feq_g = self._equilibrium_planes(rho_in_g, u_gas)
        fs_g = reconstruct_gas(g_g)
        # Momentum given to each lattice by its interface links (viscous part only; the equilibrium part carries the interface pressure) and
        # by the body force beyond eps rho_l a_l + (1 - eps) r rho_g a_g, both forcing the whole interface cell. The viscous part is 2 g_opp(i)
        # per link, with g taken as the post-collision value (1 - s_nu) fneq of the reconstructed populations of this step rather than the g of
        # the last collision: the same steady value, but without the one-step lag, which feeds back with a factor 1 - s_nu close to -1 at low
        # viscosity and makes the interface unstable
        s_l, s_g = self._shear_relaxation()
        h_l = [(1.0 - s_l) * (f - feq) for f, feq in zip(fs_l, self._equilibrium_planes(*self._macroscopic_planes(fs_l)), strict=True)]
        h_g = [(1.0 - s_g) * (f - feq) for f, feq in zip(fs_g, self._equilibrium_planes(*self._macroscopic_planes(fs_g)), strict=True)]
        if self.consistent_link_stress:
            # The missing populations are rebuilt with this h, so they carry exactly the viscous part 2 h that transfer returns below: the
            # viscous momentum the links give each lattice is returned in full. With 2 g (the last collision) the difference 2 (g - h), zero
            # only in steady state, stays in the liquid every step: moving or oscillating droplets with surface tension gain or lose momentum
            # (25 % in 1e4 steps for a 40-cell droplet carried by the gas). The rebuild with h feeds back with 1 - s_nu, so it is unstable
            # where 1 - s_nu approaches -1 (low viscosity)
            fs_l, fs_g = reconstruct(rho_boundary, h_l), reconstruct_gas(h_g)
        transfer = [0.0] * dim
        for i in neighbors_i:
            j = opposite[i]
            link = jnp.where(missing_l[i], 2.0 * h_l[j], 0.0) + r * jnp.where(missing_g[i], 2.0 * h_g[j], 0.0)
            for axis, component in enumerate(directions[i]):
                if component != 0:
                    transfer[axis] = transfer[axis] + component * link
        transfer = [
            jnp.where(interface, t + (1.0 - e_i) * rho_in_l * (2.0 * al) + e_i * r * rho_in_g * (2.0 * ag), 0.0)
            for t, al, ag in zip(transfer, half_a_l, half_a_g, strict=True)
        ]

        if self._has_evaporation:
            # Discrete interfacial area of a cell (paper Eq. 22), normalized so that a planar interface has unit area per cell column, from
            # |grad(eps)| (solid neighbors read as the cell itself)
            gradient = [0.0] * dim
            for i in neighbors_i:
                neighbor = jnp.where(flags_i[i] == SOLID, e_i, source_eps[i])
                for axis, component in enumerate(directions[i]):
                    if component != 0:
                        gradient[axis] = gradient[axis] - (weights[i] / self.lattice.cs2 * component) * neighbor
            area = 2.0 * jnp.sqrt(sum(g * g for g in gradient))
            mdot = jnp.where(interface, jnp.minimum(flux * area, jnp.maximum(m_i, 0.0)), 0.0)
            # Source with zeroth moment mdot and first moment mdot u_I (paper Eqs. 18-19), converted to gas units by 1 / r
            evaporation = [mdot * wi * (1.0 + 3.0 * _lattice_dot(d, u_interface)) for d, wi in zip(directions, weights, strict=True)]
            fs_l = [f - s for f, s in zip(fs_l, evaporation, strict=True)]
            fs_g = [f + s / r for f, s in zip(fs_g, evaporation, strict=True)]
            m_i = m_i - mdot

        # Cell type conversion (paper Eq. 14, with the hysteresis of Thuerey 2007 and closure of isolated interface cells)
        rho_l_i = sum(fs_l)
        eps_new = m_i / rho_l_i
        has_gas = reduce(jnp.logical_or, (from_gas[i] for i in neighbors_i))
        has_liquid = reduce(jnp.logical_or, (from_liquid[i] for i in neighbors_i))
        fill_i = interface & ((eps_new > 1.0 + self.alpha) | ~has_gas)
        empty_i = interface & ~fill_i & ((eps_new < -self.alpha) | ~has_liquid)
        # Gas neighbors of filled cells become interface cells, so a cell next to a filled one must not empty in the same step
        near_fill = reduce(jnp.logical_or, (from_band_neighbors(pad_rows(fill_i.astype(dtype)))[i] > 0.0 for i in neighbors_i))
        empty = pad_rows(empty_i, False) & ~near_fill
        near_empty = reduce(jnp.logical_or, (from_band_neighbors(empty.astype(dtype))[i] > 0.0 for i in neighbors_i))
        fill = pad_rows(fill_i, False)
        new_from_gas = (cell == GAS) & near_fill
        new_from_liquid = (cell == LIQUID) & near_empty
        cell_new = jnp.where(fill, LIQUID, jnp.where(empty, GAS, jnp.where(new_from_gas | new_from_liquid, INTERFACE, cell))).astype(cell.dtype)
        if self.zero_net_capillary_force:
            # Droplet labels follow the interface: every band row that holds liquid takes the smallest label of itself and its labeled band
            # neighbors (cells that become interface inherit it, merged droplets take the smaller one over the following steps), and rows
            # that become gas lose theirs
            none = float(self.max_droplets + 1)
            own = rows(label).astype(dtype)
            own = jnp.where(own > 0.0, own, none)
            smallest = own
            for value in (from_band_neighbors(own)[i] for i in neighbors_i):
                smallest = jnp.minimum(smallest, jnp.where(value > 0.0, value, none))
            label_new = jnp.where((cell_new == GAS) | (smallest >= none), 0.0, smallest)

        # Liquid density of every row: from the recomputed populations (interface rows) or the dense result (neighbor rows, which are not
        # recomputed; only needed when a liquid cell becomes an interface cell)
        rho_nbr = sum(plane.at[tuple(c[n_int:] for c in coords)].get(mode="fill", fill_value=1) for plane in liquid["new"])
        rho_l = jnp.concatenate((rho_l_i, rho_nbr.astype(dtype)))
        m = jnp.concatenate((m_i, m[n_int:]))
        excess = jnp.where(fill, m - rho_l, jnp.where(empty, m, 0.0))
        m = jnp.where((cell_new == LIQUID) | new_from_liquid, rho_l, jnp.where((cell_new == GAS) | new_from_gas, 0.0, m))
        receiver = cell_new == INTERFACE
        count = sum(from_band_neighbors(receiver.astype(dtype))[i] for i in neighbors_i)
        share = jnp.where(count > 0.0, excess / jnp.maximum(count, 1.0), 0.0)
        received = from_band_neighbors(share)
        m = jnp.where(receiver, m + sum(received[i] for i in neighbors_i), m)

        # New interface cells (neighbor rows) take the equilibrium of the mean density and velocity of the neighbors that were interface cells
        rho_li, u_li = self._macroscopic_planes(fs_l)
        rho_gi, u_gi = self._macroscopic_planes(fs_g)
        sources_count = sum(from_interface[i].astype(dtype) for i in neighbors_i)[n_int:]
        inv_sources = 1.0 / jnp.maximum(sources_count, 1.0)
        has_sources = sources_count > 0.0

        def neighbor_mean(values, default):
            gathered = from_band_neighbors(pad_rows(values))
            total_ = sum(jnp.where(from_interface[i], gathered[i], 0.0) for i in neighbors_i)[n_int:]
            return jnp.where(has_sources, total_ * inv_sources, default)

        rho_mean_l, rho_mean_g = neighbor_mean(rho_li, 1.0), neighbor_mean(rho_gi, 1.0)
        u_mean_l, u_mean_g = [neighbor_mean(v, 0.0) for v in u_li], [neighbor_mean(v, 0.0) for v in u_gi]
        new_nbr_l, new_nbr_g = new_from_gas[n_int:], new_from_liquid[n_int:]
        init_l = self._collide_planes(self._equilibrium_planes(rho_mean_l, u_mean_l), 0)
        init_g = self._collide_planes(self._equilibrium_planes(rho_mean_g, u_mean_g), 1)

        # Shared interface velocity (paper Eq. 13) of the interface rows, imposed on the momentum moment only before the rows collide
        receiver_i = receiver[:n_int]
        rho_li = sum(fs_l)
        rho_gi = sum(fs_g)
        u_l = [v + a for v, a in zip(u_li, half_a_l, strict=True)]
        u_g = [v + a for v, a in zip(u_gi, half_a_g, strict=True)]
        u_shared = [(rho_li * vl + r * rho_gi * vg - t) / (rho_li + r * rho_gi) for vl, vg, t in zip(u_l, u_g, transfer, strict=True)]
        fs_l = self._shift_velocity_planes(fs_l, rho_li, [jnp.where(receiver_i, s - v, 0.0) for s, v in zip(u_shared, u_l, strict=True)])
        fs_g = self._shift_velocity_planes(fs_g, rho_gi, [jnp.where(receiver_i, s - v, 0.0) for s, v in zip(u_shared, u_g, strict=True)])
        rho_l = jnp.concatenate((rho_li, jnp.where(new_nbr_l, rho_mean_l, rho_l[n_int:])))
        e = jnp.where(cell_new == LIQUID, 1.0, jnp.where(receiver, jnp.clip(m / rho_l, 0.0, 1.0), 0.0))

        rest = self._lattice_weights
        cell_new_i = cell_new[:n_int]
        post_l = [jnp.where(cell_new_i == GAS, rest[i], p) for i, p in enumerate(self._collide_planes(fs_l, 0))]
        post_g = [jnp.where(cell_new_i == LIQUID, rest[i], p) for i, p in enumerate(self._collide_planes(fs_g, 1))]
        drop_l = tuple(jnp.where(new_nbr_l, d[n_int:], 1 << 30) for d in drop)
        drop_g = tuple(jnp.where(new_nbr_g, d[n_int:], 1 << 30) for d in drop)
        new_l = [
            p.at[drop_int].set(v.astype(p.dtype), mode="drop").at[drop_l].set(n.astype(p.dtype), mode="drop")
            for p, v, n in zip(liquid["new"], post_l, init_l, strict=True)
        ]
        new_g = [
            p.at[drop_int].set(v.astype(p.dtype), mode="drop").at[drop_g].set(n.astype(p.dtype), mode="drop")
            for p, v, n in zip(gas["new"], post_g, init_g, strict=True)
        ]
        out = (new_l, new_g, scatter(mass, m), scatter(flags, cell_new), scatter(eps, e))
        if self.zero_net_capillary_force:
            out = out + (scatter(label, label_new), shift)
        return out

    def _link_area(self, active, from_gas, dtype):
        """Gas-link area sum_{i: x - c_i gas} 2 w_i of the active interface rows (0 elsewhere), shape (n,)."""
        area = sum(jnp.where(from_gas[i], 2.0 * self._lattice_weights[i], 0.0) for i in self._neighbor_indices)
        return jnp.where(active, area, 0.0).astype(dtype)

    def _droplet_sums(self, segment, values):
        """
        Per-droplet sums of the rows of values (n, m) by droplet segment (n,), shape (max_droplets + 1, m), as a one-hot matrix product:
        nearly all rows belong to one or two labels, so a scatter-add (segment_sum) serializes on atomics, while the product runs as a dense
        matrix multiply (in full float32 precision).
        """
        onehot = (segment[:, None] == jnp.arange(self.max_droplets + 1, dtype=segment.dtype)[None, :]).astype(values.dtype)
        return jnp.matmul(onehot.T, values, precision=jax.lax.Precision.HIGHEST)

    def _fitted_curvature(self, curvature, active, from_gas, label, normal, shift):
        """
        Low-order curvature of every droplet (capillary_curvature_degree), see capillary_curvature_degree and capillary_shift_relaxation.

        kappa is fitted per droplet by sum_k c_k phi_k(n), with phi_k the monomials of degree d and d - 1 of the unit normal, in least
        squares weighted by the gas-link area a of the rows (reduced over all shards). The fit is then shifted by the droplet's relaxed shift,
        which follows the projected-area (P = -S . n, the cell's surface element) mean of kappa - fit.

        Parameters
        ----------
        curvature, active, label (jax.numpy.ndarray): Curvature, active interface rows and droplet labels, shape (n,).

        from_gas (dict): Per neighbor direction, whether the row's upstream neighbor is a gas cell.

        normal (list of jax.numpy.ndarray): Interface normal components of the rows, each of shape (n,).

        shift (jax.numpy.ndarray): Relaxed shift of every droplet label, shape (max_droplets + 1,).

        Returns
        -------
        (jax.numpy.ndarray, jax.numpy.ndarray): Curvature (n,) and updated shifts.
        """
        dtype = curvature.dtype
        # Unit normals; on the unit sphere the monomials of degree d and d - 1 span the harmonics up to l = d without redundancy
        # (2d + 1 terms in 2D, (d + 1)^2 in 3D). All monomials up to d are linearly dependent there (e.g. 1 = |n|^2), which makes
        # the least-squares matrix singular and the solve backend-dependent (NaN with LAPACK on CPU).
        norm = jnp.sqrt(sum(c.astype(dtype) ** 2 for c in normal))
        n = [jnp.where(active & (norm > 0), c.astype(dtype) / jnp.where(norm > 0, norm, 1.0), 0.0) for c in normal]
        degree = self.capillary_curvature_degree
        basis = []
        for powers in itertools.product(range(degree + 1), repeat=self.dim):
            if sum(powers) in (degree, degree - 1):
                term = jnp.ones_like(curvature)
                for component, k in zip(n, powers, strict=True):
                    if k:
                        term = term * component**k
                basis.append(term)
        basis = jnp.stack(basis, axis=-1)
        k = basis.shape[-1]
        area = self._link_area(active, from_gas, dtype)
        segments = self.max_droplets + 1
        segment = jnp.where(active, label, 0).astype(jnp.int32)
        weighted = area[:, None] * basis
        sums = self._droplet_sums(
            segment, jnp.concatenate([(weighted[:, :, None] * basis[:, None, :]).reshape(-1, k * k), weighted * curvature[:, None]], axis=-1)
        )
        matrix, rhs = sums[:, : k * k].reshape(segments, k, k), sums[:, k * k :]
        matrix, rhs = jax.lax.psum((matrix, rhs), "x")
        ridge = 1.0e-6 * jnp.trace(matrix, axis1=1, axis2=2) / k + 1.0e-30
        coefficients = jnp.linalg.solve(matrix + ridge[:, None, None] * jnp.eye(k, dtype=dtype), rhs[..., None])[..., 0]
        fitted = jnp.sum(coefficients[segment] * basis, axis=-1)
        link = [
            sum(
                jnp.where(from_gas[i], 2.0 * self._lattice_weights[i] * self._lattice_directions[i][axis], 0.0)
                for i in self._neighbor_indices
                if self._lattice_directions[i][axis]
            )
            for axis in range(self.dim)
        ]
        projected = jnp.where(active, jnp.maximum(-sum(l_ * c for l_, c in zip(link, n, strict=True)), 0.0), 0.0).astype(dtype)
        sums = jax.lax.psum(self._droplet_sums(segment, jnp.stack([projected * (curvature - fitted), projected], axis=-1)), "x")
        instantaneous = sums[:, 0] / jnp.maximum(sums[:, 1], 1.0e-30)
        shift = shift + self.capillary_shift_relaxation * (instantaneous - shift)
        return jnp.where(active, fitted + shift[segment], curvature), shift

    def _balanced_curvature(self, curvature, active, from_gas, label, normal):
        """
        Curvature of the interface rows corrected so that every droplet feels no net capillary force (zero_net_capillary_force).

        The capillary pressure enters the liquid through the reconstructed populations of the links from gas cells, feq_i + feq_opp(i) =
        2 w_i rho_b (to O(u^2)), each carrying the momentum c_i. The net force on a droplet is therefore (sigma / cs2) sum_x kappa(x) S(x), with
        S(x) = sum_{i: x - c_i gas} 2 w_i c_i the link vector of an interface cell (the uniform part of rho_b cancels exactly over a closed
        surface: every lattice direction has as many links into the droplet as its opposite). It is cancelled per droplet L by the dipole
        correction kappa -> kappa - lambda_L . n, with n the (smooth) outward interface normal: A_L lambda_L = b_L, A_L = sum S n^T and b_L =
        sum kappa S over the droplet's interface rows, reduced over all shards (solved in least squares, A^T A lambda = A^T b, with a tiny
        relative ridge). The correction is a smooth l = 1 pressure mode over the droplet: correcting along S instead, which jumps from cell to
        cell with the staircase of the interface, adds a lattice-locked cell-scale pressure pattern that slowly excites a growing interface mode
        (a static droplet goes unstable after ~2e4 steps).

        Parameters
        ----------
        curvature (jax.numpy.ndarray): Curvature of the interface rows, shape (n,).

        active (jax.numpy.ndarray): Interface rows with a droplet label, shape (n,).

        from_gas (dict): Per neighbor direction, whether the population of the row streams from a gas cell, shape (n,).

        label (jax.numpy.ndarray): Droplet label of the rows, shape (n,).

        normal (list of jax.numpy.ndarray): Outward interface normal components of the rows, each of shape (n,).

        Returns
        -------
        (jax.numpy.ndarray): Corrected curvature, shape (n,).
        """
        dim = self.dim
        dtype = curvature.dtype
        link = [
            sum(
                jnp.where(from_gas[i], (2.0 * self._lattice_weights[i] * self._lattice_directions[i][axis]), 0.0)
                for i in self._neighbor_indices
                if self._lattice_directions[i][axis] != 0
            )
            for axis in range(dim)
        ]
        S = jnp.where(active[:, None], jnp.stack(link, axis=-1).astype(dtype), 0.0)
        segments = self.max_droplets + 1
        segment = jnp.where(active, label, 0).astype(jnp.int32)
        n = jnp.where(active[:, None], jnp.stack(normal, axis=-1).astype(dtype), 0.0)
        sums = self._droplet_sums(segment, jnp.concatenate([curvature[:, None] * S, (S[:, :, None] * n[:, None, :]).reshape(-1, dim * dim)], axis=-1))
        b, A = sums[:, :dim], sums[:, dim:].reshape(segments, dim, dim)
        b, A = jax.lax.psum((b, A), "x")
        At = jnp.swapaxes(A, 1, 2)
        normal_matrix = At @ A
        ridge = 1.0e-10 * jnp.trace(normal_matrix, axis1=1, axis2=2) / dim + 1.0e-30
        lam = jnp.linalg.solve(normal_matrix + ridge[:, None, None] * jnp.eye(dim, dtype=dtype), (At @ b[..., None]))[..., 0]
        return curvature - jnp.sum(lam[segment] * n, axis=-1)

    def _compact_band(self, band, n_int, n_nbr, tile=1024):
        """
        Compaction of the band (flat, 1: interface rows, 2: neighbor rows): the cell index of every row, interface rows first and padded to
        n_int, then neighbor rows padded to n_nbr (padding rows hold band.size), and the row of every cell (n_int + n_nbr outside the band).
        Both running counts come from one scan within tiles plus a scan over the tile totals.
        """
        size = band.size
        tiles = -(-size // tile)
        kinds = jnp.stack(((band == 1).astype(jnp.int32), (band == 2).astype(jnp.int32)))
        flat = jnp.pad(kinds, ((0, 0), (0, tiles * tile - size))).reshape(2, tiles, tile)
        within = jax.lax.associative_scan(jnp.add, flat, axis=2)
        totals = within[:, :, -1]
        # Materialized: fused into its consumers, XLA recomputes the scan levels for every one of them
        inclusive = jax.lax.optimization_barrier((within + (jnp.cumsum(totals, axis=1) - totals)[:, :, None]).reshape(2, -1)[:, :size])
        position = inclusive - kinds
        total = n_int + n_nbr
        rows = jnp.where(band == 1, position[0], jnp.where(band == 2, n_int + position[1], total))
        rows = jnp.where((band == 1) & (position[0] >= n_int) | (band == 2) & (position[1] >= n_nbr), total, rows)
        # Cell of every row by binary search in the running counts (a scatter would visit every cell of the shard)
        index = jnp.concatenate((
            jnp.searchsorted(inclusive[0], jnp.arange(1, n_int + 1, dtype=jnp.int32), side="left", method="scan_unrolled"),
            jnp.searchsorted(inclusive[1], jnp.arange(1, n_nbr + 1, dtype=jnp.int32), side="left", method="scan_unrolled"),
        )).astype(jnp.int32)
        return index, rows

    @staticmethod
    def _compact(mask, capacity, tile=1024):
        """
        Indices of the set entries of a flat mask, in order, padded to capacity with mask.size, and the row of every entry (capacity where
        unset). The running count is a scan within tiles plus a scan over the tile totals (jnp.nonzero and jnp.cumsum lower to a far slower
        reduce-window over the whole array).
        """
        size = mask.size
        tiles = -(-size // tile)
        flat = jnp.pad(mask.astype(jnp.int32), (0, tiles * tile - size)).reshape(tiles, tile)
        within = jax.lax.associative_scan(jnp.add, flat, axis=1)
        totals = within[:, -1]
        inclusive = (within + (jnp.cumsum(totals) - totals)[:, None]).reshape(-1)[:size]
        rows = jnp.where(mask, inclusive - 1, capacity)
        # Cell of every row by binary search in the running count (a scatter would visit every entry of the mask)
        index = jnp.searchsorted(inclusive, jnp.arange(1, capacity + 1, dtype=jnp.int32), side="left", method="scan_unrolled").astype(jnp.int32)
        return index, rows

    def _shift_velocity_planes(self, planes, rho, shift):
        """Direction planes with their velocity shifted by du at fixed density and higher moments: f_i + 3 w_i rho c_i . du."""
        return [f + 3.0 * w * rho * _lattice_dot(d, shift) for f, d, w in zip(planes, self._lattice_directions, self._lattice_weights, strict=True)]

    def _normal_strain_planes(self, planes, normal, s_nu):
        """Normal strain rate n . E . n from the non-equilibrium part of direction planes, E = -(3 s_nu / (2 rho)) sum_i c_i c_i fneq_i."""
        rho, velocity = self._macroscopic_planes(planes)
        feq = self._equilibrium_planes(rho, velocity)
        total = sum((f - fe) * _lattice_dot(d, normal) ** 2 for f, fe, d in zip(planes, feq, self._lattice_directions, strict=True))
        return (-1.5 * s_nu) / rho * total

    def initialize_volume_fraction(self):
        """
        Initial liquid volume fraction, to be overridden by the user.

        Returns
        -------
        (numpy.ndarray): Volume fraction in [0, 1], shape (nx, ny[, nz]) or (nx, ny[, nz], 1).
        """
        raise NotImplementedError("Override initialize_volume_fraction to define the initial liquid distribution.")

    def initialize_macroscopic_fields(self):
        """
        Initial densities and velocities of both lattices, ([rho_l, rho_g], [u_l, u_g]). May be overridden by the user.

        Returns
        -------
        None, None: Both lattices start at rest with unit density.
        """
        return None, None

    def assign_fields_sharded(self):
        """
        Initialize both lattices and the interface state from initialize_volume_fraction and initialize_macroscopic_fields.

        Cells with eps >= 1 - alpha are liquid, eps <= alpha gas, and the rest interface; liquid cells next to gas become full interface cells so
        that the interface layer is closed. Each lattice is set to equilibrium where it is active and to rest elsewhere, with the interface
        velocity shared by both.

        Returns
        -------
        f_tree (list of tuple of jax.numpy.ndarray): Liquid and gas populations, one plane per lattice direction.

        state (dict): Liquid mass ("mass"), cell types ("flags") and volume fraction ("eps"), shape (..., 1), the largest numbers of
        interface and neighbor cells over all shards ("band_count") and, with zero_net_capillary_force, the droplet labels ("label") and their
        relaxed Laplace shifts ("capillary_shift").
        """
        spatial = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        eps = np.clip(np.reshape(np.asarray(self.initialize_volume_fraction(), dtype=np.float64), spatial), 0.0, 1.0)
        solid = self._solid_host
        flags = np.where(eps >= 1.0 - self.alpha, LIQUID, np.where(eps <= self.alpha, GAS, INTERFACE)).astype(np.int8)
        flags[solid] = SOLID
        near_gas = np.zeros(spatial, dtype=bool)
        for d in self._lattice_directions:
            if any(d):
                near_gas |= self._host_neighbor(flags, d) == GAS
        flags[(flags == LIQUID) & near_gas] = INTERFACE
        eps = np.where(flags == LIQUID, 1.0, np.where(flags == INTERFACE, eps, 0.0))

        dtype = self.precision_policy.compute_dtype
        rho_tree, u_tree = self.initialize_macroscopic_fields()
        if rho_tree is None or u_tree is None:
            rho_tree = [np.ones((*spatial, 1)), np.ones((*spatial, 1))]
            u_tree = [np.zeros((*spatial, self.dim)), np.zeros((*spatial, self.dim))]
        if len(rho_tree) != 2 or len(u_tree) != 2:
            raise ValueError("initialize_macroscopic_fields must return [rho_l, rho_g], [u_l, u_g].")

        flags_device = self.distributed_array_init((*spatial, 1), jnp.int8, init_val=flags[..., None])
        eps_device = self.distributed_array_init((*spatial, 1), dtype, init_val=eps[..., None])
        rho_l = self.distributed_array_init((*spatial, 1), dtype, init_val=np.reshape(np.asarray(rho_tree[0], dtype=np.float64), (*spatial, 1)))
        mass = eps_device * rho_l
        plane_sharding = NamedSharding(self.mesh, self._plane_spec())
        f_tree = []
        for phase, (rho, u) in enumerate(zip(rho_tree, u_tree, strict=True)):
            rho = np.reshape(np.asarray(rho, dtype=np.float64), (*spatial, 1))
            u = np.reshape(np.asarray(u, dtype=np.float64), (*spatial, self.dim))
            cu = 3.0 * np.einsum("...a,ai->...i", u, np.asarray(self.c, dtype=np.float64))
            feq = rho * np.asarray(self.lattice.w) * (1.0 + cu * (1.0 + 0.5 * cu) - 1.5 * np.sum(u * u, axis=-1, keepdims=True))
            feq[flags == (GAS if phase == 0 else LIQUID)] = np.asarray(self.lattice.w)
            f_tree.append([feq[..., i] for i in range(self.q)])
        f_tree = self._share_interface_velocity_host(f_tree, flags)
        f_tree = [tuple(jax.device_put(np.asarray(p, dtype=self.precision_policy.output_dtype), plane_sharding) for p in planes) for planes in f_tree]

        interface = flags == INTERFACE
        near = np.zeros_like(interface)
        for d in self._neighbor_indices:
            near |= self._host_neighbor(flags, self._lattice_directions[d]) == INTERFACE
        near &= ~interface & (flags != SOLID)
        count = tuple(int(mask.reshape(self.n_devices, -1).sum(axis=1).max()) for mask in (interface, near))
        state = {"mass": mass, "flags": flags_device, "eps": eps_device, "band_count": count}
        if self.zero_net_capillary_force:
            state["label"] = self._label_droplets(flags_device)
            state["capillary_shift"] = jax.device_put(jnp.zeros(self.max_droplets + 1, dtype), NamedSharding(self.mesh, PartitionSpec()))
        return f_tree, state

    @property
    def _label_dtype(self):
        """Smallest integer type holding the droplet labels 0..max_droplets (labels are read and written every step)."""
        return jnp.int8 if self.max_droplets <= 127 else jnp.int16 if self.max_droplets <= 32767 else jnp.int32

    def _label_droplets(self, flags):
        """
        Droplet labels 1..K of the connected liquid and interface cells (lattice connectivity, periodic), 0 elsewhere, on the device:
        min-label propagation over the occupied cells until it converges (as many passes as the widest droplet spans), then the roots ranked
        to 1..K. Runs once, at initialization.
        """
        directions = [d for d in self._lattice_directions if any(d)]
        axes = tuple(range(self.dim))

        @jit
        def label(flags):
            occupied = (flags[..., 0] == LIQUID) | (flags[..., 0] == INTERFACE)
            size = occupied.size
            index = jnp.arange(size, dtype=jnp.int32).reshape(occupied.shape)
            empty = jnp.int32(size)

            def sweep(field):
                smallest = field
                for d in directions:
                    smallest = jnp.minimum(smallest, jnp.roll(field, tuple(-c for c in d), axis=axes))
                return jnp.where(occupied, smallest, empty)

            def body(carry):
                field, _ = carry
                new = sweep(field)
                return new, jnp.any(new != field)

            field, _ = jax.lax.while_loop(lambda carry: carry[1], body, (jnp.where(occupied, index, empty), jnp.bool_(True)))
            roots = occupied & (field == index)
            rank = jnp.cumsum(roots.reshape(-1), dtype=jnp.int32)
            labels = jnp.where(occupied, rank[jnp.minimum(field, size - 1).reshape(-1)].reshape(occupied.shape), 0)
            return labels[..., None], jnp.sum(roots, dtype=jnp.int32)

        labels, count = label(flags)
        if int(count) > self.max_droplets:
            raise ValueError(f"{int(count)} droplets exceed max_droplets = {self.max_droplets}")
        return jax.device_put(labels.astype(self._label_dtype), flags.sharding)

    def _share_interface_velocity_host(self, f_tree, flags):
        """Impose the momentum-conserving interface velocity (paper Eq. 13) on both lattices at the interface cells (initialization only)."""
        c = np.asarray(self.c, dtype=np.float64)
        w = np.asarray(self.lattice.w, dtype=np.float64)
        r = self.density_ratio
        f = [np.stack(planes, axis=-1) for planes in f_tree]
        rho = [np.sum(x, axis=-1, keepdims=True) for x in f]
        u = [
            np.einsum("...i,ai->...a", x, c) / d + np.asarray(self._half_acceleration(phase)) for phase, (x, d) in enumerate(zip(f, rho, strict=True))
        ]
        u_shared = (rho[0] * u[0] + r * rho[1] * u[1]) / (rho[0] + r * rho[1])
        interface = (flags == INTERFACE)[..., None]
        out = []
        for x, d, v in zip(f, rho, u, strict=True):
            shifted = x + 3.0 * w * d * np.einsum("...a,ai->...i", u_shared - v, c)
            x = np.where(interface, shifted, x)
            out.append([x[..., i] for i in range(self.q)])
        return out

    @partial(jit, static_argnums=(0,))
    def _dense_stage(self, f_tree, state, timestep):
        """Interface geometry and the dense part of a time step (_dense_m), returning what the band part needs."""
        dtype = self.precision_policy.compute_dtype
        geometry = self.interface_geometry(
            state["eps"],
            self.fluid_field,
            self.contact_mask,
            self.wall_normal,
            self.wall_cos,
            self.wall_sin,
            self.boundary_cells,
            self.boundary_normal,
            self.boundary_cot,
            self.boundary_fluid_dirs,
        )
        new_l, new_g, band, count = self.dense_kernel(tuple(f_tree[0]), tuple(f_tree[1]), state["flags"], self.bounceback_walls, self._open_bc_data)
        flux = jnp.where(timestep >= self._evaporation_start, self._evaporation_flux, 0.0).astype(dtype)
        return new_l, new_g, geometry, band, count, flux

    @partial(jit, static_argnums=(0, 4))
    def _speculative_step(self, f_tree, state, timestep, capacity):
        """
        One time step with static band capacities (interface rows, neighbor rows) as a single program, returning the new populations and
        state and the band size; the result is exact only if the band size fits the capacities (see step).
        """
        new_l, new_g, geometry, band, count, flux = self._dense_stage(f_tree, state, timestep)
        fields = (state["mass"], state["flags"], state["eps"])
        if self.zero_net_capillary_force:
            fields = fields + (state["label"], state["capillary_shift"])
        out = self._band_kernel(capacity)(tuple(f_tree[0]), tuple(f_tree[1]), new_l, new_g, *fields, geometry, band, flux)
        new_l, new_g, mass, flags, eps = out[:5]
        new_state = {"mass": mass, "flags": flags, "eps": eps, "band_count": count}
        if self.zero_net_capillary_force:
            new_state["label"], new_state["capillary_shift"] = out[5], out[6]
        return [new_l, new_g], new_state

    def _capacity_for_counts(self, counts, headroom=1.0):
        """Static band capacities for the given numbers of interface and neighbor cells (times headroom)."""
        return tuple(self._band_capacity_for(int(headroom * int(n))) for n in counts)

    def _verified(self, f_tree, state, timestep, capacity, result):
        """
        Check a speculative step: if its band outgrew the capacities, recompute it with capacities that fit. Returns the exact result and
        the capacities to queue the next step with (25 % headroom, shrunk when the band becomes much smaller).
        """
        count = tuple(int(n) for n in result[1]["band_count"])
        if any(n > c for n, c in zip(count, capacity, strict=True)):
            capacity = self._capacity_for_counts(count)
            result = self._speculative_step(f_tree, state, timestep, capacity)
        capacity = tuple(self._band_capacity_for(int(1.25 * n)) if n > 0.8 * c or n < 0.3 * c else c for n, c in zip(count, capacity, strict=True))
        return result, capacity

    def step(self, f_tree, state, timestep):
        """
        Perform one VoF time step: interface geometry, then one fused pass per lattice (pull streaming, collision, full-way bounce-back and
        reset of the inactive cells) with the open boundaries recomputed row by row (_dense_m), then the band of interface cells recomputed
        row by row over the dense result (_band_m), all as one program. The band capacities are those of the previous step; the band size is
        read back and, in the rare step where the band outgrew them, the step is recomputed with capacities that fit. (run() defers this check
        by one step so that the device never waits for the host.)

        Parameters
        ----------
        f_tree (list of tuple of jax.numpy.ndarray): Post-collision liquid and gas populations, one plane per lattice direction.

        state (dict): Liquid mass ("mass"), cell types ("flags"), volume fraction ("eps"), band size ("band_count") and, with
            zero_net_capillary_force, the droplet labels ("label") and their relaxed Laplace shifts ("capillary_shift").

        timestep (int): Current timestep.

        Returns
        -------
        f_tree (list of tuple of jax.numpy.ndarray): Updated populations.

        state (dict): Updated state.
        """
        capacity = self._next_capacity or self._capacity_for_counts(state["band_count"], 1.25)
        result, self._next_capacity = self._verified(f_tree, state, timestep, capacity, self._speculative_step(f_tree, state, timestep, capacity))
        return result

    @partial(jit, static_argnums=(0,))
    def compute_macroscopic_fields(self, f_tree, state):
        """
        Macroscopic fields for post-processing.

        Parameters
        ----------
        f_tree (list of tuple of jax.numpy.ndarray): Post-collision liquid and gas populations, one plane per lattice direction.

        state (dict): Liquid mass, cell types and volume fraction.

        Returns
        -------
        (dict): rho_tree and u_tree (physical, force-corrected velocity) of both lattices, the combined velocity "u" and gauge pressure "p"
        (liquid units), the liquid "volume_fraction", the cell types "flags" and the liquid "mass".
        """
        mass, flags, eps = state["mass"], state["flags"], state["eps"]
        r = self.density_ratio
        moments = []
        for phase, planes in enumerate(f_tree):
            f = jnp.stack([self.precision_policy.cast_to_compute(p) for p in planes], axis=-1)
            rho, u = self._moments_q(f)
            # Post-collision momentum carries the full body force; the physical velocity is half a step behind
            moments.append((rho, u - self._half_acceleration(phase)))
        (rho_l, u_l), (rho_g, u_g) = moments
        liquid_side = (flags == LIQUID) | ((flags == INTERFACE) & (eps >= 0.5))
        u_mixture = (rho_l * u_l + r * rho_g * u_g) / (rho_l + r * rho_g)
        u = jnp.where(flags == LIQUID, u_l, jnp.where(flags == GAS, u_g, jnp.where(flags == INTERFACE, u_mixture, 0.0)))
        p = jnp.where(flags == SOLID, 0.0, jnp.where(liquid_side, self.lattice.cs2 * (rho_l - 1.0), r * self.lattice.cs2 * (rho_g - 1.0)))
        return {
            "rho_tree": [rho_l, rho_g],
            "u_tree": [u_l, u_g],
            "u": u,
            "p": p,
            "volume_fraction": eps,
            "flags": flags,
            "mass": mass,
            **({"droplet_label": state["label"]} if "label" in state else {}),
        }

    @property
    def _state_keys(self):
        """Persistent per-cell state fields (checkpointed): liquid mass, cell types, volume fraction (and droplet labels)."""
        return ("mass", "flags", "eps") + (("label", "capillary_shift") if self.zero_net_capillary_force else ())

    def _checkpoint_state(self, f_tree, state):
        return {"component_0": f_tree[0], "component_1": f_tree[1], **{key: state[key] for key in self._state_keys}}

    def run(self, t_max):
        """
        Run the VoF simulation for t_max timesteps, with optional output, checkpointing and MLUPS measurement.

        Parameters
        ----------
        t_max (int): Total number of timesteps.

        Returns
        -------
        f_tree (list of tuple of jax.numpy.ndarray): Post-collision populations after t_max timesteps.

        state (dict): Liquid mass, cell types and volume fraction after t_max timesteps.
        """
        f_tree, state = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:
                assert self.mngr is not None, "Checkpoint manager does not exist."
                restore_target = lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding)
                target = jax.tree.map(restore_target, self._checkpoint_state(f_tree, state))
                try:
                    restored = self.mngr.restore(latest_step, args=orb.args.StandardRestore(target))
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")
                f_tree = [tuple(restored["component_0"]), tuple(restored["component_1"])]
                state = {**state, **{key: restored[key] for key in self._state_keys}}
                logger.info(f"Restored checkpoint at step {latest_step}.")
                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max = {t_max}). Consider increasing t_max.")

        if self.compute_MLUPS:
            start = time.time()

        pending = None
        for timestep in range(start_step, t_max + 1):
            io_flag = self.io_rate > 0 and (timestep % self.io_rate == 0 or timestep == t_max)
            print_iter_flag = self.print_info_rate > 0 and timestep % self.print_info_rate == 0
            checkpoint_flag = self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0

            # Queue this step with the capacities of the last verified one, then verify the previous step while this one runs; in the rare case
            # the previous band outgrew its capacities, both steps are recomputed from the inputs kept for that purpose
            inputs = (f_tree, state)
            capacity = self._next_capacity or self._capacity_for_counts(state["band_count"], 1.25)
            result = self._speculative_step(*inputs, timestep, capacity)
            if pending is not None:
                previous_inputs, previous_result = pending
                verified, self._next_capacity = self._verified(*previous_inputs, previous_result)
                if verified is not previous_result:
                    inputs, capacity = verified, self._next_capacity
                    result = self._speculative_step(*inputs, timestep, capacity)
                # Drop the previous step's buffers so they are not held through the next step
                del previous_inputs, previous_result, verified
            pending = ((*inputs, timestep, capacity), result)
            f_tree, state = result
            if io_flag or checkpoint_flag or timestep == t_max:
                # Results that leave the loop are verified first
                (f_tree, state), self._next_capacity = self._verified(*pending[0], pending[1])
                pending = None

            if print_iter_flag:
                logger.info(
                    colored("Timestep ", "blue")
                    + colored(f"{timestep}", "green")
                    + colored(" of ", "blue")
                    + colored(f"{t_max}", "green")
                    + colored(" completed", "blue")
                )

            if io_flag:
                logger.info(f"Saving data at timestep {timestep}/{t_max}")
                self.handle_io_timestep(timestep, f_tree, state)

            if checkpoint_flag:
                logger.info(f"Saving checkpoint at timestep {timestep}/{t_max}")
                self.mngr.save(timestep, args=orb.args.StandardSave(self._checkpoint_state(f_tree, state)))

            # Warm-up: the timer starts after the first steps, which include compilation and the band capacities settling
            if self.compute_MLUPS and timestep == start_step + 10:
                jax.block_until_ready(f_tree)
                start = time.time()

        if self.compute_MLUPS:
            jax.block_until_ready(f_tree)
            end = time.time()
            n_cells = self.nx * self.ny * (self.nz if self.dim == 3 else 1)
            domain = f"{self.nx} x {self.ny}" if self.dim == 2 else f"{self.nx} x {self.ny} x {self.nz}"
            logger.info(colored("Domain: ", "blue") + colored(domain, "green"))
            logger.info(colored("Number of voxels: ", "blue") + colored(f"{n_cells}", "green"))
            mlups = n_cells * (t_max - start_step - 10) / (end - start) / 1e6
            # Cell updates per second, and lattice updates (both lattices, the convention of Multiphase with n_components = 2)
            logger.info(colored("MLUPS: ", "blue") + colored(f"{mlups} (cells), {2 * mlups} (lattice updates, both lattices)", "red"))
        self._output_writer.finish()
        if self.mngr is not None:
            self.mngr.wait_until_finished()
        return f_tree, state

    def handle_io_timestep(self, timestep, f_tree, state):
        """
        Compute, downsample and gather the macroscopic fields, then pass them to output_data.

        Parameters
        ----------
        timestep (int): Current timestep.

        f_tree (list of tuple of jax.numpy.ndarray): Post-collision populations, one plane per lattice direction.

        state (dict): Liquid mass, cell types and volume fraction.
        """
        fields = self.compute_macroscopic_fields(f_tree, state)
        gather = lambda field: process_allgather(downsample_field(field, self.downsampling_factor))
        kwargs = {"timestep": timestep, "f_postcollision_tree": f_tree}
        for key, value in fields.items():
            kwargs[key] = [gather(v) for v in value] if isinstance(value, list) else gather(value)
        self._output_writer.submit(self.output_data, kwargs)

    @partial(jit, static_argnums=(0,))
    def collision(self, fin_tree):
        """Collision of both lattices (tuples of planes, one per lattice direction), without boundary conditions."""
        return [
            tuple(self.precision_policy.cast_to_output(p) for p in self._collide_planes([self.precision_policy.cast_to_compute(p) for p in f], phase))
            for phase, f in enumerate(fin_tree)
        ]

    def _collide_planes(self, planes, phase):
        """Collision of one lattice given as a list of q direction planes (any shape), overridden by VoFBGK and VoFMRT."""
        raise NotImplementedError

    def _forcing_planes(self, rho, velocity, acceleration):
        """Exact-difference forcing feq(rho, u + a) - feq(rho, u) per direction plane, for a constant acceleration (tuple of floats)."""
        delta_usqr = 1.5 * sum(2.0 * v * a + a * a for v, a in zip(velocity, acceleration, strict=True))
        planes = []
        for direction, weight in zip(self._lattice_directions, self._lattice_weights, strict=True):
            cu = 3.0 * _lattice_dot(direction, velocity)
            dcu = 3.0 * sum(c * a for c, a in zip(direction, acceleration, strict=True))
            planes.append(rho * weight * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr))
        return planes

    def _macroscopic_planes(self, planes):
        """Density and velocity (list of dim fields) of a list of q direction planes, as additions only."""
        rho = sum(planes)
        velocity = []
        for axis in range(self.dim):
            terms = [p if d[axis] > 0 else -p for p, d in zip(planes, self._lattice_directions, strict=True) if d[axis] != 0]
            velocity.append(reduce(lambda a, b: a + b, terms) / rho)
        return rho, velocity

    def _equilibrium_planes(self, rho, velocity):
        """Equilibrium as a list of q direction planes."""
        usqr = 1.5 * sum(v * v for v in velocity)
        planes = []
        for direction, weight in zip(self._lattice_directions, self._lattice_weights, strict=True):
            cu = 3.0 * _lattice_dot(direction, velocity)
            planes.append(rho * weight * (1.0 + cu * (1.0 + 0.5 * cu) - usqr))
        return planes


class VoFBGK(VoF):
    """
    VoF model with BGK collision on both lattices.

    Parameters
    ----------
    omega (list of float): Relaxation rates [liquid, gas], nu = cs2 (1 / omega - 1 / 2).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _shear_relaxation(self):
        return list(self.omega)

    def _collide_planes(self, planes, phase):
        """BGK collision with exact-difference forcing, per direction plane with the lattice velocities as compile-time constants."""
        omega, acceleration = self.omega[phase], self._acceleration[phase]
        rho, velocity = self._macroscopic_planes(planes)
        out = [f + omega * (feq - f) for f, feq in zip(planes, self._equilibrium_planes(rho, velocity), strict=True)]
        if acceleration is not None:
            out = [f + df for f, df in zip(out, self._forcing_planes(rho, velocity, acceleration), strict=True)]
        return out


class VoFMRT(VoF):
    """
    VoF model with MRT collision on both lattices.

    The relaxation applies the fused collision matrix K = M^T S M^-T built from the user's M and S (any invertible moment basis), through its
    nonzero entries per direction plane.

    Parameters
    ----------
    M (list of numpy.ndarray): Moment matrices [liquid, gas].

    s_rho, s_e, s_eta, s_j, s_q, s_v (list of float): Relaxation rates [liquid, gas]. s_v sets the viscosity, nu = cs2 (1 / s_v - 1 / 2).

    s_pi, s_m (list of float): Additional relaxation rates [liquid, gas] for D3Q19.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.s_rho = kwargs.get("s_rho")
        self.s_e = kwargs.get("s_e")
        self.s_eta = kwargs.get("s_eta")
        self.s_j = kwargs.get("s_j")
        self.s_q = kwargs.get("s_q")
        self.s_v = kwargs.get("s_v")
        M = kwargs.get("M")
        if not isinstance(M, list) or len(M) != 2:
            raise ValueError("M must be a list of two moment matrices [liquid, gas]")
        if isinstance(self.lattice, LatticeD2Q9):
            diagonals = [
                [s_rho, s_e, s_eta, s_j, s_q, s_j, s_q, s_v, s_v]
                for s_rho, s_e, s_eta, s_j, s_q, s_v in zip(self.s_rho, self.s_e, self.s_eta, self.s_j, self.s_q, self.s_v, strict=True)
            ]
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_pi = kwargs.get("s_pi")
            self.s_m = kwargs.get("s_m")
            diagonals = [
                [s_rho, s_e, s_eta, s_j, s_q, s_j, s_q, s_j, s_q, s_v, s_pi, s_v, s_pi, s_v, s_v, s_v, s_m, s_m, s_m]
                for s_rho, s_e, s_eta, s_j, s_q, s_v, s_pi, s_m in zip(
                    self.s_rho, self.s_e, self.s_eta, self.s_j, self.s_q, self.s_v, self.s_pi, self.s_m, strict=True
                )
            ]
        else:
            raise NotImplementedError("VoFMRT is implemented for D2Q9 and D3Q19 only.")

        # Nonzero entries of the fused collision matrix K = M^T S M^-T, relaxed = sum_j K[j, i] (f_j - feq_j) for each output direction i
        self.collision_terms = []
        for matrix, diagonal in zip(M, diagonals, strict=True):
            matrix = np.asarray(matrix, dtype=np.float64)
            K = matrix.T @ np.diag(diagonal) @ np.linalg.inv(matrix).T
            self.collision_terms.append(
                tuple(tuple((j, float(K[j, i])) for j in range(self.q) if not np.isclose(K[j, i], 0.0, atol=1e-7)) for i in range(self.q))
            )

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value=None):
        self._omega = value

    def _shear_relaxation(self):
        return list(self.s_v)

    def _collide_planes(self, planes, phase):
        """
        MRT collision with exact-difference forcing, fout = f - (f - feq) @ K + delta_feq, per direction plane with the nonzero entries of K
        as compile-time constants. The density lost to roundoff is restored on the rest population.
        """
        acceleration = self._acceleration[phase]
        rho, velocity = self._macroscopic_planes(planes)
        difference = [f - feq for f, feq in zip(planes, self._equilibrium_planes(rho, velocity), strict=True)]
        out = [f - sum(difference[j] * coefficient for j, coefficient in terms) for f, terms in zip(planes, self.collision_terms[phase], strict=True)]
        if acceleration is not None:
            out = [f + df for f, df in zip(out, self._forcing_planes(rho, velocity, acceleration), strict=True)]
        out[0] = out[0] + (rho - sum(out))
        return out
