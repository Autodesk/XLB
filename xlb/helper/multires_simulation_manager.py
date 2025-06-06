import neon
import warp as wp
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
from xlb.operator.macroscopic import MultiresMacroscopic


class MultiresSimulationManager(MultiresIncompressibleNavierStokesStepper):
    """
    A simulation manager for multiresolution simulations using the Neon backend in XLB.
    """

    def __init__(
        self,
        omega,
        grid,
        boundary_conditions=None,
        collision_type="BGK",
        forcing_scheme="exact_difference",
        force_vector=None,
    ):
        boundary_conditions = boundary_conditions or []
        super().__init__(grid, boundary_conditions, collision_type, forcing_scheme, force_vector)

        self.omega = omega
        self.count_levels = grid.count_levels
        self.iteration_idx = 0

        self._initialize_fields()
        self._setup_operators()
        self._construct_stepper()

    def _initialize_fields(self):
        """Initialize simulation fields with proper initial conditions."""
        self.rho = self.grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u = self.grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)
        self.coalescence_factor = self.grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)

        for level in range(self.count_levels):
            self.u.fill_run(level, 0.0, 0)
            self.rho.fill_run(level, 1.0, 0)
            self.coalescence_factor.fill_run(level, 0.0, 0)

        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.prepare_fields(self.rho, self.u)
        self.prepare_coalescence_count(coalescence_factor=self.coalescence_factor, bc_mask=self.bc_mask)

    def _setup_operators(self):
        """Setup macroscopic operators for output processing."""
        self.macro = MultiresMacroscopic(
            compute_backend=self.compute_backend,
            precision_policy=self.precision_policy,
            velocity_set=self.velocity_set,
        )

    def _construct_stepper(self):
        """Construct the stepper with a clean recursive structure."""
        self.app = []
        self._build_multires_skeleton(self.count_levels - 1, self.app)

        bk = self.grid.get_neon_backend()
        self.sk = neon.Skeleton(backend=bk)
        self.sk.sequence("mres_nse_stepper", self.app)

    def _build_multires_skeleton(self, level, app):
        """
        Build the multires skeleton using a clean recursive approach.
        This mimics the structure of nested time-stepping for different levels.
        """
        if level < 0:
            return

        self._add_collision_step(level, app)

        self._build_multires_skeleton(level - 1, app)
        self._build_multires_skeleton(level - 1, app)

        self._add_streaming_step(level, app)

    def _add_collision_step(self, level, app):
        """Add collision step for given level."""
        self.add_to_app(
            app=app,
            op_name="collide_coarse",
            mres_level=level,
            f_0=self.f_0,
            f_1=self.f_1,
            bc_mask=self.bc_mask,
            missing_mask=self.missing_mask,
            omega=self.omega,
            timestep=0,
        )

    def _add_streaming_step(self, level, app):
        """Add streaming step for given level."""
        self.add_to_app(
            app=app,
            op_name="stream_coarse_step_ABC",
            mres_level=level,
            f_0=self.f_1,
            f_1=self.f_0,
            bc_mask=self.bc_mask,
            missing_mask=self.missing_mask,
            omega=self.coalescence_factor,
            timestep=0,
        )

    def step(self):
        """Execute one simulation step."""
        self.iteration_idx += 1
        self.sk.run()

    def export_macroscopic(self, fname_prefix):
        """Export macroscopic variables to VTK file."""
        self.macro(self.f_0, self.bc_mask, self.rho, self.u, streamId=0)

        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"{fname_prefix}{self.iteration_idx}.vti", "u")

    def get_iteration(self):
        """Get current iteration number."""
        return self.iteration_idx

    def get_grid_info(self):
        """Get information about the multires grid."""
        return {"count_levels": self.count_levels, "shape": self.grid.shape, "refinement_factor": self.grid.refinement_factor}
