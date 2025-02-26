from xlb import DefaultConfig
from xlb.grid import grid_factory
from xlb.precision_policy import Precision
from typing import Tuple


def create_nse_fields(
    grid_shape: Tuple[int, int, int] = None,
    grid=None,
    velocity_set=None,
    compute_backend=None,
    precision_policy=None,
):
    """Create fields for Navier-Stokes equation solver.

    Args:
        grid_shape: Tuple of grid dimensions. Required if grid is not provided.
        grid: Optional Grid object. If provided, will be used instead of creating new grid.
        velocity_set: Optional velocity set. Defaults to DefaultConfig.velocity_set.
        compute_backend: Optional compute backend. Defaults to DefaultConfig.default_backend.
        precision_policy: Optional precision policy. Defaults to DefaultConfig.default_precision_policy.

    Returns:
        Tuple of (grid, f_0, f_1, missing_mask, bc_mask)
    """
    velocity_set = velocity_set or DefaultConfig.velocity_set
    compute_backend = compute_backend or DefaultConfig.default_backend
    precision_policy = precision_policy or DefaultConfig.default_precision_policy

    if grid is None:
        if grid_shape is None:
            raise ValueError("grid_shape must be provided when grid is None")
        grid = grid_factory(grid_shape, compute_backend=compute_backend, velocity_set=velocity_set)

    # Create fields
    f_0 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    missing_mask = grid.create_field(cardinality=velocity_set.q, dtype=Precision.UINT8)
    bc_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8)

    return grid, f_0, f_1, missing_mask, bc_mask

class Nse_simulation:
    def __init__(self, grid, velocity_set, stepper, omega):
        self.stepper = stepper
        self.grid = stepper.get_grid()
        self.precision_policy = stepper.get_precision_policy()
        self.velocity_set = velocity_set
        self.omega = omega

        # Create fields
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = stepper.prepare_fields()
        # self.f_0 = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        # self.f_1 = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        # self.missing_mask = grid.create_field(cardinality=self.velocity_set.q, dtype=Precision.UINT8)
        # self.bc_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8)

        self.rho = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)

        self.odd_step = None
        self.even_step = None
        self.iteration_idx = -1
        from xlb.operator.macroscopic import Macroscopic

        self.macro = Macroscopic(
            compute_backend=self.grid.compute_backend,
            precision_policy=self.precision_policy,
            velocity_set=self.velocity_set,
        )

        self.__init_containers()

    def __init_containers(self):
        containers = self.stepper.get_containers(self.f_0, self.f_1, self.bc_mask, self.missing_mask, self.omega, self.iteration_idx)
        self.even_step = containers['even']
        self.odd_step = containers['odd']

        containers = self.macro.get_containers(self.f_0, self.f_1,self.rho, self.u)

        self.even_macroscopic = containers['even']
        self.odd_macroscopic = containers['odd']

    def export_macroscopic(self, fname_prefix):
        if self.iteration_idx % 2 == 0:
            self.even_macroscopic.run(0)
        else:
            self.odd_macroscopic.run(0)

        import warp as wp
        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"{fname_prefix}{self.iteration_idx}.vti", 'u')

        return

    def step(self):
        self.iteration_idx += 1
        if self.iteration_idx % 2 == 0:
            print("running even")
            self.even_step.run(0)
        else:
            print("running odd")
            self.odd_step.run(0)
