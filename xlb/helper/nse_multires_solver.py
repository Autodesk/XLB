import numpy as np

from xlb import DefaultConfig
from xlb.grid.multires_grid import NeonMultiresGrid
from xlb.precision_policy import Precision
from typing import Tuple, List
import neon
import warp as wp

class Nse_multires_simulation:
    def __init__(self, grid, velocity_set, stepper, omega):
        self.stepper = stepper
        self.grid = stepper.get_grid()
        self.precision_policy = stepper.get_precision_policy()
        self.velocity_set = velocity_set
        self.omega = omega
        count_levels = grid.count_levels
        # Create fields
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = stepper.prepare_fields()
        # self.f_0 = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        # self.f_1 = grid.create_field(cardinality=self.velocity_set.q, dtype=self.precision_policy.store_precision)
        # self.missing_mask = grid.create_field(cardinality=self.velocity_set.q, dtype=Precision.UINT8)
        # self.bc_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8)

        self.rho = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)

        fname_prefix='test'
        self.rho.fill_run(0, 0.0, 0)
        self.rho.fill_run(0, 1.0, 0)
        wp.synchronize()
        self.rho.update_host(0)
        wp.synchronize()
        self.rho.export_vti(f"{fname_prefix}_topology.vti", 'u')

        self.odd_step = None
        self.even_step = None
        self.iteration_idx = -1
        from xlb.operator.macroscopic import MultiresMacroscopic

        self.macro = MultiresMacroscopic(
            compute_backend=self.grid.compute_backend,
            precision_policy=self.precision_policy,
            velocity_set=self.velocity_set,
        )

        self.__init_containers(count_levels)

    def __init_containers(self, num_levels):
        # working only with level 0 for now
        target_level = 0
        containers = self.stepper.get_containers(target_level,
                                                 self.f_0,
                                                 self.f_1,
                                                 self.bc_mask,
                                                 self.missing_mask,
                                                 self.omega,
                                                 self.iteration_idx)

        self.even_step = containers['even']
        self.odd_step = containers['odd']

        containers = self.macro.get_containers(target_level, self.f_0, self.f_1,self.rho, self.u)

        self.even_macroscopic = containers['even']
        self.odd_macroscopic = containers['odd']

        self.skeleton_even = neon.Skeleton(self.grid.get_neon_backend())
        self.skeleton_odd = neon.Skeleton(self.grid.get_neon_backend())

        self.skeleton_even.sequence(name="even lbm", containers=[self.even_step])
        self.skeleton_odd.sequence(name="odd lbm", containers=[self.odd_step])

    def export_macroscopic(self, fname_prefix):
        # if self.iteration_idx % 2 == 0:
        #     self.even_macroscopic.run(0)
        # else:
        #     self.odd_macroscopic.run(0)
        #
        # import warp as wp
        # wp.synchronize()
        # self.u.update_host(0)
        # wp.synchronize()
        # self.u.export_vti(f"{fname_prefix}{self.iteration_idx}.vti", 'u')

        return

    def step(self):
        self.iteration_idx += 1
        if self.iteration_idx % 2 == 0:
            self.even_step.run(0)
        else:
            self.odd_step.run(0)
