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
        self.rho = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)
        fname_prefix='test'
        self.u.fill_run(0, 0.0, 0)
        self.u.fill_run(1, 0.0, 0)
        self.rho.fill_run(0, 1.0, 0)
        self.rho.fill_run(1, 1.0, 0)
        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"u_{fname_prefix}_topology.vti", 'u')


        self.f_0, self.f_1, self.bc_mask, self.missing_mask = stepper.prepare_fields(rho=self.rho,u=self.u)
        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"u_t2_{fname_prefix}_topology.vti", 'u')

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
        self.containers = {}
        self.macroscopics = {}

        for target_level in range(num_levels):
            self.containers[f"{target_level}"] = self.stepper.get_containers(target_level,
                                                     self.f_0,
                                                     self.f_1,
                                                     self.bc_mask,
                                                     self.missing_mask,
                                                     self.omega,
                                                     self.iteration_idx)
            pass

        for target_level in range(num_levels):
            self.macroscopics[f"{target_level}"] = self.macro.get_containers(target_level, self.f_0, self.f_1, self.bc_mask, self.rho, self.u)


    def export_macroscopic(self, fname_prefix):
        print(f"exporting macroscopic: #levels {self.grid.count_levels}")
        for target_level in range(self.grid.count_levels):
            if self.iteration_idx % 2 == 0:
                self.macroscopics[f"{target_level}"]['even'][0].run(0)
            else:
                self.macroscopics[f"{target_level}"]['odd'][0].run(0)


        import warp as wp
        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"{fname_prefix}{self.iteration_idx}.vti", 'u')
        print("DONE exporting macroscopic")

        return

    # one step at the corase level
    def step(self):
        self.iteration_idx += 1

        if self.iteration_idx % 2 == 0:
            self.containers["1"]["even"]['collide_coarse'].run(0)
            self.containers["0"]["even"]['collide_coarse'].run(0)
            self.containers["0"]["even"]['stream_coarse'].run(0)
            self.containers["0"]["odd"]['collide_coarse'].run(0)
            self.containers["0"]["odd"]['stream_coarse'].run(0)
            self.containers["1"]["even"]['stream_coarse'].run(0)
        else:
            self.containers["1"]["odd"]["collide_coarse"].run(0)
            self.containers["0"]["even"]["collide_coarse"].run(0)
            self.containers["0"]["even"]["stream_coarse"].run(0)
            self.containers["0"]["odd"]["collide_coarse"].run(0)
            self.containers["0"]["odd"]["stream_coarse"].run(0)
            self.containers["1"]["odd"]["stream_coarse"].run(0)
