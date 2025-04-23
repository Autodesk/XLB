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
        self.count_levels = grid.count_levels
        # Create fields
        self.rho = grid.create_field(cardinality=1, dtype=self.precision_policy.store_precision)
        self.u = grid.create_field(cardinality=3, dtype=self.precision_policy.store_precision)
        fname_prefix='test'

        for level in range(self.count_levels):
            self.u.fill_run(level, 0.0, 0)
            self.rho.fill_run(level, 1.0, 0)
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

        self.__init_containers(self.count_levels)

    def __init_containers(self, num_levels):
        # working only with level 0 for now
        self.containers = {}
        self.macroscopics = {}

        # for target_level in range(num_levels):
        #     self.containers[f"{target_level}"] = self.stepper.get_containers(target_level,
        #                                              self.f_0,
        #                                              self.f_1,
        #                                              self.bc_mask,
        #                                              self.missing_mask,
        #                                              self.omega,
        #                                              self.iteration_idx)
        #     pass

        # for target_level in range(num_levels):
        #     self.macroscopics[f"{target_level}"] = self.macro.get_containers(target_level, self.f_0, self.f_1, self.bc_mask, self.rho, self.u)
        self.stepper.init_containers()
        self.macro.init_containers()

    def export_macroscopic(self, fname_prefix):
        print(f"exporting macroscopic: #levels {self.grid.count_levels}")
        self.macro.launch_container(streamId = 0, f_0 = self.f_0, bc_mask = self.bc_mask, rho = self.rho, u = self.u)

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
        iteration_id = self.iteration_idx % 2

        # op_name, mres_level, f_0, f_1, bc_mask, missing_mask,  omega, timestep
        ## LEVEL 1
        if  self.count_levels == 2:
            self.stepper.launch_container(
            streamId= 0,
            op_name="collide_coarse",
            mres_level=1,
            f_0=self.f_0,
            f_1=self.f_1,
            bc_mask=self.bc_mask,
            missing_mask=self.missing_mask,
            omega=self.omega,
            timestep=iteration_id,
            )
        ## LEVEL 0
        self.stepper.launch_container(
            streamId= 0,
            op_name="collide_coarse",
            mres_level=0,
            f_0=self.f_0,
            f_1=self.f_1,
            bc_mask=self.bc_mask,
            missing_mask=self.missing_mask,
            omega=self.omega,
            timestep=iteration_id,
        )
        self.stepper.launch_container(
                            streamId= 0,
            op_name="stream_coarse",
            mres_level=0,
            f_0=self.f_1,
            f_1=self.f_0,
            bc_mask=self.bc_mask,
            missing_mask=self.missing_mask,
            omega=self.omega,
            timestep=iteration_id,
        )
        self.stepper.launch_container(
                            streamId= 0,
            op_name="collide_coarse",
            mres_level=0,
            f_0=self.f_0,
            f_1=self.f_1,
            bc_mask=self.bc_mask,
            missing_mask=self.missing_mask,
            omega=self.omega,
            timestep=iteration_id,
        )
        self.stepper.launch_container(
            streamId= 0,
            op_name="stream_coarse",
            mres_level=0,
            f_0=self.f_1,
            f_1=self.f_0,
            bc_mask=self.bc_mask,
            missing_mask=self.missing_mask,
            omega=self.omega,
            timestep=iteration_id,
        )
        # LEVEL 0
        if  self.count_levels == 2:
            self.stepper.launch_container(
                streamId= 0,
                op_name="stream_coarse",
                mres_level=1,
                f_0=self.f_1,
                f_1=self.f_0,
                bc_mask=self.bc_mask,
                missing_mask=self.missing_mask,
                omega=self.omega,
                timestep=iteration_id,
            )


