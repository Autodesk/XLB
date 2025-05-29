import neon
import warp as wp


class MultiresSimulationManager:
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
        self.coalescence_factor = grid.create_field(cardinality=velocity_set.q, dtype=self.precision_policy.store_precision)

        fname_prefix = "test"

        for level in range(self.count_levels):
            self.u.fill_run(level, 0.0, 0)
            self.rho.fill_run(level, 1.0, 0)
            self.coalescence_factor.fill_run(level, 0.0, 0)

        # wp.synchronize()
        # self.u.update_host(0)
        # wp.synchronize()
        # self.u.export_vti(f"u_{fname_prefix}_topology.vti", 'u')

        self.f_0, self.f_1, self.bc_mask, self.missing_mask = stepper.prepare_fields(rho=self.rho, u=self.u)
        stepper.prepare_coalescence_count(coalescence_factor=self.coalescence_factor, bc_mask=self.bc_mask)

        # wp.synchronize()
        # self.u.update_host(0)
        # wp.synchronize()
        # self.u.export_vti(f"u_t2_{fname_prefix}_topology.vti", 'u')

        self.iteration_idx = -1
        from xlb.operator.macroscopic import MultiresMacroscopic

        self.macro = MultiresMacroscopic(
            compute_backend=self.grid.compute_backend,
            precision_policy=self.precision_policy,
            velocity_set=self.velocity_set,
        )

        self.__init_containers(self.count_levels)
        self._step_init()

    def __init_containers(self, num_levels):
        # working only with level 0 for now
        self.containers = {}
        self.macroscopics = {}

        self.stepper.init_containers()

    def export_macroscopic(self, fname_prefix):
        print(f"exporting macroscopic: #levels {self.grid.count_levels}")
        self.macro(self.f_0, self.bc_mask, self.rho, self.u, streamId=0)

        wp.synchronize()
        self.u.update_host(0)
        wp.synchronize()
        self.u.export_vti(f"{fname_prefix}{self.iteration_idx}.vti", "u")
        print("DONE exporting macroscopic")

        return

    def step(self):
        self.iteration_idx = self.iteration_idx + 1
        self.sk.run()

    # one step at the corase level
    def _step_init(self):
        self.app = []

        def recurtion(level, app):
            if level < 0:
                return
            print(f"RECURTION down to level {level}")
            print(f"RECURTION Level {level}, COLLIDE")

            self.stepper.add_to_app(
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
            # if(level == 0):
            #     wp.synchronize()
            #     self.f_0.update_host(0)
            #     self.f_1.update_host(0)
            #     wp.synchronize()
            #     self.f_0.export_vti(f"pop_0_", "pop_0")
            #     self.f_1.export_vti(f"pop_1_", "pop_1")
            #     # exit
            #     import sys
            #     print("exit")
            #     #sys.exit()
            #     pass

            recurtion(level - 1, app)
            recurtion(level - 1, app)

            # Important: swapping of f_0 and f_1 is done here
            print(f"RECURTION Level {level}, stream_coarse_step_ABC")
            self.stepper.add_to_app(
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
            # print(f"RECURTION Level {level}, stream_coarse_step_B")
            #
            # self.stepper.add_to_app(
            #     app=app,
            #     op_name="stream_coarse_step_B",
            #     mres_level=level,
            #     f_0=self.f_1,
            #     f_1=self.f_0,
            #     bc_mask=self.bc_mask,
            #     missing_mask=self.missing_mask,
            #     omega=self.coalescence_factor,
            #     timestep=0,
            # )

            # print(f"RECURTION Level {level}, stream_coarse_step_C")
            #
            # self.stepper.add_to_app(
            #     app=app,
            #     op_name="stream_coarse_step_C",
            #     mres_level=level,
            #     f_0=self.f_1,
            #     f_1=self.f_0,
            #     bc_mask=self.bc_mask,
            #     missing_mask=self.missing_mask,
            #     omega=self.omega,
            #     timestep=0,
            # )
            # if(level == 1):
            #     wp.synchronize()
            #     self.f_0.update_host(0)
            #     self.f_1.update_host(0)
            #     wp.synchronize()
            #     self.f_0.export_vti(f"pop_0_qq", "pop_0")
            #     self.f_1.export_vti(f"pop_1_qq", "pop_1")
            #     # exit
            #     import sys
            #     print("exit")
            #     sys.exit()
            #     pass

        recurtion(self.count_levels - 1, app=self.app)
        bk = self.grid.get_neon_backend()
        self.sk = neon.Skeleton(backend=bk)
        self.sk.sequence("mres_nse_stepper", self.app)
