# GPU only baseline, Basically the same problem but at the highest possible resolution for a single GH200

import numpy as np
import tabulate
import warp as wp

wp.init()

from windtunnel3d import WindTunnel

if __name__ == '__main__':

    # Parameters
    #inlet_velocity = 27.78 # m/s
    inlet_velocity = 0.001 # m/s
    stl_filename = "./run_1/drivaer_1_single_solid.stl"
    output_directory = "./output"
    origin = (-8.0, -2.0, -1.0)
    upper_bounds = (10.0, 2.0, 3.0)
    dx = 1.0 / (128.0)
    shape = (int((upper_bounds[0] - origin[0]) / dx),
             int((upper_bounds[1] - origin[1]) / dx),
             int((upper_bounds[2] - origin[2]) / dx))
    solve_time = 100.0
    save_q_criterion_frequency = 0.1
    amr_block_shape = (256, 256, 256)
    #amr_block_shape = (512, 512, 512)
    amr_ghost_cell_thickness = 12
    nr_streams=2

    # Make wind tunnel
    wind_tunnel = WindTunnel(
        stl_filename=stl_filename,
        output_directory=output_directory,
        no_slip_walls=True,
        inlet_velocity=inlet_velocity,
        origin=origin,
        dx=dx,
        shape=shape,
        solve_time=solve_time,
        save_q_criterion_frequency=save_q_criterion_frequency,
        collision="BGK",
        velocity_set="D3Q19",
        amr_block_shape=amr_block_shape,
        amr_ghost_cell_thickness=amr_ghost_cell_thickness,
        nr_streams=nr_streams,
        comm=None
    )

    # Run MLUPs test
    wind_tunnel.run_ooc_mlups()

    ## Run the simulation
    #wind_tunnel.run()
