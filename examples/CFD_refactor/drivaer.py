# Driver car simulation

import numpy as np
import tabulate

from windtunnel3d import WindTunnel

if __name__ == '__main__':

    # Parameters
    #inlet_velocity = 27.78 # m/s
    inlet_velocity = 1.0 # m/s
    #inlet_velocity = 0.1 # m/s
    stl_filename = "./run_1/drivaer_1_single_solid.stl"
    #lower_bounds = (-15.0, -5.0, 0.0)
    #upper_bounds = (35.0, 5.0, 10.0)
    lower_bounds = (-3.0, -2.0, -1.0)
    upper_bounds = (10.0, 2.0, 2.0)
    dx = 0.01
    solve_time = 10000.0

    # Make wind tunnel
    wind_tunnel = WindTunnel(
        stl_filename=stl_filename,
        no_slip_walls=False,
        inlet_velocity=inlet_velocity,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solve_time=solve_time,
        dx=dx,
        collision="SmagorinskyLESBGK",
        velocity_set="D3Q19",
    )

    # Run the simulation
    wind_tunnel.run()
