# Sphere simulation using the XLB library

import numpy as np
import tabulate

from windtunnel3d import WindTunnel

if __name__ == '__main__':

    # Parameters
    stl_filename = "sphere.stl"
    no_slip_walls = False
    sphere_radius = 16.0
    solve_time = 300.0 # Larger then in paper http://jtam.pl/Application-of-the-Lattice-Boltzmann-Method-to-the-flow-past-a-sphere,101879,0,2.html
    reynolds_number = [30.0, 50.0, 100.0, 300.0, 500.0, 1000.0, 3000.0, 10000.0]
    les = [False, False, False, False, False, True, True, True]
    #reynolds_number = [30.0, 50.0]
    #les = [False, False]
    lower_bounds = (-164.0, -122.0, -122.0)
    upper_bounds = (480.0, 122.0, 122.0)
    dx = 1.0
    inlet_velocity = 1.0
    density = 1.0

    drag_coefficients = []
    for r, l in zip(reynolds_number, les):
        # Calculate viscosity
        viscosity = (density * inlet_velocity * 2.0 * sphere_radius) / r

        # Print simulation parameters
        print("Simulation parameters:")
        print(f"Reynolds number: {r}")
        print(f"LES: {l}")
        print(f"Viscosity: {viscosity}")

        # Get collision model
        if l:
            collision = "SmagorinskyLESBGK"
        else:
            collision = "BGK"

        # Make wind tunnel
        wind_tunnel = WindTunnel(
            stl_filename=stl_filename,
            no_slip_walls=no_slip_walls,
            inlet_velocity=inlet_velocity,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            solve_time=solve_time,
            dx=dx,
            density=density,
            viscosity=viscosity,
            collision=collision,
            velocity_set="D3Q19",
            save_state_frequency=1e10,
            monitor_frequency=10,
        )

        # Run the simulation 
        wind_tunnel.run()

        # Get drag coefficient over last fourth of the simulation
        drag_coefficient = np.mean(wind_tunnel.drag_coefficients[-len(wind_tunnel.drag_coefficients) // 4:])
        drag_coefficients.append(drag_coefficient)
        print(f"Drag coefficient: {drag_coefficient}")

    # Print drag coefficients in a table
    # Reference values from http://jtam.pl/Application-of-the-Lattice-Boltzmann-Method-to-the-flow-past-a-sphere,101879,0,2.html
    experimental_drag_coefficients = [2.12, 1.57, 1.09, 0.65, 0.55, 0.47, 0.40, 0.41]
    reference_drag_coefficients = [2.08, 1.55, 1.08, 0.67, 0.59, 0.55, 0.53, 0.54]
    table = []
    for r, d, e, ref in zip(reynolds_number, drag_coefficients, experimental_drag_coefficients, reference_drag_coefficients):
        table.append([r, e, ref, d])
    print(tabulate.tabulate(table, headers=["Reynolds number", "Experimental drag coefficient", "Reference drag coefficient", "XLB drag coefficient"], tablefmt="fancy_grid"))
