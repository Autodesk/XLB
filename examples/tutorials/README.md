# XLB Tutorials

Seven step-by-step Jupyter notebooks teaching lattice Boltzmann simulation, building from steady flow through coupled multiphysics.

| Notebook                            | Topic                          | Key Concepts                                                                                   |
| ----------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------- |
| **01_lid_driven_cavity_2d**         | Steady enclosed flow            | Velocity set (D2Q9), boundary conditions, convergence detection, Ghia benchmark                |
| **02_heat_conduction**              | Heat diffusion (no flow)        | Scalar solver (D2Q5), ScalarDirichletBC/NeumannBC, omega_from_diffusivity                      |
| **03_flow_past_cylinder_2d**        | External flow, unsteady         | Channel inlet/outlet, RegularizedBC, ExtrapolationOutflowBC, MomentumTransfer, Strouhal number |
| **04_pass_scalar_transp_cavity_2d** | Thermal transport in flow       | Two-solver coupling (D2Q9 flow + D2Q5 thermal), Péclet number, Nusselt number                  |
| **05_flow_past_square_thermal_2d**  | Cold obstacle in warm flow      | External flow + passive scalar on interior body, thermal wake visualization                    |
| **06_buoyant_cavity**               | Buoyancy-driven flow            | Two-way thermal-flow coupling, Boussinesq force, Richardson/Grashof/Rayleigh numbers            |
| **07_rayleighbenard**               | Rayleigh–Bénard convection      | Two-way coupling (bottom-heated cell), convection rolls, Nusselt number validation              |

**Quick start:** Read the markdown in each notebook first (explains physics and API), then run the cells. Modify parameters to experiment.

**Output:** Each notebook saves figures to `figures/<notebook-name>/` subdirectory.
