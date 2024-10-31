import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.distribute import distribute
from lid_driven_cavity_2d import LidDrivenCavity2D


class LidDrivenCavity2D_distributed(LidDrivenCavity2D):
    def __init__(self, omega, prescribed_vel, grid_shape, velocity_set, backend, precision_policy):
        super().__init__(omega, prescribed_vel, grid_shape, velocity_set, backend, precision_policy)

    def setup_stepper(self, omega):
        stepper = IncompressibleNavierStokesStepper(omega, boundary_conditions=self.boundary_conditions)
        distributed_stepper = distribute(
            stepper,
            self.grid,
            self.velocity_set,
        )
        self.stepper = distributed_stepper
        return


if __name__ == "__main__":
    # Running the simulation
    grid_size = 512
    grid_shape = (grid_size, grid_size)
    backend = ComputeBackend.JAX  # Must be JAX for distributed multi-GPU computations. Distributed computations on WARP are not supported yet!
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=backend)

    # Setting fluid viscosity and relaxation parameter.
    Re = 200.0
    prescribed_vel = 0.05
    clength = grid_shape[0] - 1
    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    simulation = LidDrivenCavity2D_distributed(omega, prescribed_vel, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps=50000, post_process_interval=1000)
