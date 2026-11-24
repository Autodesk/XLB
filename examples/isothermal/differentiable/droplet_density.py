import os
import subprocess
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
from jax import config, debug, jit, lax, nn, random

from jax_lab.core.shanchen.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.shanchen.multiphase import MultiphaseBGK
from jax_lab.core.utils import save_fields_vtk

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

np.random.seed(42)

# Input dimensions (nx, ny, 1)
# Output dimensions (nx, ny, 2)

config.update("jax_default_matmul_precision", "float32")


class NeuralNetwork(eqx.Module):
    layers: list
    nx: int
    ny: int

    def __init__(self, key, nx, ny, hidden_features=512, in_channels=1, out_channels=1):
        self.nx = nx
        self.ny = ny
        in_features = nx * ny * in_channels
        out_features = nx * ny * out_channels

        k1, k2, k3, k4 = random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(in_features, hidden_features, key=k1),
            eqx.nn.Lambda(nn.sigmoid),
            eqx.nn.Linear(hidden_features, hidden_features // 2, key=k2),
            eqx.nn.Lambda(nn.sigmoid),
            eqx.nn.Linear(hidden_features // 2, hidden_features, key=k3),
            eqx.nn.Lambda(nn.sigmoid),
            eqx.nn.Linear(hidden_features, out_features, key=k4),
            eqx.nn.Lambda(nn.sigmoid),
        ]

    def __call__(self, x):
        x_flat = x.flatten() / rho_l
        for layer in self.layers:
            x_flat = layer(x_flat)
        x_out = x_flat.reshape((self.nx, self.ny, 1))
        return rho_l * x_out


class GroundTruthBGK(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)
        x, y = x.T, y.T
        dist = np.sqrt((x - self.nx // 2) ** 2 + (y - self.ny // 2) ** 2)

        rho = np.zeros((self.nx, self.ny, 1))
        rho[..., 0] = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        rho = self.distributed_array_init(rho.shape, self.precision_policy.compute_dtype, init_val=rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init(u.shape, self.precision_policy.compute_dtype, init_val=u)
        return [self.precision_policy.cast_to_output(rho)], [self.precision_policy.cast_to_output(u)]

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0])
        u = np.array(kwargs["u_tree"][0])
        fields = {"rho": rho[0, ..., 0], "ux": u[0, ..., 0], "uy": u[0, ..., 1]}
        # HDF5/XDMF output option:
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(kwargs["timestep"], fields, "output_actual", "data")
        save_fields_vtk(kwargs["timestep"], fields, "output_actual", "data")


class AutodiffMultiphaseBGK(MultiphaseBGK):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.rho_init_rand = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho_init_rand)

    def initialize_macroscopic_fields(self):
        rho = self.model(self.rho_init_rand)
        rho = self.distributed_array_init(rho.shape, self.precision_policy.compute_dtype, init_val=rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init(u.shape, self.precision_policy.compute_dtype, init_val=u)
        return [self.precision_policy.cast_to_output(rho)], [self.precision_policy.cast_to_output(u)]

    def initialize_macroscopic_fields_from_rho(self, rho):
        rho = self.distributed_array_init(rho.shape, self.precision_policy.compute_dtype, init_val=rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init(u.shape, self.precision_policy.compute_dtype, init_val=u)
        return [self.precision_policy.cast_to_output(rho)], [self.precision_policy.cast_to_output(u)]

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0])
        u = np.array(kwargs["u_tree"][0])
        fields = {"rho": rho[0, ..., 0], "ux": u[0, ..., 0], "uy": u[0, ..., 1]}
        # HDF5/XDMF output option:
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(kwargs["timestep"], fields, "output_predicted", "data")
        save_fields_vtk(kwargs["timestep"], fields, "output_predicted", "data")


@jit
def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def run_simulation(model, sim_obj, rho_init_rand, t_designated):
    rho_init_pred = model(rho_init_rand)

    # Initialize fields and distribution functions
    rho_init_tree, u_init_tree = sim_obj.initialize_macroscopic_fields_from_rho(rho_init_pred)
    f_init_tree = sim_obj.equilibrium(rho_init_tree, u_init_tree)

    # Simulation loop
    def sim_step(state_tree, step):
        state_tree, _ = sim_obj.step(state_tree, step)
        return state_tree, None

    state_tree, _ = lax.scan(sim_step, sim_obj.state_from_populations(f_init_tree), jnp.arange(t_designated))
    f_final_tree = sim_obj.poststreaming(state_tree)

    # Get final density
    rho_final_tree, _ = sim_obj.update_macroscopic(f_final_tree)
    return rho_final_tree[0]


@eqx.filter_jit
def make_step(model, opt_state, sim_obj, rho_init_rand, rho_ground_truth, t_designated):
    def loss_fn(model):
        rho_pred = run_simulation(model, sim_obj, rho_init_rand, t_designated)
        return mse_loss(rho_pred, rho_ground_truth)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


if __name__ == "__main__":
    r = 25
    nx = 100
    ny = 100

    width = 4
    a, b, R = 9 / 49, 2 / 21, 1.0
    rho_l, rho_g = 6.764470400, 0.838834226
    Tc = 0.5714285714
    T = 0.8 * Tc
    precision = "f32/f32"

    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "lambda_": [0.1**0.5],  # k = lambda_**2 = 0.1
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": VanderWaals(**{"a": [a], "b": [b], "R": [R], "T": T}),
        "body_force": [0.0, 0.0],
        "omega": [1.0],
        "precision": precision,
        "io_rate": 100,
        "compute_MLUPS": False,
        "print_info_rate": 100,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    sim_actual = GroundTruthBGK(**kwargs)
    sim_actual.run(0)

    key = random.PRNGKey(10465)
    model = NeuralNetwork(key, nx, ny)

    f_ground_truth_tree = sim_actual.assign_fields_sharded()
    rho_ground_truth_tree, _ = sim_actual.update_macroscopic(f_ground_truth_tree)
    rho_ground_truth = rho_ground_truth_tree[0]

    # Setup optimizer
    learning_rate = 1e-3
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # designated_timestep
    t_designated = 900

    # Initial random density field for the CNN
    rho_init_rand = 0.5 * (rho_l + rho_g) + 0.1 * np.random.rand(nx, ny, 1)

    sim_pred = AutodiffMultiphaseBGK(model, **kwargs)

    epochs = 300
    for epoch in range(epochs):
        model, optimizer_state, loss = make_step(model, optimizer_state, sim_pred, rho_init_rand, rho_ground_truth, t_designated)
        debug.print("Epoch {}/{}, Loss: {}", epoch + 1, epochs, loss)

    # Simulations with trained model
    kwargs.update({"io_rate": 1, "print_info_rate": 1})

    subprocess.run(["rm", "-rf", "output_predicted/"], check=True)
    sim_pred = AutodiffMultiphaseBGK(model, **kwargs)
    sim_pred.run(t_designated + 20)
