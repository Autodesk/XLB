"""Singlephase lattice Boltzmann simulation structure, later modified for multiphase implementation."""

import copy
import logging
import time
import warnings
from contextlib import contextmanager
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import jit, lax, shard_map
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from .boundary_conditions import (
    INLET_OUTLET_BC_TYPES,
    WALL_BC_TYPES,
    BoundaryCondition,
    BounceBack,
    BounceBackHalfway,
    BounceBackMoving,
    DoNothing,
    ExactNonEquilibriumExtrapolation,
    ExtrapolationOutflow,
    ExtrapolationOutflowMultiphase,
    NonEquilibriumExtrapolation,
    EquilibriumBC,
    ZouHe,
    link_fullway_walls,
)
from .precision_policy import PrecisionPolicy
from .utils import AsyncOutputWriter, colored, downsample_field

logger = logging.getLogger(__name__)


def quiet_steps(timestep, t_max, io_rate, print_info_rate, checkpoint_rate):
    """
    Number of consecutive timesteps from timestep on (up to t_max) that have no host-side event: output, progress print,
    checkpoint, the start of the MLUPS timer (timestep 1) or the last step. run() advances those in one device loop.
    """

    def event(t):
        return (
            t == t_max
            or t == 1
            or (io_rate > 0 and t % io_rate == 0)
            or (print_info_rate > 0 and t % print_info_rate == 0)
            or (checkpoint_rate > 0 and t % checkpoint_rate == 0)
        )

    n = 0
    while timestep + n <= t_max and not event(timestep + n):
        n += 1
    return n


def advance_quiet_steps(owner, carry, timestep, n_steps, advance, step):
    """
    Advance run()'s state by n_steps steps without host-side events, either per step (one dispatch per step) or with the
    device loop (advance, two steps per iteration), whichever is faster for this simulation: the device loop removes the
    per-step kernel-launch overhead, which dominates small or boundary-condition-heavy steps, but costs some throughput on
    large steps. The choice is measured once, on real steps, and kept in owner._device_loop.

    Parameters
    ----------
    owner (object): The solver whose run() calls this (holds _device_loop).

    carry (pytree): The state advanced.

    timestep (int): The first timestep.

    n_steps (int): Number of steps.

    advance (callable): advance(carry, timestep, n_pairs) -> carry.

    step (callable): step(carry, timestep) -> carry.

    Returns
    -------
    (pytree, int): The state and the next timestep.
    """
    if getattr(owner, "_device_loop", None) is None and n_steps >= 17:
        # One untimed step (compilation), then 8 steps each way
        carry = step(carry, timestep)
        jax.block_until_ready(carry)
        begin = time.perf_counter()
        carry = advance(carry, timestep + 1, 4)
        jax.block_until_ready(carry)
        loop_time = time.perf_counter() - begin
        begin = time.perf_counter()
        for k in range(8):
            carry = step(carry, timestep + 9 + k)
        jax.block_until_ready(carry)
        owner._device_loop = loop_time < time.perf_counter() - begin
        timestep, n_steps = timestep + 17, n_steps - 17
    if not getattr(owner, "_device_loop", True):
        for k in range(n_steps):
            carry = step(carry, timestep + k)
        return carry, timestep + n_steps
    if n_steps >= 2:
        carry = advance(carry, timestep, n_steps // 2)
    if n_steps % 2:
        carry = step(carry, timestep + n_steps - 1)
    return carry, timestep + n_steps


class PopulationState(NamedTuple):
    """
    The time-stepping state of one population set: its post-collision populations, one field per lattice direction
    (sharded along x), and the x-halo planes they are pulled from across the shard faces: lo, the last x-plane of the left
    neighbour for each direction with c_x > 0, and hi, the first x-plane of the right neighbour for each direction with
    c_x < 0 (shapes (n_devices, k, ...plane)). See LBMBase.step.
    """

    fields: tuple
    lo: jax.Array
    hi: jax.Array


# Boundary conditions evaluated row by row (no reduction over their nodes), which the compact evaluation can split by shard
ROW_LOCAL_BC_TYPES = (DoNothing, ExtrapolationOutflow, ExtrapolationOutflowMultiphase, NonEquilibriumExtrapolation, ExactNonEquilibriumExtrapolation)


class StreamingBCs(NamedTuple):
    """
    Boundary passes of one population set for LBMBase.prepare_streaming: whether it has a full-way bounce-back mask
    (applied by the fused collisions), the PostCollision passes without and with the full-way bounce-back (used when the
    collision does not apply it), and the PostStreaming passes. Each is a tuple of (kind, data) passes.
    """

    bounceback_mask: bool
    postcollision: tuple
    postcollision_bounceback: tuple
    poststreaming: tuple


class Components(tuple):
    """
    The components of a (..., n) field held as n separate arrays, indexable like the stacked array: c[..., k] is the k-th
    component. Used by the per-direction (component-mode) collisions to run the transforms written against (..., n) arrays
    without materializing any (..., n) intermediate.
    """

    def __new__(cls, values, dtype=None):
        obj = super().__new__(cls, values)
        obj.dtype = dtype
        return obj

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 2 and index[0] is Ellipsis:
            return tuple.__getitem__(self, index[1])
        return tuple.__getitem__(self, index)


_component_mode = [False]


@contextmanager
def component_mode():
    """While active (during the tracing of a component-mode collision), stack_components returns Components."""
    previous = _component_mode[0]
    _component_mode[0] = True
    try:
        yield
    finally:
        _component_mode[0] = previous


def matvec_components(components, terms, dtype):
    """
    Components of x @ A for x given as components, with A given by its sparse columns (LBMBase.sparse_collision_terms):
    out_j = sum_i x_i A[i, j] over the nonzero A[i, j]. Replaces a (cells x n) @ (n x n) matrix product.
    """
    out = []
    for column in terms:
        total = None
        for i, coefficient in column:
            term = components[..., i] * coefficient
            total = term if total is None else total + term
        out.append(jnp.zeros((), dtype=dtype) if total is None else total)
    return Components(out, dtype)


def _lattice_dot(direction, vector):
    """c_i . v for an integer lattice direction and per-axis fields, as additions only (0.0 for the rest direction)."""
    total = None
    for component, value in zip(direction, vector, strict=True):
        if component == 0:
            continue
        term = value * float(component) if abs(component) != 1 else (value if component > 0 else -value)
        total = term if total is None else total + term
    return 0.0 if total is None else total


def _lattice_sum(directions, channels, axis):
    """sum_i c_i[axis] f_i over per-direction fields."""
    total = None
    for direction, value in zip(directions, channels, strict=True):
        component = direction[axis]
        if component == 0:
            continue
        term = value * float(component) if abs(component) != 1 else (value if component > 0 else -value)
        total = term if total is None else total + term
    return total


def lattice_density_momentum(c, f):
    """
    Density rho = sum_i f_i and momentum sum_i c_i f_i of populations f (..., q), summed per lattice direction: the lattice
    velocities (0 and +-1 entries) are compile-time constants, so this is additions and subtractions of the q fields, fused
    into one pass. Replaces jnp.dot(f, c), a matrix product that runs at reduced (TF32) precision for f32 on recent GPUs by
    default and does not fuse with its neighbors.

    Parameters
    ----------
    c (array-like): Lattice velocities, shape (dim, q).

    f (jax.numpy.ndarray): Populations, shape (..., q).

    Returns
    -------
    rho (jax.numpy.ndarray): Density, shape (..., 1).

    momentum (jax.numpy.ndarray): Momentum, shape (..., dim).
    """
    c = np.asarray(c)
    directions = tuple(tuple(int(v) for v in c[:, i]) for i in range(c.shape[1]))
    channels = [f[..., i] for i in range(len(directions))]
    rho = sum(channels)
    momentum = []
    for axis in range(c.shape[0]):
        total = _lattice_sum(directions, channels, axis)
        momentum.append(jnp.zeros_like(rho) if total is None else total)
    return rho[..., None], jnp.stack(momentum, axis=-1)


def stack_components(components, like):
    """
    Assemble a (..., n) array from per-component expressions in one jnp.stack, with the semantics of filling
    jnp.zeros_like(like) component by component via .at[..., k].set(value): each value is cast to like.dtype and
    broadcast to like.shape[:-1], and components that are not given are zero.

    Building such arrays with a chain of .at[..., k].set() updates makes XLA emit one full-array pass (a dynamic-update-slice
    kernel) per component; computing the components first and stacking them once lets the whole transform fuse into the
    surrounding kernel.

    Parameters
    ----------
    components (dict): Component index -> value (array or scalar).

    like (jax.numpy.ndarray): Array whose shape and dtype the result takes.

    Returns
    -------
    (jax.numpy.ndarray): The assembled array, shape and dtype of like.
    """
    if _component_mode[0]:
        n = len(like) if isinstance(like, Components) else like.shape[-1]
        return Components([jnp.asarray(components.get(k, 0.0), dtype=like.dtype) for k in range(n)], like.dtype)
    shape = like.shape[:-1]
    zero = jnp.zeros(shape, dtype=like.dtype)
    return jnp.stack(
        [jnp.broadcast_to(jnp.asarray(components[k], dtype=like.dtype), shape) if k in components else zero for k in range(like.shape[-1])],
        axis=-1,
    )


class LBMBase(object):
    """
    Base class for lattice Boltzmann simulations.

    Parameters
    ----------
    lattice (Lattice): Lattice structure and weights.

    omega (float): Relaxation parameter.

    nx, ny (int): Grid points in the x and y directions.

    nz (int): Grid points in the z direction. Use 0 for two-dimensional simulations.

    precision (str): Compute/storage precision policy, such as ``"f32/f32"``.

    checkpoint_rate (int, optional): Timesteps between checkpoints. Defaults to 0, which disables checkpointing.

    checkpoint_dir (str, optional): Checkpoint directory. Defaults to ``"./checkpoints"``.

    downsampling_factor (int, optional): Spatial output downsampling factor. Defaults to 1.

    print_info_rate (int, optional): Timesteps between progress messages. Defaults to 100.

    io_rate (int, optional): Timesteps between output operations. Defaults to 0, which disables output.

    return_fpost (bool, optional): Whether simulation steps return post-collision populations. Defaults to False.

    compute_MLUPS (bool, optional): Whether to run in performance-measurement mode. Defaults to False.

    restore_checkpoint (bool, optional): Whether to restore the latest checkpoint. Defaults to False.

    async_io (bool, optional): Run output_data on a background thread so compute continues while files are written.
        Each output frame is copied to the host first. Defaults to True; False calls output_data synchronously.

    io_queue_size (int, optional): Output frames that may wait to be written before the simulation pauses for the
        writer. Defaults to 2.

    io_populations (bool, optional): With async_io, also copy the population arrays passed to output_data to the host.
        Defaults to False, since they are q times larger than a scalar field.

    """

    def __init__(self, **kwargs):
        self.omega = kwargs.get("omega")
        self.nx = kwargs.get("nx")
        self.ny = kwargs.get("ny")
        self.nz = kwargs.get("nz")

        self.precision = kwargs.get("precision")
        self.precision_policy = PrecisionPolicy.from_string(self.precision)

        self.lattice = kwargs.get("lattice")
        self.checkpoint_rate = kwargs.get("checkpoint_rate", 0)
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "./checkpoints")
        self.downsampling_factor = kwargs.get("downsampling_factor", 1)
        self.print_info_rate = kwargs.get("print_info_rate", 100)
        self.io_rate = kwargs.get("io_rate", 0)
        self.return_fpost = kwargs.get("return_fpost", False)
        self.compute_MLUPS = kwargs.get("compute_MLUPS", False)
        self.restore_checkpoint = kwargs.get("restore_checkpoint", False)
        self._output_writer = AsyncOutputWriter(kwargs.get("async_io", True), kwargs.get("io_queue_size", 2), kwargs.get("io_populations", False))
        self.n_devices = jax.device_count()
        self.backend = jax.default_backend()

        if self.compute_MLUPS:
            self.restore_checkpoint = False
            self.io_rate = 0
            self.checkpoint_rate = 0
            self.print_info_rate = 0

        # Check for distributed mode
        if self.n_devices > jax.local_device_count():
            warnings.warn(
                "Running in distributed mode. Call jax.distributed.initialize before performing JAX computations.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.c = self.lattice.c
        self.q = self.lattice.q
        self.w = self.lattice.w
        self.dim = self.lattice.d

        # Set the checkpoint manager
        if self.checkpoint_rate > 0:
            mngr_options = orb.CheckpointManagerOptions(save_interval_steps=self.checkpoint_rate, max_to_keep=1)
            # self.mngr = orb.CheckpointManager(self.checkpoint_dir, orb.PyTreeCheckpointer(), options=mngr_options)
            self.mngr = orb.CheckpointManager(self.checkpoint_dir, options=mngr_options)
        else:
            self.mngr = None

        # Adjust the number of grid points in the x direction, if necessary.
        # If the number of grid points is not divisible by the number of devices
        # it increases the number of grid points to the next multiple of the number of devices.
        # This is done in order to accommodate the domain sharding per XLA device
        nx, ny, nz = kwargs.get("nx"), kwargs.get("ny"), kwargs.get("nz")
        if None in {nx, ny, nz}:
            raise ValueError("nx, ny, and nz must be provided. For 2D examples, nz must be set to 0.")
        self.nx = nx
        if nx % self.n_devices:
            self.nx = nx + (self.n_devices - nx % self.n_devices)
            warnings.warn(
                f"nx increased from {nx} to {self.nx} to accommodate domain sharding per XLA device.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.ny = ny
        self.nz = nz

        # self.show_simulation_parameters()

        # Store grid information
        self.grid_info = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dim": self.lattice.d,
            "lattice": self.lattice,
        }

        P = PartitionSpec

        # Define the right permutation
        self.right_perm = [(i, (i + 1) % self.n_devices) for i in range(self.n_devices)]
        # Define the left permutation
        self.left_perm = [((i + 1) % self.n_devices, i) for i in range(self.n_devices)]

        # Set up the device mesh and sharding
        if self.dim == 2:
            self.mesh = jax.make_mesh((self.n_devices, 1, 1), ("x", "y", "value"), axis_types=(AxisType.Auto,) * 3)
            self.devices = self.mesh.devices
            self.sharding = NamedSharding(self.mesh, P("x", "y", "value"))

        elif self.dim == 3:
            self.mesh = jax.make_mesh((self.n_devices, 1, 1, 1), ("x", "y", "z", "value"), axis_types=(AxisType.Auto,) * 4)
            self.devices = self.mesh.devices
            self.sharding = NamedSharding(self.mesh, P("x", "y", "z", "value"))

        else:
            raise ValueError(f"dim = {self.dim} not supported")

        # Pull streaming and halo exchange (see step)
        self._build_streaming_kernels()

        # Compute the bounding box indices for boundary conditions
        self.bounding_box_indices = self.compute_bounding_box_indices()
        # Create boundary data for the simulation
        self._create_boundary_data()
        # Solid-node mask of the full-way BounceBack nodes, applied inside the fused collisions (see apply_fused_bounceback)
        self.bounceback_mask = self._make_bounceback_mask()
        # Boundary data of the time step (see prepare_streaming)
        self.streaming_bc_data = self._make_streaming_bc_data()
        # Gravity of the density-deviation buoyancy force, set by Thermal (see get_force_field); None otherwise
        self.buoyancy_gravity = None
        self.force = self.get_force()

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, value):
        if value is None:
            raise ValueError("Lattice type must be provided.")
        if self.nz == 0 and value.name not in ["D2Q9"]:
            raise ValueError("For 2D simulations, lattice type must be LatticeD2Q9.")
        if self.nz != 0 and value.name not in ["D3Q19", "D3Q15", "D3Q27"]:
            raise ValueError("For 3D simulations, lattice type must be LatticeD3Q19, or LatticeD3Q27.")

        self._lattice = value

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if value is None:
            raise ValueError("omega must be provided")
        self._omega = value

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        if value is None:
            raise ValueError("nx must be provided")
        if not isinstance(value, int):
            raise TypeError("nx must be an integer")
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        if value is None:
            raise ValueError("ny must be provided")
        if not isinstance(value, int):
            raise TypeError("ny must be an integer")
        self._ny = value

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value is None:
            raise ValueError("nz must be provided")
        if not isinstance(value, int):
            raise TypeError("nz must be an integer")
        self._nz = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        if not isinstance(value, str):
            raise TypeError("precision must be a string")
        self._precision = value

    @property
    def checkpoint_rate(self):
        return self._checkpoint_rate

    @checkpoint_rate.setter
    def checkpoint_rate(self, value):
        if not isinstance(value, int):
            raise TypeError("checkpoint_rate must be an integer")
        self._checkpoint_rate = value

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value):
        if not isinstance(value, str):
            raise TypeError("checkpoint_dir must be a string")
        self._checkpoint_dir = value

    @property
    def downsampling_factor(self):
        return self._downsampling_factor

    @downsampling_factor.setter
    def downsampling_factor(self, value):
        if not isinstance(value, int):
            raise TypeError("downsampling_factor must be an integer")
        self._downsampling_factor = value

    @property
    def print_info_rate(self):
        return self._print_info_rate

    @print_info_rate.setter
    def print_info_rate(self, value):
        if not isinstance(value, int):
            raise TypeError("print_info_rate must be an integer")
        self._print_info_rate = value

    @property
    def io_rate(self):
        return self._io_rate

    @io_rate.setter
    def io_rate(self, value):
        if not isinstance(value, int):
            raise TypeError("io_rate must be an integer")
        self._io_rate = value

    @property
    def return_fpost(self):
        return self._return_fpost

    @return_fpost.setter
    def return_fpost(self, value):
        if not isinstance(value, bool):
            raise TypeError("return_fpost must be a boolean")
        self._return_fpost = value

    @property
    def compute_MLUPS(self):
        return self._compute_MLUPS

    @compute_MLUPS.setter
    def compute_MLUPS(self, value):
        if not isinstance(value, bool):
            raise TypeError("compute_MLUPS must be a boolean")
        self._compute_MLUPS = value

    @property
    def restore_checkpoint(self):
        return self._restore_checkpoint

    @restore_checkpoint.setter
    def restore_checkpoint(self, value):
        if not isinstance(value, bool):
            raise TypeError("restore_checkpoint must be a boolean")
        self._restore_checkpoint = value

    @property
    def n_devices(self):
        return self._n_devices

    @n_devices.setter
    def n_devices(self, value):
        if not isinstance(value, int):
            raise TypeError("n_devices must be an integer")
        self._n_devices = value

    def show_simulation_parameters(self):
        attributes_to_show = [
            "omega",
            "nx",
            "ny",
            "nz",
            "dim",
            "precision",
            "lattice",
            "checkpoint_rate",
            "checkpoint_dir",
            "downsampling_factor",
            "print_info_rate",
            "io_rate",
            "compute_MLUPS",
            "restore_checkpoint",
            "backend",
            "n_devices",
        ]

        descriptive_names = {
            "omega": "Omega",
            "nx": "Grid Points in X",
            "ny": "Grid Points in Y",
            "nz": "Grid Points in Z",
            "dim": "Dimensionality",
            "precision": "Precision Policy",
            "lattice": "Lattice Type",
            "checkpoint_rate": "Checkpoint Rate",
            "checkpoint_dir": "Checkpoint Directory",
            "downsampling_factor": "Downsampling Factor",
            "print_info_rate": "Print Info Rate",
            "io_rate": "I/O Rate",
            "compute_MLUPS": "Compute MLUPS",
            "restore_checkpoint": "Restore Checkpoint",
            "backend": "Backend",
            "n_devices": "Number of Devices",
        }
        simulation_name = self.__class__.__name__

        logger.info(colored(f"**** Simulation Parameters for {simulation_name} ****", "green"))

        header = f"{colored('Parameter', 'blue'):>30} | {colored('Value', 'yellow')}"
        logger.info(header)
        logger.info("-" * 50)

        for attr in attributes_to_show:
            value = getattr(self, attr, "Attribute not set")
            descriptive_name = descriptive_names.get(attr, attr)  # Use the attribute name as a fallback
            row = f"{colored(descriptive_name, 'blue'):>30} | {colored(value, 'yellow')}"
            logger.info(row)

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.BCs = []
        self.set_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        solid_halo_list = [np.array(bc.indices).T for bc in self.BCs if bc.is_solid]
        solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

        # Create the grid mask on each process
        start = time.time()
        grid_mask = self.create_grid_mask(solid_halo_voxels)
        logger.info("Time to create the grid mask: %.6f seconds", time.time() - start)

        start = time.time()
        for bc in self.BCs:
            assert bc.implementation_step in ["PostStreaming", "PostCollision"]
            bc.create_local_mask_and_normal_arrays(grid_mask)
        link_fullway_walls(self.BCs)
        logger.info("Time to create the local masks and normal arrays: %.6f seconds", time.time() - start)

    @partial(jit, static_argnums=(0, 1, 2, 4))
    def distributed_array_init(self, shape, ttype, init_val=0, sharding=None):
        """
        Initialize a distributed array using JAX, with a specified shape, data type, and initial value.
        Optionally, provide a custom sharding strategy.

        Parameters
        ----------
        shape (tuple): The shape of the array to be created.

        ttype (dtype): The data type of the array to be created.

        init_val (scalar, optional): The initial value to fill the array with. Defaults to 0.

        sharding (Sharding, optional): The sharding strategy to use. Defaults to `self.sharding`.

        Returns
        -------
        jax.numpy.ndarray: A JAX array with the specified shape, data type, initial value, and sharding strategy.
        """
        if sharding is None:
            sharding = self.sharding
        x = jnp.full(shape=shape, fill_value=init_val, dtype=ttype)
        return jax.lax.with_sharding_constraint(x, sharding)

    @partial(jit, static_argnums=(0,))
    def create_grid_mask(self, solid_halo_voxels):
        """
        This function creates a mask for the background grid that accounts for the location of the boundaries.

        Parameters
        ----------
        solid_halo_voxels (numpy.ndarray): A NumPy array representing the voxels in the halo of the solid object.

        Returns
        -------
        A JAX array representing the grid mask of the grid.
        """
        # Halo width (hw_x is different to accommodate the domain sharding per XLA device)
        hw_x = self.n_devices
        hw_y = hw_z = 1
        if self.dim == 2:
            grid_mask = self.distributed_array_init((self.nx + 2 * hw_x, self.ny + 2 * hw_y, self.lattice.q), jnp.bool_, init_val=True)
            grid_mask = grid_mask.at[(slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(None))].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)

            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

        elif self.dim == 3:
            grid_mask = self.distributed_array_init(
                (
                    self.nx + 2 * hw_x,
                    self.ny + 2 * hw_y,
                    self.nz + 2 * hw_z,
                    self.lattice.q,
                ),
                jnp.bool_,
                init_val=True,
            )
            grid_mask = grid_mask.at[
                (
                    slice(hw_x, -hw_x),
                    slice(hw_y, -hw_y),
                    slice(hw_z, -hw_z),
                    slice(None),
                )
            ].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                solid_halo_voxels = solid_halo_voxels.at[:, 2].add(hw_z)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)
            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

    def compute_bounding_box_indices(self):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Returns
        -------
        boundingBox (dict): A dictionary where keys are the names of the bounding box faces
        ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
        are numpy arrays of indices corresponding to each face.
        """
        if self.dim == 2:
            # For a 2D grid, the bounding box consists of four edges: bottom, top, left, and right.
            # Each edge is represented as an array of indices. For example, the bottom edge includes
            # all points where the y-coordinate is 0, so its indices are [[i, 0] for i in range(self.nx)].
            bounding_box = {
                "bottom": np.array([[i, 0] for i in range(self.nx)], dtype=int),
                "top": np.array([[i, self.ny - 1] for i in range(self.nx)], dtype=int),
                "left": np.array([[0, i] for i in range(self.ny)], dtype=int),
                "right": np.array([[self.nx - 1, i] for i in range(self.ny)], dtype=int),
            }

            return bounding_box

        elif self.dim == 3:
            # For a 3D grid, the bounding box consists of six faces: bottom, top, left, right, front, and back.
            # Each face is represented as an array of indices. For example, the bottom face includes all points
            # where the z-coordinate is 0, so its indices are [[i, j, 0] for i in range(self.nx) for j in range(self.ny)].
            bounding_box = {
                "bottom": np.array(
                    [[i, j, 0] for i in range(self.nx) for j in range(self.ny)],
                    dtype=int,
                ),
                "top": np.array(
                    [[i, j, self.nz - 1] for i in range(self.nx) for j in range(self.ny)],
                    dtype=int,
                ),
                "left": np.array(
                    [[0, j, k] for j in range(self.ny) for k in range(self.nz)],
                    dtype=int,
                ),
                "right": np.array(
                    [[self.nx - 1, j, k] for j in range(self.ny) for k in range(self.nz)],
                    dtype=int,
                ),
                "front": np.array(
                    [[i, 0, k] for i in range(self.nx) for k in range(self.nz)],
                    dtype=int,
                ),
                "back": np.array(
                    [[i, self.ny - 1, k] for i in range(self.nx) for k in range(self.nz)],
                    dtype=int,
                ),
            }

            return bounding_box

    def initialize_macroscopic_fields(self):
        """
        This function initializes the macroscopic fields (density and velocity) to their default values.
        The default density is 1 and the default velocity is 0.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
        None, None: The default density and velocity, both None. This indicates that the actual values should be set elsewhere.
        """
        warnings.warn(
            "Default fluid initial conditions assumed: density = 1 and velocity = 0. Override initialize_macroscopic_fields to set explicit values.",
            UserWarning,
            stacklevel=2,
        )
        return None, None

    def assign_fields_sharded(self):
        """
        This function is used to initialize the simulation by assigning the macroscopic fields and populations.

        The function first initializes the macroscopic fields, which are the density (rho0) and velocity (u0).
        Depending on the dimension of the simulation (2D or 3D), it then sets the shape of the array that will hold the
        distribution functions (f).

        If the density or velocity are not provided, the function initializes the distribution functions with a default
        value (self.w), representing density=1 and velocity=0. Otherwise, it uses the provided density and velocity to initialize the populations.

        Returns
        -------
        f: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the distribution functions for the simulation.
        """
        rho0, u0 = self.initialize_macroscopic_fields()

        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)

        if rho0 is None or u0 is None:
            f = self.distributed_array_init(shape, self.precision_policy.output_dtype, init_val=self.w)
        else:
            f = self.initialize_populations(rho0, u0)

        return f

    def initialize_populations(self, rho0, u0):
        """
        This function initializes the populations (distribution functions) for the simulation.
        It uses the equilibrium distribution function, which is a function of the macroscopic
        density and velocity.

        Parameters
        ----------
        rho0 (jax.numpy.ndarray): Initial density field.

        u0 (jax.numpy.ndarray): Initial velocity field.

        Returns
        -------
        f (jax.numpy.ndarray): The array holding the initialized distribution functions for the simulation.
        """
        return self.equilibrium(rho0, u0)

    def send_right(self, x, axis_name):
        """
        This function sends the data to the right neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x (jax.numpy.ndarray): The data to be sent.

        axis_name (str): The name of the axis along which the data is sent.

        Returns
        -------
        (jax.numpy.ndarray): The data after being sent to the right neighboring process.
        """
        return lax.ppermute(x, perm=self.right_perm, axis_name=axis_name)

    def send_left(self, x, axis_name):
        """
        This function sends the data to the left neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x (jax.numpy.ndarray): The data to be sent.

        axis_name (str): The name of the axis along which the data is sent.

        Returns
        -------
        The data after being sent to the left neighboring process.
        """
        return lax.ppermute(x, perm=self.left_perm, axis_name=axis_name)

    def _split_local_indices(self, indices, *aux_arrays):
        """
        Split global node indices (and any per-row auxiliary data aligned with them) by x-shard, padding each
        device's list to a common length. Shared by every local (per-shard, int32) boundary condition path -
        full-way BounceBack (no auxiliary data) and the wall boundary conditions in WALL_BC_TYPES (imissing,
        iknown, vel, and optionally interpolation weights).

        Parameters
        ----------
        indices (numpy.ndarray): Global node coordinates, shape (n, dim).

        *aux_arrays (numpy.ndarray): Per-row data aligned with indices, each shape (n, ...).

        Returns
        -------
        (numpy.ndarray or None, tuple of numpy.ndarray or None): Padded local indices, shape
        (n_devices, max_local, dim), and one padded local array per aux_arrays entry, shape
        (n_devices, max_local, ...) zero-filled at padded rows. Both are None if indices is empty.
        """
        if len(indices) == 0:
            return None, tuple(None for _ in aux_arrays)
        local_nx = self.nx // self.n_devices
        owner = indices[:, 0] // local_nx
        by_device = [(indices[owner == device], [aux[owner == device] for aux in aux_arrays]) for device in range(self.n_devices)]
        max_local = max(len(idx) for idx, _ in by_device)
        if max_local == 0:
            return None, tuple(None for _ in aux_arrays)

        local_indices = np.zeros((self.n_devices, max_local, self.dim), dtype=np.int32)
        local_indices[..., 0] = local_nx
        local_aux = [np.zeros((self.n_devices, max_local, *aux.shape[1:]), dtype=aux.dtype) for aux in aux_arrays]
        for device, (idx, auxs) in enumerate(by_device):
            count = len(idx)
            local_indices[device, :count] = idx
            local_indices[device, :count, 0] -= device * local_nx
            for a, aux in enumerate(auxs):
                local_aux[a][device, :count] = aux
        return local_indices, tuple(local_aux)

    def _distribute_local(self, array, dtype=None):
        """
        Distribute a padded local (per-shard) host array sharded along x, blocking until it is ready.

        Parameters
        ----------
        array (numpy.ndarray or None): Padded local array, shape (n_devices, max_local, ...), or None.

        dtype (jax.numpy.dtype, optional): Target dtype. Defaults to array's own dtype, so floating-point
            auxiliary data (vel, weights) is distributed at whatever precision apply()'s global path already
            uses it at, rather than being independently rounded to a different precision.

        Returns
        -------
        (jax.numpy.ndarray or None): Distributed array, or None if array is None.
        """
        if array is None:
            return None
        if dtype is None:
            dtype = array.dtype
        sharding = NamedSharding(self.mesh, PartitionSpec("x", *([None] * (array.ndim - 1))))
        distributed = self.distributed_array_init(array.shape, dtype, init_val=array, sharding=sharding)
        distributed.block_until_ready()
        return distributed

    def _bounceback_mask_for(self, BCs):
        """
        Boolean mask (spatial shape, sharded along x) of the full-way BounceBack nodes of BCs, for applying the bounce-back
        inside the fused collisions. None when BCs has no BounceBack, or when another boundary condition could interact
        with it at the post-collision stage (one applied post-collision, or one preparing populations), so that moving the
        bounce-back into the collision cannot change the order in which boundary conditions act.

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        Returns
        -------
        (jax.Array or None): The mask.
        """
        bounceback = [bc for bc in BCs if isinstance(bc, BounceBack)]
        if not bounceback:
            return None
        for bc in BCs:
            if isinstance(bc, BounceBack):
                continue
            if bc.implementation_step == "PostCollision" or type(bc).prepare_populations is not BoundaryCondition.prepare_populations:
                return None
        spatial = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        mask = np.zeros(spatial, dtype=bool)
        for bc in bounceback:
            mask[tuple(np.asarray(index) for index in bc.indices)] = True
        sharding = NamedSharding(self.mesh, PartitionSpec("x", *([None] * (self.dim - 1))))
        mask = jax.device_put(mask, sharding)
        mask.block_until_ready()
        return mask

    def _make_bounceback_mask(self):
        """
        The full-way BounceBack mask used by the fused collisions (see _bounceback_mask_for). Overridden by Multiphase to
        return one mask per component.
        """
        return self._bounceback_mask_for(self.BCs)

    def apply_fused_bounceback(self, outputs, channels, mask):
        """
        Full-way bounce-back inside a fused collision: at the BounceBack nodes the post-collision population of direction i
        is the pre-collision population of the opposite direction (fout_i = fin_opp(i)). Applied
        per direction before the populations are stacked, with the pre-collision populations the collision already reads,
        so it adds no memory pass; applying it afterwards as a scatter into the stacked array makes XLA materialize the
        whole post-collision array once more.

        Parameters
        ----------
        outputs (list): Post-collision populations, one field per direction.

        channels (list): Pre-collision populations (the collision input), one field per direction.

        mask (jax.Array or None): BounceBack node mask (see _bounceback_mask_for); None leaves outputs unchanged.

        Returns
        -------
        (list): The post-collision populations with the bounce-back applied.
        """
        if mask is None:
            return outputs
        opposite = np.asarray(self.lattice.opp_indices)
        return [jnp.where(mask, channels[int(opposite[i])], output) for i, output in enumerate(outputs)]

    def _collision_applies_bounceback(self):
        """
        Whether collision() applies the full-way bounce-back itself (with apply_fused_bounceback), so the boundary pass
        of prepare_streaming skips it.
        Each model returns True exactly when its collision takes the path that does; False here (the generic collision).
        """
        return False

    def _collect_wall_bc_data(self, BCs, bc_type, has_weights):
        """
        Build padded local (per-shard, int32) fluid-node indices and aligned auxiliary data (imissing, iknown,
        vel, and optionally interpolation weights) for every boundary condition of exactly bc_type in BCs.

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        bc_type (type): Exact wall boundary condition class to match (subclasses are matched by their own entry
            in WALL_BC_TYPES instead, so each concrete formula gets its own local kernel).

        has_weights (bool): Whether bc_type carries per-node interpolation weights (set_proximity_ratio is
            called eagerly here if a matching boundary condition hasn't computed them yet).

        Returns
        -------
        (numpy.ndarray or None, tuple): Padded local indices and (imissing, iknown, vel, weights) padded local
        arrays (weights is None when has_weights is False). All None if BCs has no matching boundary condition.
        """
        matches = [bc for bc in BCs if type(bc) is bc_type]
        if not matches:
            return None, (None, None, None, None)

        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        imissing = np.vstack([np.asarray(bc.imissing) for bc in matches])
        iknown = np.vstack([np.asarray(bc.iknown) for bc in matches])
        # Not cast to compute_dtype here: kept at whatever dtype apply()'s global path already uses (self.vel /
        # self.weights, untouched), so the local path is numerically identical to it, not independently rounded.
        vel = np.vstack([
            np.broadcast_to(np.asarray(bc.vel), (len(bc.indices[0]), self.dim)) if bc.vel is not None else np.zeros((len(bc.indices[0]), self.dim))
            for bc in matches
        ])
        if has_weights:
            for bc in matches:
                if bc.weights is None:
                    bc.set_proximity_ratio()
            weights = np.vstack([np.asarray(bc.weights) for bc in matches])
        else:
            # Unused by apply_fn for this bc_type (see WALL_BC_TYPES), but always materialized so every wall
            # kernel shares the same argument signature.
            weights = np.zeros_like(imissing, dtype=np.float64)

        local_indices, (local_imissing, local_iknown, local_vel, local_weights) = self._split_local_indices(indices, imissing, iknown, vel, weights)
        return local_indices, (local_imissing, local_iknown, local_vel, local_weights)

    def _collect_solid_pin_indices(self, BCs):
        """
        Build padded local (per-shard, int32) solid-node indices for every wall boundary condition in
        WALL_BC_TYPES present in BCs, used to pin those nodes to the rest equilibrium after streaming (see
        BounceBackHalfway.prepare_populations).

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        Returns
        -------
        (numpy.ndarray or None): Padded local indices, shape (n_devices, max_local, dim), or None if BCs has no
        matching boundary condition.
        """
        solid_indices = [np.asarray(bc.solid_indices, dtype=np.int32).T for wall_type, _, _ in WALL_BC_TYPES for bc in BCs if type(bc) is wall_type]
        if not solid_indices:
            return None
        local_indices, _ = self._split_local_indices(np.vstack(solid_indices))
        return local_indices

    def _collect_inlet_outlet_bc_data(self, BCs, bc_type, ttype):
        """
        Build padded local (per-shard, int32) fluid-node indices and aligned auxiliary data (normals,
        imiddle_mask, iknown_mask, imissing, iknown, prescribed) for every boundary condition of exactly
        bc_type with a matching prescribed-value type (bc.type) in BCs.

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        bc_type (type): Exact ZouHe/Regularized class to match (matched by (type, ttype) instead of type alone,
            since the formula branch is selected by the instance's own bc.type, not the class).

        ttype (str): 'velocity' or 'pressure', matched against each candidate boundary condition's bc.type.

        Returns
        -------
        (numpy.ndarray or None, tuple): Padded local indices and (imissing, iknown, vel, weights)-shaped
        (normals, imiddle_mask, iknown_mask, imissing, iknown, prescribed) padded local arrays. All None if BCs
        has no matching boundary condition.
        """
        matches = [bc for bc in BCs if type(bc) is bc_type and bc.type == ttype]
        if not matches:
            return None, (None, None, None, None, None, None)

        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        normals = np.vstack([np.asarray(bc.normals) for bc in matches])
        imiddle_mask = np.vstack([np.asarray(bc.imiddle_mask) for bc in matches])
        iknown_mask = np.vstack([np.asarray(bc.iknown_mask) for bc in matches])
        imissing = np.vstack([np.asarray(bc.imissing) for bc in matches])
        iknown = np.vstack([np.asarray(bc.iknown) for bc in matches])
        prescribed = np.vstack([np.asarray(bc.prescribed) for bc in matches])

        local_indices, (local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed) = (
            self._split_local_indices(indices, normals, imiddle_mask, iknown_mask, imissing, iknown, prescribed)
        )
        return local_indices, (local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed)

    def _collect_equilibrium_bc_data(self, BCs):
        """
        Build padded local (per-shard, int32) fluid-node indices and the aligned precomputed equilibrium values
        for every EquilibriumBC in BCs.

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        Returns
        -------
        (numpy.ndarray or None, numpy.ndarray or None): Padded local indices and padded local equilibrium
        values. Both None if BCs has no EquilibriumBC.
        """
        matches = [bc for bc in BCs if type(bc) is EquilibriumBC]
        if not matches:
            return None, None
        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        out = np.vstack([np.asarray(bc.out) for bc in matches])
        local_indices, (local_out,) = self._split_local_indices(indices, out)
        return local_indices, local_out

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, rho, u, cast_output=True):
        """
        This function computes the equilibrium distribution function in the Lattice Boltzmann Method.
        The equilibrium distribution function is a function of the macroscopic density and velocity.

        The function first casts the density and velocity to the compute precision if the cast_output flag is True.
        The function finally casts the equilibrium distribution function to the output precision if the cast_output
        flag is True.

        Parameters
        ----------
        rho (jax.numpy.ndarray): The macroscopic density.

        u (jax.numpy.ndarray): The macroscopic velocity.

        cast_output (bool, optional): A flag indicating whether to cast the density, velocity, and equilibrium
            distribution function to the
        compute and output precisions. Default is True.

        Returns
        -------
        feq (jax.numpy.ndarray): The equilibrium distribution function.
        """
        # Cast the density and velocity to the compute precision if the cast_output flag is True
        if cast_output:
            rho, u = self.precision_policy.cast_to_compute((rho, u))

        # Cast c to compute precision so that XLA call FXX matmul,
        # which is faster (it is faster in some older versions of JAX, newer versions are smart enough to do this automatically)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        feq = rho * self.w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

        if cast_output:
            return self.precision_policy.cast_to_output(feq)
        else:
            return feq

    @partial(jit, static_argnums=(0,))
    def momentum_flux(self, fneq):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann
        Method (LBM).

        Parameters
        ----------
        fneq (jax.numpy.ndarray): The non-equilibrium distribution functions.

        Returns
        -------
        (jax.numpy.ndarray): The computed momentum flux.
        """
        return jnp.dot(fneq, self.lattice.cc)

    @partial(jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f):
        """
        This function computes the macroscopic variables (density and velocity) based on the
        distribution functions (f).

        The density is computed as the sum of the distribution functions over all lattice directions.
        The velocity is computed as the dot product of the distribution functions and the lattice
        velocities, divided by the density.

        Parameters
        ----------
        f (jax.numpy.ndarray): The distribution functions.

        Returns
        -------
        rho (jax.numpy.ndarray): Computed density.

        u (jax.numpy.ndarray): Computed velocity.
        """
        f = self.precision_policy.cast_to_compute(f)
        rho, momentum = lattice_density_momentum(self.lattice.c, f)
        u = momentum / rho

        return rho, u

    # ------------------------------------------------------------------------------------------------------------------
    # Time stepping (pull scheme)
    #
    # The state of a simulation is its post-collision populations, held as one field per lattice direction together with
    # the two x-halo planes received from the neighbouring shards (PopulationState). A step
    #
    #     f  = stream(state)                      post-streaming populations, f_i(x) = g_i(x - c_i)   (pull)
    #     g' = PostCollisionBCs(collision(f))
    #     state' = prepare_streaming(g')           split into per-direction fields, halo exchange, PostStreaming BCs
    #
    # compiles to one pass over the populations: the shifted reads of stream() are fused into the collision kernel
    # (see _pull_shift), which writes the q output fields directly.
    #
    # Boundary conditions are applied by prepare_streaming, on the nodes they act on only. Slot g_i(x - c_i) is read by
    # node x only, so the PostStreaming boundary conditions are applied ahead of time: the populations each boundary node
    # will pull are gathered from their slots (in the fields or in the received halo planes), the boundary conditions are
    # applied to them, and the results are written back into the same slots. The order is, at PostCollision: the
    # boundary conditions that read other nodes than their own (outflows, extrapolations, ...; _compact_bc), in their
    # list order, then the full-way bounce-back unless the collision applies it; at PostStreaming: those same boundary
    # conditions, then the solid pins, wall, equilibrium and inlet/outlet boundary conditions (_boundary_pass_m).
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _pull_shift(a, shift, axis, edge=None):
        """
        Shift a by one node along axis, b[x] = a[x - shift] for shift in {-1, 0, 1}: periodic within a when edge is None,
        otherwise the plane entering at the boundary is edge (a halo plane from the neighbouring shard).

        Written as a select between two zero pads rather than a concatenate (as jnp.roll is): XLA fuses pads and selects
        into the consuming kernel, but emits concatenates as separate passes that materialize the shifted field.
        """
        if shift == 0:
            return a
        n = a.shape[axis]
        zero = jnp.zeros((), a.dtype)
        main_pad = [(0, 0, 0)] * a.ndim
        edge_pad = [(0, 0, 0)] * a.ndim
        if shift > 0:
            main = lax.slice_in_dim(a, 0, n - 1, axis=axis)
            edge = lax.slice_in_dim(a, n - 1, n, axis=axis) if edge is None else edge
            main_pad[axis], edge_pad[axis], edge_index = (1, 0, 0), (0, n - 1, 0), 0
        else:
            main = lax.slice_in_dim(a, 1, n, axis=axis)
            edge = lax.slice_in_dim(a, 0, 1, axis=axis) if edge is None else edge
            main_pad[axis], edge_pad[axis], edge_index = (0, 1, 0), (n - 1, 0, 0), n - 1
        at_edge = lax.broadcasted_iota(jnp.int32, a.shape, axis) == edge_index
        return jnp.where(at_edge, lax.pad(edge, zero, edge_pad), lax.pad(main, zero, main_pad))

    def _halo_directions(self):
        """Directions entering a shard through its low x face (c_x > 0) and through its high x face (c_x < 0)."""
        cx = np.asarray(self.c)[0]
        return tuple(int(i) for i in np.flatnonzero(cx > 0)), tuple(int(i) for i in np.flatnonzero(cx < 0))

    def _halos_m(self, fields):
        """
        Shard-local halo exchange: the planes of the neighbouring shards that this shard's nodes pull along x. lo holds the
        last x-plane of the left neighbour for each direction with c_x > 0, hi the first x-plane of the right neighbour
        for each direction with c_x < 0; shapes (1, k, ...plane).
        """
        right, left = self._halo_directions()
        lo = self.send_right(jnp.stack([fields[i][-1] for i in right]), "x")
        hi = self.send_left(jnp.stack([fields[i][0] for i in left]), "x")
        return lo[None], hi[None]

    def _pull_m(self, fields, lo, hi):
        """
        Shard-local pull streaming, g_i(x) = f_i(x - c_i), of per-direction fields with their halo planes. Fused into the
        kernel that consumes it (see _pull_shift).
        """
        right, left = self._halo_directions()
        c = np.asarray(self.c)
        out = []
        for i, f in enumerate(fields):
            if c[0, i] > 0:
                f = self._pull_shift(f, 1, 0, lo[0, right.index(i)][None])
            elif c[0, i] < 0:
                f = self._pull_shift(f, -1, 0, hi[0, left.index(i)][None])
            for axis in range(1, self.dim):
                shift = int(c[axis, i])
                if shift:
                    # Periodic within the shard: a gather with a constant index vector, one load per value (nested pads
                    # would make one predicated load per combination of edges)
                    n = f.shape[axis]
                    f = jnp.take(f, (np.arange(n) - shift) % n, axis=axis)
            out.append(f)
        return tuple(out)

    def _build_streaming_kernels(self):
        """shard_map'd halo exchange and pull streaming (P("x") partitions the x axis of fields and halos alike)."""
        spec = PartitionSpec("x")
        self._halo_kernel = shard_map(self._halos_m, mesh=self.mesh, in_specs=(spec,), out_specs=spec, check_vma=False)
        self._pull_kernel = shard_map(self._pull_m, mesh=self.mesh, in_specs=(spec, spec, spec), out_specs=spec, check_vma=False)

    def _split_directions(self, f, per_direction=False):
        """
        (..., q) populations as a tuple of q fields. per_direction: f is a per-direction stack (stack_populations), whose
        stacking the slices cancel. Otherwise f is a materialized (..., q) array, split with one transpose: slicing it
        directly would be q strided passes, each reading all of f.
        """
        if per_direction:
            return tuple(f[..., i] for i in range(self.q))
        f = lax.optimization_barrier(jnp.moveaxis(f, -1, 0))
        return tuple(f[i] for i in range(self.q))

    @partial(jit, static_argnums=(0,))
    def streaming(self, f):
        """
        Stream a (..., q) array along the lattice directions, out[..., i](x) = f[..., i](x - c_i), periodic across the
        domain. The same pull streaming as the time step, for arrays that are not the simulation state (e.g. neighbour
        values of a scalar field repeated q times).
        """
        fields = self._split_directions(f)
        return jnp.stack(self._pull_kernel(fields, *self._halo_kernel(fields)), axis=-1)

    def stream(self, state):
        """
        Post-streaming populations of a state, f_i(x) = g_i(x - c_i), as a (..., q) array. Inside a jitted step the
        stacking cancels against the collision's per-direction reads, so the shifted reads fuse into the collision.
        """
        return jnp.stack(self._pull_kernel(state.fields, state.lo, state.hi), axis=-1)

    @partial(jit, static_argnums=(0,))
    def state_from_populations(self, f):
        """
        The state whose streamed populations are f (post-streaming populations, e.g. the initial ones or a restored
        checkpoint): g_i(x) = f_i(x + c_i), so that stream(state) == f and the next step collides f.
        """
        axes = tuple(range(self.dim))
        c = np.asarray(self.c)
        fields = tuple(jnp.roll(f[..., i], tuple(-int(v) for v in c[: self.dim, i]), axis=axes) for i in range(self.q))
        return PopulationState(fields, *self._halo_kernel(fields))

    def prepare_streaming(self, f_postcollision, previous, timestep, bc_data=None):
        """The state after a collision (see _prepare_streaming)."""
        return self._prepare_streaming(f_postcollision, previous, timestep, bc_data)[0]

    def _prepare_streaming(self, f_postcollision, previous, timestep, bc_data=None):
        """
        The state after a collision: the post-collision populations split into per-direction fields, with the boundary
        conditions applied in this order:

        1. PostCollision (writing the nodes' own populations), with the streamed populations of the previous state as
           input: the boundary conditions that read other nodes than their own (outflows, extrapolations, ...) in list
           order, the full-way bounce-back unless the collision applies it, the moving bounce-back.
        2. The halo exchange.
        3. PostStreaming (writing the pull slots): the same non-node-local boundary conditions, then the solid pins, wall,
           equilibrium and inlet/outlet boundary conditions.

        Each phase is one shard-local pass (_boundary_pass_m); boundary conditions that reduce over their nodes are
        evaluated on a global compact array first (_global_bc_pass).

        Parameters
        ----------
        f_postcollision (jax.numpy.ndarray): Post-collision populations (..., q).

        previous (PopulationState): The state that was streamed and collided.

        timestep (int): The current timestep.

        bc_data (StreamingBCs, optional): The component's boundary data; defaults to this simulation's.

        Returns
        -------
        (PopulationState, tuple): The state, and the post-collision fields after the PostCollision boundary conditions.
        """
        bc_data = self.streaming_bc_data if bc_data is None else bc_data
        fields = self._split_directions(f_postcollision, self._postcollision_per_direction())
        bounceback_fused = bc_data.bounceback_mask and self._collision_applies_bounceback()
        target = PopulationState(fields, None, None)
        for kind, data in bc_data.postcollision if bounceback_fused else bc_data.postcollision_bounceback:
            target = self._run_pass(kind, data, target, previous, timestep)
        postcollision_fields = target.fields
        state = PopulationState(postcollision_fields, *self._halo_kernel(postcollision_fields))
        for kind, data in bc_data.poststreaming:
            state = self._run_pass(kind, data, state, state, timestep)
        return state, postcollision_fields

    def _run_pass(self, kind, data, target, source, timestep):
        """Run one boundary pass (see _prepare_streaming)."""
        if kind == "global":
            return self._global_bc_pass(data, target, source)
        arrays, static = data
        moving = []
        for bc in static[1]:
            indices, vel = bc.update_function(timestep)
            indices = jnp.stack(indices, axis=-1).astype(jnp.int32)
            # Same term as BounceBackMoving.apply, per node (vel may be one velocity for all nodes)
            cu = 6.0 * self.lattice.w * jnp.dot(vel, jnp.array(self.lattice.c, dtype=self.precision_policy.compute_dtype))
            moving.append((indices, jnp.broadcast_to(cu, (indices.shape[0], self.q))))
        spec = PartitionSpec("x")
        kernel = shard_map(
            partial(self._boundary_pass_m, static=static),
            mesh=self.mesh,
            in_specs=(spec, spec, spec, PartitionSpec()),
            out_specs=spec,
            check_vma=False,
        )
        return kernel(target, source, arrays, moving)

    # Whether an overridden post_collision returns per-direction stacks made with stack_populations (fused like the
    # collisions); see _postcollision_per_direction
    post_collision_per_direction = False

    def post_collision(self, f_postcollision, f_poststreaming, timestep):
        """
        Hook applied to the post-collision populations before the PostCollision boundary conditions (e.g. a time-dependent
        forcing). Returns f_postcollision unchanged by default. An override that computes per direction and returns
        stack_populations(...) should set the class attribute post_collision_per_direction = True, so that its output is
        split into the per-direction state fields without a transpose.
        """
        return f_postcollision

    def _postcollision_per_direction(self):
        """Whether the post-collision populations are a per-direction stack: the collision takes its fused path (which
        also applies the bounce-back), and post_collision is the default or declares post_collision_per_direction."""
        hook = type(self).post_collision is LBMBase.post_collision or self.post_collision_per_direction
        return self._collision_applies_bounceback() and hook

    @partial(jit, static_argnums=(0,))
    def poststreaming(self, state):
        """Post-streaming populations of a state (after the PostStreaming boundary conditions) as a (..., q) array: what
        output, checkpoints and run() return."""
        return self.stream(state)

    @partial(jit, static_argnums=(0, 3))
    def step(self, state, timestep, return_fpost=False):
        """
        One time step: pull streaming, collision, and the boundary conditions, fused into one pass over the populations.

        Parameters
        ----------
        state (PopulationState): Post-collision populations of the previous step.

        timestep (int): The current timestep.

        return_fpost (bool, optional): Also return the post-collision populations as a (..., q) array.

        Returns
        -------
        state (PopulationState): The state after this step.

        f_postcollision (jax.numpy.ndarray or None): The post-collision populations (after the PostCollision boundary
        conditions), or None if return_fpost is False.
        """
        f_poststreaming = self.stream(state)
        f_postcollision = self.post_collision(self.collision(f_poststreaming), f_poststreaming, timestep)
        new_state, postcollision_fields = self._prepare_streaming(f_postcollision, state, timestep)
        return new_state, (jnp.stack(postcollision_fields, axis=-1) if return_fpost else None)

    @partial(jit, static_argnums=(0,))
    def advance(self, state, timestep, n_pairs):
        """
        2 n_pairs time steps from timestep, as one loop on the device: no host round trip per step, so short steps (small
        domains) are not limited by kernel launches. Used by run() between host-side events (see quiet_steps).

        Two steps per loop iteration: a pull step cannot write over the populations it reads, so with one step per
        iteration the loop would copy its state every iteration; with two, the second step writes into the buffers of the
        first one's input.
        """

        def two_steps(k, state):
            state = self.step(state, timestep + 2 * k)[0]
            return self.step(state, timestep + 2 * k + 1)[0]

        return lax.fori_loop(0, n_pairs, two_steps, state)

    # Boundary-condition data

    _NODE_LOCAL_SKIPPED = (BounceBack, BounceBackHalfway, EquilibriumBC, ZouHe)

    def _make_streaming_bc_data(self):
        """Boundary data of the time step (see prepare_streaming). Overridden by Multiphase (one per component)."""
        return self._streaming_bcs(self.BCs, self.bounceback_mask is not None)

    def _streaming_bcs(self, BCs, has_bounceback_mask):
        """
        Set up the boundary passes of one component for prepare_streaming.

        Parameters
        ----------
        BCs (list): The component's boundary conditions.

        has_bounceback_mask (bool): Whether the component has a full-way bounce-back mask, which the fused collisions
            apply themselves (decided when tracing, see _prepare_streaming).

        Returns
        -------
        (StreamingBCs): The data.
        """
        generic = [bc for bc in BCs if not isinstance(bc, self._NODE_LOCAL_SKIPPED) and not bc.is_dynamic]
        for bc in generic:
            if hasattr(bc, "find_neighbors") and not getattr(bc, "neighbors_found", True):
                bc.find_neighbors()
                bc.neighbors_found = True
        prepares = lambda bc: type(bc).prepare_populations is not BoundaryCondition.prepare_populations
        postcollision = [bc for bc in generic if bc.implementation_step == "PostCollision" or prepares(bc)]
        poststreaming = [bc for bc in generic if bc.implementation_step == "PostStreaming" or prepares(bc)]
        bounceback = [bc for bc in BCs if isinstance(bc, BounceBack)]
        moving = []
        for bc in BCs:
            if bc.is_dynamic:
                if not isinstance(bc, BounceBackMoving):
                    raise NotImplementedError(f"Dynamic boundary condition {type(bc).__name__} is not supported.")
                moving.append(bc)
        without_bounceback = self._phase_passes("PostCollision", postcollision, [], moving, [])
        return StreamingBCs(
            has_bounceback_mask,
            without_bounceback,
            self._phase_passes("PostCollision", postcollision, bounceback, moving, []) if bounceback else without_bounceback,
            self._phase_passes("PostStreaming", poststreaming, [], [], BCs),
        )

    def _phase_passes(self, implementation_step, generic, bounceback, moving, node_local_bcs):
        """
        The boundary passes of one phase: one shard-local pass (_boundary_pass_data), preceded by a global compact pass
        (_compact_bc_data) for the non-node-local boundary conditions when one of them is not row-local (see
        ROW_LOCAL_BC_TYPES).
        """
        passes = []
        if generic and not self._row_local(generic):
            passes.append(("global", self._compact_bc_data(implementation_step, generic, bounceback)))
            generic, bounceback = [], []
        shard_pass = self._boundary_pass_data(implementation_step, generic, bounceback, moving, node_local_bcs)
        if shard_pass is not None:
            passes.append(("shard", shard_pass))
        return tuple(passes)

    def _row_local(self, BCs):
        """Whether BCs are row-local and every node and the neighbours it reads are on the same shard."""
        local_nx = self.nx // self.n_devices
        return all(type(bc) in ROW_LOCAL_BC_TYPES for bc in BCs) and all(
            np.array_equal(np.asarray(bc.indices)[0] // local_nx, np.asarray(getattr(bc, name))[0] // local_nx)
            for bc in BCs
            for name in ("indices_nbr", "indices_next_nbr")
            if getattr(bc, name, None) is not None
        )

    def _node_lin(self, indices):
        """Owning shard and shard-local linear index of global node indices (tuple of dim arrays)."""
        local_nx = self.nx // self.n_devices
        idx = np.asarray(indices, dtype=np.int64).T
        device = idx[:, 0] // local_nx
        local = idx.copy()
        local[:, 0] -= device * local_nx
        return device, np.ravel_multi_index(tuple(local.T), self._local_shape())

    def _boundary_pass_data(self, implementation_step, generic, bounceback, moving, node_local_bcs):
        """
        Host-side setup of one shard-local boundary pass (see _boundary_pass_m).

        The pass gathers the populations of a set of rows (nodes) per shard: the nodes of the row-local boundary
        conditions and the neighbours they read, the full-way bounce-back nodes, and the general nodes of the node-local
        PostStreaming boundary conditions (see _node_local_stages). Row-local boundary conditions are evaluated by their
        own prepare_populations and apply on a copy whose node indices point into the shard's rows; one copy per shard,
        with the rows on other shards pointing to a spare row m (written and read there, never scattered back).

        Returns
        -------
        (tuple or None): (arrays, static), or None when the pass has nothing to do.
        """
        index_names = ("indices", "indices_nbr", "indices_next_nbr")
        n = self.n_devices
        stages, general, constants = self._node_local_stages(node_local_bcs) if node_local_bcs else ([], [np.zeros(0, np.int64)] * n, None)
        if not (generic or bounceback or moving or stages or constants is not None):
            return None
        rows = [list(g) for g in general]
        node_sets = [getattr(bc, name) for bc in generic for name in index_names if getattr(bc, name, None) is not None]
        node_sets += [bc.indices for bc in bounceback]
        for indices in node_sets:
            device, lin = self._node_lin(indices)
            for d in range(n):
                rows[d].extend(lin[device == d].tolist())
        rows = [np.unique(np.asarray(r, dtype=np.int64)) for r in rows]
        index_arrays, m = self._row_index_arrays(rows)

        def positions_on(d, indices):
            """Positions of nodes in shard d's rows; m (the spare row) for nodes of other shards."""
            device, lin = self._node_lin(indices)
            found = np.minimum(np.searchsorted(rows[d], lin), max(len(rows[d]) - 1, 0))
            member = (device == d) & ((rows[d][found] == lin) if len(rows[d]) else False)
            return np.where(member, found, m).astype(np.int32)

        per_device = None
        if generic or bounceback:
            per_device = []
            for d in range(n):
                clones = []
                for bc in generic:
                    clone = copy.copy(bc)
                    for name in index_names:
                        if getattr(bc, name, None) is not None:
                            setattr(clone, name, (positions_on(d, getattr(bc, name)),))
                    clones.append(clone)
                bounceback_positions = np.concatenate([positions_on(d, bc.indices) for bc in bounceback]) if bounceback else None
                per_device.append((tuple(clones), bounceback_positions))
            per_device = tuple(per_device)
        arrays = {"rows": {key: self._distribute_local(value, jnp.int32) for key, value in index_arrays.items()}}
        if stages:
            arrays["stages"] = self._distribute_stage_arrays(stages, rows, m)
        if constants is not None:
            arrays["constants"] = constants
        static = (implementation_step, tuple(moving), per_device, tuple((kind, formula) for kind, formula, _, _ in stages), m)
        return arrays, static

    def _boundary_pass_m(self, target, source, arrays, moving, static):
        """
        Shard-local boundary pass (see _boundary_pass_data and _prepare_streaming).

        PostCollision: fout is the rows' own post-collision populations (target) and fin the populations they were
        collided from (pulled from source); the results, and the moving bounce-back, are written to the rows' own
        populations. PostStreaming: fout is the populations the rows pull (from target) and fin their own post-collision
        populations; the row-local boundary conditions, then the node-local stages, are applied and the results, with the
        constant nodes' values, are written into the pull slots.

        The block is direction-major, (q, m), so each row maps to one population field; only the (m, q) blocks taken by
        the boundary-condition formulas are transposed.
        """
        implementation_step, _, per_device, stages, m = static
        rows = arrays["rows"]
        postcollision = implementation_step == "PostCollision"
        if not postcollision and per_device is None and not stages:
            # Constant nodes only: nothing to gather
            return self._scatter_slots(target, None, None, arrays["constants"])
        if postcollision:
            fout, fin = self._gather_own(target.fields, rows), self._gather_slots(source, rows)
        else:
            fout, fin = self._gather_slots(target, rows), self._gather_own(target.fields, rows)
        if per_device is not None:
            spare = lambda block: jnp.concatenate([block.T, jnp.zeros((1, self.q), block.dtype)])
            branches = [
                partial(self._evaluate_bcs, clones=clones, bounceback=bounceback, implementation_step=implementation_step)
                for clones, bounceback in per_device
            ]
            fout = lax.switch(lax.axis_index("x"), branches, spare(fout), spare(fin))[:m].T
        if stages:
            fout = self._apply_node_local_stages(fout, fin, stages, arrays["stages"])
        if not postcollision:
            return self._scatter_slots(target, fout, rows, arrays.get("constants"))
        index = [rows["own"][0]] * self.q
        values = list(fout)
        if moving:
            local_nx = target.fields[0].shape[0]
            pulled = self._pull_m(source.fields, source.lo, source.hi)
            opposite = np.asarray(self.lattice.opp_indices)
            for indices, cu in moving:
                x = indices[:, 0] - lax.axis_index("x") * local_nx
                inside = (x >= 0) & (x < local_nx)
                node = (jnp.where(inside, x, 0), *(indices[:, axis] for axis in range(1, self.dim)))
                lin = jnp.where(inside, jnp.ravel_multi_index(node, target.fields[0].shape, mode="clip"), target.fields[0].size)
                for i in range(self.q):
                    bounced = self._get(pulled[int(opposite[i])].reshape(-1), lin) + cu[:, i]
                    index[i] = jnp.concatenate([index[i], lin])
                    values[i] = jnp.concatenate([values[i], bounced.astype(values[i].dtype)])
        fields = tuple(f.reshape(-1).at[index[i]].set(values[i].astype(f.dtype), mode="drop").reshape(f.shape) for i, f in enumerate(target.fields))
        return PopulationState(fields, target.lo, target.hi)

    # Rows: gathering and scattering the populations of a set of nodes, shard-locally

    def _local_shape(self):
        return (self.nx // self.n_devices, *((self.ny,) if self.dim == 2 else (self.ny, self.nz)))

    def _row_index_arrays(self, rows):
        """
        Index data for gathering and scattering the populations of given nodes (per device, local linear indices), as a
        direction-major (q, m) block per device: "own" (the node's own populations, as in the post-collision fields) and
        the pull slots: "slot", x - c_i in the fields (out of range when the slot is in a halo plane), "slot_lo"/"slot_hi",
        its index in the lo/hi halo plane (out of range when it is not there), and the same halo slots as flat lists for
        scattering, "lo_src"/"hi_src" with their block positions "lo_dst"/"hi_dst". Padding is out of range everywhere.

        Returns
        -------
        (dict, int): Host arrays (n_devices, ...), and the block width m.
        """
        shape = self._local_shape()
        local_nx = shape[0]
        plane = int(np.prod(shape[1:]))
        field_size = int(np.prod(shape))
        right, left = self._halo_directions()
        n, q = self.n_devices, self.q
        m = max(1, max(len(r) for r in rows))
        own = np.full((n, m), field_size, dtype=np.int32)
        slot = np.full((n, q, m), field_size, dtype=np.int32)
        halo = {-1: ([], [], len(right) * plane, right), 1: ([], [], len(left) * plane, left)}
        slot_halo = {side: np.full((n, q, m), size, dtype=np.int32) for side, (_, _, size, _) in halo.items()}
        c = np.asarray(self.c).T.astype(np.int64)
        for d in range(n):
            lin = np.asarray(rows[d], dtype=np.int64)
            own[d, : len(lin)] = lin
            x = np.stack(np.unravel_index(lin, shape), axis=-1).reshape(-1, self.dim)
            y = x[:, None, :] - c[None]
            for axis in range(1, self.dim):
                y[..., axis] %= shape[axis]
            place = np.where(y[..., 0] < 0, -1, np.where(y[..., 0] >= local_nx, 1, 0))
            inside = place == 0
            slot[d][:, : len(lin)] = np.where(
                inside, np.ravel_multi_index(tuple(np.moveaxis(np.where(inside[..., None], y, 0), -1, 0)), shape), field_size
            ).T
            in_plane = np.ravel_multi_index(tuple(np.moveaxis(y[..., 1:], -1, 0)), shape[1:]) if len(lin) else np.zeros((0, q), np.int64)
            for side, (src, dst, _, directions) in halo.items():
                s, t = [], []
                for k, i in enumerate(directions):
                    nodes = np.flatnonzero(place[:, i] == side)
                    s.extend((k * plane + in_plane[nodes, i]).tolist())
                    t.extend((i * m + nodes).tolist())
                    slot_halo[side][d, i, nodes] = k * plane + in_plane[nodes, i]
                src.append(s)
                dst.append(t)
        arrays = {"own": own, "slot": slot, "slot_lo": slot_halo[-1], "slot_hi": slot_halo[1]}
        for side, name in ((-1, "lo"), (1, "hi")):
            src, dst, size, _ = halo[side]
            width = max(1, max(len(s) for s in src))
            arrays[f"{name}_src"] = np.full((n, width), size, dtype=np.int32)
            arrays[f"{name}_dst"] = np.full((n, width), q * m, dtype=np.int32)
            for d in range(n):
                arrays[f"{name}_src"][d, : len(src[d])] = src[d]
                arrays[f"{name}_dst"][d, : len(dst[d])] = dst[d]
        return arrays, m

    @staticmethod
    def _get(x, index):
        return x.at[index].get(mode="fill", fill_value=0.0)

    def _gather_own(self, fields, rows):
        """Shard-local: own populations of the rows, (q, m)."""
        return jnp.stack([self._get(f.reshape(-1), rows["own"][0]) for f in fields])

    def _gather_slots(self, state, rows):
        """Shard-local: populations the rows pull (from the fields and the halo planes), (q, m)."""
        lo, hi = state.lo.reshape(-1), state.hi.reshape(-1)
        columns = []
        for i, f in enumerate(state.fields):
            slot_lo, slot_hi = rows["slot_lo"][0][i], rows["slot_hi"][0][i]
            # Gathers and selects only, which XLA fuses into one kernel (a scatter of the halo values would be one more)
            value = jnp.where(
                slot_lo < lo.size,
                self._get(lo, slot_lo),
                jnp.where(slot_hi < hi.size, self._get(hi, slot_hi), self._get(f.reshape(-1), rows["slot"][0][i])),
            )
            columns.append(value)
        return jnp.stack(columns)

    def _scatter_own(self, fields, block, rows):
        """Shard-local inverse of _gather_own."""
        return tuple(f.reshape(-1).at[rows["own"][0]].set(block[i].astype(f.dtype), mode="drop").reshape(f.shape) for i, f in enumerate(fields))

    def _scatter_slots(self, state, block, rows, constants=None):
        """
        Shard-local inverse of _gather_slots (writes the fields and the halo planes). constants optionally adds fixed
        values written in the same scatters: per direction "field_index"/"field_value" (q, k) and per halo plane
        "<lo|hi>_index"/"<lo|hi>_value" (k,).
        """
        field_index = [rows["slot"][0][i] for i in range(self.q)] if rows is not None else [None] * self.q
        field_value = [block[i] for i in range(self.q)] if rows is not None else [None] * self.q
        if constants is not None:
            join = lambda a, b: b if a is None else jnp.concatenate([a, b.astype(a.dtype)])
            field_index = [join(a, constants["field_index"][0][i]) for i, a in enumerate(field_index)]
            field_value = [join(a, constants["field_value"][0][i]) for i, a in enumerate(field_value)]
        fields = tuple(
            f.reshape(-1).at[index].set(value.astype(f.dtype), mode="drop").reshape(f.shape)
            for f, index, value in zip(state.fields, field_index, field_value, strict=True)
        )
        halos = []
        for name, halo in (("lo", state.lo), ("hi", state.hi)):
            index = rows[f"{name}_src"][0] if rows is not None else None
            value = self._get(block.reshape(-1), rows[f"{name}_dst"][0]) if rows is not None else None
            if constants is not None:
                index = constants[f"{name}_index"][0] if index is None else jnp.concatenate([index, constants[f"{name}_index"][0]])
                value = (
                    constants[f"{name}_value"][0] if value is None else jnp.concatenate([value, constants[f"{name}_value"][0].astype(value.dtype)])
                )
            halos.append(halo.reshape(-1).at[index].set(value.astype(halo.dtype), mode="drop").reshape(halo.shape))
        return PopulationState(fields, *halos)

    # Boundary conditions that reduce over their nodes (or read nodes on other shards): global compact evaluation

    def _compact_bc_data(self, implementation_step, BCs, bounceback):
        """
        Host-side setup of the global compact evaluation (_global_bc_pass) of boundary conditions that are not row-local.

        The nodes these boundary conditions read or write (their indices and neighbour indices) are gathered shard-locally
        into a compact (N, q) array, in device-major blocks of m rows; each boundary condition is evaluated, in list order,
        by its own prepare_populations and apply on a copy of it whose node indices point into that array.

        Returns
        -------
        (tuple): (rows arrays, static (implementation_step, (clones, bounce-back positions), m)).
        """
        index_names = ("indices", "indices_nbr", "indices_next_nbr")
        node_sets = [getattr(bc, name) for bc in BCs for name in index_names if getattr(bc, name, None) is not None]
        node_sets += [bc.indices for bc in bounceback]
        rows = [[] for _ in range(self.n_devices)]
        for indices in node_sets:
            device, lin = self._node_lin(indices)
            for d in range(self.n_devices):
                rows[d].extend(lin[device == d].tolist())
        rows = [np.unique(np.asarray(r, dtype=np.int64)) for r in rows]
        index_arrays, m = self._row_index_arrays(rows)

        def remap(indices):
            device, lin = self._node_lin(indices)
            return ((device * m + np.array([np.searchsorted(rows[d], v) for d, v in zip(device, lin, strict=True)])).astype(np.int32),)

        clones = []
        for bc in BCs:
            clone = copy.copy(bc)
            for name in index_names:
                if getattr(bc, name, None) is not None:
                    setattr(clone, name, remap(getattr(bc, name)))
            clones.append(clone)
        bounceback_positions = np.concatenate([remap(bc.indices)[0] for bc in bounceback]) if bounceback else None
        arrays = {key: self._distribute_local(value, jnp.int32) for key, value in index_arrays.items()}
        return arrays, (implementation_step, (tuple(clones), bounceback_positions), m)

    def _global_bc_pass(self, data, target, source):
        """
        Global compact evaluation of boundary conditions (see _compact_bc_data), with the inputs and outputs of
        _boundary_pass_m: own populations at PostCollision, pull slots at PostStreaming.
        """
        arrays, (implementation_step, (clones, bounceback), m) = data
        postcollision = implementation_step == "PostCollision"
        spec = PartitionSpec("x")

        def gather_m(target, source, rows):
            own = self._gather_own(target.fields, rows).T
            pulled = self._gather_slots(source, rows).T
            return (own[None], pulled[None]) if postcollision else (pulled[None], own[None])

        def scatter_m(target, block, rows):
            if postcollision:
                return PopulationState(self._scatter_own(target.fields, block[0], rows), target.lo, target.hi)
            return self._scatter_slots(target, block[0], rows)

        gather = shard_map(gather_m, mesh=self.mesh, in_specs=(spec, spec, spec), out_specs=spec, check_vma=False)
        fout, fin = (b.reshape(-1, self.q) for b in gather(target, source, arrays))
        fout = self._evaluate_bcs(fout, fin, clones, bounceback, implementation_step)
        block = fout.reshape(self.n_devices, m, self.q).transpose(0, 2, 1)
        scatter = shard_map(scatter_m, mesh=self.mesh, in_specs=(spec, spec, spec), out_specs=spec, check_vma=False)
        return scatter(target, block, arrays)

    def _evaluate_bcs(self, fout, fin, clones, bounceback, implementation_step):
        """Boundary conditions on compact (rows, q) arrays, in list order, then the full-way bounce-back (PostCollision)."""
        for bc in clones:
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementation_step == implementation_step:
                fout = fout.at[bc.indices].set(bc.apply(fout, fin))
        if implementation_step == "PostCollision" and bounceback is not None:
            fout = fout.at[bounceback].set(fin[bounceback][..., self.lattice.opp_indices])
        return fout

    # Node-local PostStreaming boundary conditions (solid pins, walls, equilibrium, inlet/outlet)

    def _node_local_stages(self, BCs):
        """
        Host-side setup of the node-local PostStreaming boundary conditions of one component: the stages (solid pins,
        walls, equilibrium, inlet/outlet, in this order), and the split of their nodes in two sets, per shard:

        - general nodes, touched by a wall or inlet/outlet boundary condition: the populations they pull are gathered and
          the stages applied to them (see _apply_node_local_stages).
        - constant nodes, touched only by solid pins and EquilibriumBC: the populations they pull are constants, computed
          here in stage order, and are only written.

        Returns
        -------
        (list, list, dict or None): The stages (kind, formula, padded local indices, aux), the general nodes per device
        (local linear indices), and the distributed constant writes (see _scatter_slots), or None.
        """
        n, q = self.n_devices, self.q
        stages = []

        def add(kind, formula, local_indices, *aux):
            if local_indices is not None:
                stages.append((kind, formula, local_indices, aux))

        add("pin", None, self._collect_solid_pin_indices(BCs))
        for bc_type, formula, has_weights in WALL_BC_TYPES:
            local_indices, aux = self._collect_wall_bc_data(BCs, bc_type, has_weights)
            add("wall", formula, local_indices, *aux)
        local_indices, local_out = self._collect_equilibrium_bc_data(BCs)
        add("equilibrium", None, local_indices, local_out)
        for bc_type, ttype, formula in INLET_OUTLET_BC_TYPES:
            local_indices, aux = self._collect_inlet_outlet_bc_data(BCs, bc_type, ttype)
            add("inlet_outlet", formula, local_indices, *aux)
        if not stages:
            return [], [np.zeros(0, np.int64)] * n, None

        general = [
            np.unique(
                np.concatenate(
                    [self._stage_rows(li, d)[1] for kind, _, li, _ in stages if kind in ("wall", "inlet_outlet")] or [np.zeros(0, np.int64)]
                )
            )
            for d in range(n)
        ]
        w_out = np.asarray(self.precision_policy.cast_to_output(self.w))
        constant = []
        for d in range(n):
            values = {}
            for kind, _, li, aux in stages:
                if kind in ("pin", "equilibrium"):
                    for k, node in zip(*self._stage_rows(li, d), strict=True):
                        values[int(node)] = w_out if kind == "pin" else np.asarray(aux[0][d, k])
            general_nodes = set(general[d].tolist())
            constant.append({node: value for node, value in values.items() if node not in general_nodes})
        constants = None
        if any(constant):
            # Written through the slot index data of the constant nodes, as flat per-direction lists
            const_rows, m_const = self._row_index_arrays([list(c.keys()) for c in constant])
            const_values = np.zeros((n, q, m_const), dtype=w_out.dtype)
            for d in range(n):
                if constant[d]:
                    const_values[d, :, : len(constant[d])] = np.asarray(list(constant[d].values())).T
            constants = {"field_index": const_rows["slot"], "field_value": const_values}
            for name in ("lo", "hi"):
                # Halo slots: the value at each (direction, node) block position (padding is out of range)
                constants[f"{name}_index"] = const_rows[f"{name}_src"]
                position = np.minimum(const_rows[f"{name}_dst"], q * m_const - 1)
                constants[f"{name}_value"] = np.take_along_axis(const_values.reshape(n, -1), position, axis=1).astype(w_out.dtype)
            constants = {k: self._distribute_local(v, v.dtype if k.endswith("value") else jnp.int32) for k, v in constants.items()}
        # The stages are applied in the pass only when there are general nodes (constant nodes are only written)
        return (stages if any(len(g) for g in general) else []), general, constants

    def _stage_rows(self, local_indices, device):
        """Valid rows of a padded local index array on one device, and their linear indices."""
        shape = self._local_shape()
        valid = local_indices[device, :, 0] < shape[0]
        return np.flatnonzero(valid), np.ravel_multi_index(tuple(local_indices[device, valid].T), shape)

    def _distribute_stage_arrays(self, stages, rows, m):
        """Per stage: the positions of its nodes in the pass rows (padded with m) and its auxiliary data, distributed."""
        aux_dtypes = {
            "pin": (),
            "wall": (jnp.uint8, jnp.uint8, None, None),
            "equilibrium": (None,),
            "inlet_outlet": (None, jnp.bool_, jnp.bool_, jnp.uint8, jnp.uint8, None),
        }

        def positions(local_indices):
            pos = np.full(local_indices.shape[:2], m, dtype=np.int32)
            for d in range(self.n_devices):
                valid, lin = self._stage_rows(local_indices, d)
                found = np.minimum(np.searchsorted(rows[d], lin), max(len(rows[d]) - 1, 0))
                member = (rows[d][found] == lin) if len(rows[d]) else np.zeros(len(lin), bool)
                pos[d, valid[member]] = found[member]
            return pos

        return [
            (
                self._distribute_local(positions(li), jnp.int32),
                tuple(self._distribute_local(a, dtype) for a, dtype in zip(aux, aux_dtypes[kind], strict=True)),
            )
            for kind, _, li, aux in stages
        ]

    def _apply_node_local_stages(self, fb, fin, stages, stage_arrays):
        """
        Apply the node-local stages to the (q, m) block fb of pulled populations (fin: the rows' own post-collision
        populations): solid pins, the wall formulas of WALL_BC_TYPES, equilibrium values, the inlet/outlet formulas of
        INLET_OUTLET_BC_TYPES.
        """
        take = lambda block, pos: jnp.take(block, pos, axis=1, mode="fill", fill_value=0.0).T
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cc = jnp.array(self.lattice.cc, dtype=self.precision_policy.compute_dtype)
        for (kind, formula), (pos, aux) in zip(stages, stage_arrays, strict=True):
            pos = pos[0]
            aux = tuple(a[0] for a in aux)
            if kind == "pin":
                value = jnp.broadcast_to(self.precision_policy.cast_to_output(self.w), (pos.shape[0], self.q))
            elif kind == "wall":
                imissing, iknown, vel, weights = aux
                value = formula(take(fb, pos), take(fin, pos), imissing, iknown, vel, weights, self.lattice.w, c)
            elif kind == "equilibrium":
                value = aux[0]
            else:
                normals, imiddle_mask, iknown_mask, imissing, iknown, prescribed = aux
                value = formula(take(fb, pos), prescribed, normals, imiddle_mask, iknown_mask, imissing, iknown, self.lattice.w, c, cc, self.dim)
            fb = fb.at[:, pos].set(value.astype(fb.dtype).T, mode="drop")
        return fb

    def run(self, t_max):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the
        simulation steps (streaming, collision and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max (int): The total number of time steps to run the simulation.

        Returns
        -------
        f (jax.numpy.ndarray): The (post-streaming) distribution functions after the simulation.
        """
        f = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                restore_target = lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=self.sharding)
                state = jax.tree.map(restore_target, {"f": f})
                try:
                    f = self.mngr.restore(latest_step, args=orb.args.StandardRestore(state))["f"]
                    logger.info(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")

                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max = {t_max}). Consider increasing t_max.")
        # The loop advances the state (post-collision populations, see step); the post-streaming populations f are formed
        # from it only where they are used (output, checkpoints, the returned value)
        state = self.state_from_populations(f)
        del f
        # Compile the device loop up front (a zero-step call), so that it is not timed (MLUPS)
        state = self.advance(state, start_step, 0)
        if self.compute_MLUPS:
            start = time.time()
        # Loop over all time steps
        timestep = start_step
        while timestep <= t_max:
            # Steps without host-side events, see advance_quiet_steps
            quiet = quiet_steps(timestep, t_max, self.io_rate, self.print_info_rate, self.checkpoint_rate)
            if quiet:
                state, timestep = advance_quiet_steps(self, state, timestep, quiet, self.advance, lambda s, t: self.step(s, t)[0])
                continue
            io_flag = self.io_rate > 0 and (timestep % self.io_rate == 0 or timestep == t_max)
            print_iter_flag = self.print_info_rate > 0 and timestep % self.print_info_rate == 0
            checkpoint_flag = self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev, u_prev = self.update_macroscopic(self.poststreaming(state))
                rho_prev = downsample_field(rho_prev, self.downsampling_factor)
                u_prev = downsample_field(u_prev, self.downsampling_factor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)

            # Perform one time-step (streaming, collision, and boundary conditions)
            state, fstar = self.step(state, timestep, io_flag and self.return_fpost)
            f = self.poststreaming(state) if (io_flag or checkpoint_flag) else None
            # Print the progress of the simulation
            if print_iter_flag:
                logger.info(
                    colored("Timestep ", "blue")
                    + colored(f"{timestep}", "green")
                    + colored(" of ", "blue")
                    + colored(f"{t_max}", "green")
                    + colored(" completed", "blue")
                )

            if io_flag:
                # Save the simulation data
                logger.info(f"Saving data at timestep {timestep}/{t_max}")
                rho, u = self.update_macroscopic(f)
                rho = downsample_field(rho, self.downsampling_factor)
                u = downsample_field(u, self.downsampling_factor)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, rho, u, rho_prev, u_prev)

            if checkpoint_flag:
                # Save the checkpoint
                logger.info(f"Saving checkpoint at timestep {timestep}/{t_max}")
                self.mngr.save(timestep, args=orb.args.StandardSave({"f": f}))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.compute_MLUPS and timestep == 1:
                jax.block_until_ready(state)
                start = time.time()
            timestep += 1

        if self.compute_MLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(state)
            end = time.time()
            n_voxels = self.nx * self.ny if self.dim == 2 else self.nx * self.ny * self.nz
            domain = f"{self.nx} x {self.ny}" if self.dim == 2 else f"{self.nx} x {self.ny} x {self.nz}"
            logger.info(colored("Domain: ", "blue") + colored(domain, "green"))
            logger.info(colored("Number of voxels: ", "blue") + colored(f"{n_voxels}", "green"))
            logger.info(colored("MLUPS: ", "blue") + colored(f"{n_voxels * t_max / (end - start) / 1e6}", "red"))
        self._output_writer.finish()
        if self.mngr is not None:
            self.mngr.wait_until_finished()
        return self.poststreaming(state)

    def handle_io_timestep(self, timestep, f, fstar, rho, u, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep (int): Current time step of the simulation.

        f (jax.numpy.ndarray): Post-streaming distribution functions at the current time step.

        fstar (jax.numpy.ndarray): Post-collision distribution functions at the current time step.

        rho (jax.numpy.ndarray): Density field at the current time step.

        u (jax.numpy.ndarray): Velocity field at the current time step.

        rho_prev (jax.numpy.ndarray): Density field at the previous I/O time step.

        u_prev (jax.numpy.ndarray): Velocity field at the previous I/O time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "u": u,
            "u_prev": u_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar,
        }
        self._output_writer.submit(self.output_data, kwargs)

    def output_data(self, **kwargs):
        """
        This function is intended to be overwritten by the user to customize the input/output (I/O)
        operations of the simulation.

        By default, it does nothing. When overwritten, it could save the simulation data to files,
        display the simulation results in real time, send the data to another process for analysis, etc.

        Parameters
        ----------
        **kwargs (dict): A dictionary containing the simulation data to be outputted. The keys are the names of the
        data fields, and the values are the data fields themselves.
        """
        pass

    def set_boundary_conditions(self):
        """
        This function sets the boundary conditions for the simulation.

        It is intended to be overwritten by the user to specify the boundary conditions according to
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip
        boundaries, inflow/outflow boundaries, etc.
        """
        pass

    @partial(jit, static_argnums=(0,))
    def collision(self, fin):
        """
        This function performs the collision step in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the collision operator according to
        the specific LBM model being used.

        By default, it does nothing. When overwritten, it could implement the BGK collision operator,
        the MRT collision operator, etc.

        Parameters
        ----------
        fin (jax.numpy.ndarray): Pre-collision distribution functions.

        Returns
        -------
        fin (jax.numpy.ndarray): Post-collision distribution functions.
        """
        pass

    @staticmethod
    def stack_populations(populations):
        """
        Stack post-collision populations into a single (..., q) array, with all q populations of a node computed together.

        Without the barrier, XLA fuses the whole collision into this stack (and into the per-direction fields that consume
        it) and emits one GPU thread per (node, direction) element, so every thread recomputes the node's macroscopic fields,
        equilibrium, forcing and relaxation for the single population it writes - q times the necessary arithmetic. The
        optimization barrier ends that fusion at the q populations, which XLA then computes as one multi-output fusion with
        one thread per node. The values are unchanged (same operations, same precision); only the kernel boundary moves.

        Parameters
        ----------
        populations (jax.numpy.ndarray or sequence of jax.numpy.ndarray): Either a (..., q) array or a sequence of q arrays
            of shape (...), already cast to the output precision.

        Returns
        -------
        (jax.numpy.ndarray): The populations stacked along the last axis.
        """
        if not isinstance(populations, (list, tuple)):
            populations = [populations[..., i] for i in range(populations.shape[-1])]
        return jnp.stack(lax.optimization_barrier(tuple(populations)), axis=-1)

    def sparse_collision_terms(self, collision_matrix):
        """
        Sparse per-direction columns of a fused MRT collision matrix K = M @ S @ M_inv, used by the symbolic MRT collisions.

        Column j lists the (input direction i, coefficient K[i, j]) pairs with a nonzero coefficient. Coefficients are kept in
        the compute precision (float64 for f64 policies, float32 otherwise). Entries that agree up to roundoff (relative
        1e-12, the noise left by the numerical inverse of M) are first set to one common value: K has only a handful of
        distinct values, and identical coefficients let XLA share the products difference_i * K[i, j] across output
        directions, which roundoff-distinct float64 coefficients would prevent.

        Parameters
        ----------
        collision_matrix (array-like): The (q, q) fused collision matrix (or any (n_in, n_out) matrix applied as x @ A).

        Returns
        -------
        (tuple): n_out tuples of (input_direction, coefficient) pairs, one per output column.
        """
        coefficient_dtype = np.float64 if self.precision_policy.compute_dtype == jnp.float64 else np.float32
        matrix = np.asarray(collision_matrix)
        flat = matrix.ravel().astype(np.float64)
        order = np.argsort(flat, kind="stable")
        snapped = flat.copy()
        start = 0
        for k in range(1, flat.size + 1):
            if k == flat.size or not np.isclose(flat[order[k]], flat[order[start]], rtol=1e-12, atol=0.0):
                group = order[start:k]
                if group.size > 1:
                    snapped[group] = flat[group].mean()
                start = k
        matrix = snapped.reshape(matrix.shape)
        n_in, n_out = matrix.shape
        return tuple(
            tuple(
                (input_direction, coefficient_dtype(matrix[input_direction, output_direction]))
                for input_direction in range(n_in)
                if not np.isclose(matrix[input_direction, output_direction], 0.0, atol=1e-7)
            )
            for output_direction in range(n_out)
        )

    def get_force(self):
        """
        This function computes the force to be applied to the fluid in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the force according to the specific
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a constant
        force term.

        Returns
        -------
        force (jax.numpy.ndarray): The force to be applied to the fluid.
        """
        pass

    def get_force_field(self, rho):
        """
        Velocity increment du of the body force, given the density field: used by apply_force and the fused collision
        kernels. By default it is get_force() (uniform, or whatever the user returns). When Thermal enables buoyancy
        (buoyancy_gravity is set) it is the density-deviation buoyancy plus the fluid force, the same formula as
        Thermal.apply_force_thermal - provided as a force field so that the fused per-direction collisions keep applying.

        Parameters
        ----------
        rho (jax.numpy.ndarray): Density field, shape (..., 1).

        Returns
        -------
        (jax.numpy.ndarray or None): du, broadcastable to (..., dim).
        """
        if self.buoyancy_gravity is None:
            return self.get_force()
        buoyancy = jnp.repeat(rho - rho.mean(), repeats=self.dim, axis=-1) * self.buoyancy_gravity
        return buoyancy + self.force

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, f_postcollision, feq, rho, u):
        """
        add force based on exact-difference method due to Kupershtokh

        Computes delta_feq = feq(rho, u + du) - feq(rho, u) directly from cu, dcu and delta_usqr instead of
        building a second full equilibrium distribution and subtracting feq from it. Lattice-generic: works for
        any lattice this class supports (D2Q9, D3Q19, D3Q27), since it only uses self.c/self.w.

        Parameters
        ----------
        f_postcollision (jax.numpy.ndarray): Post-collision distribution functions.

        feq (jax.numpy.ndarray): Equilibrium distribution functions. Unused - kept for interface compatibility
            with existing callers, since the compact difference formula only needs rho, u and the force.

        rho (jax.numpy.ndarray): Density field.

        u (jax.numpy.ndarray): Velocity field.

        Returns
        -------
        f_postcollision (jax.numpy.ndarray): Post-collision distribution functions with the force applied.

        References
        ----------
        1. Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann equation. In
        Proceedings of the 5th International EHD Workshop (pp. 241-246). University of Poitiers, Poitiers, France.
        Chikatamarla, S. S., & Karlin, I. V. (2013). Entropic lattice Boltzmann method for turbulent flow simulations:
        Boundary conditions. Physica A, 392, 1925-1930.
        2. Krüger, T., et al. (2017). The lattice Boltzmann method. Springer International Publishing, 10.978-3, 4-15.
        """
        du = self.get_force_field(rho)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        dcu = 3.0 * jnp.dot(du, c)
        delta_usqr = 1.5 * (2.0 * jnp.sum(u * du, axis=-1, keepdims=True) + jnp.sum(jnp.square(du), axis=-1, keepdims=True))
        delta_feq = rho * self.w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)
        return f_postcollision + delta_feq
