# Wind tunnel simulation using the XLB library

from functools import partial
from typing import Any
import os
import trimesh
from time import time
import numpy as np
import warp as wp
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import time
import xml.etree.ElementTree as ET
import itertools
from mpi4py import MPI

wp.init()

import xlb
from xlb.operator import Operator
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep

from amr import AMR, Block, Box
from memory_pool import MemoryPool
from operators import UniformInitializer, MomentumTransfer, QCriterion, GridToPoint, MyCopy
from utils import combine_vtks


class IncompressibleNavierStokesStepper(Stepper):
    """
    Fast NS stepper with only equilibrium and full way BC
    """

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(
            self.velocity_set.q, dtype=wp.uint8
        )  # TODO fix vec bool

        # Get the boundary condition ids
        _equilibrium_bc = wp.uint8(self.equilibrium_bc.id)
        _do_nothing_bc = wp.uint8(self.do_nothing_bc.id)
        _halfway_bounce_back_bc = wp.uint8(self.halfway_bounce_back_bc.id)
        _fullway_bounce_back_bc = wp.uint8(self.fullway_bounce_back_bc.id)

        # Construct the kernel
        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            boundary_id: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            timestep: int,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO warp should fix this

            # Get the boundary id and missing mask
            _boundary_id = boundary_id[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply streaming boundary conditions
            if (_boundary_id == wp.uint8(0)) or _boundary_id == _fullway_bounce_back_bc:
                # Regular streaming
                f_post_stream = self.stream.warp_functional(f_0, index)
            elif _boundary_id == _equilibrium_bc:
                # Equilibrium boundary condition
                f_post_stream = self.equilibrium_bc.warp_functional(
                    f_0, _missing_mask, index
                )

            # Compute rho and u
            rho, u = self.macroscopic.warp_functional(f_post_stream)

            # Compute equilibrium
            feq = self.equilibrium.warp_functional(rho, u)

            # Apply collision
            f_post_collision = self.collision.warp_functional(
                f_post_stream,
                feq,
                rho,
                u,
            )

            # Apply collision type boundary conditions
            if _boundary_id == _fullway_bounce_back_bc:
                # Full way boundary condition
                f_post_collision = self.fullway_bounce_back_bc.warp_functional(
                    f_post_stream,
                    f_post_collision,
                    _missing_mask,
                )

            # Set the output
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1], index[2]] = f_post_collision[l]

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, boundary_id, missing_mask, timestep):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f_0,
                f_1,
                boundary_id,
                missing_mask,
                timestep,
            ],
            dim=f_0.shape[1:],
        )
        return f_1

class LDC:
    """
    LDC
    """

    def __init__(
        self, 
        output_directory: str,
        base_velocity: float = 0.06, # m/s
        shape: int = 512,
        tau: float = 0.505,
        nr_steps: int = 1024,
        save_q_criterion_frequency: int = None,
        q_criterion_threshold: float = 1e-6,
        collision="BGK",
        equilibrium="Quadratic",
        velocity_set="D3Q19",
        use_amr=True,
        amr_block_shape=(128, 128, 128),
        amr_ghost_cell_thickness=8,
        nr_streams=3,
        comm=None,
    ):

        # Set parameters
        self.output_directory = output_directory
        self.base_velocity = base_velocity
        self.shape = (shape, shape, shape)
        self.tau = tau
        self.q_criterion_threshold = q_criterion_threshold
        self.use_amr = use_amr
        self.amr_block_shape = amr_block_shape
        self.amr_ghost_cell_thickness = amr_ghost_cell_thickness
        self.nr_streams = nr_streams
        self.comm = comm

        # Get fluid properties needed for the simulation
        self.omega = 1.0 / self.tau
        self.density = 1.0
        self.nr_steps = (nr_steps // self.amr_ghost_cell_thickness) * self.amr_ghost_cell_thickness # Make sure steps is divisible by ghost cell thickness
        if save_q_criterion_frequency is None:
            self.save_q_criterion_frequency = -1
        else:
            self.save_q_criterion_frequency = (save_q_criterion_frequency // self.amr_ghost_cell_thickness) * self.amr_ghost_cell_thickness

        # Make output directory
        os.makedirs(self.output_directory, exist_ok=True)

        # Make logging
        logging.basicConfig(level=logging.INFO)

        # Log the parameters
        logging.info(f"Base velocity: {self.base_velocity}")
        logging.info(f"Shape: {self.shape}")
        logging.info(f"Tau: {self.tau}")
        logging.info(f"Omega: {self.omega}")
        logging.info(f"Nr steps: {self.nr_steps}")
        logging.info(f"Save q criterion frequency: {self.save_q_criterion_frequency}")
        logging.info(f"Collision: {collision}")
        logging.info(f"Equilibrium: {equilibrium}")
        logging.info(f"Velocity set: {velocity_set}")
        logging.info(f"AMR block shape: {self.amr_block_shape}")
        logging.info(f"AMR ghost cell thickness: {self.amr_ghost_cell_thickness}")
        logging.info(f"Nr streams: {self.nr_streams}")

        # Set the compute backend NOTE: hard coded for now
        self.compute_backend = xlb.ComputeBackend.WARP

        # Set the precision policy NOTE: hard coded for now
        self.precision_policy = xlb.PrecisionPolicy.FP32FP32

        # Set the velocity set
        if velocity_set == "D3Q27":
            self.velocity_set = xlb.velocity_set.D3Q27()
        elif velocity_set == "D3Q19":
            self.velocity_set = xlb.velocity_set.D3Q19()
        else:
            raise ValueError("Invalid velocity set")

        # Make warp streams
        self.wp_streams = [wp.Stream() for _ in range(self.nr_streams)]

        # Make operators
        self.initializer = UniformInitializer(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        if collision == "BGK":
            self.collision = xlb.operator.collision.BGK(
                omega=self.omega,
                velocity_set=self.velocity_set,
                precision_policy=self.precision_policy,
                compute_backend=self.compute_backend,
            )
        elif collision == "KBC":
            self.collision = xlb.operator.collision.KBC(
                omega=self.omega,
                velocity_set=self.velocity_set,
                precision_policy=self.precision_policy,
                compute_backend=self.compute_backend,
            )
        elif collision == "SmagorinskyLESBGK":
            self.collision = xlb.operator.collision.SmagorinskyLESBGK(
                omega=self.omega,
                velocity_set=self.velocity_set,
                precision_policy=self.precision_policy,
                compute_backend=self.compute_backend,
            )
        self.equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.macroscopic = xlb.operator.macroscopic.Macroscopic(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.stream = xlb.operator.stream.Stream(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.equilibrium_bc = xlb.operator.boundary_condition.EquilibriumBC(
            rho=self.density,
            u=(0.0, self.base_velocity, 0.0),
            equilibrium_operator=self.equilibrium,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.full_way_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.stepper = IncompressibleNavierStokesStepper(
            collision=self.collision,
            equilibrium=self.equilibrium,
            macroscopic=self.macroscopic,
            stream=self.stream,
            boundary_conditions=[
                self.full_way_bc,
                self.equilibrium_bc,
            ],
        )
        self.q_criterion = QCriterion(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.grid_to_point = GridToPoint(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.my_copy = MyCopy( # NOTE Removes performance issues with 4d wp.copy of non-contiguous arrays
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.planar_boundary_masker = xlb.operator.boundary_masker.PlanarBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.stl_boundary_masker = xlb.operator.boundary_masker.STLBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

        # Make AMR
        if self.use_amr:
            self.amr = AMR(
                shape=self.shape,
                block_shape=amr_block_shape,
                nr_levels=1,
                ghost_cell_thickness=amr_ghost_cell_thickness,
                comm=comm,
            )

            # Make memory pool for each stream
            self.memory_pools = [MemoryPool() for _ in range(self.nr_streams)]

            # Initialize boxes for the AMR
            self.amr.initialize_boxes(
                name="f",
                dtype=wp.float32,
                cardinality=self.velocity_set.q,
                ordering="SOA",
            )
            self.amr.initialize_boxes(
                name="boundary_id",
                dtype=wp.uint8,
                cardinality=1,
                ordering="SOA",
            )
            self.amr.initialize_boxes(
                name="missing_mask",
                dtype=wp.bool,
                cardinality=self.velocity_set.q,
                ordering="SOA",
            )

    def allocate(self):
        """
        Allocate the flow field
        """

        # Allocate amr
        self.amr.allocate()

    def save_vtm(
        self,
        file_name: str="initial.vtm",
    ):
        """
        Save the solid id array.
        """
        file_name = os.path.join(self.output_directory, file_name)
        self.amr.save_vtm(file_name)

    def initialize(self):
        """
        Initialize the flow field

        # TODO: Memory limited so just using 1 stream right now
        """

        # Make stream idx
        stream_idx = 0

        # MPI communication parameters
        comm_tag = 0
        requests = []

        # Set initial conditions
        logging.info("Setting initial conditions")
        for block in self.amr.blocks:

            # Set warp stream
            with wp.ScopedStream(self.wp_streams[stream_idx]):

                # Check if block matches pid 
                if block.pid == self.amr.pid:

                    # Get compute arrays
                    rho = self.memory_pools[stream_idx].get((1, *block.shape), wp.float32)
                    u = self.memory_pools[stream_idx].get((3, *block.shape), wp.float32)
                    f = self.memory_pools[stream_idx].get((self.velocity_set.q, *block.shape), wp.float32)
                    boundary_id = self.memory_pools[stream_idx].get((1, *block.shape), wp.uint8)
                    missing_mask = self.memory_pools[stream_idx].get((self.velocity_set.q, *block.shape), wp.bool)
                    f_ghost = {}
                    boundary_id_ghost = {}
                    missing_mask_ghost = {}
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        f_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (self.velocity_set.q, *ghost_boxes["f"].shape),
                            wp.float32
                        )
                        boundary_id_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (1, *ghost_boxes["boundary_id"].shape),
                            wp.uint8
                        )
                        missing_mask_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (self.velocity_set.q, *ghost_boxes["missing_mask"].shape),
                            wp.bool
                        )

                    # Initialize Inlet bc (bottom x face)
                    lower_bound = (0, 1, 1) # no edges
                    upper_bound = (0, self.shape[1]-1, self.shape[2]-1)
                    direction = (1, 0, 0)
                    boundary_id, missing_mask = self.planar_boundary_masker(
                        lower_bound,
                        upper_bound,
                        direction,
                        self.equilibrium_bc.id,
                        boundary_id,
                        missing_mask,
                        block.offset,
                    )
        
                    # Set full way bc (top x face)
                    lower_bound = (self.shape[0]-1, 1, 1)
                    upper_bound = (self.shape[0]-1, self.shape[1]-1, self.shape[2]-1)
                    boundary_id, missing_mask = self.planar_boundary_masker(
                        lower_bound,
                        upper_bound,
                        direction,
                        self.full_way_bc.id,
                        boundary_id,
                        missing_mask,
                        block.offset,
                    )
    
                    # Set full way bc (bottom y face)
                    lower_bound = (0, 0, 0)
                    upper_bound = (self.shape[0], 0, self.shape[2])
                    direction = (0, 1, 0)
                    boundary_id, missing_mask = self.planar_boundary_masker(
                        lower_bound,
                        upper_bound,
                        direction,
                        self.full_way_bc.id,
                        boundary_id,
                        missing_mask,
                        block.offset,
                    )
                    
                    # Set full way bc (top y face)
                    lower_bound = (0, self.shape[1]-1, 0)
                    upper_bound = (self.shape[0], self.shape[1]-1, self.shape[2])
                    direction = (0, -1, 0)
                    boundary_id, missing_mask = self.planar_boundary_masker(
                        lower_bound,
                        upper_bound,
                        direction,
                        self.full_way_bc.id,
                        boundary_id,
                        missing_mask,
                        block.offset,
                    )
        
                    # Set full way bc (bottom z face)
                    lower_bound = (0, 0, 0)
                    upper_bound = (self.shape[0], self.shape[1], 0)
                    direction = (0, 0, 1)
                    boundary_id, missing_mask = self.planar_boundary_masker(
                        lower_bound,
                        upper_bound,
                        direction,
                        self.full_way_bc.id,
                        boundary_id,
                        missing_mask,
                        block.offset,
                    )
        
                    # Set full way bc (top z face)
                    lower_bound = (0, 0, self.shape[2]-1)
                    upper_bound = (self.shape[0], self.shape[1], self.shape[2]-1)
                    direction = (0, 0, -1)
                    boundary_id, missing_mask = self.planar_boundary_masker(
                        lower_bound,
                        upper_bound,
                        direction,
                        self.full_way_bc.id,
                        boundary_id,
                        missing_mask,
                        block.offset,
                    )
    
                    # Initialize the flow field
                    rho, u = self.initializer(rho, u, boundary_id, self.base_velocity)
                    f = self.equilibrium(rho, u, f)

                    # Copy to local ghost boxes
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():

                        # Get slice start and stop
                        slice_start = ghost_boxes["f"].offset - block.offset
                        slice_stop = slice_start + ghost_boxes["f"].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])

                        # Copy
                        self.my_copy(
                            f_ghost[ghost_block],
                            f[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )
                        self.my_copy(
                            boundary_id_ghost[ghost_block],
                            boundary_id[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )
                        self.my_copy(
                            missing_mask_ghost[ghost_block],
                            missing_mask[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )

                    # Copy to block
                    wp.copy(block.boxes["f"].data, f)
                    wp.copy(block.boxes["boundary_id"].data, boundary_id)
                    wp.copy(block.boxes["missing_mask"].data, missing_mask)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        wp.copy(ghost_boxes["f"].data, f_ghost[ghost_block])
                        wp.copy(ghost_boxes["boundary_id"].data, boundary_id_ghost[ghost_block])
                        wp.copy(ghost_boxes["missing_mask"].data, missing_mask_ghost[ghost_block])

                    # Return arrays
                    self.memory_pools[stream_idx].ret(rho, zero=True)
                    self.memory_pools[stream_idx].ret(u, zero=True)
                    self.memory_pools[stream_idx].ret(f, zero=True)
                    self.memory_pools[stream_idx].ret(boundary_id, zero=True)
                    self.memory_pools[stream_idx].ret(missing_mask, zero=True)
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        self.memory_pools[stream_idx].ret(f_ghost[ghost_block], zero=True)
                        self.memory_pools[stream_idx].ret(boundary_id_ghost[ghost_block], zero=True)
                        self.memory_pools[stream_idx].ret(missing_mask_ghost[ghost_block], zero=True)

                    # Update stream idx
                    stream_idx = (stream_idx + 1) % self.nr_streams

                ## Send ghost boxes
                #wp.synchronize() # TODO remove this
                #r, comm_tag = block.send_ghost_boxes(
                #    self.amr.comm,
                #    comm_tag=comm_tag,
                #    names=["f", "boundary_id", "missing_mask"],
                #)
                #requests.extend(r)

        # Send blocks
        wp.synchronize()
        for block in self.amr.blocks:
            r, comm_tag = block.send_ghost_boxes(
                self.amr.comm,
                comm_tag=comm_tag,
                names=["f", "boundary_id", "missing_mask"],
            )
            requests.extend(r)

        # Wait for requests
        if self.amr.comm is not None:
            self.comm.Barrier()
            MPI.Request.Waitall(requests)
            pass
        else:
            assert len(requests) == 0

        # Swap neighbour buffers
        for block in self.amr.blocks:
            if block.pid == self.amr.pid:
                block.swap_buffers()

        # Clear memory pools
        for memory_pool in self.memory_pools:
            memory_pool.clear()

    def step(self, debug=False):

        # Make stream idx
        stream_idx = 0

        # MPI communication parameters
        comm_tag = 0
        requests = []

        # Make event
        event = None

        # Keep track of previous block
        previous_block = None

        # Set Perform steps equal to the number of ghost cell thickness
        for block in self.amr.blocks:

            # Set warp stream
            with wp.ScopedStream(self.wp_streams[stream_idx]):

                # Check if block matches pid 
                if block.pid == self.amr.pid:

                    # Get total box offset and extent
                    offset = block.offset_with_ghost
                    extent = block.extent_with_ghost
                    #offset = block.offset
                    #bound = block.offset + block.extent
                    #for neighbour_boxes in block.neighbour_ghost_boxes.values():
                    #    box = neighbour_boxes["f"]
                    #    offset = np.minimum(offset, box.offset)
                    #    bound = np.maximum(bound, box.offset + box.extent)
                    #extent = bound - offset

                    # Get compute arrays
                    f0 = self.memory_pools[stream_idx].get((self.velocity_set.q, *extent), wp.float32)
                    f1 = self.memory_pools[stream_idx].get((self.velocity_set.q, *extent), wp.float32)
                    boundary_id = self.memory_pools[stream_idx].get((1, *extent), wp.uint8)
                    missing_mask = self.memory_pools[stream_idx].get((self.velocity_set.q, *extent), wp.bool)
                    f_block = self.memory_pools[stream_idx].get((self.velocity_set.q, *block.shape), wp.float32)
                    boundary_id_block = self.memory_pools[stream_idx].get((1, *block.shape), wp.uint8)
                    missing_mask_block = self.memory_pools[stream_idx].get((self.velocity_set.q, *block.shape), wp.bool)
                    f_ghost = {}
                    boundary_id_ghost = {}
                    missing_mask_ghost = {}
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        f_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (self.velocity_set.q, *ghost_boxes["f"].shape),
                            wp.float32
                        )
                        boundary_id_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (1, *ghost_boxes["boundary_id"].shape),
                            wp.uint8
                        )
                        missing_mask_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (self.velocity_set.q, *ghost_boxes["missing_mask"].shape),
                            wp.bool
                        )

                    # Copy from block
                    wp.copy(
                        f_block,
                        block.boxes["f"].data
                    )
                    wp.copy(
                        boundary_id_block,
                        block.boxes["boundary_id"].data
                    )
                    wp.copy(
                        missing_mask_block,
                        block.boxes["missing_mask"].data
                    )
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        wp.copy(
                            f_ghost[ghost_block],
                            ghost_boxes["f"].data
                        )
                        wp.copy(
                            boundary_id_ghost[ghost_block],
                            ghost_boxes["boundary_id"].data
                        )
                        wp.copy(
                            missing_mask_ghost[ghost_block],
                            ghost_boxes["missing_mask"].data
                        )

                    # Wait for previous event
                    if event is not None:
                        self.wp_streams[stream_idx].wait_event(event)

                    # Copy to compute arrays
                    self.my_copy(
                        f0[
                            :,
                            int(block.offset[0] - offset[0]):int(block.offset[0] - offset[0] + block.extent[0]),
                            int(block.offset[1] - offset[1]):int(block.offset[1] - offset[1] + block.extent[1]),
                            int(block.offset[2] - offset[2]):int(block.offset[2] - offset[2] + block.extent[2]),
                        ],
                        f_block
                    )
                    self.my_copy(
                        boundary_id[
                            :,
                            int(block.offset[0] - offset[0]):int(block.offset[0] - offset[0] + block.extent[0]),
                            int(block.offset[1] - offset[1]):int(block.offset[1] - offset[1] + block.extent[1]),
                            int(block.offset[2] - offset[2]):int(block.offset[2] - offset[2] + block.extent[2]),
                        ],
                        boundary_id_block
                    )
                    self.my_copy(
                        missing_mask[
                            :,
                            int(block.offset[0] - offset[0]):int(block.offset[0] - offset[0] + block.extent[0]),
                            int(block.offset[1] - offset[1]):int(block.offset[1] - offset[1] + block.extent[1]),
                            int(block.offset[2] - offset[2]):int(block.offset[2] - offset[2] + block.extent[2]),
                        ],
                        missing_mask_block
                    )
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        self.my_copy(
                            f0[
                                :,
                                int(ghost_boxes["f"].offset[0] - offset[0]):int(ghost_boxes["f"].offset[0] - offset[0] + ghost_boxes["f"].shape[0]),
                                int(ghost_boxes["f"].offset[1] - offset[1]):int(ghost_boxes["f"].offset[1] - offset[1] + ghost_boxes["f"].shape[1]),
                                int(ghost_boxes["f"].offset[2] - offset[2]):int(ghost_boxes["f"].offset[2] - offset[2] + ghost_boxes["f"].shape[2]),
                            ],
                            f_ghost[ghost_block]
                        )
                        self.my_copy(
                            boundary_id[
                                :,
                                int(ghost_boxes["boundary_id"].offset[0] - offset[0]):int(ghost_boxes["boundary_id"].offset[0] - offset[0] + ghost_boxes["boundary_id"].shape[0]),
                                int(ghost_boxes["boundary_id"].offset[1] - offset[1]):int(ghost_boxes["boundary_id"].offset[1] - offset[1] + ghost_boxes["boundary_id"].shape[1]),
                                int(ghost_boxes["boundary_id"].offset[2] - offset[2]):int(ghost_boxes["boundary_id"].offset[2] - offset[2] + ghost_boxes["boundary_id"].shape[2]),
                            ],
                            boundary_id_ghost[ghost_block]
                        )
                        self.my_copy(
                            missing_mask[
                                :,
                                int(ghost_boxes["missing_mask"].offset[0] - offset[0]):int(ghost_boxes["missing_mask"].offset[0] - offset[0] + ghost_boxes["missing_mask"].shape[0]),
                                int(ghost_boxes["missing_mask"].offset[1] - offset[1]):int(ghost_boxes["missing_mask"].offset[1] - offset[1] + ghost_boxes["missing_mask"].shape[1]),
                                int(ghost_boxes["missing_mask"].offset[2] - offset[2]):int(ghost_boxes["missing_mask"].offset[2] - offset[2] + ghost_boxes["missing_mask"].shape[2]),
                            ],
                            missing_mask_ghost[ghost_block]
                        )

                    # Perform update
                    for _ in range(self.amr_ghost_cell_thickness):

                        # Perform stepper
                        f1 = self.stepper(f0, f1, boundary_id, missing_mask, 0)
                        f0, f1 = f1, f0

                    # Copy from compute arrays
                    self.my_copy(
                        f_block,
                        f0[
                            :,
                            int(block.offset[0] - offset[0]):int(block.offset[0] - offset[0] + block.extent[0]),
                            int(block.offset[1] - offset[1]):int(block.offset[1] - offset[1] + block.extent[1]),
                            int(block.offset[2] - offset[2]):int(block.offset[2] - offset[2] + block.extent[2]),
                        ]
                    )
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():

                        # Get slice start and stop
                        slice_start = ghost_boxes["f"].offset - block.offset
                        slice_stop = slice_start + ghost_boxes["f"].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])

                        # Copy f
                        self.my_copy(
                            f_ghost[ghost_block],
                            f_block[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )

                    # Wait for previous event
                    if event is None:
                        event = wp.Event()
                    self.wp_streams[stream_idx].record_event(event)

                    # Copy to block
                    wp.copy(
                        block.boxes["f"].data,
                        f_block
                    )
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        wp.copy(
                            ghost_boxes["f"].data,
                            f_ghost[ghost_block]
                        )
 
                    # Return arrays
                    self.memory_pools[stream_idx].ret(f0, zero=False)
                    self.memory_pools[stream_idx].ret(f1, zero=False)
                    self.memory_pools[stream_idx].ret(boundary_id, zero=False)
                    self.memory_pools[stream_idx].ret(missing_mask, zero=False)
                    self.memory_pools[stream_idx].ret(f_block, zero=False)
                    self.memory_pools[stream_idx].ret(boundary_id_block, zero=False)
                    self.memory_pools[stream_idx].ret(missing_mask_block, zero=False)
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        self.memory_pools[stream_idx].ret(f_ghost[ghost_block], zero=False)
                        self.memory_pools[stream_idx].ret(boundary_id_ghost[ghost_block], zero=False)
                        self.memory_pools[stream_idx].ret(missing_mask_ghost[ghost_block], zero=False)

                    # Update stream idx
                    stream_idx = (stream_idx + 1) % self.nr_streams

                ## Send previous ghost boxes
                #if previous_block is not None:
                #    wp.synchronize()
                #    r, comm_tag = previous_block.send_ghost_boxes(
                #        self.amr.comm,
                #        comm_tag=comm_tag,
                #        names=["f"],
                #    )
                #    requests.extend(r)
                #previous_block = block

        # Send blocks
        wp.synchronize()
        for block in self.amr.blocks:
            r, comm_tag = block.send_ghost_boxes(
                self.amr.comm,
                comm_tag=comm_tag,
                names=["f"],
            )
            requests.extend(r)

        # Wait for requests
        if self.amr.comm is not None:
            self.comm.Barrier()
            MPI.Request.Waitall(requests)
            pass
        else:
            assert len(requests) == 0

        # Swap neighbour buffers
        for block in self.amr.blocks:
            if block.pid == self.amr.pid:
                block.swap_buffers(names=["f"])

    def save_q_criterion(self, file_name: str):

        # Make directory
        os.makedirs(os.path.join(self.output_directory, file_name), exist_ok=True)

        # Clear memory pools
        for memory_pool in self.memory_pools:
            memory_pool.clear()

        # Store the files
        files = []

        # Loop over blocks
        for idx, block in enumerate(self.amr.blocks):

            # Check if block matches pid 
            if block.pid != self.amr.pid:
                continue

            # TODO: remove this
            extent = tuple([int(e) for e in block.extent])

            # Get compute arrays
            f = self.memory_pools[0].get((self.velocity_set.q, *extent), wp.float32)
            boundary_id = self.memory_pools[0].get((1, *extent), wp.uint8)
            rho = self.memory_pools[0].get((1, *extent), wp.float32)
            u = self.memory_pools[0].get((3, *extent), wp.float32)
            norm_mu = self.memory_pools[0].get((1, *extent), wp.float32)
            q = self.memory_pools[0].get((1, *extent), wp.float32)

            # Get marching cubes arrays
            mc = wp.MarchingCubes(
                nx=extent[0],
                ny=extent[1],
                nz=extent[2],
                max_verts=extent[0] * extent[1] * extent[2] * 5,
                max_tris=extent[0] * extent[1] * extent[2] * 3,
            )

            # Copy from block
            wp.copy(
                f,
                block.boxes["f"].data
            )
            wp.copy(
                boundary_id,
                block.boxes["boundary_id"].data
            )

            # Compute q criterion
            rho, u = self.macroscopic(f, rho, u)
            norm_mu, q = self.q_criterion(u, boundary_id, norm_mu, q)

            # Perform marching cubes
            mc.surface(q[0], self.q_criterion_threshold)

            # Get point data
            scalars = wp.zeros((mc.verts.shape[0]), wp.float32)
            scalars = self.grid_to_point(norm_mu, mc.verts, scalars)

            # Check if there are any vertices
            if mc.verts.shape[0] == 0:
                continue

            # Save the marching cubes
            vertices = mc.verts.numpy()
            vertices = vertices + block.offset
            indices = mc.indices.numpy()
            indices = indices.reshape(-1, 3)
            faces_flat = np.concatenate([
                np.full((indices.shape[0], 1), 3, dtype=np.uint32),
                indices
            ], axis=1).flatten().astype(np.uint32)

            # Create a PyVista PolyData object
            poly_data = pv.PolyData(vertices, faces_flat)

            # Add scalars to the PolyData object
            poly_data.point_data["norm_vort"] = scalars.numpy()

            # Save the PolyData object with colors to a VTK file
            vtk_file_name = os.path.join(file_name, f"{file_name}_{idx}.vtp")
            poly_data.save(os.path.join(self.output_directory, vtk_file_name))
            files.append(vtk_file_name)

            # Return arrays
            self.memory_pools[0].ret(f)
            self.memory_pools[0].ret(rho)
            self.memory_pools[0].ret(u)
            self.memory_pools[0].ret(norm_mu)
            self.memory_pools[0].ret(q)

        # Clear memory pools
        for memory_pool in self.memory_pools:
            memory_pool.clear()

        # Check if no files
        if files == []:
            return

        # Get all the files
        if self.comm is not None:
            files = self.comm.gather(files, root=0)
            if self.comm.rank == 0:
                files = list(itertools.chain(*files))

        # Combine the files
        if self.comm is not None:
            if self.comm.rank == 0:
                combine_vtks(files, os.path.join(self.output_directory, f"{file_name}.vtm"))
        else:
            combine_vtks(files, os.path.join(self.output_directory, f"{file_name}.vtm"))

    def run_non_amr(self):
        # Make arrays
        rho = wp.zeros((1, *self.shape), wp.float32)
        u = wp.zeros((3, *self.shape), wp.float32)
        f0 = wp.zeros((self.velocity_set.q, *self.shape), wp.float32)
        f1 = wp.zeros((self.velocity_set.q, *self.shape), wp.float32)
        boundary_id = wp.zeros((1, *self.shape), wp.uint8)
        missing_mask = wp.zeros((self.velocity_set.q, *self.shape), wp.bool)

        # Initialize Inlet bc (bottom x face)
        lower_bound = (0, 1, 1) # no edges
        upper_bound = (0, self.shape[1]-1, self.shape[2]-1)
        direction = (1, 0, 0)
        boundary_id, missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.equilibrium_bc.id,
            boundary_id,
            missing_mask,
            (0, 0, 0),
        )
        
        # Set full way bc (top x face)
        lower_bound = (self.shape[0]-1, 1, 1)
        upper_bound = (self.shape[0]-1, self.shape[1]-1, self.shape[2]-1)
        boundary_id, missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            boundary_id,
            missing_mask,
            (0, 0, 0),
        )
    
        # Set full way bc (bottom y face)
        lower_bound = (0, 0, 0)
        upper_bound = (self.shape[0], 0, self.shape[2])
        direction = (0, 1, 0)
        boundary_id, missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            boundary_id,
            missing_mask,
            (0, 0, 0),
        )
        
        # Set full way bc (top y face)
        lower_bound = (0, self.shape[1]-1, 0)
        upper_bound = (self.shape[0], self.shape[1]-1, self.shape[2])
        direction = (0, -1, 0)
        boundary_id, missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            boundary_id,
            missing_mask,
            (0, 0, 0),
        )
        
        # Set full way bc (bottom z face)
        lower_bound = (0, 0, 0)
        upper_bound = (self.shape[0], self.shape[1], 0)
        direction = (0, 0, 1)
        boundary_id, missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            boundary_id,
            missing_mask,
            (0, 0, 0),
        )
        
        # Set full way bc (top z face)
        lower_bound = (0, 0, self.shape[2]-1)
        upper_bound = (self.shape[0], self.shape[1], self.shape[2]-1)
        direction = (0, 0, -1)
        boundary_id, missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.full_way_bc.id,
            boundary_id,
            missing_mask,
            (0, 0, 0),
        )
    
        # Initialize the flow field
        rho, u = self.initializer(rho, u, boundary_id, self.base_velocity)
        f0 = self.equilibrium(rho, u, f0)

        # Warm up
        for i in range(8 * self.amr_ghost_cell_thickness):
            f1 = self.stepper(f0, f1, boundary_id, missing_mask, 0)
            f0, f1 = f1, f0

        # Start timer
        wp.synchronize()
        start = time.time()
        for i in tqdm(range(self.nr_steps)):
            f1 = self.stepper(f0, f1, boundary_id, missing_mask, 0)
            f0, f1 = f1, f0
        wp.synchronize()
        end = time.time()

        # Compute MLUPS
        nr_cells = self.shape[0] * self.shape[1] * self.shape[2]
        nr_updates = self.nr_steps * nr_cells
        mlups = nr_updates / (end - start) / 1e6
        logging.info(f"MLUPS: {mlups}")
        return mlups

    def run_amr_mlups(self):

        # Allocate amr fields
        self.allocate()

        # Initialize the boundary conditions and flow field
        self.initialize()

        # Take a few steps to warm up
        for i in range(8):
            self.step()

        # Perform update
        wp.synchronize()
        start = time.time()
        for i in tqdm(range(self.nr_steps // self.amr_ghost_cell_thickness)):

            # Step
            self.step()

            # Save q
            if self.save_q_criterion_frequency != -1 and i * self.amr_ghost_cell_thickness % self.save_q_criterion_frequency == 0:
                self.save_q_criterion(f"q_criterion_{i * self.amr_ghost_cell_thickness}")
                wp.synchronize()
                end = time.time()
                nr_cells = self.shape[0] * self.shape[1] * self.shape[2]
                nr_updates = nr_cells * (i + 1) * self.amr_ghost_cell_thickness
                mlups = nr_updates / (end - start) / 1e6
                logging.info(f"MLUPS: {mlups}")
 
        wp.synchronize()
        if self.comm is not None:
            self.comm.Barrier()
        end = time.time()

        # Compute MLUPS
        nr_cells = self.shape[0] * self.shape[1] * self.shape[2]
        nr_updates = self.nr_steps * nr_cells
        mlups = nr_updates / (end - start) / 1e6
        logging.info(f"MLUPS: {mlups}")
        return mlups

    def run(self):
        if self.use_amr:
            return self.run_amr_mlups()
        else:
            return self.run_non_amr()
