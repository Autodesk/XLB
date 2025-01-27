# Wind tunnel simulation using the XLB library

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

wp.set_mempool_release_threshold("cuda:0", 0.95)
wp.init()

import xlb

from amr import AMR, Block, Box
from memory_pool import MemoryPool
from operators import UniformInitializer, MomentumTransfer, QCriterion, GridToPoint, MyCopy
from utils import combine_vtks

class WindTunnel:
    """
    Wind tunnel simulation using the XLB library
    """

    def __init__(
        self, 
        stl_filename: str,
        output_directory: str,
        base_velocity: float = 0.0289, # m/s
        no_slip_walls: bool = True,
        inlet_velocity: float = 27.78, # m/s
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0), # m
        dx: float = 0.01, # m
        shape: tuple[int, int, int] = (512, 128, 128),
        viscosity: float = 1.42e-5, # air at 20 degrees Celsius
        density: float = 1.2754, # kg/m^3
        solve_time: float = 1.0, # s
        save_q_criterion_frequency: float = 0.01, # s
        q_criterion_threshold: float = 1e-5,
        save_drag_coefficient_frequency: float = 0.01, # s
        collision="SmagorinskyLESBGK",
        equilibrium="Quadratic",
        velocity_set="D3Q27",
        amr_block_shape=(128, 128, 128),
        amr_ghost_cell_thickness=8,
        nr_streams=3,
        comm=None,
    ):

        # Set parameters
        self.stl_filename = stl_filename
        self.output_directory = output_directory
        self.no_slip_walls = no_slip_walls
        self.inlet_velocity = inlet_velocity
        self.origin = origin
        self.dx = dx
        self.spacing = (dx, dx, dx)
        self.shape = shape
        self.solve_time = solve_time
        self.viscosity = viscosity
        self.density = density
        self.save_q_criterion_frequency = save_q_criterion_frequency
        self.q_criterion_threshold = q_criterion_threshold
        self.save_drag_coefficient_frequency = save_drag_coefficient_frequency
        self.amr_block_shape = amr_block_shape
        self.amr_ghost_cell_thickness = amr_ghost_cell_thickness
        self.nr_streams = nr_streams
        self.comm = comm

        # Get fluid properties needed for the simulation
        self.base_velocity = base_velocity
        self.velocity_conversion = self.base_velocity / inlet_velocity
        self.dt = self.dx * self.velocity_conversion
        self.lbm_viscosity = self.viscosity * self.dt / (self.dx ** 2)
        self.tau = 0.5 + 3.0 * self.lbm_viscosity # tau = 0.5 + 3 * viscosity
        self.omega = 1.0 / self.tau
        self.lbm_density = 1.0
        self.mass_conversion = self.dx ** 3 * (self.density / self.lbm_density)
        self.nr_steps = int(solve_time / self.dt)
        self.nr_steps_save_q_criterion = int(self.save_q_criterion_frequency / self.dt)
        self.nr_steps_save_drag_coefficient = int(self.save_drag_coefficient_frequency / self.dt)

        # Make output directory
        os.makedirs(self.output_directory, exist_ok=True)

        # Make logging
        logging.basicConfig(level=logging.INFO)

        # Log the parameters
        logging.info(f"STL filename: {self.stl_filename}")
        logging.info(f"Base velocity: {self.base_velocity}")
        logging.info(f"No slip walls: {self.no_slip_walls}")
        logging.info(f"Inlet velocity: {self.inlet_velocity}")
        logging.info(f"Origin: {self.origin}")
        logging.info(f"dx: {self.dx}")
        logging.info(f"Shape: {self.shape}")
        logging.info(f"Solve time: {self.solve_time}")
        logging.info(f"Viscosity: {self.viscosity}")
        logging.info(f"Density: {self.density}")
        logging.info(f"Collision: {collision}")
        logging.info(f"Equilibrium: {equilibrium}")
        logging.info(f"Velocity set: {velocity_set}")
        logging.info(f"AMR block shape: {amr_block_shape}")
        logging.info(f"AMR ghost cell thickness: {amr_ghost_cell_thickness}")
        logging.info(f"Comm: {comm}")
        logging.info(f"Base velocity: {self.base_velocity}")
        logging.info(f"Velocity conversion: {self.velocity_conversion}")
        logging.info(f"dt: {self.dt}")
        logging.info(f"LBM viscosity: {self.lbm_viscosity}")
        logging.info(f"Tau: {self.tau}")
        logging.info(f"LBM density: {self.lbm_density}")
        logging.info(f"Mass conversion: {self.mass_conversion}")
        logging.info(f"Nr steps: {self.nr_steps}")
        logging.info(f"Million cells: {self.shape[0] * self.shape[1] * self.shape[2] / 1e6}")

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

        # Make AMR
        self.amr = AMR(
            shape=self.shape,
            block_shape=amr_block_shape,
            origin=self.origin,
            spacing=self.spacing,
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
            rho=self.lbm_density,
            u=(self.base_velocity, 0.0, 0.0),
            equilibrium_operator=self.equilibrium,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.half_way_bc = xlb.operator.boundary_condition.HalfwayBounceBackBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.full_way_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.do_nothing_bc = xlb.operator.boundary_condition.DoNothingBC(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )
        self.stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
            collision=self.collision,
            equilibrium=self.equilibrium,
            macroscopic=self.macroscopic,
            stream=self.stream,
            boundary_conditions=[
                self.half_way_bc,
                self.full_way_bc,
                self.equilibrium_bc,
                self.do_nothing_bc
            ],
        )
        self.momentum_transfer = MomentumTransfer(
            halfway_bounce_back=self.half_way_bc,
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
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
        #self.my_copy = MyCopy(
        #    velocity_set=self.velocity_set,
        #    precision_policy=self.precision_policy,
        #    compute_backend=self.compute_backend,
        #)
        self.my_copy = wp.copy
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

        # Make list to store drag coefficients
        self.time_steps = []
        self.drag_coefficients = []

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

            # Check if block matches pid 
            if block.pid == self.amr.pid:

                # Set warp stream
                with wp.ScopedStream(self.wp_streams[stream_idx]):

                    # Get arrays
                    rho = wp.zeros((1, *block.shape), wp.float32)
                    u = wp.zeros((3, *block.shape), wp.float32)
                    f = wp.zeros((self.velocity_set.q, *block.shape), wp.float32)
                    boundary_id = wp.zeros((1, *block.shape), wp.uint8)
                    missing_mask = wp.zeros((self.velocity_set.q, *block.shape), wp.bool)
                    f_ghost = {}
                    boundary_id_ghost = {}
                    missing_mask_ghost = {}
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        f_ghost[ghost_block] = wp.zeros(
                            (self.velocity_set.q, *ghost_boxes["f"].shape),
                            wp.float32
                        )
                        boundary_id_ghost[ghost_block] = wp.zeros(
                            (1, *ghost_boxes["boundary_id"].shape),
                            wp.uint8
                        )
                        missing_mask_ghost[ghost_block] = wp.zeros(
                            (self.velocity_set.q, *ghost_boxes["missing_mask"].shape),
                            wp.bool
                        )

                    # Initialize Inlet bc (bottom x face)
                    if self.no_slip_walls:
                        lower_bound = (0, 1, 1) # no edges
                        upper_bound = (0, self.shape[1]-1, self.shape[2]-1)
                    else:
                        lower_bound = (0, 0, 0) # edges
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
        
                    # Set outlet bc (top x face)
                    if self.no_slip_walls:
                        lower_bound = (self.shape[0]-1, 1, 1)
                        upper_bound = (self.shape[0]-1, self.shape[1]-1, self.shape[2]-1)
                    else:
                        lower_bound = (self.shape[0]-1, 0, 0)
                        upper_bound = (self.shape[0]-1, self.shape[1]-1, self.shape[2]-1)
                    boundary_id, missing_mask = self.planar_boundary_masker(
                        lower_bound,
                        upper_bound,
                        direction,
                        self.do_nothing_bc.id,
                        boundary_id,
                        missing_mask,
                        block.offset,
                    )
    
                    # Set no slip walls
                    if self.no_slip_walls:
    
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
    
                    # Set stl half way bc
                    boundary_id, missing_mask = self.stl_boundary_masker(
                        self.stl_filename,
                        self.origin,
                        self.spacing,
                        self.half_way_bc.id,
                        #self.full_way_bc.id,
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
                        wp.copy(
                            f_ghost[ghost_block],
                            f[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )
                        wp.copy(
                            boundary_id_ghost[ghost_block],
                            boundary_id[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )
                        wp.copy(
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

                    # Delete arrays
                    del rho
                    del u
                    del f
                    del boundary_id
                    del missing_mask
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        del f_ghost[ghost_block]
                        del boundary_id_ghost[ghost_block]
                        del missing_mask_ghost[ghost_block]
                    wp.synchronize()

                # Send ghost boxes
                r, comm_tag = block.send_ghost_boxes(
                    self.amr.comm,
                    comm_tag=comm_tag,
                    names=["f", "boundary_id", "missing_mask"],
                )
                requests.extend(r)

        # Wait sync
        wp.synchronize()

        # Wait for requests
        if self.amr.comm is not None:
            self.amr.comm.wait_all(requests)
        else:
            assert len(requests) == 0

        # Swap neighbour buffers
        for block in self.amr.blocks:
            if block.pid == self.amr.pid:
                block.swap_buffers()


    def step(self, debug=False):

        # Make stream idx
        stream_idx = 0

        # MPI communication parameters
        comm_tag = 0
        requests = []

        # Make event
        event = None

        # Set Perform steps equal to the number of ghost cell thickness
        for block in self.amr.blocks:

            # Check if block matches pid 
            if block.pid == self.amr.pid:

                # Set warp stream
                with wp.ScopedStream(self.wp_streams[stream_idx]):

                    # Get total box offset and extent
                    offset = block.offset
                    bound = block.offset + block.extent
                    for neighbour_boxes in block.neighbour_ghost_boxes.values():
                        box = neighbour_boxes["f"]
                        offset = np.minimum(offset, box.offset)
                        bound = np.maximum(bound, box.offset + box.extent)
                    extent = bound - offset

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
                    for _ in range(self.amr_ghost_cell_thickness-1):

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

                # Send ghost boxes
                r, comm_tag = block.send_ghost_boxes(
                    self.amr.comm,
                    comm_tag=comm_tag,
                    names=["f"],
                )
                requests.extend(r)

        # Wait sync
        wp.synchronize()

        # Wait for requests
        if self.amr.comm is not None:
            self.amr.comm.wait_all(requests)
        else:
            assert len(requests) == 0

        # Swap neighbour buffers
        for block in self.amr.blocks:
            if block.pid == self.amr.pid:
                block.swap_buffers()

    def save_drag_coefficient(self):
        raise NotImplementedError

    def save_boundary_id(self, file_name: str):

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
            if block.pid == self.amr.pid:
                continue

            # Get compute arrays
            boundary_id = self.memory_pools[0].get((1, *block.shape), wp.uint8)

            # TODO: remove this
            extent = tuple([int(e) for e in block.extent])

            # Make marching cubes
            mc = wp.MarchingCubes(
                nx=extent[0],
                ny=extent[1],
                nz=extent[2],
                max_verts=extent[0] * extent[1] * extent[2] * 5,
                max_tris=extent[0] * extent[1] * extent[2] * 3,
            )

            # Copy from block
            wp.copy(
                boundary_id,
                block.boxes["boundary_id"].data
            )
            np_boundary_id = boundary_id.numpy()
            np_boundary_id = np_boundary_id.astype(np.float32)
            boundary_id = wp.from_numpy(np_boundary_id, dtype=wp.float32)

            # Perform marching cubes
            mc.surface(boundary_id[0], float(self.full_way_bc.id - 0.5))

            # Check if there are any vertices
            if mc.verts.shape[0] == 0 or mc.indices.shape[0] == 0:
                continue

            # Save the marching cubes
            vertices = mc.verts.numpy()
            vertices = vertices * self.spacing + block.offset * self.spacing + self.origin
            indices = mc.indices.numpy()
            indices = indices.reshape(-1, 3)
            faces_flat = np.concatenate([
                np.full((indices.shape[0], 1), 3, dtype=np.uint32),
                indices
            ], axis=1).flatten().astype(np.uint32)

            # Create a PyVista PolyData object
            poly_data = pv.PolyData(vertices, faces_flat)

            # Save the PolyData object with colors to a VTK file
            vtk_file_name = os.path.join(file_name, f"{file_name}_{idx}.vtp")
            poly_data.save(os.path.join(self.output_directory, vtk_file_name))
            files.append(vtk_file_name)

            # Return arrays
            self.memory_pools[0].ret(boundary_id)

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

            # Compute q criterion
            rho, u = self.macroscopic(f, rho, u)
            norm_mu, q = self.q_criterion(u, norm_mu, q)

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
            vertices = vertices * self.spacing + block.offset * self.spacing + self.origin
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

    def run_baseline_mlups(self, nr_steps: int = 512):
        # NOTE: This is purely for MLUPs tests and should not be used for actual simulations

        # Make arrays
        rho = wp.zeros((1, *self.shape), wp.float32)
        u = wp.zeros((3, *self.shape), wp.float32)
        f0 = wp.zeros((self.velocity_set.q, *self.shape), wp.float32)
        f1 = wp.zeros((self.velocity_set.q, *self.shape), wp.float32)
        boundary_id = wp.zeros((1, *self.shape), wp.uint8)
        missing_mask = wp.zeros((self.velocity_set.q, *self.shape), wp.bool)

        # Initialize Inlet bc (bottom x face)
        if self.no_slip_walls:
            lower_bound = (0, 1, 1) # no edges
            upper_bound = (0, self.shape[1]-1, self.shape[2]-1)
        else:
            lower_bound = (0, 0, 0) # edges
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
        
        # Set outlet bc (top x face)
        if self.no_slip_walls:
            lower_bound = (self.shape[0]-1, 1, 1)
            upper_bound = (self.shape[0]-1, self.shape[1]-1, self.shape[2]-1)
        else:
            lower_bound = (self.shape[0]-1, 0, 0)
            upper_bound = (self.shape[0]-1, self.shape[1]-1, self.shape[2]-1)
        boundary_id, missing_mask = self.planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            self.do_nothing_bc.id,
            boundary_id,
            missing_mask,
            (0, 0, 0),
        )
    
        # Set no slip walls
        if self.no_slip_walls:
    
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
    
        # Set stl half way bc
        boundary_id, missing_mask = self.stl_boundary_masker(
            self.stl_filename,
            self.origin,
            self.spacing,
            self.half_way_bc.id,
            #self.full_way_bc.id,
            boundary_id,
            missing_mask,
            (0, 0, 0),
        )
   
        # Initialize the flow field
        rho, u = self.initializer(rho, u, boundary_id, self.base_velocity)
        f0 = self.equilibrium(rho, u, f0)

        # Warm up
        for i in range(32):
            f1 = self.stepper(f0, f1, boundary_id, missing_mask, 0)
            f0, f1 = f1, f0

        # Start timer
        wp.synchronize()
        start = time.time()
        for i in tqdm(range(nr_steps)):
            f1 = self.stepper(f0, f1, boundary_id, missing_mask, 0)
            f0, f1 = f1, f0
        wp.synchronize()
        end = time.time()

        # Compute MLUPS
        nr_cells = self.shape[0] * self.shape[1] * self.shape[2]
        nr_updates = nr_steps * nr_cells
        mlups = nr_updates / (end - start) / 1e6
        logging.info(f"MLUPS: {mlups}")
        return mlups


    def run_ooc_mlups(self, nr_steps: int = 32):
        # NOTE: This is purely for MLUPs tests and should not be used for actual simulations

        # Allocate amr fields
        self.allocate()

        # Initialize the boundary conditions and flow field
        self.initialize()

        # Take a few steps to warm up
        for i in range(3):
            self.step()

        # Perform update
        wp.synchronize()
        start = time.time()
        for i in range(nr_steps):
            self.step()
        wp.synchronize()
        end = time.time()

        # Compute MLUPS
        nr_cells = self.shape[0] * self.shape[1] * self.shape[2]
        nr_updates = nr_steps * self.amr_ghost_cell_thickness * nr_cells
        mlups = nr_updates / (end - start) / 1e6
        logging.info(f"MLUPS: {mlups}")
        return mlups

    def run(self):

        # Allocate amr fields
        self.allocate()

        # Initialize the boundary conditions and flow field
        self.initialize()

        # Save vtm of initial state
        #self.save_vtm("initial.vtm")

        # Save boundary id
        self.save_boundary_id("boundary_id")

        # Perform update
        for i in range(5000):
            self.step()
            if i % 10 == 0:
                logging.info(f"Step: {i}")
                self.save_q_criterion(f"q_criterion_{str(i).zfill(8)}")
                #self.save_vtm(f"state_{str(i).zfill(8)}.vtm")

        # Save the initial state

        ## Compute cross section
        #for l in range(self.velocity_set.q):
        #    if self.velocity_set.c[0, l] == 0 and self.velocity_set.c[1, l] == 0 and self.velocity_set.c[2, l] == 0:
        #        zero_index = l
        #np_missing_mask = self.missing_mask.numpy()
        #np_boundary_id = self.boundary_id.numpy()
        #is_solid = np.logical_and(np_missing_mask[zero_index] == 1, np_boundary_id[0] == self.half_way_bc.id)
        #cross_section = np.sum(is_solid, axis=(0))
        #self.cross_section = np.sum(cross_section > 0)

        ## Run the simulation
        #for i in tqdm.tqdm(range(self.nr_steps)):

        #    # Step
        #    self.step()

        #    # Monitor
        #    if i % self.monitor_frequency == 0:
        #        self.monitor(i)

        #    # Save monitor plot
        #    if i % (self.monitor_frequency) == 0:
        #        self.plot_drag_coefficient()

        #    # Save state
        #    if i % self.save_state_frequency == 0:
        #        self.compute_rho_u()
        #        self.save_state(str(i).zfill(8))

        ## Delete all the fields
        #del self.rho
        #del self.u
        #del self.f0
        #del self.f1
        #del self.boundary_id
        #del self.missing_mask
