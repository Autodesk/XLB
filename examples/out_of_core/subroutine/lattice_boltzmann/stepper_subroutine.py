from typing import List
from mpi4py import MPI
import warp as wp

from ds.ooc_grid import MemoryPool
from operators.operator import Operator
from ..subroutine import Subroutine
from operators.copy.soa_copy import SOACopy

class StepperSubroutine(Subroutine):

    def __init__(
        self,
        stepper,
        my_copy: Operator = SOACopy(),
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.stepper = stepper
        self.my_copy = my_copy
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        amr_grid,
        nr_steps=None,
        f_name = "f",
        boundary_id_name = "boundary_id",
        missing_mask_name = "missing_mask",
        clear_memory_pools = True,
    ):

        # Get number of steps
        if nr_steps is None:
            nr_steps = min(amr_grid.ghost_cell_thickness)
        assert nr_steps <= min(amr_grid.ghost_cell_thickness)

        # Make stream idx
        stream_idx = 0

        # MPI communication parameters
        comm_tag = 0
        requests = []

        # Make event
        event = None

        # Set Perform steps equal to the number of ghost cell thickness
        for block in amr_grid.blocks.values():

            # Set warp stream
            with wp.ScopedStream(self.wp_streams[stream_idx]):

                # Check if block matches pid 
                if block.pid == amr_grid.pid:

                    # Get block cardinality
                    q = block.boxes[f_name].cardinality

                    # Get total box offset, extent and shape
                    offset = block.offset_with_ghost
                    extent = block.extent_with_ghost
                    shape = extent

                    # Get compute arrays
                    f0 = self.memory_pools[stream_idx].get((q, *shape), wp.float32)
                    f1 = self.memory_pools[stream_idx].get((q, *shape), wp.float32)
                    boundary_id = self.memory_pools[stream_idx].get((1, *shape), wp.uint8)
                    missing_mask = self.memory_pools[stream_idx].get((q, *shape), wp.bool)

                    # Fill boundary_id with -1
                    #boundary_id.fill_(wp.uint8(-1))

                    # Get transmit arrays
                    f_block = self.memory_pools[stream_idx].get((q, *block.shape), wp.float32)
                    boundary_id_block = self.memory_pools[stream_idx].get((1, *block.shape), wp.uint8)
                    missing_mask_block = self.memory_pools[stream_idx].get((q, *block.shape), wp.bool)
                    f_neighbour_ghost = {}
                    boundary_id_neighbour_ghost = {}
                    missing_mask_neighbour_ghost = {}
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        f_neighbour_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes["f"].shape),
                            wp.float32
                        )
                        boundary_id_neighbour_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (1, *ghost_boxes["boundary_id"].shape),
                            wp.uint8
                        )
                        missing_mask_neighbour_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes["missing_mask"].shape),
                            wp.bool
                        )
                    f_local_ghost = {}
                    boundary_id_local_ghost = {}
                    missing_mask_local_ghost = {}
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        f_local_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes["f"].shape),
                            wp.float32
                        )
                        boundary_id_local_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (1, *ghost_boxes["boundary_id"].shape),
                            wp.uint8
                        )
                        missing_mask_local_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes["missing_mask"].shape),
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
                            f_neighbour_ghost[ghost_block],
                            ghost_boxes["f"].data
                        )
                        wp.copy(
                            boundary_id_neighbour_ghost[ghost_block],
                            ghost_boxes["boundary_id"].data
                        )
                        wp.copy(
                            missing_mask_neighbour_ghost[ghost_block],
                            ghost_boxes["missing_mask"].data
                        )

                    # Wait for previous event
                    if event is not None:
                        self.wp_streams[stream_idx].wait_event(event)

                    # Copy to compute arrays
                    start_1 = int(block.offset[0] - offset[0])
                    stop_1 = start_1 + block.extent[0]
                    start_2 = int(block.offset[1] - offset[1])
                    stop_2 = start_2 + block.extent[1]
                    start_3 = int(block.offset[2] - offset[2])
                    stop_3 = start_3 + block.extent[2]
                    self.my_copy(
                        f0[
                            :,
                            start_1:stop_1,
                            start_2:stop_2,
                            start_3:stop_3,
                        ],
                        f_block
                    )
                    self.my_copy(
                        boundary_id[
                            :,
                            start_1:stop_1,
                            start_2:stop_2,
                            start_3:stop_3,
                        ],
                        boundary_id_block
                    )
                    self.my_copy(
                        missing_mask[
                            :,
                            start_1:stop_1,
                            start_2:stop_2,
                            start_3:stop_3,
                        ],
                        missing_mask_block
                    )
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        start_1 = int(ghost_boxes["f"].offset[0] - offset[0])
                        stop_1 = start_1 + ghost_boxes["f"].extent[0]
                        start_2 = int(ghost_boxes["f"].offset[1] - offset[1])
                        stop_2 = start_2 + ghost_boxes["f"].extent[1]
                        start_3 = int(ghost_boxes["f"].offset[2] - offset[2])
                        stop_3 = start_3 + ghost_boxes["f"].extent[2]
                        self.my_copy(
                            f0[
                                :,
                                start_1:stop_1,
                                start_2:stop_2,
                                start_3:stop_3,
                            ],
                            f_neighbour_ghost[ghost_block],
                        )
                        self.my_copy(
                            boundary_id[
                                :,
                                start_1:stop_1,
                                start_2:stop_2,
                                start_3:stop_3,
                            ],
                            boundary_id_neighbour_ghost[ghost_block],
                        )
                        self.my_copy(
                            missing_mask[
                                :,
                                start_1:stop_1,
                                start_2:stop_2,
                                start_3:stop_3,
                            ],
                            missing_mask_neighbour_ghost[ghost_block],
                        )

                    # Perform update
                    for _ in range(nr_steps):

                        # Perform stepper
                        f1 = self.stepper(f0, f1, boundary_id, missing_mask, 0)
                        f0, f1 = f1, f0

                    # Copy from compute arrays
                    start_1 = int(block.offset[0] - offset[0])
                    stop_1 = start_1 + block.extent[0]
                    start_2 = int(block.offset[1] - offset[1])
                    stop_2 = start_2 + block.extent[1]
                    start_3 = int(block.offset[2] - offset[2])
                    stop_3 = start_3 + block.extent[2]
                    self.my_copy(
                        f_block,
                        f0[
                            :,
                            start_1:stop_1,
                            start_2:stop_2,
                            start_3:stop_3,
                        ]
                    )
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():

                        # Get slice start and stop
                        slice_start = (ghost_boxes[f_name].offset - block.offset)
                        slice_stop = slice_start + ghost_boxes[f_name].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])

                        # Copy
                        self.my_copy(
                            f_local_ghost[ghost_block],
                            f0[
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
                            ghost_boxes[f_name].data,
                            f_local_ghost[ghost_block]
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
                        self.memory_pools[stream_idx].ret(f_neighbour_ghost[ghost_block], zero=False)
                        self.memory_pools[stream_idx].ret(boundary_id_neighbour_ghost[ghost_block], zero=False)
                        self.memory_pools[stream_idx].ret(missing_mask_neighbour_ghost[ghost_block], zero=False)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        self.memory_pools[stream_idx].ret(f_local_ghost[ghost_block], zero=False)
                        self.memory_pools[stream_idx].ret(boundary_id_local_ghost[ghost_block], zero=False)
                        self.memory_pools[stream_idx].ret(missing_mask_local_ghost[ghost_block], zero=False)

                    # Update stream idx
                    stream_idx = (stream_idx + 1) % self.nr_streams

        # Send blocks
        wp.synchronize()
        for block in amr_grid.blocks.values():
            r, comm_tag = block.send_ghost_boxes(
                amr_grid.comm,
                comm_tag=comm_tag,
                names=["f"],
            )
            requests.extend(r)

        # Wait for requests
        if amr_grid.comm is not None:
            amr_grid.comm.Barrier()
            MPI.Request.Waitall(requests)
            pass
        else:
            assert len(requests) == 0

        # Swap neighbour buffers
        for block in amr_grid.blocks.values():
            if block.pid == amr_grid.pid:
                block.swap_buffers(names=["f"])
