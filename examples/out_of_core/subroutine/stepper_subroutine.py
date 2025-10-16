from typing import List, Callable
from mpi4py import MPI
import warp as wp

from ds.ooc_grid import MemoryPool
from subroutine.subroutine import Subroutine
from operators.soa_copy import SOACopy


class StepperSubroutine(Subroutine):
    def __init__(
        self,
        stepper: Callable,
        omega: float,
        my_copy: Callable = SOACopy(),
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.stepper = stepper
        self.omega = omega
        self.my_copy = my_copy
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        ooc_grid,
        nr_steps=None,
        f_name="f",
        boundary_id_name="boundary_id",
        missing_mask_name="missing_mask",
        clear_memory_pools=True,
    ):
        # Get number of steps
        if nr_steps is None:
            nr_steps = min(ooc_grid.ghost_cell_thickness)
        assert nr_steps <= min(ooc_grid.ghost_cell_thickness)

        # Make stream idx
        stream_idx = 0

        # MPI communication parameters
        comm_tag = 0
        requests = []

        # Make event
        event = None

        # Set Perform steps equal to the number of ghost cell thickness
        for block in ooc_grid.blocks.values():
            # Set warp stream
            with wp.ScopedStream(self.wp_streams[stream_idx]):
                # Check if block matches pid
                if block.pid == ooc_grid.pid:
                    # Get block cardinality
                    q = block.boxes[f_name].cardinality

                    # Get total box offset, extent and shape
                    offset = block.offset_with_ghost
                    extent = block.extent_with_ghost

                    # Get compute arrays
                    f0 = self.memory_pools[stream_idx].get((q, *extent), wp.float32)
                    f1 = self.memory_pools[stream_idx].get((q, *extent), wp.float32)
                    boundary_id = self.memory_pools[stream_idx].get((1, *extent), wp.uint8)
                    missing_mask = self.memory_pools[stream_idx].get((q, *extent), wp.bool)

                    # Get transmit arrays
                    f_block = self.memory_pools[stream_idx].get((q, *block.shape), wp.float32)
                    boundary_id_block = self.memory_pools[stream_idx].get((1, *block.shape), wp.uint8)
                    missing_mask_block = self.memory_pools[stream_idx].get((q, *block.shape), wp.bool)
                    f_neighbour_ghost = {}
                    boundary_id_neighbour_ghost = {}
                    missing_mask_neighbour_ghost = {}
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        f_neighbour_ghost[ghost_block] = self.memory_pools[stream_idx].get((q, *ghost_boxes["f"].shape), wp.float32)
                        boundary_id_neighbour_ghost[ghost_block] = self.memory_pools[stream_idx].get((1, *ghost_boxes["boundary_id"].shape), wp.uint8)
                        missing_mask_neighbour_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes["missing_mask"].shape), wp.bool
                        )
                    f_local_ghost = {}
                    boundary_id_local_ghost = {}
                    missing_mask_local_ghost = {}
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        f_local_ghost[ghost_block] = self.memory_pools[stream_idx].get((q, *ghost_boxes["f"].shape), wp.float32)
                        boundary_id_local_ghost[ghost_block] = self.memory_pools[stream_idx].get((1, *ghost_boxes["boundary_id"].shape), wp.uint8)
                        missing_mask_local_ghost[ghost_block] = self.memory_pools[stream_idx].get((q, *ghost_boxes["missing_mask"].shape), wp.bool)

                    # Copy from block
                    wp.copy(f_block, block.boxes["f"].data)
                    wp.copy(boundary_id_block, block.boxes["boundary_id"].data)
                    wp.copy(missing_mask_block, block.boxes["missing_mask"].data)
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        wp.copy(f_neighbour_ghost[ghost_block], ghost_boxes["f"].data)
                        wp.copy(boundary_id_neighbour_ghost[ghost_block], ghost_boxes["boundary_id"].data)
                        wp.copy(missing_mask_neighbour_ghost[ghost_block], ghost_boxes["missing_mask"].data)

                    # Wait for previous event
                    if event is not None:
                        self.wp_streams[stream_idx].wait_event(event)

                    # Copy to compute arrays
                    slice_start = block.offset - offset
                    slice_stop = slice_start + block.shape
                    slice_start = tuple([int(s) for s in slice_start])
                    slice_stop = tuple([int(s) for s in slice_stop])
                    self.my_copy(
                        f0[
                            :,
                            slice_start[0] : slice_stop[0],
                            slice_start[1] : slice_stop[1],
                            slice_start[2] : slice_stop[2],
                        ],
                        f_block,
                    )
                    self.my_copy(
                        boundary_id[
                            :,
                            slice_start[0] : slice_stop[0],
                            slice_start[1] : slice_stop[1],
                            slice_start[2] : slice_stop[2],
                        ],
                        boundary_id_block,
                    )
                    self.my_copy(
                        missing_mask[
                            :,
                            slice_start[0] : slice_stop[0],
                            slice_start[1] : slice_stop[1],
                            slice_start[2] : slice_stop[2],
                        ],
                        missing_mask_block,
                    )
                    for ghost_block, ghost_boxes in block.neighbour_ghost_boxes.items():
                        slice_start = ghost_boxes["f"].offset - offset
                        slice_stop = slice_start + ghost_boxes["f"].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])
                        self.my_copy(
                            f0[
                                :,
                                slice_start[0] : slice_stop[0],
                                slice_start[1] : slice_stop[1],
                                slice_start[2] : slice_stop[2],
                            ],
                            f_neighbour_ghost[ghost_block],
                        )
                        self.my_copy(
                            boundary_id[
                                :,
                                slice_start[0] : slice_stop[0],
                                slice_start[1] : slice_stop[1],
                                slice_start[2] : slice_stop[2],
                            ],
                            boundary_id_neighbour_ghost[ghost_block],
                        )
                        self.my_copy(
                            missing_mask[
                                :,
                                slice_start[0] : slice_stop[0],
                                slice_start[1] : slice_stop[1],
                                slice_start[2] : slice_stop[2],
                            ],
                            missing_mask_neighbour_ghost[ghost_block],
                        )

                    # Perform update
                    for _ in range(nr_steps):
                        # Perform stepper
                        f0, f1 = self.stepper(f0, f1, boundary_id, missing_mask, self.omega, 0)
                        f0, f1 = f1, f0

                    # Copy from compute arrays
                    slice_start = block.offset - offset
                    slice_stop = slice_start + block.shape
                    slice_start = tuple([int(s) for s in slice_start])
                    slice_stop = tuple([int(s) for s in slice_stop])
                    self.my_copy(
                        f_block,
                        f0[
                            :,
                            slice_start[0] : slice_stop[0],
                            slice_start[1] : slice_stop[1],
                            slice_start[2] : slice_stop[2],
                        ],
                    )
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        # Get slice start and stop
                        slice_start = ghost_boxes[f_name].offset - offset
                        slice_stop = slice_start + ghost_boxes[f_name].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])

                        # Copy
                        self.my_copy(
                            f_local_ghost[ghost_block],
                            f0[
                                :,
                                slice_start[0] : slice_stop[0],
                                slice_start[1] : slice_stop[1],
                                slice_start[2] : slice_stop[2],
                            ],
                        )

                    # Wait for previous event
                    if event is None:
                        event = wp.Event()
                    self.wp_streams[stream_idx].record_event(event)

                    # Copy to block
                    wp.copy(block.boxes["f"].data, f_block)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        wp.copy(ghost_boxes[f_name].data, f_local_ghost[ghost_block])

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
        for block in ooc_grid.blocks.values():
            r, comm_tag = block.send_ghost_boxes(
                ooc_grid.comm,
                comm_tag=comm_tag,
                names=["f"],
            )
            requests.extend(r)

        # Wait for requests
        if ooc_grid.comm is not None:
            ooc_grid.comm.Barrier()
            MPI.Request.Waitall(requests)
            pass
        else:
            assert len(requests) == 0

        # Swap neighbour buffers
        for block in ooc_grid.blocks.values():
            if block.pid == ooc_grid.pid:
                block.swap_buffers(names=["f"])
