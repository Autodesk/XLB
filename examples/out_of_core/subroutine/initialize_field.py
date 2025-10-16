from typing import List, Callable
from mpi4py import MPI
import warp as wp

from ds.ooc_grid import MemoryPool
from subroutine.subroutine import Subroutine
from operators.soa_copy import SOACopy


class InitializeFieldSubroutine(Subroutine):
    def __init__(
        self,
        initializer: Callable,
        my_copy: Callable = SOACopy(),
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.initializer = initializer
        self.my_copy = my_copy
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        ooc_grid,
        field_name="field",
        clear_memory_pools=True,
    ):
        # Make stream idx
        stream_idx = 0

        # MPI communication parameters
        comm_tag = 0
        requests = []

        # Set initial conditions
        for block in ooc_grid.blocks.values():
            # Set warp stream
            with wp.ScopedStream(self.wp_streams[stream_idx]):
                # Check if block matches pid
                if block.pid == ooc_grid.pid:
                    # Get compute arrays
                    field = self.memory_pools[stream_idx].get(block.boxes[field_name].data_shape, block.boxes[field_name].dtype)
                    field_ghost = {}
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        field_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            ghost_boxes[field_name].data_shape, ghost_boxes[field_name].dtype
                        )

                    # Initialize the field
                    field = self.initializer(
                        field,
                        block.local_origin,
                        block.local_spacing,
                    )

                    # Copy to local ghost boxes
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        # Get slice start and stop
                        slice_start = ghost_boxes[field_name].offset - block.boxes[field_name].offset
                        slice_stop = slice_start + ghost_boxes[field_name].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])

                        # Copy
                        self.my_copy(
                            field_ghost[ghost_block],
                            field[
                                :,
                                slice_start[0] : slice_stop[0],
                                slice_start[1] : slice_stop[1],
                                slice_start[2] : slice_stop[2],
                            ],
                        )

                    # Copy to block
                    wp.copy(block.boxes[field_name].data, field)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        wp.copy(ghost_boxes[field_name].data, field_ghost[ghost_block])

                    # Return arrays
                    self.memory_pools[stream_idx].ret(field, zero=True)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        self.memory_pools[stream_idx].ret(field_ghost[ghost_block], zero=True)

                    # Update stream idx
                    stream_idx = (stream_idx + 1) % self.nr_streams

        # Send blocks
        wp.synchronize()
        for block in ooc_grid.blocks.values():
            r, comm_tag = block.send_ghost_boxes(
                ooc_grid.comm,
                comm_tag=comm_tag,
                names=[field_name],
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
                block.swap_buffers(
                    names=[field_name],
                )

        # Clear memory pools
        if clear_memory_pools:
            for memory_pool in self.memory_pools:
                memory_pool.clear()
