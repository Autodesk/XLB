from typing import List
from mpi4py import MPI
import warp as wp

from ds.amr_grid import MemoryPool
from operator.operator import Operator
from subroutine.amr_grid.amr_grid_subroutine import AMRGridSubroutine
from operator.optimize.gradient_descent import GradientDescent

class GradientDescentSubroutine(AMRGridSubroutine):

    def __init__(
        self,
        gradient_descent: Operator = GradientDescent(),
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.gradient_descent = gradient_descent
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        amr_grid,
        field_name,
        adj_field_name,
        learning_rate=1e-3,
        clear_memory_pools = True,
    ):

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
                    q = block.boxes[field_name].cardinality

                    # Get compute arrays
                    field = self.memory_pools[stream_idx].get((q, *block.shape), wp.float32)
                    adj_field = self.memory_pools[stream_idx].get((q, *block.shape), wp.float32)
                    field_ghost = {}
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        field_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes[field_name].shape),
                            wp.float32
                        )

                    # Copy from block
                    wp.copy(field, block.boxes[field_name].data)
                    wp.copy(adj_field, block.boxes[adj_field_name].data)

                    # Perform gradient decent
                    self.gradient_descent(field, adj_field, learning_rate)

                    # Copy to local ghost boxes
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        
                        # Get slice start and stop
                        slice_start = (ghost_boxes[field_name].offset - block.offset) // 2 ** block.amr_level
                        slice_stop = slice_start + ghost_boxes[field_name].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])

                        # Copy
                        wp.copy(
                            field_ghost[ghost_block],
                            field[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )

                    # Copy to block
                    wp.copy(block.boxes[field_name].data, field)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        wp.copy(ghost_boxes[field_name].data, field_ghost[ghost_block])

                    # Return arrays
                    self.memory_pools[stream_idx].ret(field, zero=True)
                    self.memory_pools[stream_idx].ret(adj_field, zero=True)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        self.memory_pools[stream_idx].ret(field_ghost[ghost_block], zero=True)

                    # Update stream idx
                    stream_idx = (stream_idx + 1) % self.nr_streams

        # Send blocks
        wp.synchronize()
        comm_tag = 0
        requests = []
        for block in amr_grid.blocks.values():
            r, comm_tag = block.send_ghost_boxes(
                amr_grid.comm,
                comm_tag=comm_tag,
                names=[field_name],
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
                block.swap_buffers(
                    names=[field_name],
                )

        # Clear memory pools
        if clear_memory_pools:
            for memory_pool in self.memory_pools:
                memory_pool.clear()
