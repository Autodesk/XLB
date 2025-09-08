from typing import List
from mpi4py import MPI
import warp as wp

from ds.ooc_grid import MemoryPool
from operators.operator import Operator
from ..subroutine import Subroutine
from operators.copy.soa_copy import SOACopy

class ForwardRhoLossSubroutine(Subroutine):

    def __init__(
        self,
        macroscopic: Operator,
        loss: Operator,
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.macroscopic = macroscopic
        self.loss = loss
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        amr_grid,
        loss,
        f_name = "f_0000",
        boundary_id_name = "boundary_id",
        target_rho_name = "target_rho",
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
                    q = block.boxes[f_name].cardinality

                    # Get compute arrays
                    rho = self.memory_pools[stream_idx].get((1, *block.shape), wp.float32)
                    target_rho = self.memory_pools[stream_idx].get((1, *block.shape), wp.float32)
                    u = self.memory_pools[stream_idx].get((3, *block.shape), wp.float32)
                    f = self.memory_pools[stream_idx].get((q, *block.shape), wp.float32)
                    boundary_id = self.memory_pools[stream_idx].get((1, *block.shape), wp.uint8)

                    # Copy from block
                    wp.copy(f, block.boxes[f_name].data)
                    wp.copy(target_rho, block.boxes[target_rho_name].data)
                    wp.copy(boundary_id, block.boxes[boundary_id_name].data)

                    # Get rho and u
                    rho, u = self.macroscopic(f, boundary_id, rho, u)

                    # Compute loss
                    loss = self.loss(rho, target_rho, boundary_id, loss)

                    # Return arrays
                    self.memory_pools[stream_idx].ret(rho, zero=True)
                    self.memory_pools[stream_idx].ret(u, zero=True)
                    self.memory_pools[stream_idx].ret(f, zero=True)
                    self.memory_pools[stream_idx].ret(target_rho, zero=True)
                    self.memory_pools[stream_idx].ret(boundary_id, zero=True)

                    # Update stream idx
                    stream_idx = (stream_idx + 1) % self.nr_streams

        # Synchronize
        wp.synchronize()

        # Clear memory pools
        if clear_memory_pools:
            for memory_pool in self.memory_pools:
                memory_pool.clear()

class BackwardRhoLossSubroutine(Subroutine):

    def __init__(
        self,
        macroscopic: Operator,
        loss: Operator,
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.macroscopic = macroscopic
        self.loss = loss
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        amr_grid,
        loss,
        f_name = "f",
        adj_f_name = "adj_f",
        boundary_id_name = "boundary_id",
        target_rho_name = "target_rho",
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
                    q = block.boxes[f_name].cardinality

                    # Get compute arrays
                    rho = self.memory_pools[stream_idx].get((1, *block.shape), wp.float32, requires_grad=True)
                    target_rho = self.memory_pools[stream_idx].get((1, *block.shape), wp.float32, requires_grad=True)
                    u = self.memory_pools[stream_idx].get((3, *block.shape), wp.float32, requires_grad=True)
                    f = self.memory_pools[stream_idx].get((q, *block.shape), wp.float32, requires_grad=True)
                    boundary_id = self.memory_pools[stream_idx].get((1, *block.shape), wp.uint8)
                    adj_f_ghost = {}
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        adj_f_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes[f_name].shape),
                            wp.float32
                        )

                    # Copy from block
                    wp.copy(f, block.boxes[f_name].data)
                    wp.copy(target_rho, block.boxes[target_rho_name].data)
                    wp.copy(boundary_id, block.boxes[boundary_id_name].data)

                    # Make gradient tape
                    with wp.Tape() as tape:
                        rho, u = self.macroscopic(f, boundary_id, rho, u)
                        loss = self.loss(rho, target_rho, boundary_id, loss)

                    # Compute gradients
                    tape.backward()

                    # Copy to local ghost boxes
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():

                        # Get slice start and stop
                        slice_start = (ghost_boxes[f_name].offset - block.offset)
                        slice_stop = slice_start + ghost_boxes[f_name].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])

                        # Copy
                        wp.copy(
                            adj_f_ghost[ghost_block],
                            f.grad[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )

                    # Copy to block
                    wp.copy(block.boxes[adj_f_name].data, f.grad)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        wp.copy(ghost_boxes[adj_f_name].data, adj_f_ghost[ghost_block])

                    # Return arrays
                    self.memory_pools[stream_idx].ret(rho, zero=True)
                    self.memory_pools[stream_idx].ret(u, zero=True)
                    self.memory_pools[stream_idx].ret(f, zero=True)
                    self.memory_pools[stream_idx].ret(target_rho, zero=True)
                    self.memory_pools[stream_idx].ret(boundary_id, zero=True)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        self.memory_pools[stream_idx].ret(adj_f_ghost[ghost_block], zero=True)

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
                names=[adj_f_name],
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
                    names=[adj_f_name],
                )

        # Clear memory pools
        if clear_memory_pools:
            for memory_pool in self.memory_pools:
                memory_pool.clear()
