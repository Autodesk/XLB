from typing import List, Callable
from mpi4py import MPI
import warp as wp

from ds.ooc_grid import MemoryPool
from subroutine.subroutine import Subroutine
from operators.soa_copy import SOACopy

class PrepareFieldsSubroutine(Subroutine):

    def __init__(
        self,
        initializer: Callable,
        equilibrium: Callable,
        boundary_conditions: List[Callable],
        indices_boundary_masker: Callable,
        my_copy: Callable = SOACopy(),
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.initializer = initializer
        self.equilibrium = equilibrium
        self.boundary_conditions = boundary_conditions
        self.indices_boundary_masker = indices_boundary_masker
        self.my_copy = my_copy
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        ooc_grid,
        f_name = "f",
        boundary_id_name = "boundary_id",
        missing_mask_name = "missing_mask",
        clear_memory_pools = True,
    ):

        # Make stream idx
        stream_idx = 0

        # Set initial conditions
        for block in ooc_grid.blocks.values():

            # Set warp stream
            with wp.ScopedStream(self.wp_streams[stream_idx]):

                # Check if block matches pid 
                if block.pid == ooc_grid.pid:

                    # Get q value
                    q = block.boxes[f_name].cardinality

                    # Get total box offset, extent and shape
                    offset = block.offset_with_ghost
                    extent = block.extent_with_ghost

                    # Get compute arrays
                    rho = self.memory_pools[stream_idx].get((1, *extent), wp.float32)
                    u = self.memory_pools[stream_idx].get((3, *extent), wp.float32)
                    f = self.memory_pools[stream_idx].get((q, *extent), wp.float32)
                    boundary_id = self.memory_pools[stream_idx].get((1, *extent), wp.uint8)
                    missing_mask = self.memory_pools[stream_idx].get((q, *extent), wp.bool)

                    # Get transmit arrays
                    f_block = self.memory_pools[stream_idx].get((q, *block.shape), wp.float32)
                    boundary_id_block = self.memory_pools[stream_idx].get((1, *block.shape), wp.uint8)
                    missing_mask_block = self.memory_pools[stream_idx].get((q, *block.shape), wp.bool)
                    f_ghost = {}
                    boundary_id_ghost = {}
                    missing_mask_ghost = {}
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        f_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes[f_name].shape),
                            wp.float32
                        )
                        boundary_id_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (1, *ghost_boxes[boundary_id_name].shape),
                            wp.uint8
                        )
                        missing_mask_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (q, *ghost_boxes[missing_mask_name].shape),
                            wp.bool
                        )

                    # Initialize boundary id and missing mask
                    boundary_id, missing_mask = self.indices_boundary_masker(
                        self.boundary_conditions,
                        boundary_id,
                        missing_mask,
                        offset,
                    )
    
                    # Initialize the flow field
                    rho, u = self.initializer(rho, u, boundary_id)
                    f = self.equilibrium(rho, u, f)

                    # Copy to block
                    slice_start = (block.offset - offset)
                    slice_stop = slice_start + block.extent
                    slice_start = tuple([int(s) for s in slice_start])
                    slice_stop = tuple([int(s) for s in slice_stop])
                    self.my_copy(
                        f_block,
                        f[
                            :,
                            slice_start[0]:slice_stop[0],
                            slice_start[1]:slice_stop[1],
                            slice_start[2]:slice_stop[2],
                        ],
                    )
                    self.my_copy(
                        boundary_id_block,
                        boundary_id[
                            :,
                            slice_start[0]:slice_stop[0],
                            slice_start[1]:slice_stop[1],
                            slice_start[2]:slice_stop[2],
                        ],
                    )
                    self.my_copy(
                        missing_mask_block,
                        missing_mask[
                            :,
                            slice_start[0]:slice_stop[0],
                            slice_start[1]:slice_stop[1],
                            slice_start[2]:slice_stop[2],
                        ],
                    )

                    # Copy to local ghost boxes
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():

                        # Get slice start and stop
                        slice_start = (ghost_boxes[f_name].offset - offset)
                        slice_stop = slice_start + ghost_boxes[f_name].shape
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
                    wp.copy(block.boxes[f_name].data, f_block)
                    wp.copy(block.boxes[boundary_id_name].data, boundary_id_block)
                    wp.copy(block.boxes[missing_mask_name].data, missing_mask_block)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        wp.copy(ghost_boxes[f_name].data, f_ghost[ghost_block])
                        wp.copy(ghost_boxes[boundary_id_name].data, boundary_id_ghost[ghost_block])
                        wp.copy(ghost_boxes[missing_mask_name].data, missing_mask_ghost[ghost_block])

                    # Return arrays
                    self.memory_pools[stream_idx].ret(rho, zero=True)
                    self.memory_pools[stream_idx].ret(u, zero=True)
                    self.memory_pools[stream_idx].ret(f, zero=True)
                    self.memory_pools[stream_idx].ret(boundary_id, zero=True)
                    self.memory_pools[stream_idx].ret(missing_mask, zero=True)
                    self.memory_pools[stream_idx].ret(f_block, zero=True)
                    self.memory_pools[stream_idx].ret(boundary_id_block, zero=True)
                    self.memory_pools[stream_idx].ret(missing_mask_block, zero=True)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        self.memory_pools[stream_idx].ret(f_ghost[ghost_block], zero=True)
                        self.memory_pools[stream_idx].ret(boundary_id_ghost[ghost_block], zero=True)
                        self.memory_pools[stream_idx].ret(missing_mask_ghost[ghost_block], zero=True)

                    # Update stream idx
                    stream_idx = (stream_idx + 1) % self.nr_streams

        # Send blocks
        wp.synchronize()
        comm_tag = 0
        requests = []
        for block in ooc_grid.blocks.values():
            r, comm_tag = block.send_ghost_boxes(
                ooc_grid.comm,
                comm_tag=comm_tag,
                names=[f_name, boundary_id_name, missing_mask_name],
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
                    names=[f_name, boundary_id_name, missing_mask_name],
                )

        # Clear memory pools
        if clear_memory_pools:
            for memory_pool in self.memory_pools:
                memory_pool.clear()
