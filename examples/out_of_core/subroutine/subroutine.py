# Description: Subroutine class, used to execute complex operations on out of core grids

from typing import List
import warp as wp

from ds.ooc_grid import MemoryPool


class Subroutine:
    def __init__(
        self,
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.nr_streams = nr_streams
        self.wp_streams = wp_streams if wp_streams is not None else [wp.get_stream() for _ in range(nr_streams)]
        self.memory_pools = memory_pools if memory_pools is not None else [MemoryPool() for _ in range(nr_streams)]

    def __call__(self, *args):
        raise NotImplementedError
