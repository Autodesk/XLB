import numpy as np
import cupy as cp

# from mpi4py import MPI
import itertools

from xlb.experimental.ooc.tiles.dense_tile import DenseTile, DenseGPUTile, DenseCPUTile
from xlb.experimental.ooc.tiles.compressed_tile import (
    CompressedTile,
    CompressedGPUTile,
    CompressedCPUTile,
)


class OOCArray:
    """An out-of-core distributed array class.

    Parameters
    ----------
    shape : tuple
        The shape of the array.
    dtype : cp.dtype
        The data type of the array.
    tile_shape : tuple
        The shape of the tiles. Should be a factor of the shape.
    padding : int or tuple
        The padding of the tiles.
    comm : MPI communicator
        The MPI communicator.
    devices : list of cp.cuda.Device
        The list of GPU devices to use.
    codec : Codec
        The codec to use for compression. None for no compression (Dense tiles).
    nr_compute_tiles : int
        The number of compute tiles used for asynchronous copies.
        TODO currently only 1 is supported when using JAX.
    """

    def __init__(
        self,
        shape,
        dtype,
        tile_shape,
        padding=1,
        comm=None,
        devices=[cp.cuda.Device(0)],
        codec=None,
        nr_compute_tiles=1,
    ):
        self.shape = shape
        self.tile_shape = tile_shape
        self.dtype = dtype
        if isinstance(padding, int):
            padding = (padding,) * len(shape)
        self.padding = padding
        self.comm = comm
        self.devices = devices
        self.codec = codec
        self.nr_compute_tiles = nr_compute_tiles

        # Set tile class
        if self.codec is None:
            self.Tile = DenseTile
            self.DeviceTile = DenseGPUTile
            self.HostTile = DenseCPUTile  # TODO: Possibly make HardDiskTile or something

        else:
            self.Tile = CompressedTile
            self.DeviceTile = CompressedGPUTile
            self.HostTile = CompressedCPUTile

        # Get process id and number of processes
        self.pid = self.comm.Get_rank()
        self.nr_proc = self.comm.Get_size()

        # Check that the tile shape divides the array shape.
        if any([shape[i] % tile_shape[i] != 0 for i in range(len(shape))]):
            raise ValueError(f"Tile shape {tile_shape} does not divide shape {shape}.")
        self.tile_dims = tuple([shape[i] // tile_shape[i] for i in range(len(shape))])
        self.nr_tiles = np.prod(self.tile_dims)

        # Get number of tiles per process
        if self.nr_tiles % self.nr_proc != 0:
            raise ValueError(f"Number of tiles {self.nr_tiles} does not divide number of processes {self.nr_proc}.")
        self.nr_tiles_per_proc = self.nr_tiles // self.nr_proc

        # Make the tile mapppings
        self.tile_process_map = {}
        self.tile_device_map = {}
        for i, tile_index in enumerate(itertools.product(*[range(n) for n in self.tile_dims])):
            self.tile_process_map[tile_index] = i % self.nr_proc
            self.tile_device_map[tile_index] = devices[i % len(devices)]  # Checkoboard pattern, TODO: may not be optimal

        # Get my device
        if self.nr_proc != len(self.devices):
            raise ValueError(f"Number of processes {self.nr_proc} does not equal number of devices {len(self.devices)}.")
        self.device = self.devices[self.pid]

        # Make the tiles
        self.tiles = {}
        for tile_index in self.tile_process_map.keys():
            if self.pid == self.tile_process_map[tile_index]:
                self.tiles[tile_index] = self.HostTile(self.tile_shape, self.dtype, self.padding, self.codec)

        # Make GPU tiles for copying data between CPU and GPU
        if self.nr_tiles % self.nr_compute_tiles != 0:
            raise ValueError(
                f"Number of tiles {self.nr_tiles} does not divide number of compute tiles {self.nr_compute_tiles}. This is used for asynchronous copies."
            )
        compute_array_shape = [s + 2 * p for (s, p) in zip(self.tile_shape, self.padding)]
        self.compute_tiles_htd = []
        self.compute_tiles_dth = []
        self.compute_streams_htd = []
        self.compute_streams_dth = []
        self.compute_arrays = []
        self.current_compute_index = 0
        with cp.cuda.Device(self.device):
            for i in range(self.nr_compute_tiles):
                # Make compute tiles for copying data
                compute_tile = self.DeviceTile(self.tile_shape, self.dtype, self.padding, self.codec)
                self.compute_tiles_htd.append(compute_tile)
                compute_tile = self.DeviceTile(self.tile_shape, self.dtype, self.padding, self.codec)
                self.compute_tiles_dth.append(compute_tile)

                # Make cupy stream
                self.compute_streams_htd.append(cp.cuda.Stream(non_blocking=True))
                self.compute_streams_dth.append(cp.cuda.Stream(non_blocking=True))

                # Make compute array

                self.compute_arrays.append(cp.empty(compute_array_shape, self.dtype))

        # Make compute tile mappings
        self.compute_tile_mapping_htd = {}
        self.compute_tile_mapping_dth = {}
        self.compute_stream_mapping_htd = {}

    def size(self):
        """Return number of allocated bytes for all host tiles."""
        return sum([tile.size() for tile in self.tiles.values()])

    def nbytes(self):
        """Return number of bytes for all host tiles."""
        return sum([tile.nbytes for tile in self.tiles.values()])

    def compression_ratio(self):
        """Return the compression ratio for all host tiles."""
        return self.nbytes() / self.size()

    def compression_ratio(self):
        """Return the compression ratio aggregated over all tiles."""

        if self.codec is None:
            return 1.0
        else:
            total_bytes = 0
            total_uncompressed_bytes = 0
            for tile in self.tiles.values():
                (
                    tile_total_bytes_uncompressed,
                    tile_total_bytes_compressed,
                ) = tile.compression_ratio()
                total_bytes += tile_total_bytes_compressed
                total_uncompressed_bytes += tile_total_bytes_uncompressed
            return total_uncompressed_bytes / total_bytes

    def update_compute_index(self):
        """Update the current compute index."""
        self.current_compute_index = (self.current_compute_index + 1) % self.nr_compute_tiles

    def _guess_next_tile_index(self, tile_index):
        """Guess the next tile index to use for the compute array."""
        # TODO: This assumes access is sequential
        tile_indices = list(self.tiles.keys())
        current_ind = tile_indices.index(tile_index)
        next_ind = current_ind + 1
        if next_ind >= len(tile_indices):
            return None
        else:
            return tile_indices[next_ind]

    def reset_queue_htd(self):
        """Reset the queue for host to device copies."""

        self.compute_tile_mapping_htd = {}
        self.compute_stream_mapping_htd = {}
        self.current_compute_index = 0

    def managed_compute_tiles_htd(self, tile_index):
        """Get the compute tiles needed for computation.

        Parameters
        ----------
        tile_index : tuple
            The tile index.

        Returns
        -------
        compute_tile : ComputeTile
            The compute tile needed for computation.
        """

        ###################################################
        # TODO: This assumes access is sequential for tiles
        ###################################################

        # Que up the next tiles
        cur_tile_index = tile_index
        cur_compute_index = self.current_compute_index
        for i in range(self.nr_compute_tiles):
            # Check if already in compute tile map and if not que it
            if cur_tile_index not in self.compute_tile_mapping_htd.keys():
                # Get the store tile
                tile = self.tiles[cur_tile_index]

                # Get the compute tile
                compute_tile = self.compute_tiles_htd[cur_compute_index]

                # Get the compute stream
                compute_stream = self.compute_streams_htd[cur_compute_index]

                # Copy the tile to the compute tile using the compute stream
                with compute_stream:
                    tile.to_gpu_tile(compute_tile)
                tile.to_gpu_tile(compute_tile)

                # Set the compute tile mapping
                self.compute_tile_mapping_htd[cur_tile_index] = compute_tile
                self.compute_stream_mapping_htd[cur_tile_index] = compute_stream

            # Update the tile index and compute index
            cur_tile_index = self._guess_next_tile_index(cur_tile_index)
            if cur_tile_index is None:
                break
            cur_compute_index = (cur_compute_index + 1) % self.nr_compute_tiles

        # Get the compute tile
        self.compute_stream_mapping_htd[tile_index].synchronize()
        compute_tile = self.compute_tile_mapping_htd[tile_index]

        # Pop the tile from the compute tile map
        self.compute_tile_mapping_htd.pop(tile_index)
        self.compute_stream_mapping_htd.pop(tile_index)

        # Return the compute tile
        return compute_tile

    def get_compute_array(self, tile_index):
        """Given a tile index, copy the tile to the compute array.

        Parameters
        ----------
        tile_index : tuple
            The tile index.

        Returns
        -------
        compute_array : array
            The compute array.
        global_index : tuple
            The lower bound index that the compute array corresponds to in the global array.
            For example, if the compute array is the 0th tile and has padding 1, then the
            global index will be (-1, -1, ..., -1).
        """

        # Get the compute tile
        compute_tile = self.managed_compute_tiles_htd(tile_index)

        # Concatenate the sub-arrays to make the compute array
        compute_tile.to_array(self.compute_arrays[self.current_compute_index])

        # Return the compute array index in global array
        global_index = tuple([i * s - p for (i, s, p) in zip(tile_index, self.tile_shape, self.padding)])

        return self.compute_arrays[self.current_compute_index], global_index

    def set_tile(self, compute_array, tile_index):
        """Given a tile index, copy the compute array to the tile.

        Parameters
        ----------
        compute_array : array
            The compute array.
        tile_index : tuple
            The tile index.
        """

        # Syncronize the current stream dth stream
        stream = self.compute_streams_dth[self.current_compute_index]
        stream.synchronize()
        cp.cuda.get_current_stream().synchronize()

        # Set the compute tile to the correct one
        compute_tile = self.compute_tiles_dth[self.current_compute_index]

        # Split the compute array into a tile
        compute_tile.from_array(compute_array)

        # Syncronize the current stream and the compute stream
        cp.cuda.get_current_stream().synchronize()

        # Copy the tile from the compute tile to the store tile
        with stream:
            compute_tile.to_cpu_tile(self.tiles[tile_index])
        compute_tile.to_cpu_tile(self.tiles[tile_index])

    def update_padding(self):
        """Perform a padding swap between neighboring tiles."""

        # Get padding indices
        pad_ind = self.compute_tiles_htd[0].pad_ind

        # Loop over tiles
        comm_tag = 0
        for tile_index in self.tile_process_map.keys():
            # Loop over all padding
            for pad_index in pad_ind:
                # Get neighboring tile index
                neigh_tile_index = tuple([(i + p) % s for (i, p, s) in zip(tile_index, pad_index, self.tile_dims)])
                neigh_pad_index = tuple([-p for p in pad_index])  # flip

                # 4 cases:
                # 1. the tile and neighboring tile are on the same process
                # 2. the tile is on this process and the neighboring tile is on another process
                # 3. the tile is on another process and the neighboring tile is on this process
                # 4. the tile and neighboring tile are on different processes

                # Case 1: the tile and neighboring tile are on the same process
                if self.pid == self.tile_process_map[tile_index] and self.pid == self.tile_process_map[neigh_tile_index]:
                    # Get the tile and neighboring tile
                    tile = self.tiles[tile_index]
                    neigh_tile = self.tiles[neigh_tile_index]

                    # Get pointer to padding and neighboring padding
                    padding = tile._padding[pad_index]
                    neigh_padding = neigh_tile._buf_padding[neigh_pad_index]

                    # Swap padding
                    tile._padding[pad_index] = neigh_padding
                    neigh_tile._buf_padding[neigh_pad_index] = padding

                # Case 2: the tile is on this process and the neighboring tile is on another process
                if self.pid == self.tile_process_map[tile_index] and self.pid != self.tile_process_map[neigh_tile_index]:
                    # Get the tile and padding
                    tile = self.tiles[tile_index]
                    padding = tile._padding[pad_index]

                    # Send padding to neighboring process
                    self.comm.Send(
                        padding,
                        dest=self.tile_process_map[neigh_tile_index],
                        tag=comm_tag,
                    )

                # Case 3: the tile is on another process and the neighboring tile is on this process
                if self.pid != self.tile_process_map[tile_index] and self.pid == self.tile_process_map[neigh_tile_index]:
                    # Get the neighboring tile and padding
                    neigh_tile = self.tiles[neigh_tile_index]
                    neigh_padding = neigh_tile._buf_padding[neigh_pad_index]

                    # Receive padding from neighboring process
                    self.comm.Recv(
                        neigh_padding,
                        source=self.tile_process_map[tile_index],
                        tag=comm_tag,
                    )

                # Case 4: the tile and neighboring tile are on different processes
                if self.pid != self.tile_process_map[tile_index] and self.pid != self.tile_process_map[neigh_tile_index]:
                    pass

                # Increment the communication tag
                comm_tag += 1

        # Shuffle padding with buffers
        for tile in self.tiles.values():
            tile.swap_buf_padding()

    def get_array(self):
        """Get the full array out from all the sub-arrays. This should only be used for testing."""

        # Get the full array
        if self.comm.rank == 0:
            array = np.ones(self.shape, dtype=self.dtype)
        else:
            array = None

        # Loop over tiles
        comm_tag = 0
        for tile_index in self.tile_process_map.keys():
            # Set the center array in the full array
            slice_index = tuple([slice(i * s, (i + 1) * s) for (i, s) in zip(tile_index, self.tile_shape)])

            # if tile on this process compute the center array
            if self.comm.rank == self.tile_process_map[tile_index]:
                # Get the tile
                tile = self.tiles[tile_index]

                # Copy the tile to the compute tile
                tile.to_gpu_tile(self.compute_tiles_htd[0])

                # Get the compute array
                self.compute_tiles_htd[0].to_array(self.compute_arrays[0])

                # Get the center array
                center_array = self.compute_arrays[0][tile._slice_center].get()

            # 4 cases:
            # 1. the tile is on rank 0 and this process is rank 0
            # 2. the tile is on another rank and this process is rank 0
            # 3. the tile is on this rank and this process is not rank 0
            # 4. the tile is not on rank 0 and this process is not rank 0

            # Case 1: the tile is on rank 0
            if self.comm.rank == 0 and self.tile_process_map[tile_index] == 0:
                # Set the center array in the full array
                array[slice_index] = center_array

            # Case 2: the tile is on another rank and this process is rank 0
            if self.comm.rank == 0 and self.tile_process_map[tile_index] != 0:
                # Get the data from the other rank
                center_array = np.empty(self.tile_shape, dtype=self.dtype)
                self.comm.Recv(center_array, source=self.tile_process_map[tile_index], tag=comm_tag)

                # Set the center array in the full array
                array[slice_index] = center_array

            # Case 3: the tile is on this rank and this process is not rank 0
            if self.comm.rank != 0 and self.tile_process_map[tile_index] == self.comm.rank:
                # Send the data to rank 0
                self.comm.Send(center_array, dest=0, tag=comm_tag)

            # Case 4: the tile is not on rank 0 and this process is not rank 0
            if self.comm.rank != 0 and self.tile_process_map[tile_index] != 0:
                pass

            # Update the communication tag
            comm_tag += 1

        return array
