import numpy as np
import warp as wp
from tqdm import tqdm
import logging
import itertools
import pyvista as pv
import gc


class Box:
    def __init__(
        self,
        extent,
        offset,
        origin,
        spacing,
        cardinality,
        ordering,
        dtype,
        device,
    ):
        # Check valid ordering
        assert ordering in ["AOS", "SOA"], f"Unknown ordering {ordering}, must be 'AOS' or 'SOA'."

        # Parameters
        self.extent = extent
        self.offset = offset
        self.origin = origin
        self.spacing = spacing
        self.cardinality = cardinality
        self.ordering = ordering
        self.dtype = dtype
        self.device = device

        # Allocate data
        self.data = None

    @property
    def local_origin(self):
        return self.origin + np.array(self.offset) * np.array(self.spacing)

    @property
    def local_spacing(self):
        return np.array(self.spacing)

    @property
    def shape(self):
        return tuple(self.extent)

    @property
    def data_shape(self):
        if self.ordering == "AOS":
            return list(self.shape) + [self.cardinality]
        elif self.ordering == "SOA":
            return [self.cardinality] + list(self.shape)

    @property
    def nbytes(self):
        if self.data is None:
            return 0
        else:
            return self.data.capacity

    def allocate(
        self,
    ):
        # Delete data if exists
        if self.data is not None:
            del self.data

        # Allocate data
        if not np.any([s == 0 for s in self.data_shape]):
            self.data = wp.zeros(
                self.data_shape,
                dtype=self.dtype,
                pinned=True if self.device == "cpu" else None,
                device=self.device,
            )

    @staticmethod
    def _box_intersection(
        extent_1,
        offset_1,
        extent_2,
        offset_2,
        global_extent,
    ):
        # Get min and max of boxes
        min_1 = offset_1
        max_1 = offset_1 + extent_1
        min_2 = offset_2
        max_2 = offset_2 + extent_2

        # Get intersection
        min_intersection = np.maximum(min_1, min_2)
        max_intersection = np.minimum(max_1, max_2)
        extent_intersection = np.maximum(max_intersection - min_intersection, 0)
        offset_intersection = min_intersection

        # Check if intersection is valid
        return extent_intersection, offset_intersection


class Block:
    def __init__(
        self,
        extent,
        offset,
        origin,
        spacing,
        ghost_cell_thickness,
        device,
        pid=0,
    ):
        # Set the parameters
        self.extent = np.array(extent, dtype=np.int32)
        self.offset = np.array(offset, dtype=np.int32)
        self.origin = np.array(origin, dtype=np.float32)
        self.spacing = np.array(spacing, dtype=np.float32)
        self.ghost_cell_thickness = np.array(ghost_cell_thickness, dtype=np.int32)
        self.device = device
        self.pid = pid

        # Make set for neighbour blocks
        self.neighbour_blocks = set()

        # Make dict for boxes
        self.boxes = {}

        # Make dict for particles
        self.particles = {}

        # Make list for ghost boxes/cells
        self.local_ghost_boxes = {}  # Ghost boxes from local edges, these will be sent to neighbours
        self.neighbour_ghost_boxes = {}  # Neighbour blocks
        self.neighbour_ghost_boxes_buffer = {}  # Buffer for receiving ghost boxes

    @property
    def extent_with_ghost(self):
        return self.extent + 2 * self.ghost_cell_thickness

    @property
    def shape(self):
        return tuple(self.extent)

    @property
    def shape_with_ghost(self):
        return tuple(self.extent_with_ghost)

    @property
    def offset_with_ghost(self):
        return self.offset - self.ghost_cell_thickness

    @property
    def local_origin(self):
        return self.origin + np.array(self.offset) * np.array(self.spacing)

    @property
    def local_spacing(self):
        return np.array(self.spacing)

    @property
    def nbytes(self):
        nbytes = 0
        for box in self.boxes.values():
            nbytes += box.nbytes
        for boxes in self.local_ghost_boxes.values():
            for box in boxes.values():
                nbytes += box.nbytes
        for boxes in self.neighbour_ghost_boxes.values():
            for box in boxes.values():
                nbytes += box.nbytes
        for boxes in self.neighbour_ghost_boxes_buffer.values():
            for box in boxes.values():
                nbytes += box.nbytes
        return nbytes

    def add_neighbour_block(self, block):
        self.neighbour_blocks.add(block)
        self.local_ghost_boxes[block] = {}
        self.neighbour_ghost_boxes[block] = {}
        self.neighbour_ghost_boxes_buffer[block] = {}

    def remove_neighbour_block(self, block):
        if block in self.neighbour_blocks:
            self.neighbour_blocks.remove(block)
            self.local_ghost_boxes.pop(block)
            self.neighbour_ghost_boxes.pop(block)
            self.neighbour_ghost_boxes_buffer.pop(block)

    def initialize_box(
        self,
        name,
        dtype,
        cardinality,
        ordering,
        global_extent,
        extent=None,
        offset=None,
    ):
        # Get extent and offset, if None use whole domain
        extent = extent if extent is not None else global_extent
        offset = offset if offset is not None else np.zeros(len(global_extent), dtype=np.int32)

        # Get intersection
        local_extent, local_offset = Box._box_intersection(
            extent,
            offset,
            self.extent,
            self.offset,
            global_extent,
        )

        # Add box to block
        self.boxes[name] = Box(
            local_extent,
            local_offset,
            origin=self.origin,
            spacing=self.spacing,
            cardinality=cardinality,
            ordering=ordering,
            dtype=dtype,
            device=self.device,
        )

        # Initialize local ghost boxes
        for neighbour_block in self.neighbour_blocks:
            # Get intersections
            local_ghost_extent, local_ghost_offset = Box._box_intersection(
                extent,
                offset,
                neighbour_block.extent_with_ghost,
                neighbour_block.offset_with_ghost,
                global_extent,
            )
            local_ghost_extent, local_ghost_offset = Box._box_intersection(
                self.extent,
                self.offset,
                local_ghost_extent,
                local_ghost_offset,
                global_extent,
            )
            neighbour_ghost_extent, neighbour_ghost_offset = Box._box_intersection(
                extent,
                offset,
                self.extent_with_ghost,
                self.offset_with_ghost,
                global_extent,
            )
            neighbour_ghost_extent, neighbour_ghost_offset = Box._box_intersection(
                neighbour_block.extent,
                neighbour_block.offset,
                neighbour_ghost_extent,
                neighbour_ghost_offset,
                global_extent,
            )

            # Add local ghost box
            self.local_ghost_boxes[neighbour_block][name] = Box(
                extent=local_ghost_extent,
                offset=local_ghost_offset,
                origin=self.origin,
                spacing=self.spacing,
                cardinality=cardinality,
                ordering=ordering,
                dtype=dtype,
                device=self.device,
            )

            # Add neighbour ghost box
            self.neighbour_ghost_boxes[neighbour_block][name] = Box(
                extent=neighbour_ghost_extent,
                offset=neighbour_ghost_offset,
                origin=self.origin,
                spacing=self.spacing,
                cardinality=cardinality,
                ordering=ordering,
                dtype=dtype,
                device=self.device,
            )
            self.neighbour_ghost_boxes_buffer[neighbour_block][name] = Box(
                extent=neighbour_ghost_extent,
                offset=neighbour_ghost_offset,
                origin=self.origin,
                spacing=self.spacing,
                cardinality=cardinality,
                ordering=ordering,
                dtype=dtype,
                device=self.device,
            )

    def initialize_particles(
        self,
    ):
        pass

    def allocate(
        self,
    ):
        for box in self.boxes.values():
            box.allocate()
        for boxes in self.local_ghost_boxes.values():
            for box in boxes.values():
                box.allocate()
        for boxes in self.neighbour_ghost_boxes.values():
            for box in boxes.values():
                box.allocate()
        for boxes in self.neighbour_ghost_boxes_buffer.values():
            for box in boxes.values():
                box.allocate()

    def send_ghost_boxes(self, comm=None, comm_tag=0, names=None):
        # Get names if None
        if names is None:
            names = list(self.boxes.keys())

        # Get pid
        if comm is not None:
            pid = comm.Get_rank()
        else:
            pid = 0

        # Make list for send requests
        requests = []

        # Loop over neighbour blocks
        for neighbour_block, ghost_boxes in self.local_ghost_boxes.items():
            # Loop over ghost boxes
            for name, ghost_box in ghost_boxes.items():
                # Check if required to send
                if name not in names:
                    continue

                # 4 Cases:
                # 1. Current pid is the same as block and neighbour
                # 2. Current pid is the same as block but different than neighbour
                # 3. Current pid is different than block but the same as neighbour
                # 4. Current pid is different than block and neighbour
                # Case 1
                if (pid == self.pid) and (pid == neighbour_block.pid):
                    # Swap data
                    local_data = self.local_ghost_boxes[neighbour_block][name].data
                    neighbour_data = neighbour_block.neighbour_ghost_boxes_buffer[self][name].data
                    self.local_ghost_boxes[neighbour_block][name].data = neighbour_data
                    neighbour_block.neighbour_ghost_boxes_buffer[self][name].data = local_data

                # Case 2
                if (pid == self.pid) and (pid != neighbour_block.pid):
                    # Send data
                    requests.append(
                        comm.Isend(
                            self.local_ghost_boxes[neighbour_block][name].data,
                            dest=neighbour_block.pid,
                            tag=comm_tag,
                        )
                    )

                # Case 3
                if (pid != self.pid) and (pid == neighbour_block.pid):
                    # Receive data
                    requests.append(
                        comm.Irecv(
                            neighbour_block.neighbour_ghost_boxes_buffer[self][name].data,
                            source=self.pid,
                            tag=comm_tag,
                        )
                    )

                # Case 4
                if (pid != self.pid) and (pid != neighbour_block.pid):
                    pass

                # Update tag
                comm_tag += 1

        return requests, comm_tag

    def to_image_data(
        self,
        include_ghost=False,  # Just for debugging
    ):
        # Return grids
        grids = []

        # Make function for converting data
        def _convert_data(box):
            if box.data is not None:
                if box.ordering == "AOS":
                    return box.data.numpy().reshape((-1, box.cardinality), order="F")
                elif box.ordering == "SOA":
                    np_data = box.data.numpy()
                    aos_data = np.stack([np_data[i, ...] for i in range(box.cardinality)], axis=-1)
                    return aos_data.reshape((-1, box.cardinality), order="F")

        # Make center image data
        grid = pv.ImageData(
            dimensions=np.array(self.shape) + 1,
            origin=self.local_origin,
            spacing=self.local_spacing,
        )

        # Add data
        for name, box in self.boxes.items():
            grid.cell_data[name] = _convert_data(box)

        # Add grid to grids
        grids.append(grid)

        # Add ghost data
        if include_ghost:
            # Add local ghost data
            for ghost_boxes in self.local_ghost_boxes.values():
                for ghost_name, ghost_box in ghost_boxes.items():
                    grid = pv.ImageData(
                        dimensions=np.array(ghost_box.shape) + 1,
                        origin=ghost_box.local_origin,
                        spacing=ghost_box.local_spacing,
                    )
                    grid.cell_data[ghost_name + "_local_ghost"] = _convert_data(ghost_box)
                    grids.append(grid)
                for ghost_boxes in self.neighbour_ghost_boxes.values():
                    for ghost_name, ghost_box in ghost_boxes.items():
                        grid = pv.ImageData(
                            dimensions=np.array(ghost_box.shape) + 1,
                            origin=ghost_box.local_origin,
                            spacing=ghost_box.local_spacing,
                        )
                        grid.cell_data[name + "_neighbour_ghost"] = _convert_data(ghost_box)
                        grids.append(grid)
        return grids

    def swap_buffers(self, names=None):
        for neighbour_boxes, neighbour_boxes_buffer in zip(self.neighbour_ghost_boxes.values(), self.neighbour_ghost_boxes_buffer.values()):
            if names is None:
                names = list(neighbour_boxes.keys())
            for name in names:
                neighbour_boxes[name].data, neighbour_boxes_buffer[name].data = neighbour_boxes_buffer[name].data, neighbour_boxes[name].data


class OOCGrid:
    """An out-of-core Adaptive Mesh Refinement grid data structure."""

    def __init__(
        self,
        shape,
        block_shape,
        origin=None,
        spacing=None,
        ghost_cell_thickness=1,
        comm=None,
        pid_device_mapping=None,
    ):
        """Initialize the out-of-core data structure."""

        # Set the parameters
        self.shape = shape
        self.block_shape = block_shape
        self.origin = origin if origin is not None else tuple(np.zeros(len(shape)))
        self.spacing = spacing if spacing is not None else tuple(np.ones(len(shape)))
        if isinstance(ghost_cell_thickness, int):
            ghost_cell_thickness = (ghost_cell_thickness,) * len(shape)
        self.ghost_cell_thickness = ghost_cell_thickness
        self.comm = comm

        # Check that the block shape divides the shape
        self.block_dims = tuple([shape[i] // block_shape[i] for i in range(len(shape))])

        # Get process id and number of processes
        if comm is None:
            self.pid = 0
            self.size = 1
        else:
            self.pid = comm.Get_rank()
            self.size = comm.Get_size()
        if pid_device_mapping is None:
            pid_device_mapping = ["cpu" for _ in range(self.size)]

        # dist = np.arange(self.block_dims[0] * self.block_dims[1] * self.block_dims[2])
        # np.random.shuffle(dist)

        # Initialize blocks and connections
        logging.info("Initializing blocks and connections...")
        self.blocks = {}
        for index, block_index in tqdm(enumerate(itertools.product(*[range(n) for n in self.block_dims]))):
            # Get dist
            # dist = hilb.distance_from_point([block_index[2], block_index[1], block_index[0]])
            dist = index

            # Get block pid
            # block_pid = (dist // self.size) % self.size
            block_pid = (dist) % self.size

            # Get device
            device = pid_device_mapping[block_pid]

            # Create block
            block = Block(
                extent=block_shape,
                offset=[i * s for i, s in zip(block_index, block_shape)],
                origin=self.origin,
                spacing=self.spacing,
                ghost_cell_thickness=ghost_cell_thickness,
                device=device,
                pid=block_pid,
            )

            # Add block to blocks
            self.blocks[block_index] = block

        # Initialize connections
        for block_index in self.blocks.keys():
            # Get neighbour block indices
            for direction in itertools.product(*[range(-1, 2) for _ in self.block_dims]):
                # Skip if no neighbour
                if np.all([d == 0 for d in direction]):
                    continue

                # Get neighbour block index
                # neigh_block_index = tuple([(i + d) % n for i, d, n in zip(block_index, direction, self.block_dims)])
                neigh_block_index = tuple([(i + d) for i, d in zip(block_index, direction)])

                # Add neighbour block to block
                if neigh_block_index in self.blocks:
                    self.blocks[block_index].add_neighbour_block(self.blocks[neigh_block_index])

        # Barrier
        if self.comm is not None:
            self.comm.Barrier()

    @property
    def nbytes(self):
        nbytes = 0
        for block in self.blocks.values():
            nbytes += block.nbytes
        # if self.comm is not None:
        #    nbytes = self.comm.allreduce(nbytes, op=MPI.SUM)
        return nbytes

    def initialize_boxes(
        self,
        name,
        dtype,
        cardinality,
        ordering="SOA",
        extent=None,
        offset=None,
    ):
        # Initialize boxes
        for block in self.blocks.values():
            # Initialize box
            block.initialize_box(
                name=name,
                dtype=dtype,
                cardinality=cardinality,
                ordering=ordering,
                global_extent=self.shape,
                extent=extent,
                offset=offset,
            )

    def initialize_particles(
        self,
    ):
        pass

    def allocate(
        self,
    ):
        for block in self.blocks.values():
            if block.pid == self.pid:
                block.allocate()

    def save_vtm(
        self,
        filename,
    ):
        # Create multi block dataset
        mb = pv.MultiBlock()

        # Loop over blocks
        for block in self.blocks.values():
            # Add block to multi block
            mb.extend(block.to_image_data())

        # Save multi block
        mb.save(filename)


class MemoryPool:
    def __init__(self):
        self.pool = {}

    def clear(self):
        for key in list(self.pool.keys()):
            for array in self.pool[key]:
                del array
            del self.pool[key]
            self.pool[key] = []
        wp.synchronize()
        gc.collect()

    def get(self, shape, dtype, requires_grad=False):
        key = (tuple(shape), dtype, requires_grad)
        if key not in self.pool:
            self.pool[key] = []
        if len(self.pool[key]) == 0:
            self.pool[key].append(wp.zeros(shape, dtype=dtype, requires_grad=requires_grad))
        return self.pool[key].pop()

    def ret(self, array, zero=True):
        key = (tuple(array.shape), array.dtype, array.requires_grad)
        if zero:
            array.zero_()
            if array.requires_grad:
                array.grad.zero_()
        self.pool[key].append(array)

    @property
    def nbytes(self):
        nbytes = 0
        for key in self.pool.keys():
            for array in self.pool[key]:
                nbytes += array.capacity
        return nbytes
