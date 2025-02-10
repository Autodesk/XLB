import numpy as np
import warp as wp
from tqdm import tqdm
import logging
import itertools
import pyvista as pv
from dataclasses import dataclass
from mpi4py import MPI
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve

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

    ## Modulo global extent
    #min_1 = np.mod(min_1, global_extent)
    #max_1 = np.mod(max_1, global_extent)
    #min_2 = np.mod(min_2, global_extent)
    #max_2 = np.mod(max_2, global_extent)

    # Get intersection
    min_intersection = np.maximum(min_1, min_2)
    max_intersection = np.minimum(max_1, max_2)
    extent_intersection = np.maximum(max_intersection - min_intersection, 0)
    offset_intersection = min_intersection

    # Check if intersection is valid
    if np.any(extent_intersection == 0):
        return None
    else:
        return extent_intersection, offset_intersection

class Box:

    def __init__(
        self,
        extent,
        offset,
        origin,
        spacing,
        amr_level,
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
        self.amr_level = amr_level
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
        return np.array(self.spacing) * 2 ** self.amr_level

    @property
    def nbytes(self):
        if self.data is None:
            return 0
        else:
            return self.data.capacity

    @property
    def shape(self):
        return tuple([s // 2 ** self.amr_level for s in self.extent])

    @property
    def data_shape(self):
        if self.ordering == "AOS":
            return list(self.shape) + [self.cardinality]
        elif self.ordering == "SOA":
            return [self.cardinality] + list(self.shape)

    def allocate(
        self,
    ):

        # Delete data if exists
        if self.data is not None:
            del self.data

        # Allocate data
        self.data = wp.zeros(
            self.data_shape,
            dtype=self.dtype,
            pinned=True if self.device == "cpu" else None,
            device=self.device,
        )

class Block:

    def __init__(
        self,
        extent,
        offset,
        origin,
        spacing,
        amr_level,
        ghost_cell_thickness,
        device,
        pid=0,
    ):

        # Set the parameters
        self.extent = np.array(extent, dtype=np.int32)
        self.offset = np.array(offset, dtype=np.int32)
        self.origin = np.array(origin, dtype=np.float32)
        self.spacing = np.array(spacing, dtype=np.float32)
        self.amr_level = amr_level
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
        self.local_ghost_boxes = {} # Ghost boxes from local edges, these will be sent to neighbours
        self.neighbour_ghost_boxes = {} # Neighbour blocks
        self.neighbour_ghost_boxes_buffer = {} # Buffer for receiving ghost boxes

    @property
    def shape(self):
        return tuple([s // 2 ** self.amr_level for s in self.extent])

    @property
    def shape_with_ghost(self):
        return tuple([s + 2 * self.ghost_cell_thickness for s in self.shape])

    @property
    def extent_with_ghost(self):
        return self.extent + 2 * self.ghost_cell_thickness

    @property
    def offset_with_ghost(self):
        return self.offset - self.ghost_cell_thickness

    @property
    def local_origin(self):
        return self.origin + np.array(self.offset) * np.array(self.spacing)

    @property
    def local_spacing(self):
        return np.array(self.spacing) * 2 ** self.amr_level

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

        # Get extent and offset
        extent = extent if extent is not None else self.extent
        offset = offset if offset is not None else self.offset

        # Get intersection
        intersection = _box_intersection(
            extent,
            offset,
            self.extent,
            self.offset,
            global_extent,
        )

        # Add box to block
        if intersection is not None:
            self.boxes[name] = Box(
                extent=intersection[0],
                offset=intersection[1],
                origin=self.origin,
                spacing=self.spacing,
                amr_level=0,
                cardinality=cardinality,
                ordering=ordering,
                dtype=dtype,
                device=self.device,
            )

        # Initialize local ghost boxes
        for neighbour_block in self.neighbour_blocks:

            # Get intersections
            local_ghost_intersection = _box_intersection(
                extent,
                offset,
                neighbour_block.extent_with_ghost,
                neighbour_block.offset_with_ghost,
                global_extent,
            )
            neighbour_ghost_intersection = _box_intersection(
                extent + 2 * self.ghost_cell_thickness,
                offset - self.ghost_cell_thickness,
                neighbour_block.extent,
                neighbour_block.offset,
                global_extent,
            )

            # Add local ghost box
            if local_ghost_intersection is not None:
                self.local_ghost_boxes[neighbour_block][name] = Box(
                    extent=local_ghost_intersection[0],
                    offset=local_ghost_intersection[1],
                    origin=self.origin,
                    spacing=self.spacing,
                    amr_level=0,
                    cardinality=cardinality,
                    ordering=ordering,
                    dtype=dtype,
                    device=self.device,
                )

            # Add neighbour ghost box
            if neighbour_ghost_intersection is not None:
                self.neighbour_ghost_boxes[neighbour_block][name] = Box(
                    extent=neighbour_ghost_intersection[0],
                    offset=neighbour_ghost_intersection[1],
                    origin=self.origin,
                    spacing=self.spacing,
                    amr_level=0,
                    cardinality=cardinality,
                    ordering=ordering,
                    dtype=dtype,
                    device=self.device,
                )
                self.neighbour_ghost_boxes_buffer[neighbour_block][name] = Box(
                    extent=neighbour_ghost_intersection[0],
                    offset=neighbour_ghost_intersection[1],
                    origin=self.origin,
                    spacing=self.spacing,
                    amr_level=0,
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

    def send_ghost_boxes(
        self,
        comm=None,
        comm_tag=0,
        names=None
    ):

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
        include_ghost=False, # Just for debugging
    ):

        # Return grids
        grids = []

        # Make function for converting data
        def _convert_data(box):
            if box.data is not None:
                if box.ordering == "AOS":
                    return box.data.numpy().reshape((-1, box.cardinality), order='F')
                elif box.ordering == "SOA":
                    np_data = box.data.numpy()
                    aos_data = np.stack([np_data[i, ...] for i in range(box.cardinality)], axis=-1)
                    return aos_data.reshape((-1, box.cardinality), order='F')

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

    def swap_buffers(
        self,
        names=None
    ):

        for neighbour_boxes, neighbour_boxes_buffer in zip(self.neighbour_ghost_boxes.values(), self.neighbour_ghost_boxes_buffer.values()):
            if names is None:
                names = list(neighbour_boxes.keys())
            for name in names:
                neighbour_boxes[name].data, neighbour_boxes_buffer[name].data = neighbour_boxes_buffer[name].data, neighbour_boxes[name].data

    def mergable(self, block):
        pass

        ## Check if blocks are on the same level
        #if any([self.blocks[name].amr_level != block.blocks[name].amr_level for name in self.blocks.keys()]):
        #    return False

        ## Check if blocks are adjacent
        #for i in range(len(self.extent)):
        #    if self.offset[i] + self.extent[i] != block.offset[i] and block.offset[i] + block.extent[i] != self.offset[i]:
        #        return False

        #return True

    def merge(self, block):
        pass

        ## Check if blocks are mergable
        #if not self.mergable(block):
        #    raise ValueError("Blocks are not mergable.")

        ## Get new extent
        #new_extent = tuple([max(e1, e2) for e1, e2 in zip(self.extent, block.extent)])

        ## Get new offset
        #new_offset = tuple([min(o1, o2) for o1, o2 in zip(self.offset, block.offset)])

        ## Set new extent and offset
        #self.extent = new_extent
        #self.offset = new_offset

        ## Add neighbour blocks
        #self.neighbour_blocks.update(block.neighbour_blocks)

        ## Remove blocks from neighbour blocks
        ## TODO

        ### Merge boxes
        ##for key, box in self.boxes.items():
        ##    if box is not None:
        ##        if key in block.boxes:
        ##            self.boxes[key] = box.merge(block.boxes[key])

        ### Merge particles
        ##for key, particle in self.particles.items():
        ##    if key in block.particles:
        ##        self.particles[key] = particle.merge(block.particles[key])

    def split(self, amr_level):
        pass


class AMR:
    """An out-of-core distributed array class."""

    def __init__(
        self,
        shape,
        block_shape,
        origin=None,
        spacing=None,
        nr_levels=1,
        ghost_cell_thickness=1,
        comm=None,
        pid_device_mapping=None,
    ):
        """Initialize the out-of-core data structure.
        """

        # Set the parameters
        self.shape = shape
        self.block_shape = block_shape
        self.origin = origin if origin is not None else tuple(np.zeros(len(shape)))
        self.spacing = spacing if spacing is not None else tuple(np.ones(len(shape)))
        self.nr_levels = nr_levels
        if isinstance(ghost_cell_thickness, int):
            ghost_cell_thickness = (ghost_cell_thickness,) * len(shape)
        self.ghost_cell_thickness = ghost_cell_thickness
        self.comm = comm

        # Check that the block shape divides the shape
        if any([shape[i] % block_shape[i] * 2 ** self.nr_levels != 0 for i in range(len(shape))]):
            raise ValueError(f"Block shape {block_shape} does not divide shape {shape}.")
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

        # Make hilbert curb
        hilb = HilbertCurve(
            p=int(np.log2(max(self.block_dims))) + 1,
            n=len(shape)
        )

        # Initialize blocks and connections
        logging.info("Initializing blocks and connections...")
        self.blocks = {}
        for index, block_index in tqdm(enumerate(itertools.product(*[range(n) for n in self.block_dims]))):

            # Get dist
            dist = hilb.distance_from_point([block_index[2], block_index[1], block_index[0]])

            # Get block pid
            block_pid = (dist // self.size) % self.size

            # Get device
            device = pid_device_mapping[block_pid]

            # Create block
            block = Block(
                extent=block_shape,
                offset=[i * s for i, s in zip(block_index, block_shape)],
                origin=self.origin,
                spacing=self.spacing,
                amr_level=0,
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
                #neigh_block_index = tuple([(i + d) % n for i, d, n in zip(block_index, direction, self.block_dims)])
                neigh_block_index = tuple([(i + d) for i, d in zip(block_index, direction)])

                # Add neighbour block to block
                if neigh_block_index in self.blocks:
                    self.blocks[block_index].add_neighbour_block(self.blocks[neigh_block_index])

        # Convert blocks to list
        self.blocks = list(self.blocks.values())

    @property
    def nbytes(self):
        nbytes = 0
        for block in self.blocks:
            nbytes += block.nbytes
        if self.comm is not None:
            nbytes = self.comm.allreduce(nbytes, op=MPI.SUM)
        return nbytes

    @staticmethod
    def morton3D(x, y, z):
        """
        Compute a 3D Morton code (Z-order curve) for (x, y, z).

        This version uses a naive loop to interleave bits.
        For large coordinates, you may want to adjust the bit range.
        """
        morton_code = 0
        # Up to 21 bits => up to 2^21 = 2,097,152 in each dimension.
        # Adjust if you have an even larger domain.
        for i in range(21):
            # Shift bits of x, y, z up into appropriate positions
            bit_mask = 1 << i
            x_bit = (x & bit_mask) >> i
            y_bit = (y & bit_mask) >> i
            z_bit = (z & bit_mask) >> i

            # Interleave these bits
            morton_code |= (x_bit << (3*i)) | (y_bit << (3*i + 1)) | (z_bit << (3*i + 2))

        return morton_code

    def initialize_boxes(
        self,
        name,
        dtype,
        cardinality,
        ordering,
        extent=None,
        offset=None,
    ):

        # Initialize boxes
        for block in self.blocks:

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
        for block in self.blocks:
            if block.pid == self.pid:
                block.allocate()

    def save_vtm(
        self,
        filename,
    ):

        # Create multi block dataset
        mb = pv.MultiBlock()

        # Loop over blocks
        for block in self.blocks:

            # Add block to multi block
            mb.extend(block.to_image_data())

        # Save multi block
        mb.save(filename)
