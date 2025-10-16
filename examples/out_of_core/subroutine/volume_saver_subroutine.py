from typing import List
import itertools
import os
import pyvista as pv
import numpy as np
import warp as wp
import xml.etree.ElementTree as ET

from ds.ooc_grid import MemoryPool
from subroutine.subroutine import Subroutine


class VolumeSaverSubroutine(Subroutine):
    def __init__(
        self,
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        super().__init__(nr_streams, wp_streams, memory_pools)

    @staticmethod
    def combine_vtks(files, output_file):
        # Create the root element
        vtk_file = ET.Element(
            "VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian", header_type="UInt32", compressor="vtkZLibDataCompressor"
        )
        vtk_multi_block_data_set = ET.SubElement(vtk_file, "vtkMultiBlockDataSet")

        # Create the DataSet elements
        for i, file in enumerate(files):
            data_set = ET.SubElement(vtk_multi_block_data_set, "DataSet", index=str(i), name=f"Block-{str(i).zfill(5)}", file=file)

        # Create the tree
        tree = ET.ElementTree(vtk_file)

        # Write the tree to a file
        tree.write(output_file, encoding="utf-8", xml_declaration=True, method="xml")

    def __call__(
        self,
        ooc_grid,
        field_names: List[str],
        file_name: str = "initial.vtm",
        clear_memory_pools=True,
    ):
        """
        Save the solid id array.
        """

        # Make directory
        os.makedirs(file_name, exist_ok=True)

        # Clear memory pools
        for memory_pool in self.memory_pools:
            memory_pool.clear()

        # Store the files
        files = []

        # Loop over blocks
        for idx, block in enumerate(ooc_grid.blocks.values()):
            # Check if block matches pid
            if block.pid != ooc_grid.pid:
                continue

            # Make grid
            grid = pv.ImageData(
                dimensions=np.array(block.shape) + 1,
                origin=block.local_origin,
                spacing=block.local_spacing,
            )

            # Convert data
            def _convert_data(data):
                np_data = data.numpy()
                np_data = np.stack([np_data[i, ...] for i in range(np_data.shape[0])], axis=-1)
                return np_data.reshape((-1, np_data.shape[-1]), order="F")

            # Add fields
            for field_name in field_names:
                np_field = _convert_data(block.boxes[field_name].data)
                grid.cell_data[field_name] = np_field

            # Save the grid
            post_fix = f"{file_name.split('/')[-1]}"
            vtk_file_name = os.path.join(file_name, f"{post_fix}_{idx}.vti")
            grid.save(vtk_file_name)
            files.append(f"{post_fix}/{post_fix}_{idx}.vti")

        # Get all the files
        if ooc_grid.comm is not None:
            files = ooc_grid.comm.gather(files, root=0)
            if ooc_grid.comm.rank == 0:
                files = list(itertools.chain(*files))

        # Combine the files
        if ooc_grid.comm is not None:
            if ooc_grid.comm.rank == 0:
                self.combine_vtks(files, f"{file_name}.vtm")
        else:
            self.combine_vtks(files, f"{file_name}.vtm")
