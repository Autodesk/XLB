from typing import List
import os
from mpi4py import MPI
import pyvista as pv
import numpy as np
import warp as wp
import itertools
import xml.etree.ElementTree as ET

from ds.ooc_grid import MemoryPool
from operators.operator import Operator
from ..subroutine import Subroutine
from operators.copy.soa_copy import SOACopy

class VolumeSaverSubroutine(Subroutine):

    def __init__(
        self,
        macroscopic: Operator,
        q_criterion: Operator,
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.macroscopic = macroscopic
        self.q_criterion = q_criterion
        super().__init__(nr_streams, wp_streams, memory_pools)

    @staticmethod
    def combine_vtks(files, output_file):
    
        # Create the root element
        vtk_file = ET.Element('VTKFile', type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian", header_type="UInt32", compressor="vtkZLibDataCompressor")
        vtk_multi_block_data_set = ET.SubElement(vtk_file, 'vtkMultiBlockDataSet')
    
        # Create the DataSet elements
        for i, file in enumerate(files):
            data_set = ET.SubElement(vtk_multi_block_data_set, 'DataSet', index=str(i), name=f'Block-{str(i).zfill(5)}', file=file)
    
        # Create the tree
        tree = ET.ElementTree(vtk_file)
    
        # Write the tree to a file
        tree.write(output_file, encoding='utf-8', xml_declaration=True, method='xml')

    def __call__(
        self,
        amr_grid,
        file_name: str="initial.vtm",
        save_u: bool=True,
        save_rho: bool=True,
        save_q: bool=True,
        save_norm_mu: bool=True,
        f_name = "f",
        boundary_id_name = "boundary_id",
        missing_mask_name = "missing_mask",
        clear_memory_pools = True,
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
        for idx, block in enumerate(amr_grid.blocks.values()):

            # Check if block matches pid 
            if block.pid != amr_grid.pid:
                continue

            # Get block cardinality
            q = block.boxes[f_name].cardinality

            # Get compute arrays
            f = self.memory_pools[0].get((q, *block.shape), wp.float32)
            boundary_id = self.memory_pools[0].get((1, *block.shape), wp.uint8)
            rho = self.memory_pools[0].get((1, *block.shape), wp.float32)
            u = self.memory_pools[0].get((3, *block.shape), wp.float32)
            norm_mu = self.memory_pools[0].get((1, *block.shape), wp.float32)
            q = self.memory_pools[0].get((1, *block.shape), wp.float32)

            # Copy from block
            wp.copy(
                f,
                block.boxes[f_name].data
            )
            wp.copy(
                boundary_id,
                block.boxes[boundary_id_name].data
            )

            # Compute q criterion
            rho, u = self.macroscopic(f, boundary_id, rho, u)
            norm_mu, q = self.q_criterion(u, boundary_id, norm_mu, q)

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
                return np_data.reshape((-1, np_data.shape[-1]), order='F')
            np_rho = _convert_data(rho)
            np_u = _convert_data(u)
            np_norm_mu = _convert_data(norm_mu)
            np_q = _convert_data(q)

            # Add data to grid
            if save_rho:
                grid.cell_data["rho"] = np_rho
            if save_u:
                grid.cell_data["u"] = np_u
            if save_norm_mu:
                grid.cell_data["norm_mu"] = np_norm_mu
            if save_q:
                grid.cell_data["q"] = np_q

            # Save the grid
            post_fix = f"{file_name.split('/')[-1]}"
            vtk_file_name = os.path.join(file_name, f"{post_fix}_{idx}.vti")
            grid.save(vtk_file_name)
            files.append(f"{post_fix}/{post_fix}_{idx}.vti")

            # Return arrays
            self.memory_pools[0].ret(f)
            self.memory_pools[0].ret(boundary_id)
            self.memory_pools[0].ret(rho)
            self.memory_pools[0].ret(u)
            self.memory_pools[0].ret(norm_mu)
            self.memory_pools[0].ret(q)

        # Clear memory pools
        for memory_pool in self.memory_pools:
            memory_pool.clear()

        # Get all the files
        if amr_grid.comm is not None:
            files = amr_grid.comm.gather(files, root=0)
            if amr_grid.comm.rank == 0:
                files = list(itertools.chain(*files))

        # Combine the files
        if amr_grid.comm is not None:
            if amr_grid.comm.rank == 0:
                self.combine_vtks(files, f"{file_name}.vtm")
        else:
            self.combine_vtks(files, f"{file_name}.vtm")
