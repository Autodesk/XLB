from typing import List, Callable
import warp as wp
import numpy as np
from PIL import Image

from ds.ooc_grid import MemoryPool
from subroutine.subroutine import Subroutine
from operators.trilinear_interpolation import TrilinearInterpolation
from operators.mesh_renderer import MeshRenderer
from operators.color_mapper import ColorMapper
from operators.transform_mesh import TransformMesh
from operators.q_criterion import QCriterion


class RenderQCriterionSubroutine(Subroutine):
    def __init__(
        self,
        macroscopic: Callable,
        q_criterion: Callable = QCriterion(),
        mesh_renderer: Callable = MeshRenderer(),
        color_mapper: Callable = ColorMapper(),
        grid_to_point_interpolator: Callable = TrilinearInterpolation(),
        mesh_transformer: Callable = TransformMesh(),
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.macroscopic = macroscopic
        self.q_criterion = q_criterion
        self.mesh_renderer = mesh_renderer
        self.color_mapper = color_mapper
        self.grid_to_point_interpolator = grid_to_point_interpolator
        self.mesh_transformer = mesh_transformer
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        ooc_grid,
        image_name: str,
        pixel_buffer: wp.array(dtype=wp.float32),
        depth_buffer: wp.array(dtype=wp.float32),
        camera_pos: wp.vec3f,
        camera_target: wp.vec3f,
        camera_up: wp.vec3f,
        fov_degrees: float,
        ambient_intensity: float,
        edge_sharpness: float,
        gamma: float,
        q_criterion_threshold: float,
        vmin: float,
        vmax: float,
        boundary_mesh: wp.Mesh = None,
        boundary_color: wp.vec3f = None,
        f_name="f",
        boundary_id_name="boundary_id",
    ):
        # Make stream idx
        stream_idx = 0

        # Set Perform steps equal to the number of ghost cell thickness
        for block in ooc_grid.blocks.values():
            # Set warp stream
            with wp.ScopedStream(self.wp_streams[stream_idx]):
                # Check if block matches pid
                if block.pid == ooc_grid.pid:
                    # Get block cardinality
                    q = block.boxes[f_name].cardinality

                    # Get compute arrays
                    f = self.memory_pools[0].get((q, *block.shape), wp.float32)
                    boundary_id = self.memory_pools[0].get((1, *block.shape), wp.uint8)
                    rho = self.memory_pools[0].get((1, *block.shape), wp.float32)
                    u = self.memory_pools[0].get((3, *block.shape), wp.float32)
                    norm_mu = self.memory_pools[0].get((1, *block.shape), wp.float32)
                    q = self.memory_pools[0].get((1, *block.shape), wp.float32)

                    # Get marching cubes arrays
                    mc = wp.MarchingCubes(
                        nx=int(block.extent[0]),
                        ny=int(block.extent[1]),
                        nz=int(block.extent[2]),
                        max_verts=int(block.extent[0]) * int(block.extent[1]) * int(block.extent[2]) * 5,
                        max_tris=int(block.extent[0]) * int(block.extent[1]) * int(block.extent[2]) * 3,
                    )

                    # Copy from block
                    wp.copy(f, block.boxes[f_name].data)
                    wp.copy(boundary_id, block.boxes[boundary_id_name].data)

                    # Compute q criterion
                    rho, u = self.macroscopic(f, rho, u)
                    norm_mu, q = self.q_criterion(u, boundary_id, norm_mu, q)

                    # Perform marching cubes
                    mc.surface(q[0], q_criterion_threshold)

                    # Check if any vertices found
                    if mc.verts.shape[0] > 0:
                        # Make mesh
                        mesh = wp.Mesh(
                            points=mc.verts,
                            indices=mc.indices,
                        )

                        # Transform mesh
                        mesh = self.mesh_transformer(
                            mesh=mesh,
                            origin=block.local_origin,
                            scale=block.local_spacing,
                        )

                        # Get point data
                        scalars = wp.zeros((1, mc.verts.shape[0]), wp.float32)
                        scalars = self.grid_to_point_interpolator(
                            norm_mu,
                            mesh.points,
                            origin=block.local_origin,
                            spacing=block.local_spacing,
                            point_values=scalars,
                        )

                        # Map scalars to colors - reshape to 1D array first
                        vertex_colors = wp.zeros((mc.verts.shape[0], 3), wp.float32)
                        vertex_colors = self.color_mapper(
                            scalars[0, :],  # Take first channel only since q_criterion is scalar
                            vertex_colors,
                            vmin=vmin,
                            vmax=vmax,
                            colormap="jet",
                        )

                        # Render mesh
                        self.mesh_renderer(
                            mesh=mesh,
                            vertex_colors=vertex_colors,
                            pixel_buffer=pixel_buffer,
                            depth_buffer=depth_buffer,
                            camera_pos=camera_pos,
                            camera_target=camera_target,
                            camera_up=camera_up,
                            fov_degrees=fov_degrees,
                            ambient_intensity=ambient_intensity,
                            edge_sharpness=edge_sharpness,
                            gamma=gamma,
                        )

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

        # Render boundary mesh
        if boundary_mesh is not None:
            vertex_colors = wp.full((boundary_mesh.points.shape[0], 3), boundary_color, dtype=wp.float32)
            self.mesh_renderer(
                mesh=boundary_mesh,
                vertex_colors=vertex_colors,
                pixel_buffer=pixel_buffer,
                depth_buffer=depth_buffer,
                camera_pos=camera_pos,
                camera_target=camera_target,
                camera_up=camera_up,
                fov_degrees=fov_degrees,
                ambient_intensity=ambient_intensity,
                edge_sharpness=edge_sharpness,
                gamma=gamma,
            )

        # Get all the files
        if ooc_grid.comm is not None:
            # Set barrier
            ooc_grid.comm.Barrier()
            # Send buffers from non-root ranks to root
            if ooc_grid.comm.rank != 0:
                ooc_grid.comm.Send(pixel_buffer.numpy(), dest=0, tag=2 * ooc_grid.comm.rank)
                ooc_grid.comm.Send(depth_buffer.numpy(), dest=0, tag=2 * ooc_grid.comm.rank + 1)

            # Root rank receives and combines buffers
            if ooc_grid.comm.rank == 0:
                # Initialize with root rank's buffers
                np_pixel_buffer = pixel_buffer.numpy()
                np_depth_buffer = depth_buffer.numpy()

                # Receive buffers from other ranks
                for i in range(1, ooc_grid.comm.size):
                    # Create receive buffers with same shape as local buffers
                    other_pixel = np.empty_like(np_pixel_buffer)
                    other_depth = np.empty_like(np_depth_buffer)

                    # Receive pixel and depth buffers
                    ooc_grid.comm.Recv(other_pixel, source=i, tag=2 * i)
                    ooc_grid.comm.Recv(other_depth, source=i, tag=2 * i + 1)

                    # Update pixels where other depth is smaller
                    mask = other_depth < np_depth_buffer
                    np_pixel_buffer[mask] = other_pixel[mask]
                    np_depth_buffer[mask] = other_depth[mask]

                # Convert float buffer (0-1) to uint8 (0-255)
                np_pixel_buffer = (np_pixel_buffer[..., :3] * 255).astype(np.uint8)

                # Ensure correct shape and remove extra dimensions
                np_pixel_buffer = np_pixel_buffer.squeeze()

                # Save the combined image
                Image.fromarray(np_pixel_buffer).save(f"{image_name}.png")

        else:
            np_pixel_buffer = pixel_buffer.numpy()[..., :3]
            np_depth_buffer = depth_buffer.numpy()

            # Convert float buffer (0-1) to uint8 (0-255)
            np_pixel_buffer = (np_pixel_buffer * 255).astype(np.uint8)

            # Ensure correct shape and remove any extra dimensions
            np_pixel_buffer = np_pixel_buffer.squeeze()

            # Save the actual image
            Image.fromarray(np_pixel_buffer).save(f"{image_name}.png")
