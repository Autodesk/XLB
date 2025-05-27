from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
import neon


def grid_factory(
    shape: Tuple[int, ...],
    compute_backend: ComputeBackend = None,
    velocity_set=None,
):
    compute_backend = compute_backend or DefaultConfig.default_backend
    velocity_set = velocity_set or DefaultConfig.velocity_set
    if compute_backend == ComputeBackend.WARP:
        from xlb.grid.warp_grid import WarpGrid

        return WarpGrid(shape)
    elif compute_backend == ComputeBackend.NEON:
        from xlb.grid.neon_grid import NeonGrid

        return NeonGrid(shape=shape, velocity_set=velocity_set)
    elif compute_backend == ComputeBackend.JAX:
        from xlb.grid.jax_grid import JaxGrid

        return JaxGrid(shape)

    raise ValueError(f"Compute backend {compute_backend} is not supported")


def multires_grid_factory(
    shape: Tuple[int, ...],
    compute_backend: ComputeBackend = None,
    velocity_set=None,
    sparsity_pattern_list: List[np.ndarray] = [],
    sparsity_pattern_origins: List[neon.Index_3d] = [],
):
    compute_backend = compute_backend or DefaultConfig.default_backend
    velocity_set = velocity_set or DefaultConfig.velocity_set
    if compute_backend == ComputeBackend.NEON:
        from xlb.grid.multires_grid import NeonMultiresGrid

        return NeonMultiresGrid(
            shape=shape, velocity_set=velocity_set, sparsity_pattern_list=sparsity_pattern_list, sparsity_pattern_origins=sparsity_pattern_origins
        )
    else:
        raise ValueError(f"Compute backend {compute_backend} is not supported for multires grid")


class Grid(ABC):
    def __init__(
        self,
        shape: Tuple[int, ...],
        compute_backend: ComputeBackend,
    ):
        self.shape = shape
        self.dim = len(shape)
        self.compute_backend = compute_backend
        self._initialize_backend()

    @abstractmethod
    def _initialize_backend(self):
        pass

    def get_compute_backend(self):
        return self.compute_backend

    def bounding_box_indices(self, shape=None, remove_edges=False):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Parameters
        ----------
        remove_edges : bool, optional
            If True, the nodes along the edges (not just the corners) are removed from the bounding box indices.
            Default is False.

        Returns
        -------
        boundingBox (dict): A dictionary where keys are the names of the bounding box faces
        ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
        are numpy arrays of indices corresponding to each face.
        """

        # If shape is not give, use self.shape
        if shape is None:
            shape = self.shape

        # Get the shape of the grid
        origin = np.array([0, 0, 0])
        bounds = np.array(shape)
        if remove_edges:
            origin += 1
            bounds -= 1
        slice_x = slice(origin[0], bounds[0])
        slice_y = slice(origin[1], bounds[1])
        dim = len(bounds)

        # Generate bounding box indices for each face
        grid = np.indices(shape)
        boundingBoxIndices = {}

        if dim == 2:
            nx, ny = shape
            boundingBoxIndices = {
                "bottom": grid[:, slice_x, 0],
                "top": grid[:, slice_x, ny - 1],
                "left": grid[:, 0, slice_y],
                "right": grid[:, nx - 1, slice_y],
            }
        elif dim == 3:
            nx, ny, nz = shape
            slice_z = slice(origin[2], bounds[2])
            boundingBoxIndices = {
                "bottom": grid[:, slice_x, slice_y, 0].reshape(3, -1),
                "top": grid[:, slice_x, slice_y, nz - 1].reshape(3, -1),
                "left": grid[:, 0, slice_y, slice_z].reshape(3, -1),
                "right": grid[:, nx - 1, slice_y, slice_z].reshape(3, -1),
                "front": grid[:, slice_x, 0, slice_z].reshape(3, -1),
                "back": grid[:, slice_x, ny - 1, slice_z].reshape(3, -1),
            }

        return {k: v.tolist() for k, v in boundingBoxIndices.items()}
