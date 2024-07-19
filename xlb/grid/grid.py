from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision


def grid_factory(
    shape: Tuple[int, ...],
    compute_backend: ComputeBackend = None):
    compute_backend = compute_backend or DefaultConfig.default_backend
    if compute_backend == ComputeBackend.WARP:
        from xlb.grid.warp_grid import WarpGrid

        return WarpGrid(shape)
    elif compute_backend == ComputeBackend.JAX:
        from xlb.grid.jax_grid import JaxGrid

        return JaxGrid(shape)

    raise ValueError(f"Compute backend {compute_backend} is not supported")


class Grid(ABC):
    def __init__(self, shape: Tuple[int, ...], compute_backend: ComputeBackend):
        self.shape = shape
        self.dim = len(shape)
        self.compute_backend = compute_backend
        self._bounding_box_indices()
        self._initialize_backend()

    @abstractmethod
    def _initialize_backend(self):
        pass

    def _bounding_box_indices(self):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.
        
        Returns
        -------
            boundingBox (dict): A dictionary where keys are the names of the bounding box faces
            ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
            are numpy arrays of indices corresponding to each face.
        """
        def to_tuple(list):
            d = len(list[0])
            return [tuple([sublist[i] for sublist in list]) for i in range(d)]

        if self.dim == 2:
            # For a 2D grid, the bounding box consists of four edges: bottom, top, left, and right.
            # Each edge is represented as an array of indices. For example, the bottom edge includes
            # all points where the y-coordinate is 0, so its indices are [[i, 0] for i in range(nx)].
            nx, ny = self.shape
            self.boundingBoxIndices = {
                "bottom": to_tuple([[i, 0] for i in range(nx)]),
                "top": to_tuple([[i, ny - 1] for i in range(nx)]),
                "left": to_tuple([[0, i] for i in range(ny)]),
                "right": to_tuple([[nx - 1, i] for i in range(ny)])
            }
                            
        elif self.dim == 3:
            # For a 3D grid, the bounding box consists of six faces: bottom, top, left, right, front, and back.
            # Each face is represented as an array of indices. For example, the bottom face includes all points
            # where the z-coordinate is 0, so its indices are [[i, j, 0] for i in range(nx) for j in range(ny)].
            nx, ny, nz = self.shape
            self.boundingBoxIndices = {
                "bottom": to_tuple([[i, j, 0] for i in range(nx) for j in range(ny)]),
                "top": to_tuple([[i, j, nz - 1] for i in range(nx) for j in range(ny)]),
                "left": to_tuple([[0, j, k] for j in range(ny) for k in range(nz)]),
                "right": to_tuple([[nx - 1, j, k] for j in range(ny) for k in range(nz)]),
                "front": to_tuple([[i, 0, k] for i in range(nx) for k in range(nz)]),
                "back": to_tuple([[i, ny - 1, k] for i in range(nx) for k in range(nz)])
            }
        return 

