import numpy as np
from xlb.compute_backend import ComputeBackend


def check_bc_overlaps(bclist, dim, compute_backend):
    index_list = [[] for _ in range(dim)]
    for bc in bclist:
        if bc.indices is None:
            continue
        # Detect duplicates within bc.indices
        index_arr = np.unique(bc.indices, axis=-1)
        if index_arr.shape[-1] != len(bc.indices[0]):
            if compute_backend == ComputeBackend.WARP:
                raise ValueError(f"Boundary condition {bc.__class__.__name__} has duplicate indices!")
            print(f"WARNING: there are duplicate indices in {bc.__class__.__name__} and hence the order in bc list matters!")
        for d in range(dim):
            index_list[d] += bc.indices[d]

    # Detect duplicates within bclist
    index_arr = np.unique(index_list, axis=-1)
    if index_arr.shape[-1] != len(index_list[0]):
        if compute_backend == ComputeBackend.WARP:
            raise ValueError("Boundary condition list containes duplicate indices!")
        print("WARNING: there are duplicate indices in the boundary condition list and hence the order in this list matters!")
