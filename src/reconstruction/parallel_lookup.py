from numba import jit, prange
import numpy as np

@jit(nopython=True, parallelize=True)
def lookup(L, Q):
    """
    This function relies on numba to parallelize lookup on L given Q.

    Parameters
    ----------
    L : HW x Z x C numpy array of float32
        flattened lookup table
    Q : HW x C numpy array of float32
        flattened normalized image to query on the lookup table
    
    Returns
    -------
    minD : HW numpy array of int32
        index of the minimum value along Z axis
    loss_map : HW numpy array of float32
        minimum value along Z axis
    """
    HW, Z, C = L.shape

    minD = np.zeros(HW, dtype=np.int32)
    loss = np.zeros(HW, dtype=np.float32)

    for i in prange(HW):
        best_idx = -1
        best_val = 1e30

        for j in range(Z):
            dist = 0.0
            for k in range(C):
                diff = L[i, j, k] - Q[i, k]
                dist += diff * diff

            if dist < best_val:
                best_val = dist
                best_idx = j

        minD[i] = best_idx
        loss[i] = best_val

    return minD, loss