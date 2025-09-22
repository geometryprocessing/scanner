from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def lookup_no_mask(L, Q):
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

@jit(nopython=True, parallel=True)
def lookup_with_mask(L, Q, mask):
    """
    This function relies on numba to parallelize lookup on L given Q.

    Parameters
    ----------
    L : HW x Z x C numpy array of float32
        flattened lookup table
    Q : HW x C numpy array of float32
        flattened normalized image to query on the lookup table
    mask : HW numpy of bool
        flattened mask where lookup should not be performed
        if mask[i] == False, then minD[i] = -1 and loss[i] = 1e30

    
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

        if mask[i]:
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

def lookup(L, Q, mask=None):
    """
    Overloaded parallel lookup function, where mask is an optional argument.

    Parameters
    ----------
    L : HW x Z x C numpy array of float32
        flattened lookup table
    Q : HW x C numpy array of float32
        flattened normalized image to query on the lookup table
    mask : HW numpy of bool, optional
        flattened mask 

    Returns
    -------
    minD : HW numpy array of int32
        index of the minimum value along Z axis
    loss_map : HW numpy array of float32
        minimum value along Z axis
    
    """
    if mask is None:
        return lookup_no_mask(L, Q)
    else:
        return lookup_with_mask(L, Q, mask)
