from numba import jit, prange, cuda
import numpy as np

@jit(nopython=True, parallel=True)
def ray_search(L, Q):
    j = -1
    best_val = 1e30
    Z, C = L.shape
    for j in range(Z):
        dist = 0.0
        for k in range(C):
            diff = L[j, k] - Q[k]
            dist += diff * diff

        if dist < best_val:
            best_val = dist
            best_idx = j

    return best_idx, best_val

@jit(nopython=True, parallel=True)
def lookup_3dim_no_mask(L, Q):
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

    minD = np.full(HW, fill_value=-1, dtype=np.int32)
    loss = np.full(HW, fill_value=1e30, dtype=np.float32)

    for i in prange(HW):
        best_idx, best_val = ray_search(L[i], Q[i])

        minD[i] = best_idx
        loss[i] = best_val

    return minD, loss

@jit(nopython=True, parallel=True)
def lookup_3dim_with_mask(L, Q, mask):
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

    minD = np.full(HW, fill_value=-1, dtype=np.int32)
    loss = np.full(HW, fill_value=1e30, dtype=np.float32)

    for i in prange(HW):
        if mask[i]:
            best_idx, best_val = ray_search(L[i], Q[i])

            minD[i] = best_idx
            loss[i] = best_val

    return minD, loss


@jit(nopython=True, parallel=True)
def lookup_4dim_no_mask(L, Q):
    """
    This function relies on numba to parallelize lookup on L given Q.

    Parameters
    ----------
    L : H x W x Z x C numpy array of float32
        lookup table
    Q : H x W x C numpy array of float32
        normalized image to query on the lookup table
    
    Returns
    -------
    minD : H x W numpy array of int32
        index of the minimum value along Z axis
    loss_map : H x W numpy array of float32
        minimum value along Z axis
    """
    H, W, Z, C = L.shape

    minD = np.full((H,W), fill_value=-1, dtype=np.int32)
    loss = np.full((H,W), fill_value=1e30, dtype=np.float32)

    for i in prange(H):
        for j in range(W):
            best_idx, best_val = ray_search(L[i,j], Q[i,j])

            minD[i,j] = best_idx
            loss[i,j] = best_val

    return minD, loss

@jit(nopython=True, parallel=True)
def lookup_4dim_with_mask(L, Q, mask):
    """
    This function relies on numba to parallelize lookup on L given Q.

    Parameters
    ----------
    L : H x W x Z x C numpy array of float32
        lookup table
    Q : H x W x C numpy array of float32
        normalized image to query on the lookup table
    mask : H x W numpy of bool
        mask where lookup should not be performed
        if mask[i,j] == False, then minD[i,j] = -1 and loss[i,j] = 1e30

    
    Returns
    -------
    minD : H x W numpy array of int32
        index of the minimum value along Z axis
    loss_map : H x W numpy array of float32
        minimum value along Z axis
    """
    H, W, Z, C = L.shape

    minD = np.full((H,W), fill_value=-1, dtype=np.int32)
    loss = np.full((H,W), fill_value=1e30, dtype=np.float32)

    for i in prange(H):
        for j in range(W):
            if mask[i,j]:
                best_idx, best_val = ray_search(L[i,j], Q[i,j])

                minD[i,j] = best_idx
                loss[i,j] = best_val

    return minD, loss


def lookup(L, Q, mask=None):
    """
    Overloaded parallel lookup function, where mask is an optional argument.

    Parameters
    ----------
    L : H x W x Z x C or HW x Z x C numpy array of float32
        lookup table
    Q : H x W x C or HW x C numpy array of float32
        normalized image to query on the lookup table
    mask : H x W or HW numpy of bool, optional
        mask 

    Returns
    -------
    minD : H x W or HW numpy array of int32
        index of the minimum value along Z axis
    loss_map : H x W or HW numpy array of float32
        minimum value along Z axis
    """
    Lshape = L.shape
    if (Lshape[:-2] + Lshape[-1:]) != Q.shape:
        raise ValueError('L and Q do not match shapes')

    if len(Lshape) == 3:
        if mask is None:
            return lookup_3dim_no_mask(L, Q)
        else:
            return lookup_3dim_with_mask(L, Q, mask)
    elif len(Lshape) == 4:
        if mask is None:
            return lookup_4dim_no_mask(L, Q)
        else:
            return lookup_4dim_with_mask(L, Q, mask)
    else:
        raise ValueError('Unrecognized shape of LookUp Table')