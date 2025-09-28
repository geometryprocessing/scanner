from numba import jit, prange, cuda
import numba as nb
import numpy as np


####################################################################
########                  CPU FUNCTIONS BELOW               ######## 
####################################################################

@jit(nopython=True, parallel=True)
def ray_search(L, Q):
    best_idx = nb.int32(-1)
    best_val = nb.float32(float('inf'))
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
    loss = np.full(HW, fill_value=float('inf'), dtype=np.float32)

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
        if mask[i] == False, then minD[i] = -1 and loss[i] = float('inf')

    
    Returns
    -------
    minD : HW numpy array of int32
        index of the minimum value along Z axis
    loss_map : HW numpy array of float32
        minimum value along Z axis
    """
    HW, Z, C = L.shape

    minD = np.full(HW, fill_value=-1, dtype=np.int32)
    loss = np.full(HW, fill_value=float('inf'), dtype=np.float32)

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
    loss = np.full((H,W), fill_value=float('inf'), dtype=np.float32)

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
        if mask[i,j] == False, then minD[i,j] = -1 and loss[i,j] = float('inf')

    
    Returns
    -------
    minD : H x W numpy array of int32
        index of the minimum value along Z axis
    loss_map : H x W numpy array of float32
        minimum value along Z axis
    """
    H, W, Z, C = L.shape

    minD = np.full((H,W), fill_value=-1, dtype=np.int32)
    loss = np.full((H,W), fill_value=float('inf'), dtype=np.float32)

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

####################################################################
########  GPU FUNCTIONS BELOW -- THESE WORK ONLY IN CUDA    ######## 
####################################################################
    
@cuda.jit(device=True)
def ray_search_gpu_3channel(L,Q):
    Z, C = L.shape
    best_idx = nb.int32(-1)
    best_val = nb.float32(float('inf'))
    for k in range(Z):
        dist = 0.0
        dist = (L[k, 0] - Q[0])**2\
             + (L[k, 1] - Q[1])**2\
             + (L[k, 2] - Q[2])**2

        if dist < best_val:
            best_val = dist
            best_idx = k
    return best_idx, best_val

@cuda.jit(device=True)
def ray_search_gpu(L,Q):
    Z, C = L.shape
    best_idx = nb.int32(-1)
    best_val = nb.float32(float('inf'))
    for k in range(Z):
        dist = 0.0
        for c in range(C):
            diff = L[k,c] - Q[c]
            dist = cuda.fma(diff,diff,dist)
        if dist < best_val:
            best_val = dist
            best_idx = k
    return best_idx, best_val

@cuda.jit
def lookup_3dim_no_mask_gpu(L, D, Q, depth, minD, loss):
    i, j = cuda.grid(2)
    HW, Z, C = L.shape
    if i < HW:
        if C == 3:
            best_idx, best_val = ray_search_gpu_3channel(L[i], Q[i])
        else:
            best_idx, best_val = ray_search_gpu(L[i], Q[i])
            
        depth[i] = D[i,best_idx]
        minD[i] = best_idx
        loss[i] = best_val

@cuda.jit
def lookup_3dim_with_mask_gpu(L, D, Q, mask, depth, minD, loss):
    i = cuda.grid(1)
    HW, Z, C = L.shape

    if i < HW:
        if mask[i]:
            if C == 3:
                best_idx, best_val = ray_search_gpu_3channel(L[i], Q[i])
            else:
                best_idx, best_val = ray_search_gpu(L[i], Q[i])
                
            depth[i] = D[i,best_idx]
            minD[i] = best_idx
            loss[i] = best_val

@cuda.jit
def lookup_4dim_no_mask_gpu(L, D, Q, depth, minD, loss):
    i, j = cuda.grid(2)
    H, W, Z, C = L.shape
    if i < H and j < W:
        if C == 3:
            best_idx, best_val = ray_search_gpu_3channel(L[i,j], Q[i,j])
        else:
            best_idx, best_val = ray_search_gpu(L[i,j], Q[i,j])
            
        depth[i,j] = D[i,j,best_idx]
        minD[i,j] = best_idx
        loss[i,j] = best_val

@cuda.jit
def lookup_4dim_with_mask_gpu(L, D, Q, mask, depth, minD, loss):
    i, j = cuda.grid(2)
    H, W, Z, C = L.shape
    if i < H and j < W:
        if mask[i,j]:
            if C == 3:
                best_idx, best_val = ray_search_gpu_3channel(L[i,j], Q[i,j])
            else:
                best_idx, best_val = ray_search_gpu(L[i,j], Q[i,j])
                
            depth[i,j] = D[i,j,best_idx]
            minD[i,j] = best_idx
            loss[i,j] = best_val


def lookup_gpu(L, D, Q, threadsPerBlock:int | tuple[int], mask=None):
    """
    Overloaded CUDA parallel lookup function, where mask is an optional argument.

    Parameters
    ----------
    L : H x W x Z x C or HW x Z x C array of float32 on the GPU
        lookup table
    D : H x W z Z or HW x Z array of float 32 on the GPU
        depth associated with lookup table values 
    Q : H x W x C or HW x C array of float32 on the GPU
        normalized image to query on the lookup table
    mask : H x W or HW numpy of bool, optional
        mask
    threadsPerBlock : int or tuple of ints, optional
        parameter for how many threads per block of GPU
        This will dictate how many blocks we will generate.
        If lookup table is 4 dimensional, then this should be
        a tuple (threadsPerBlock_X, threadsPerBlock_Y).

    Returns
    -------
    depth : H x W or HW array of float32 on the GPU
        depth value retrieved from lookup search
    minD : H x W or HW array of int32 on the GPU
        index of the minimum value along Z axis
    loss_map : H x W or HW array of float32 on the GPU
        minimum squared euclidean distance between input and lookup
    """
    import cupy as cp
    Lshape = L.shape
    if (Lshape[:-2] + Lshape[-1:]) != Q.shape:
        raise ValueError('L and Q do not match shapes')
   
    depth = cp.full(shape=Lshape[:-2], fill_value=-1, dtype=cp.float32)
    minD = cp.full(shape=Lshape[:-2], fill_value=-1, dtype=cp.int32)
    loss = cp.full(shape=Lshape[:-2], fill_value=float('inf'), dtype=cp.float32)

    if len(Lshape) == 3:
        blockspergrid = (Lshape[0] + threadsPerBlock[0]-1)//threadsPerBlock[0]
        if mask is None:
            lookup_3dim_no_mask_gpu[blockspergrid, threadsPerBlock](L, D, Q, depth, minD, loss)
        else:
            lookup_3dim_with_mask_gpu[blockspergrid, threadsPerBlock](L, D, Q, mask, depth, minD, loss)
    elif len(Lshape) == 4:
        blockspergrid = ((Lshape[0] + threadsPerBlock[0]-1)//threadsPerBlock[0], (Lshape[1] + threadsPerBlock[1]-1)//threadsPerBlock[1])
        if mask is None:
            lookup_4dim_no_mask_gpu[blockspergrid, threadsPerBlock](L, D, Q, depth, minD, loss)
        else:
            lookup_4dim_with_mask_gpu[blockspergrid, threadsPerBlock](L, D, Q, mask, depth, minD, loss)
    else:
        raise ValueError('Unrecognized shape of LookUp Table')
    
    return depth, minD, loss