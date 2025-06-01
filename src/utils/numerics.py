import heapq
import numpy as np
import torch
import scipy.interpolate

def k_smallest_indices(array: np.ndarray, k: int):
    """
    Function to find the k largest elements in the array using a min heap.

    Parameters
    ----------
    arr: array_like
        array with elements
    k: int
        k smallest will be returned

    Returns
    -------
    k_smallest_indices : list
        list of the k smallest indices in the array

    """
    k_smallest_indices = heapq.nsmallest(k, range(len(array)), key=array.__getitem__)
    
    return k_smallest_indices

def spline_interpolant(x: np.ndarray, y: np.ndarray, knots: list[int], samples: int):
    """
    
    """
    fits = []
    y = np.atleast_2d(y)
    try:
        for ch in range(y.shape[1]):
            fit = scipy.interpolate.make_splrep(x, y[:, ch], t=np.linspace(x[2], x[-3], knots[ch]), k=3)
            fits.append(fit)
    except ValueError as e:
        print("Fit failed", e)
        return y

    d_samples = np.linspace(x[0], x[-1], samples)
    fitted_y = np.array([fits[i](d_samples) for i in range(len(fits))])
    return fitted_y

def calculate_loss(array_1: np.ndarray, array_2: np.ndarray, ord: int | str =2, axis: int=1):
    """
    Given two arrays, this function calculates

    Parameters
    ----------
    array_1 : array_like
        First input array.
    array_2 : array_like
        Second input array.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm. inf means numpy's `inf` object. The default is 2.
        `fro` is frobenius
    axis : {None, int, 2-tuple of ints}, optional.
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
        is 1.

    Returns
    -------
    n : float or ndarray
        Norm of `array_1 - array_2`.
    """
    # handle string input for infinity norm
    if ord == 'inf':
        ord = np.inf
    elif ord == '-inf':
        ord = -np.inf

    return np.linalg.norm(array_1 - array_2, ord=ord, axis=axis)


def blockLookup(L, Q, dtype, blockSize=256):
    """
    Given
        lookup table L on the cpu: H x W x Z x C 
        query image Q on the cpu: H x W x C
        dtype of the data
    Return:
        minD: H x W s.t.  minD[i,j] is argmin_k ||L[i,j,k] - Q[i,j]|| on the cpu
        loss: H x W s.t.  loss[i,j] is min_k ||L[i,j,k] - Q[i,j]|| on the cpu

    Does this in blocks on the CPU, and promotes types as needed (e.g., int16
    -> int32 and float16 -> float32)
    """
    Q = Q.cuda()
    H, W, Z, C = L.shape
    numRowBlocks = (H // blockSize) + (1 if H % blockSize != 0 else 0)
    numColBlocks = (W // blockSize) + (1 if W % blockSize != 0 else 0)
    minD = torch.zeros((H,W), dtype=torch.long, device='cuda')
    loss = torch.zeros((H,W), dtype=torch.float32, device='cuda')
    for blockI in range(numRowBlocks):
        sy, ey = blockI * blockSize, min(H, ((blockI+1) * blockSize))
        for blockJ in range(numColBlocks):
            sx, ex = blockJ * blockSize, min(W, ((blockJ+1) * blockSize))
            # we may have to promote things to avoid blowup 
            if dtype in [torch.float16, torch.float32]:
                LUp = L[sy:ey,sx:ex,:,:].cuda().type(torch.float32)
                QUp = Q[sy:ey,sx:ex,None,:].type(torch.float32)
            elif dtype in [torch.int16, torch.int32]:
                # if it's an int, do the arithmetic in int32 to avoid overflow
                LUp = L[sy:ey,sx:ex,:,:].cuda().type(torch.int32)
                QUp = Q[sy:ey,sx:ex,None,:].type(torch.int32)
            distance = torch.sum((LUp-QUp)**2, dim=-1)
            loss[sy:ey, sx:ex] = torch.min(distance , dim=-1)
            minD[sy:ey, sx:ex] = torch.argmin(distance , dim=-1)

    return minD.cpu(), loss.cpu()