import argparse
import os
import sys

import numpy as np
import scipy.signal

def low_rank(lut, r: int):
    """
    
    Parameters
    ----------

    Returns
    -------

    """
    # prepare
    original_shape = lut.shape
    # TODO: decide if depth should be part of low rank...
    num_channels = original_shape[-1] - 1 # subtract one because of depth
    num_depths = original_shape[-2]
    lut_d = lut[...,-1]

    # operate
    lowrank_lut = []
    for ch in range(num_channels):
        U, S, Vh = np.linalg.svd(lut[...,ch].reshape((-1,num_depths)), full_matrices=False)
        L = U[:, :r] @ np.diag(S[:r])
        R = Vh[:r, :]
        lowrank_channel = np.matmul(L,R)
        lowrank_lut += [lowrank_channel.reshape(original_shape[:-1])]

    # add depth back
    lowrank_lut += [lut_d]
    return np.stack(lowrank_lut, axis=-1)


def moving_average(lut, window_size: int):
    """
    
    Parameters
    ----------
    
    Returns
    -------

    """
    # prepare
    original_shape = lut.shape
    num_channels = original_shape[-1] - 1 # subtract one because of depth
    num_depths = original_shape[-2]
    lut_d = lut[...,-1]

    # operate
    convolved_lut = []
    kernel = np.ones((1,window_size)) / window_size
    for ch in range(num_channels):
        convolved_channel = scipy.signal.convolve2d(lut[...,ch].reshape((-1,num_depths)), kernel, mode='same', boundary='symm')
        convolved_lut += [convolved_channel.reshape(original_shape[:-1])]

    # add depth back
    convolved_lut += [lut_d]

    #return 
    return np.stack(convolved_lut, axis=-1)

def low_pass_filter(lut, cutoff: int):
    # prepare
    original_shape = lut.shape
    num_channels = original_shape[-1] - 1 # subtract one because of depth
    num_depths = original_shape[-2]
    # need to lose depth, for some reason (does it just need to be even?)
    lut_d = lut[...,:-1,-1] 

    # operate
    fourier_lut = []
    for ch in range(num_channels):
        f_channel = np.fft.rfft2(lut[...,ch].reshape((-1,num_depths)))
        f_channel[:,cutoff:] = 0
        # need to lose depth, for some reason (does it just need to be even?)
        fourier_lut += [np.fft.irfft2(f_channel).reshape(*original_shape[:-2], num_depths - 1)]

    # add depth back
    fourier_lut += [lut_d]

    #return 
    return np.stack(fourier_lut, axis=-1)

def spline_fitting(lut, s):
    pass

def denoise(lut, config, method):
    pass