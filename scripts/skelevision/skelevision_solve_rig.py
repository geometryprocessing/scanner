import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from src.utils.three_d_utils import normalize, rodrigues_rotation

import numpy as np
def solve_rig(points: np.ndarray, thetas: np.ndarray, zs: np.ndarray):
    """
    This function solves for a rig of cameras specifically rotating around
    an axis and moving up and down.

    For a more general function, feel free to look into the code written here
    and rewrite it for your purposes.

    Parameters
    ----------
    points : array_like
        points in shape (NUM_POSITIONS, NUM_CAMERAS, 3)
    thetas : array_like
        rotations (in radians) for all NUM_POSITIONS  
    zs : array_like
        displacement (can be viewed as up/down) foor all NUM_POSITIONS

    Returns
    -------
    center :
        origin point of system
    
    k :
        axis of rotation

    radius : 
        shape (NUM_CAMERAS, 3)
    """
    NUM_POSITIONS, NUM_CAMERAS, _ = points.shape
    
    def rig_function(x, *params):
        center, k, radius = np.array(params[:3]), np.array(params[3:5]), np.array(params[5:])
        k = np.array([k[0],k[1], np.sqrt(1-k[0]**2+k[1]**2)])
        radius = radius.reshape(NUM_CAMERAS, 3)
        
        x = x.reshape(-1,2)
        NUM_POINTS = x.shape[0]
        theta, z = x[...,0], x[..., 1, np.newaxis] # theta has shape (NUM_POINTS,) and z now has shape (NUM_POINTS,1)
        rotated_vecs = rodrigues_rotation(radius, k, theta).reshape(NUM_POINTS,NUM_CAMERAS,3)
        translated_vecs = (center + z * k).reshape(NUM_POINTS,1,3) # so the summation below can be broadcast properly
        results = translated_vecs + rotated_vecs
        return results.flatten()

    # assemble x data
    x_data = np.stack([thetas, zs], axis=0) # (2*NUM_POSITIONS)
    # flatten y data (needs to be 1D for scipy)
    y_data = points.flatten()
    # assemble initial guesses and bounds
    num_params = 3 + 2 + NUM_CAMERAS * 3
    p0 = np.zeros(num_params)
    p0[:3] = np.mean(points, axis=(0,1))
    lo_bounds = np.full(shape=(num_params,), fill_value=-np.inf)
    up_bounds = np.full(shape=(num_params,), fill_value=np.inf)
    lo_bounds[(3,4)] = -1
    up_bounds[(3,4)] = 1

    import scipy.optimize
    res = scipy.optimize.curve_fit(rig_function, x_data, y_data, p0=p0, bounds=(lo_bounds,up_bounds))
    params = res[0]
    center, k, radius = np.array(params[:3]), np.array(params[3:5]), np.array(params[5:]).reshape(NUM_CAMERAS,3)
    k = np.array([k[0],k[1], np.sqrt(1-k[0]**2+k[1]**2)])
    return center, k, radius

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/sfm_config.yaml', required=False,
                        help='path to config file')

    args, extras = parser.parse_known_args()

    config = load_yaml(args.config, cli_args=extras)
    config.cmd_args = vars(args)



if __name__ == '__main__':
    main()