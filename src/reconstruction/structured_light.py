import argparse
import cv2
import numpy as np
import structuredlight
import hilbert
import scipy
import sys
import os

from src.utils.three_d_utils import triangulate_pixels, intersect_pixels, \
    save_point_cloud, depth_map_from_point_cloud, get_origin
from src.utils.image_utils import load_ldr, convert_to_gray
from src.reconstruction.configs import StructuredLightConfig, \
    get_config, is_valid_config, apply_cmdline_args

def generate_mask(reconstruction_directory: str, config: StructuredLightConfig):
    
    white = config.white_image
    black = config.black_image
    if white is None or black is None:
        return None

    white = load_ldr(os.path.join(reconstruction_directory, white), make_gray=True)
    black = load_ldr(os.path.join(reconstruction_directory, black), make_gray=True)
    data_type = white.dtype
    m = np.iinfo(data_type).max if data_type.kind in 'iu' else np.finfo(data_type).max
    mask = (abs(white-black) > m*config.mask_thr)
    return mask

def decode(reconstruction_directory: str, config: StructuredLightConfig) -> tuple:
    """
    Given a reconstruction directory and a structured light config, 
    finds the 

    Returns
    -------
    index_x : array_like
        Array with decoded horizontal (x) projector indices. Same resolution of camera.
        Is None if index_x is not decoded.

    index_y : array_like
        Array with decoded vertical (y) projector indices. Same resolution of camera.
        Is None if index_y is not decoded.

    """
    index_x, index_y = None, None
    pattern = config.pattern.lower()
    if pattern in ['gray', 'binary', 'xor']:
        vert = config.vertical_images
        if isinstance(vert, list):
            vert = [os.path.join(reconstruction_directory, img) for img in vert]
        horz = config.horizontal_images
        if isinstance(horz, list):
            horz = [os.path.join(reconstruction_directory, img) for img in horz]
        inv_vert = config.inverse_vertical_images
        if isinstance(inv_vert, list):
            inv_vert = [os.path.join(reconstruction_directory, img) for img in inv_vert]
        inv_horz = config.inverse_horizontal_images
        if isinstance(inv_horz, list):
            inv_horz = [os.path.join(reconstruction_directory, img) for img in inv_horz]
        
        white = None if config.white_image is None else os.path.join(reconstruction_directory, config.white_image)
        black = None if config.black_image is None else os.path.join(reconstruction_directory, config.black_image)
        thr = config.binary_threshold
        
        index_x, index_y = decode_gray_binary_xor(pattern,
                                                vert,
                                                inv_vert,
                                                horz,
                                                inv_horz,
                                                white,
                                                black,
                                                thr)
    elif pattern == 'hilbert':
        # handle structure grammar for Hilbert
        index_x = decode_hilbert([os.path.join(reconstruction_directory, img) 
                                        for img in config.images],
                                        config.num_bits)
    elif pattern == 'phaseshift':
        # handle structure grammar for Phase Shift
        F = 1.0 if 'F' not in config else config['F']
        index_x = decode_phaseshift(config.projector.resx,
                                            [os.path.join(reconstruction_directory, img) 
                                            for img in config.images], 
                                            config.phaseshift_frequency)
    elif pattern in ['microphaseshift', 'mps']:
        # handle structure grammar for Micro Phase Shift
        index_x = decode_mps(
                            [os.path.join(reconstruction_directory, img) 
                            for img in config.images], 
                            config.frequency_vector,
                            config.camera.get_image_shape(),
                            config.projector.get_projector_shape(),
                            config.median_filter)
    else:
        raise ValueError("Unrecognized pattern, cannot decode structured light")
    
    return index_x, index_y

def decode_hilbert(images, num_bits):
    if isinstance(images, list):
        images = [load_ldr(img, make_gray=True) 
                    if isinstance(img, str) else np.atleast_3d(img)
                    for img in images]
        images = np.concatenate(images, axis=2)
    
    num_dims = images.shape[2]

    decoded = hilbert.encode(images, num_dims=num_dims, num_bits=num_bits)
    return decoded.reshape((images.shape[0],images.shape[1]))

def decode_gray_binary_xor(pattern,
                            vertical_images=None,
                            inverse_vertical_images=None,
                            horizontal_images=None,
                            inverse_horizontal_images=None,
                            white_image=None,
                            black_image=None,
                            thresh=None):
    """

    Parameters
    ----------
    vertical_images : list
        List of strings containing image paths (or string for folder containing images).
        These images will be used for decoding vertical structured light patterns.
    inverse_vertical_images : list, optional
        List of strings containing image paths (or string for folder containing images).
        Inverse vertical structured light pattern images will be used for 
        setting a threshold value for pixel intensity when decoding vertical patterns.
        Read the structuredlight docs for more information:
        https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image
    horizontal_images : list
        List of strings containing image paths (or string for folder containing images).
        These images will be used for decoding horizontal structured light patterns.
    inverse_horizontal_images : list, optional
        List of strings containing image paths (or string for folder containing images).
        Inverse horizontal structured light pattern images will be used for 
        setting a threshold value for pixel intensity when decoding horizontal patterns.
        Read the structuredlight docs for more information:
        https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image
    white_image : str | np.ndarray, optional
        (if str, path to) captured image of scene with an all-white pattern projected.
        The white image can be used for setting a threshold value when decoding
        structured light patterns. This has to be used in combination with black_image,
        since the  threshold for ON or OFF will be set per pixel as:
        thr = 0.5 * white_image + 0.5 * black_image.
        If negative/inverse structured light patterns are passed,
        black and white pattern images will be ignored for threshold setting.
        Read the structuredlight docs for more information:
        https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image

    
    black_image : str | np.ndarray, optional
        (if str, path to) captured image of scene with an all-black pattern projected.
        The black image can be used to extract colors for the reconstructed point cloud.
        The black image can also be used for setting a threshold value when decoding
        structured light patterns. This has to be used in combination with white_image,
        since the  threshold for ON or OFF will be set per pixel as:
        thr = 0.5 * white_image + 0.5 * black_image.
        If negative/inverse structured light patterns are passed,
        black and white pattern images will be ignored for threshold setting.
        Read the structuredlight docs for more information:
        https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image
    thresh : float, optional
        threshold value for considering a pixel ON or OFF based on its intensity.
        If black and white pattern images are set, or if negative/inverse
        patterns are passed, this threshold value will be ignored.

    Returns
    -------
    index_x, index_y
    """
    
    if isinstance(pattern, str):
        match pattern.lower():
            case 'gray':
                pattern = structuredlight.Gray()
            case 'binary':
                pattern = structuredlight.Binary()
            case 'bin':
                pattern = structuredlight.Binary()
            case 'xor':
                pattern = structuredlight.XOR()
            case _:
                print("Unrecognized structured light pattern type, defaulting to gray...\n")
                pattern = structuredlight.Gray()
    
    index_x, index_y = None, None
    if white_image and black_image:
        white_image = load_ldr(white_image, make_gray=True) \
                if isinstance(white_image, str) else convert_to_gray(white_image) 
        black_image = load_ldr(black_image, make_gray=True) \
                if isinstance(black_image, str) else convert_to_gray(black_image) 
        thresh = 0.5*white_image + 0.5*black_image

    if horizontal_images:
        gray_horizontal = [load_ldr(img, make_gray=True) 
                            if isinstance(img, str) else convert_to_gray(img) 
                            for img in horizontal_images]
        if inverse_horizontal_images:
            assert len(inverse_horizontal_images) == len(horizontal_images), \
                "Mismatch between number of horizontal patterns \
                    and inverse horizontal patterns. Must be the same"
            horizontal_second_argument = [load_ldr(img, make_gray=True) 
                                            if isinstance(img, str) else convert_to_gray(img) 
                                            for img in inverse_horizontal_images]
        else:
            horizontal_second_argument = thresh
        index_y = pattern.decode(gray_horizontal, horizontal_second_argument)

    if vertical_images:
        gray_vertical = [load_ldr(img, make_gray=True) 
                            if isinstance(img, str) else convert_to_gray(img) 
                            for img in vertical_images]
        if inverse_vertical_images:
            assert len(inverse_vertical_images) == len(vertical_images), \
                "Mismatch between number of vertical patterns \
                    and inverse vertical patterns. Must be the same"
            vertical_second_argument = [load_ldr(img, make_gray=True) 
                                            if isinstance(img, str) else convert_to_gray(img) 
                                            for img in inverse_vertical_images]
        else:
            vertical_second_argument = thresh
        index_x = pattern.decode(gray_vertical, vertical_second_argument)

    return index_x, index_y

def decode_phaseshift(width, images, F=1.0):
    pattern = structuredlight.PhaseShifting(num=len(images), F=F)
    pattern.width = width

    images = [load_ldr(img, make_gray=True) 
                if isinstance(img, str) else convert_to_gray(img) 
                for img in images]
    result = pattern.decode(images)

    return result

def decode_mps(images: list,
                frequency_vec: list,
                cam: tuple,
                pro: tuple,
                medfilt_param: int=5):
    num_frequency = len(frequency_vec)

    # Making the measurement matrix M (see paper for definition)
    M = np.zeros((num_frequency+2, num_frequency+2))
    
    # Filling the first three rows -- correpsonding to the first frequency
    M[0,:3] = [1, np.cos(2*np.pi*0/3), -np.sin(2*np.pi*0/3)]
    M[1,:3] = [1, np.cos(2*np.pi*1/3), -np.sin(2*np.pi*1/3)]
    M[2,:3] = [1, np.cos(2*np.pi*2/3), -np.sin(2*np.pi*2/3)]
    
    # Filling the remaining rows - one for each subsequent frequency
    for f in range(1, num_frequency):
        #print([1, np.zeros(f), 1, np.zeros(numFrequency-f)], M[f+1, :])
        line = [1.0]
        line.extend([0.0]*(f+1))
        line.extend([1.0])
        line.extend([0.0]*(num_frequency-f-1))
        M[f+2, :] = line

    # Making the observation matrix (captured images)
    R = np.zeros((num_frequency+2, cam[0]*cam[1]))


    # Filling the observation matrix (image intensities)
    for i, img in enumerate(images):
        img = load_ldr(img, make_gray=True) if isinstance(img, str) else convert_to_gray(img)
        # img = cv2.imread(img_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)   # reads an image in the BGR format
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float64")
        img = img / 255
        R[i,:]  = img.T.reshape(-1)
        
    # Solving the linear system
    # The unknowns are [Offset, Amp*cos(phi_1), Amp*sin(phi_1), Amp*cos(phi_2),
    # ..., Amp*cos(phi_F)], where F = numFrequency. See paper for details. 
    U = np.linalg.solve(M, R)

    # Computing the amplitude 
    Amp = np.sqrt(U[1,:]**2 + U[2,:]**2)

    # Dividing the amplitude to get the CosSinMat --- matrix containing the sin
    # and cos of the phases corresponding to different frequencies. For the
    # phase of the first frequency, we have both sin and cos. For the phases of
    # the remaining frequencies, we have cos. 

    CosSinMat = U[1:, :] / np.tile(Amp, (num_frequency+1, 1))  
    
    #  Converting the CosSinMat into column indices
    # IC            -- correspondence map (corresponding projector column (sub-pixel) for each camera pixel. Size of IC is the same as input captured imgaes.
    IC = _phase_unwrap_cos_sin_to_column_index(CosSinMat, frequency_vec, pro[0], cam[1], cam[0])
    IC = scipy.signal.medfilt2d(IC, medfilt_param) # Applying median filtering

    return IC



def _phase_unwrap_cos_sin_to_column_index(CosSinMat, frequencyVec, numProjColumns, nr, nc):

    """
    This function converts the CosSinMat into column-correspondence for MPS.  
    
    CosSinMat is the matrix containing the sin and cos of the phases 
    corresponding to different frequencies for each camera pixel. For the
    phase of the first frequency, we have both sin and cos. For the phases of
    the remaining frequencies, we have cos. 
    
    The function first performs a linear search on the projector column
    indices. Then, it adds the sub-pixel component. 
    """
    x0 = np.array([list(range(0, numProjColumns))]) # Projector column indices

    
    # Coomputing the cos and sin values for each projector column. The format 
    # is the same as in CosSinMat - for the phase of the first frequency, we 
    # have both sin and cos. For the phases of the remaining frequencies, we 
    # have cos. These will be compared against the values in CosSinMat to find 
    # the closest match. 

    TestMat = np.tile(x0, (CosSinMat.shape[0], 1)).astype("float64")
    
    TestMat[0,:] = np.cos((np.mod(TestMat[0,:], frequencyVec[0]) / frequencyVec[0]) * 2 * np.pi) # cos of the phase for the first frequency
    TestMat[1,:] = np.sin((np.mod(TestMat[1,:], frequencyVec[0]) / frequencyVec[0]) * 2 * np.pi) # sin of the phase for the first frequency

    for i in range(2, CosSinMat.shape[0]):
        TestMat[i,:] = np.cos((np.mod(TestMat[i,:], frequencyVec[i-1]) / frequencyVec[i-1]) * 2 * np.pi) # cos of the phases of the remaining frequency

    IC = np.zeros((1, nr*nc), dtype="float64") # Vector of column-values
    
    # For each camera pixel, find the closest match
    # TODO parpool(4);% This loop can be run in parallel using MATLAB parallel toolbox. The number here is the number of cores on your machine. 
    for i in range(0, CosSinMat.shape[1]):
        CosSinVec = CosSinMat[:,i]
        CosSinVec = CosSinVec.reshape((CosSinVec.shape[0], 1))
        ErrorVec = np.sum(np.abs(np.tile(CosSinVec, (1, numProjColumns)) - TestMat)**2, axis=0)
        #print(ErrorVec.shape, ErrorVec)
        Ind = np.argmin(ErrorVec)
        #print(Ind)
        IC[0, i] = Ind

    # Computing the fractional value using phase values of the first frequency 
    # since it has both cos and sin values. 

    PhaseFirstFrequency = np.arccos(CosSinMat[0,:]) # acos returns values in [0, pi] range. There is a 2 way ambiguity.
    PhaseFirstFrequency[CosSinMat[1,:]<0] = 2 * np.pi - PhaseFirstFrequency[CosSinMat[1,:]<0] # Using the sin value to resolve the ambiguity
    ColumnFirstFrequency = PhaseFirstFrequency * frequencyVec[0] / (2 * np.pi) # The phase for the first frequency, in pixel units. This is equal to mod(trueColumn, frequencyVec(1)). 

    NumCompletePeriodsFirstFreq = np.floor(IC / frequencyVec[0]) # The number of complete periods for the first frequency 
    ICFrac = NumCompletePeriodsFirstFreq * frequencyVec[0] + ColumnFirstFrequency # The final correspondence, with the fractional component

    # If the difference after fractional correction is large (because of noise), keep the original value. 
    ICFrac[np.abs(ICFrac-IC)>=1] = IC[np.abs(ICFrac-IC)>=1]
    IC = ICFrac

    IC = np.reshape(IC, [nr, nc], order='F')
    return IC

def triangulate_point_cloud(index_x = None, index_y = None, mask = None, config: StructuredLightConfig = None):
    """
    Take correspondence indices between camera and projector and use
    ray intersection to triangulate points in 3D.

    If decoding was done for both x and y coordinates, this functions does a
    direct triangulation of the projector pixels and camera pixels.
    Otherwise, it does a plane-line intersection to find the 3D points in 
    world coordinates. 
    """
    assert config is not None, "No config passed!"
    assert index_x is not None or index_y is not None, \
        "Decoding function call missing"

    cam_resolution = config.camera.get_image_shape()
    campixels_x, campixels_y = np.meshgrid(np.arange(cam_resolution[0]),
                                            np.arange(cam_resolution[1]))
    campixels = np.stack([campixels_x, campixels_y], axis=-1)
    campixels = campixels.reshape((-1,2)) if mask is None else campixels[mask].reshape((-1,2))

    if index_x is not None and index_y is not None:
        projpixels = np.stack([index_x, index_y], axis=-1)
        projpixels = projpixels.reshape((-1,2)) if mask is None else projpixels[mask].reshape((-1,2))
        point_cloud = triangulate_pixels(
            campixels,
            config.camera.K,
            config.camera.dist_coeffs,
            config.camera.R,
            config.camera.T,
            projpixels,
            config.projector.K,
            config.projector.dist_coeffs,
            config.projector.R,
            config.projector.T
        )
    elif index_x is not None:
        point_cloud = intersect_pixels(
            campixels,
            config.camera.K,
            config.camera.dist_coeffs,
            config.camera.R,
            config.camera.T,
            index_x if mask is None else index_x[mask],
            config.projector.get_projector_shape(),
            config.projector.K,
            config.projector.dist_coeffs,
            config.projector.R,
            config.projector.T,
            index = 'x'
        )
    elif index_y is not None:
        point_cloud = intersect_pixels(
            campixels,
            config.camera.K,
            config.camera.dist_coeffs,
            config.camera.R,
            config.camera.T,
            index_y if mask is None else index_y[mask],
            config.projector.get_projector_shape(),
            config.projector.K,
            config.projector.dist_coeffs,
            config.projector.R,
            config.projector.T,
            index = 'y'
        )

    return point_cloud

def save_reconstruction_outputs(folder: str,
                                mask = None,
                                depth_map = None,
                                point_cloud = None,
                                colors = None,
                                index_x = None,
                                index_y = None,
                                config: StructuredLightConfig = None):
    # TODO: this is mostly a copy of the save_reconstruction_output from lookup.py
    # is there any way to merge them? and move this to src/utils/file_io
    assert config is not None, "No config passed!"
    name = config.pattern

    if config.save_point_cloud and point_cloud is not None:
        if colors is None:
            print("No color image set, therefore no color extraction for point cloud")
            save_point_cloud(os.path.join(folder,f"structured_light_{name}_point_cloud.ply"),
                                    point_cloud)
        else:
            save_point_cloud(os.path.join(folder,f"structured_light_{name}_point_cloud.ply"),
                                    point_cloud,
                                    colors=colors)
        if config.verbose:
            print('-' * 15)
            print("Saved point cloud")

        
    if config.save_depth_map:
        if depth_map is None and point_cloud is not None and mask is not None:
            depth_map = depth_map_from_point_cloud(point_cloud,
                                    mask,
                                    get_origin(config.camera.R, config.camera.T))
        np.save(os.path.join(folder,f"structured_light_{name}_depth_map.npy"), depth_map)
        if config.verbose:
            print('-' * 15)
            print("Saved depth map")

    if config.save_index_map:
        if index_x is not None:
            np.save(os.path.join(folder,f"structured_light_{name}_index_x.npy"), index_x)
        if index_y is not None:
            np.save(os.path.join(folder,f"structured_light_{name}_index_y.npy"), index_y)
        if config.verbose:
            print('-' * 15)
            print("Saved index map")



def run(reconstruction_directory, config: StructuredLightConfig):
    index_x, index_y = decode(reconstruction_directory, config)
    mask = generate_mask(reconstruction_directory, config)
    pcd = triangulate_point_cloud(index_x, index_y, mask=mask, config=config)
    save_reconstruction_outputs(reconstruction_directory, mask=mask, point_cloud=pcd, config=config)

def main(args):
    parser = argparse.ArgumentParser(description="Reconstructs scene with Traditional Structured Light.")
    parser.add_argument('-i', '--input', type=str, default=None, required=True,
                        help='Path to input folder to run reconstruction on.' \
                        'It should contain image data for Structured Light Reconstruction')
    parser.add_argument('--configs', nargs='+', type=str,
                        help='Sturctured Light Reconstruction Reconstruction configuration -- can either be a path to JSON file ' \
                        'or a known lookup3d config name. Check src/reconstruction/configs.py file.')
    # print params good for debugging
    parser.add_argument('--print_params', '-pp', action='store_true', help='Print the parameters of the provided scene and exit.')
    args, uargs = parser.parse_known_args(args)

    if any(not is_valid_config(config) for config in args.configs):
        raise ValueError(f'Unknown lookup config detected: {args.configs}')

    assert len(args.camconfigs) == len(args.configs), "Configs and CameraConfigs should match"

    for config_name in args.configs:
        config: StructuredLightConfig = get_config(config_name)
        base_path = args.input
        
        # TODO: with multiview, command line arguments will apply to all cameras
        # how would I like for the ability to change each separately?
        remaining_args = apply_cmdline_args(config, uargs, return_dict=True)
        if config.verbose:
            print(f"Starting {base_path} folder with config {config_name}")
        
        if args.print_params:
            print(config.to_dict())
            continue

        run( base_path, config)
        config.dump_json(os.path.join(base_path), f'{config_name}_sl_reconstruction_config.json')

if __name__ == '__main__':
    main(sys.argv[1:])
