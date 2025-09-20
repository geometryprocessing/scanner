import argparse
import cv2
import concurrent.futures
from datetime import datetime
import numpy as np
CUDA_AVAILABLE = False # constant
# import torch ### swapping for cupy since we don't need autodiff and I prefer interoperability
try:
    import cupy as cp
    print("cupy imported successfully.")
    def get_array_module(*args):
        return cp.get_array_module(*args)
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    print("cupy not found. Using numpy fallback everywhere.")
    def get_array_module(*args):
        return np
import os

from src.reconstruction.configs import LookUp3DConfig, get_config
from src.utils.three_d_utils import point_cloud_from_depth_map, \
    fit_line, save_point_cloud, intersect_line_with_plane, camera_to_ray_world
from src.utils.file_io import save_json, \
    load_json, get_all_paths, get_all_folders, get_folder_from_file
from src.utils.image_utils import extract_mask, normalize_color, \
    denoise_fft, load_ldr, crop, gaussian_blur, generate_mask_binary_structure,\
          convert_to_gray, replace_with_nearest
from src.scanner.camera import Camera
from src.scanner.calibration import Calibration, CheckerBoard, Charuco

def load_lut(filename: str, is_lowrank, use_gpu: bool = False, gpu_device: int = 0):
    lookup_table = load_lowrank_table(filename) if is_lowrank else np.load(filename)

    # TODO: check if shape matches roi's shape

    lut = lookup_table[...,:-1]
    dep = lookup_table[...,-1]

    if use_gpu and CUDA_AVAILABLE:
        with cp.cuda.Device(gpu_device):
            lut = cp.ndarray(lut)
            dep = cp.ndarray(dep)

    return lut, dep
    
def load_lowrank_table(filename: str):
    data = np.load(filename)
    original_shape = data['shape']
    num_channels = original_shape[-1] - 1 # subtract one because of depth

    lut = []

    # table is saved as L_{idx}, R_{idx}, with idx starting at 1
    for ch in range(1,num_channels+1):
        lut += [np.matmul(data[f'L_{ch}'],data[f'R_{ch}']).reshape(original_shape[:-1])]

    Ld = data['L_d']
    Rd = data['R_d']
    lut_d = np.matmul(Ld,Rd).reshape(original_shape[:-1])

    lut += [lut_d]

    return np.stack(lut, axis=-1)

def concatenate_lookup_tables(lookup_tables: list[str], filename: str):
    """
    Create a lookup table (dictionary) of 2D pixel coordinates, RGB values for many patterns, and depth.
    """
    lookup_tables = [np.load(lut) for lut in lookup_tables]
    result = [lut[:, :, :, :-1] for lut in lookup_tables]
    result.append(lookup_tables[-1][:, :, :, -1, np.newaxis])

    print("Concatenating...")
    result = np.concatenate(result, axis=3)
    np.save(filename, result)

def process_position(folder: str,
                     config: LookUp3DConfig) -> tuple:
        """
        Function that accepts a folder and a config file and
        returns the normalized pattern image (HxWxN), mask (HxW), and colors (HxWx3).

        Parameters
        ----------
        folder : str
            path to folder containing the image files to be processed
        config : LookUp3DConfig
            congif class of LookUp3D Reconstruction containing processing parameters
        
        Returns
        -------
        normalized : array_like
            array of shape HxWxN, where H is height, W is width, and N is number of channels
            if use_gpu, returns cupy ndarray, else return numpy ndarray
        mask : array_like
            array of shape HxW containing mask values to reduce computation time
            if use_gpu, returns cupy ndarray, else return numpy ndarray
        colors : np.ndarray
            array of shape HxWx3 containing RGB values
            this is passed to point cloud, so that it is stored with color
            if no colors parameter is defined in config, then this is None
        """
        assert config is not None, "No config passed!"
    
        mask = None
        colors = None
        name = config.name

        if config.verbose:
            print('-' * 15)
            print("Normalizing image")

        roi = config.roi
        images: list[str] = config.images
        pattern_images = crop(np.concatenate([np.atleast_3d(load_ldr(os.path.join(folder, image))) for image in images], axis=2), roi=roi)
        white_image = crop(load_ldr(os.path.join(folder, config.white_image)), roi=roi)
        black_image = None
        if config.black_image is not None:
                black_image = crop(load_ldr(os.path.join(folder, config.black_image)), roi=roi)
        
        mask_thr: float = config.mask_thr
        image_for_mask = pattern_images if config.use_pattern_for_mask else white_image 
        mask = generate_mask_binary_structure(convert_to_gray(image_for_mask), mask_thr) if config.use_binary_mask else extract_mask(image_for_mask, mask_thr)


        normalized = normalize_color(color_image=pattern_images,
                                                white_image=white_image,
                                                black_image=black_image,
                                                mask=mask)
        
        if config.denoise_input:
            normalized = denoise_fft(normalized, int(config.denoise_cutoff))

        if config.blur_input:
            normalized = gaussian_blur(normalized, sigmas=int(config.blur_input_sigma))

        # if config.save_normalized:
            # np.savez_compressed(os.path.join(folder, f"{name}.npz"), pattern=normalized)

        if config.colors_image is not None:
            colors = crop(load_ldr(os.path.join(folder, config.colors_image)), roi=roi)

        if config.use_gpu and CUDA_AVAILABLE:
            with cp.cuda.Device(config.gpu_device):
                normalized = cp.ndarray(normalized)
                mask = cp.ndarray(mask)

        return normalized, mask, colors

def save_reconstruction_outputs(folder: str,
                                mask = None,
                                depth_map = None,
                                loss_map = None,
                                point_cloud = None,
                                colors = None,
                                index_map = None,
                                config: LookUp3DConfig = None):
    # TODO: what does cupy need to save these?
    assert config is not None, "No config passed"
    table_name = config.name

    if config.save_depth_map and depth_map is not None:
        np.save(os.path.join(folder,f"{table_name}_depth_map.npy"), depth_map)
        if config.verbose:
            print('-' * 15)
            print("Saved depth map")
    
    if config.save_loss_map and loss_map is not None:
        np.save(os.path.join(folder,f"{table_name}_loss_map.npy"), loss_map)
        if config.verbose:
            print('-' * 15)
            print("Saved loss map")

    if config.save_index_map and index_map is not None:
        np.save(os.path.join(folder,f"{table_name}_index_map.npy"), index_map)
        if config.verbose:
            print('-' * 15)
            print("Saved index map")

    if config.save_point_cloud:
        if config.verbose:
            print('-' * 15)
            print("Constructing point cloud")

        pcd_mask = (depth_map > 0).flatten() & (loss_map < config.loss_thr).flatten()

        point_cloud: np.ndarray = point_cloud_from_depth_map(depth_map=depth_map,
                                                                        K=config.camera.K,
                                                                        dist_coeffs=config.camera.dist_coeffs,
                                                                        R=config.camera.R,
                                                                        T=config.camera.T,
                                                                        roi=config.roi)

        if config.verbose:
            print("Extracting colors for point cloud")

        if colors is None:
            print("No color image set, therefore no color extraction for point cloud")
            save_point_cloud(os.path.join(folder,f"{table_name}_point_cloud.ply"),
                point_cloud.reshape((-1,3))[pcd_mask])
        else:
            save_point_cloud(os.path.join(folder,f"{table_name}_point_cloud.ply"),
                                point_cloud.reshape((-1,3))[pcd_mask],
                                colors=colors.reshape((-1,3))[pcd_mask])
        if config.verbose:
            print('-' * 15)
            print("Saved point cloud")

    # if config.save_mask:
    #     np.save(os.path.join(folder,"mask.npy"), mask)
    #     if config.verbose:
    #         print('-' * 15)
    #         print("Saved mask")

def blockLookup(L, Q, dtype, block_size: int = 256):
    """
    Parameters
    ----------
        lookup table L H x W x Z x C 
        normalized query image Q: H x W x C
        dtype of the data
        block_size
    
    Returns
    -------
        minD: H x W s.t.  minD[i,j] is argmin_k ||L[i,j,k] - Q[i,j]|| on the cpu
        loss: H x W s.t.  loss[i,j] is min_k ||L[i,j,k] - Q[i,j]|| on the cpu

    Does this in blocks on the CPU using numpy or GPU using cupy,
    and promotes types as needed (e.g., int16 -> int32 and float16 -> float32)

    NOTE: if L and Q are different modules (i.e. one is numpy the other cupy,
    this will cause an error) 
    """
    # code is GPU/CPU agnostic
    # if you send cupy arrays, does everything on GPU
    # else does it on CPU with numpy arrays
    xp = get_array_module(L)

    shape = L.shape
    assert len(shape) < 5 and len(shape) > 2, "Unrecognized shape of LookUp Table"
    if len(shape) == 3:
        HW, Z, C = L.shape
        return_shape = HW
    else:
        H, W, Z, C = L.shape
        HW = H*W
        return_shape = (H,W)
        L = L.reshape(HW, Z, C)
        Q = Q.reshape(HW, C)
    numBlocks = (HW // block_size) + (1 if HW % block_size != 0 else 0)
    minD = xp.zeros((HW), dtype=xp.long)
    loss = xp.zeros((HW), dtype=(xp.float32 if dtype in [xp.float16, xp.float32] else xp.int32))
    for block in range(numBlocks):
        sy, ey = block * block_size, min(HW, ((block+1) * block_size))
        if dtype in [xp.float16, xp.float32]:
            LUp = L[sy:ey,:,:].astype(xp.float32)
            QUp = Q[sy:ey,None,:].astype(xp.float32)
        elif dtype in [xp.int16, xp.int32]:
            # if it's an int, do the arithmetic in int32 to avoid overflow
            LUp = L[sy:ey,:,:].astype(xp.int32)
            QUp = Q[sy:ey,None,:].astype(xp.int32)
        distance = xp.sum((LUp-QUp)**2, axis=-1)
        minIndex = xp.argmin(distance , axis=-1)
        minD[sy:ey] = minIndex
        loss[sy:ey] = xp.squeeze(xp.take_along_axis(distance,minIndex[:,None],axis=1))

    return minD.reshape(return_shape), loss.reshape(return_shape)

def restrict_lut_depth_range(lut, index, delta):
    """
    Given a lookup table, a 2D array of indices, and an integer delta,
    this function returns a restricted  lookup table of the values
    around indices +- delta.

    Parameters
    ----------
    lut (H x W x Z x C)
        table to be reduced
    index (H x W) : np.ndarray of int type
        indices of the table
    delta : int
        integer value to reduce table with values only around index +- delta

    Returns
    -------
    reduced_lut (H x W x z x C)
        table where z (< Z) is now a reducded range
    start (H x W) : np.ndarray of int type
        this is useful util for TC and C2F
        it usually will be index - delta, but it gets
        clipped between 0 and max_index - 2*delta

    NOTE: if arrays are different modules
    (i.e. some are numpy, others cupy, this will cause an error)
    """
    xp = get_array_module(lut)

    shape = lut.shape
    # if lut is passed as HW x Z x C (i.e. a 3D array),
    # make it 4D so that the code works below
    # this can happen when it is passed with a mask
    assert len(shape) < 5 and len(shape) > 2, "Unrecognized shape of LookUp Table"
    if len(shape) == 3:
        lut = lut[None,...]
        index = index[None,...]
    shape = lut.shape
    # avoid underflow, cast it to int16, then back to uint16
    start = (index.astype(xp.int16)-delta).clip(0, shape[2]-2*delta).astype(xp.uint16)
    i, j = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]), indexing='ij')
    k = xp.arange(2*delta)
    # collect LUT only +- delta around previous_index
    reduced_lut = lut[i[..., None], j[..., None], start[..., None] + k]
    # in the case of original shape==3, squeeze is necessary here
    return xp.squeeze(reduced_lut), xp.squeeze(start)

def c2f_lut(lut,
            dep,
            normalized_image,
            ks,
            deltas,
            block_size: int = 65336,
            mask=None):
    """
    Function to run Coarse-to-Fine (C2F) reconstruction wiht LookUp3D.

    Returns
    -------
    depth_map: array_like

    loss_map : array_like

    index_map : array_like

    NOTE: if arrays are different modules
    (i.e. some are numpy, others cupy, this will cause an error)
    """
    xp = get_array_module(normalized_image)

    previous_index = xp.zeros(shape=(normalized_image.shape[:2]), dtype=xp.uint16)
    c2f_mask = xp.full(shape=(normalized_image.shape[:2]), fill_value=False)

    # TODO: this mask operation slows down everything -- it is not free
    # why are we doing array[mask] when mask is full 
    if mask is None:
        mask = xp.full(shape=(normalized_image.shape[:2]), fill_value=True)

    for iter in range(len(ks) - 1):
        k = ks[iter]
        delta = deltas[iter]
        c2f_mask[::k,::k] = mask[::k, ::k]
        L = lut[c2f_mask,...]
        n = normalized_image[c2f_mask,...]
        p = previous_index[c2f_mask]

        L, start = restrict_lut_depth_range(L, p, delta)
        minD, _ = blockLookup(L, n, dtype=xp.float32, block_size=block_size)
        previous_index[c2f_mask] = minD + start

        jump = k//ks[iter+1]
        previous_index = replace_with_nearest(previous_index, '=', 0)
        previous_index = gaussian_blur(previous_index, sigmas=jump)
    
    # FULL RESOLUTION
    depth_map = xp.full(shape=(normalized_image.shape[:2]), fill_value=-1., dtype=xp.float32)
    loss_map = xp.full(shape=normalized_image.shape[:2], fill_value=xp.inf, dtype=xp.float32)
    index_map = xp.zeros(shape=normalized_image.shape[:2], dtype=xp.uint16)
    
    L, start = restrict_lut_depth_range(lut, previous_index, delta)
    minD, loss = blockLookup(L[mask], normalized_image[mask], dtype=xp.float32, block_size=block_size)
    loss_map[mask] = loss
    index_map[mask] = minD + start[mask]
    depth_map[mask] = xp.squeeze(xp.take_along_axis(dep[mask],index_map[mask,None],axis=-1))

    return depth_map, loss_map, index_map

def tc_lut(lut,
           dep,
           normalized_image,
           delta,
           previous_index,
           block_size: int = 65536,
           mask=None):
    """
    TODO: write description

    Returns
    -------
    depth_map: array_like

    loss_map : array_like

    index_map : array_like

    NOTE: if arrays are different modules
    (i.e. some are numpy, others cupy, this will cause an error)
    """
    xp = get_array_module(normalized_image)

    depth_map = xp.full(shape=(normalized_image.shape[:2]), fill_value=-1., dtype=xp.float32)
    loss_map = xp.full(shape=normalized_image.shape[:2], fill_value=xp.inf, dtype=xp.float32)
    index_map = xp.zeros(shape=normalized_image.shape[:2], dtype=xp.uint16)

    # handle mask
    if mask is None:
        LUT = lut
        DEP = dep
        NORMALIZED = normalized_image
        PREVIOUS = previous_index
    else:
        LUT = lut[mask]
        DEP = dep[mask]
        NORMALIZED = normalized_image[mask]
        PREVIOUS = previous_index[mask]
    
    L, start = restrict_lut_depth_range(LUT, PREVIOUS, delta)
    _minD, _loss_map = blockLookup(L, NORMALIZED, dtype=xp.float32, block_size=block_size)
    _minD += start
    _depth_map = xp.squeeze(xp.take_along_axis(DEP,_minD[:,None],axis=-1))

    # handle mask
    if mask is None:
        depth_map = _depth_map
        loss_map = _loss_map
        index_map = _minD
    else:
        depth_map[mask] = _depth_map
        loss_map[mask] = _loss_map
        index_map[mask] = _minD

    return depth_map, loss_map, index_map

def naive_lut(lut,
              dep,
              normalized_image,
              block_size: int = 65536,
              mask = None):
    """
    NOTE: if lut, dep, normalized_image, mask are different modules
    (i.e. some are numpy, others cupy, this will cause an error)
    """
    xp = get_array_module(normalized_image)

    depth_map = xp.full(shape=(normalized_image.shape[:2]), fill_value=-1., dtype=xp.float32)
    loss_map = xp.full(shape=normalized_image.shape[:2], fill_value=xp.inf, dtype=xp.float32)
    index_map = xp.zeros(shape=normalized_image.shape[:2], dtype=xp.uint16)

    # handle mask
    if mask is None:
        LUT = lut
        DEP = dep
        NORMALIZED = normalized_image
    else:
        LUT = lut[mask]
        DEP = dep[mask]
        NORMALIZED = normalized_image[mask]

    _minD, _loss_map = blockLookup(LUT, NORMALIZED, dtype=xp.float32, block_size=block_size)
    _depth_map = xp.squeeze(xp.take_along_axis(DEP,_minD[:,None],axis=-1))
    
    # handle mask
    if mask is None:
        depth_map = _depth_map
        loss_map = _loss_map
        index_map = _minD
    else:
        depth_map[mask] = _depth_map
        loss_map[mask] = _loss_map
        index_map[mask] = _minD
    
    return depth_map, loss_map, index_map

class LookUpCalibration:
    """
    """
    def __init__(self, config: dict | str = None):
        self.camera = Camera()

        self.plane_pattern = None
        self.root = None
        self.calibration_directory = []
        self.num_directories = 0
        self.structure_grammar = {}
        self.num_channels = 0
        self.roi = ()

        # flag for verbose
        self.verbose = False
        self.debug = False
        # flag for parallelizing position processing
        self.parallelize_positions = False
        self.num_cpus = 1

        if config is not None:
            self.load_config(config)

    def load_config(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        # TODO: think about replacing these hard-coded config parsing
        # to a dynamic parsing using setattr.
        # issue here is that some, like set_camera() actually do
        # more than just set the attribute... think more about it

        # for key, value in config['look_up_calibration'].items:
        #     setattr(self, key, value)
        
        self.set_camera(config['look_up_calibration']['camera'])
        self.set_plane_pattern(config['look_up_calibration']['plane_pattern'])
        self.set_roi(config['look_up_calibration']['roi'])
        self.set_calibration_directory(config['look_up_calibration']['calibration_directory'])
        self.set_structure_grammar(config['look_up_calibration']['structure_grammar'])
        self.set_verbose(config['look_up_calibration']['verbose'])
        self.set_debug(config['look_up_calibration']['debug'])
        self.set_parallelize_positions(config['look_up_calibration']['parallelize_positions'])
        self.set_num_cpus(config['look_up_calibration']['num_cpus'])

    # setters
    def set_camera(self, camera: str | dict | Camera):
        # if string, load the json file and get the parameters
        # check if dict, get the parameters
        if isinstance(camera, (str, dict)):
            self.camera.load_config(camera)
        else:
            self.camera = camera

    def set_plane_pattern(self, pattern: dict | Charuco | CheckerBoard):
        """
        Parameters
        ----------
        pattern : dictionary | Charuco object | CheckerBoard object
            Calibration pattern used to find where the plane of calibration is.
            This is often a printed / physical board to assist camera detection 
            of the calibration plane, where the projected pattern lies.
            Function only aceppts ChArUco and Checkerboard/Chessboard.
        """
        if type(pattern) is dict:
            try:
                if pattern["type"] == "charuco":
                    pattern = Charuco(board_config=pattern)
                elif pattern["type"] == "checkerboard":
                    pattern = CheckerBoard(board_config=pattern)
            except:
                pattern = Charuco(board_config=pattern)
        self.plane_pattern = pattern

    def set_calibration_directory(self, path: str):
        """

        Parameters
        ----------
        path : str
            path to directory containing multiple scenes folders,
            each with the same structured light capture of an object
        """
        assert os.path.isdir(path), "This is not a directory. This function only works with a folder."
        self.root = os.path.abspath(path)
        self.calibration_directory = get_all_folders(self.root)
        self.num_directories = len(self.calibration_directory)

    def set_structure_grammar(self, structure_grammar: dict):
        """
        The structure grammar is the configuration to read images
        and save them to pattern/look up tables accordingly.
        Example below:
            structure_grammar = {
                "name": "gray",
                "images": ["img_02.tiff", "img_04.tiff", "img_06.tiff"],
                "num_channels": 3,
                "utils": {
                    "white": "white.tiff", (or "green.tiff" if monochormatic, for instance)
                    "black": "black.tiff",
                }
            }
        The list of strings are the images which will be used
        to create a look up table with the key name.
        """
        self.structure_grammar = structure_grammar

    def set_roi(self, roi: list | np.ndarray):
        """

        Parameters
        ----------
        roi : list or set
            xyxy region of interest to reduce size of look up tables 
        
        Note: set it to None if you would like to use the whole pixel array from the camera. 
        """
        self.roi = roi

    def set_verbose(self, verbose: bool):
        """
        Parameters
        ----------
        verbose : bool
            Whether to print messages while processing.
            Default is False
        """
        self.verbose = verbose

    def set_debug(self, debug: bool):
        """
        Parameters
        ----------
        verbose : bool
            Whether to save extra debug results.
            Default is False
        """
        self.debug = debug

    def set_parallelize_positions(self, parallelize: bool):
        """
        Parameters
        ----------
        parallelize : bool
            Whether to parallelize the position processing.
        """
        self.parallelize_positions = parallelize
    
    def set_num_cpus(self, num_cpus: int):
        """
        Parameters
        ----------
        num_cpus : int
            Number of CPUs to use for parallel processing.
        """
        self.num_cpus = int(max(1, num_cpus))

    # getters

    # functions
    def save_lookup_table(self,
                            lookup_table: np.ndarray,
                            filename):
                        #   roi: np.ndarray):
        """
        Save the lookup table as a npy file.
        """
        np.save(os.path.join(self.root, filename), lookup_table)
        if self.debug:
            x = lookup_table.shape[1]//2
            y = lookup_table.shape[0]//2
            np.save(os.path.join(self.root, f'{filename}_ray_{y}_{x}'), lookup_table[y,x,:,:])

    def normalize_positions(self):
        if self.verbose:
            print('-' * 15)
            print(f"Beginning normalization for all positions for LookUp Table {self.structure_grammar['name']}")
        # optional parallelization
        if self.parallelize_positions:
            if self.verbose:
                print(f"Parallelizing with {self.num_cpus} CPU cores")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                futures = [executor.submit(self.normalize_pattern, folder, self.structure_grammar) for folder in self.calibration_directory]
                concurrent.futures.wait(futures)  # Wait for all tasks to complete
        else:
        # original sequential processing
            for folder in self.calibration_directory:
                self.normalize_pattern(folder, self.structure_grammar)

    def find_depth_positions(self):
        if self.verbose:
            print('-' * 15)
            print(f"Beginning calibration for all positions for LookUp Table {self.structure_grammar['name']}")
        
        # optional parallelization
        if self.parallelize_positions:
            if self.verbose:
                print(f"Parallelizing with {self.num_cpus} CPU cores")
            if self.verbose:
                print(f"Detecting markers for board trajectory")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                futures = [executor.submit(self.find_plane, self.camera, self.plane_pattern, folder, self.structure_grammar) for folder in self.calibration_directory]
                concurrent.futures.wait(futures)  # Wait for all tasks to complete
            if self.verbose:
                print(f"Fitting board trajectory into line")
            self.fit_plane_trajectory(self.calibration_directory)
            if self.verbose:
                print(f"Generating depth.npz for each board position")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                futures = [executor.submit(self.find_depth, self.camera, folder) for folder in self.calibration_directory]
                concurrent.futures.wait(futures)  # Wait for all tasks to complete
        
        else:
        # original sequential processing
            if self.verbose:
                print(f"Detecting markers for board trajectory")
            for folder in self.calibration_directory:
                self.find_plane(self.camera, self.plane_pattern, folder, self.structure_grammar)
            if self.verbose:
                print(f"Fitting board trajectory into line")
            self.fit_plane_trajectory(self.calibration_directory)
            if self.verbose:
                print(f"Generating depth.npz for each board position")
            for folder in self.calibration_directory:
                self.find_depth(self.camera, folder)

    def stack_results(self):
        """
        TODO: cannot parallelize multiple function calls to the same memory address lookup_table.
        Best thing to do is similar to Yurii's legacy code, where we, at the end, concatenate all single_positions
        into a MEGA lookup table.
        """
        def _stack_single_pattern_single_position(table: np.ndarray, index: int, folder: str, table_name: str):
            """
            Util function to ...

            This function is meant to be innacessible to user.
            """
            depth = np.load(os.path.join(folder, 'depth.npz'))['depth'][y0:y0+height, x0:x0+width]
            pattern = np.load(os.path.join(folder, f'{table_name}.npz'))['pattern'][y0:y0+height, x0:x0+width]
            table[:,:,index,:] = np.concatenate([pattern, depth[:, :, np.newaxis]], axis=2)
        
        if self.verbose:
            print('-' * 15)
            print(f"Beginning stacking of all normalized patterns and depth")
        
        table_name = self.structure_grammar['name']
        width, height = self.camera.get_image_shape()
        x0, y0 = 0, 0

        if len(self.roi) == 4:
            x0 = self.roi[0]
            y0 = self.roi[1]
            width = self.roi[2] - x0
            height = self.roi[3] - y0

        # allocate memory for massive, single float precision numpy array
        lookup_table = np.full(shape=(height, width, self.num_directories, self.structure_grammar['num_channels'] + 1), fill_value=np.nan, dtype=np.float32)
        
        # optional parallelization
        if self.parallelize_positions:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                futures = [executor.submit(_stack_single_pattern_single_position, lookup_table, pos, folder, table_name) for pos, folder in enumerate(self.calibration_directory)]
                concurrent.futures.wait(futures)  # Wait for all tasks to complete

        else:
            # original sequential processing
            for pos, folder in enumerate(self.calibration_directory):
                _stack_single_pattern_single_position(lookup_table, pos, folder, table_name)

        if self.verbose:
            print('-' * 15)
            print(f"Saving LookUp Table {table_name} and Calibration Info")

        self.save_lookup_table(lookup_table, table_name)
        save_json({'roi': self.roi,
                    'structure_grammar': self.structure_grammar,
                    'date': datetime.now().strftime('%m/%d/%Y, %H:%M:%S')},
                    os.path.join(self.root, f'{table_name}_calibration_info.json'))

    @staticmethod
    def find_plane(camera: Camera,
                   calibration_board: Charuco | CheckerBoard,
                   folder: str,
                   structure_grammar: dict):
        utils: dict = structure_grammar['utils']
        depth_image = load_ldr(os.path.join(folder, utils['depth'])) if 'depth' in utils else load_ldr(os.path.join(folder, utils['white']))

        # detect plane/board markers with camera
        img_points, obj_points, _ = \
            calibration_board.detect_markers(depth_image)
        
        # although plane reconstruction requires 3 points,
        # OpenCV extrinsic calibration requires 6 points
        if len(img_points) < 6:
            rvec, tvec = None, None
        else:
            # find relative position of camera and board
            result = Calibration.calibrate_extrinsic(obj_points,
                                                    img_points,
                                                    camera.K,
                                                    camera.dist_coeffs)
            rvec = result['rvec'].ravel()
            tvec = result['tvec'].ravel()

        save_json({
            'original': {
                'rvec': rvec,
                'tvec': tvec,
            }
        }, os.path.join(folder, 'board.json'))

    @staticmethod
    def fit_plane_trajectory(calibration_directory):

        rvecs = []
        tvecs = []
        
        for folder in calibration_directory:
            d = load_json(os.path.join(folder, 'board.json'))
            tvec = np.full(shape=(3,), fill_value=np.nan) if d['original']['tvec'] is None else d['original']['tvec']
            rvec = np.full(shape=(3,), fill_value=np.nan) if d['original']['rvec'] is None else d['original']['rvec']
            rvecs.append(rvec)
            tvecs.append(tvec)
        
        tvecs = np.squeeze(np.array(tvecs))
        rvecs = np.squeeze(np.array(rvecs))
        median_rvec = np.nanmean(rvecs, axis=0)

        C, N = fit_line(tvecs[~np.isnan(tvecs).any(axis=1)])
        if np.dot(N, [0, 0, 1]) > 0:
            # either direction in LINE can be returned, so we need to
            # make sure our line is moving in the direction we want
            N = -N
        middle_index = np.nanargmin(np.linalg.norm(tvecs - C, axis=1))

        steps = np.linalg.norm(np.diff(tvecs, axis=0), axis=1)
        idx = np.abs(steps - np.nanmedian(steps)) < np.nanstd(steps)
        stride = np.nanmean(steps[idx])

        save_json({
            'stride': stride,
            'number_of_folders': len(calibration_directory),
            'number_of_identified_planes': len(steps),
            'number_steps_within_1_std': len(steps[idx])
        }, os.path.join(get_folder_from_file(calibration_directory[0]), 'board_trajectory.json'))

        for i, folder in enumerate(calibration_directory):
            d = load_json(os.path.join(folder, 'board.json'))
            d['fitted'] = {
                'rvec': median_rvec,
                'tvec': C + (i - middle_index) * stride * N
            }
            save_json(d, os.path.join(folder, 'board.json'))

    @staticmethod
    def find_depth(camera: Camera,
                   folder: str):
        plane_data = load_json(os.path.join(folder, 'board.json'))
        rvec, tvec = plane_data['fitted']['rvec'], plane_data['fitted']['tvec']
        R, _ = cv2.Rodrigues(rvec)
        T = np.array(tvec).reshape(3,1)

        width, height = camera.get_image_shape()
        campixels_x, campixels_y = np.meshgrid(np.arange(width),
                                               np.arange(height))
        campixels = np.stack([campixels_x, campixels_y], axis=-1).reshape((-1,2))

        origin, rays = camera_to_ray_world(campixels,
                                                       R,
                                                       T,
                                                       camera.K,
                                                       camera.dist_coeffs)
        points = intersect_line_with_plane(origin,
                                                       rays,
                                                       np.array([0,0,0], dtype=np.float32),
                                                       np.array([0,0,1], dtype=np.float32))

        # save only Z of the depth
        result3D = np.matmul(points, R.T) + T.T
        depth = result3D[:,2]

        # result3D = np.matmul((points - T.T), R)
        # # depth = result3D[:,2]
        # # TODO: fix this -- should not be abs, it should simply already be positive
        # depth = np.abs(result3D[:,2])
        
        # bs = np.matmul(R.T, (xs - T).T).T
        # idx = (roi[0] < bs[:, 0]) & (bs[:, 0] < roi[2]) & \
        #     (roi[1] < bs[:, 1]) & (bs[:, 1] < roi[3])
        # bs = None

        # depth[~idx] = 0

        depth = depth.reshape((height, width))
        np.savez_compressed(os.path.join(folder, f"depth.npz"), depth=depth)

    @staticmethod
    def normalize_pattern(folder: str,
                          structure_grammar: dict):
        table_name = structure_grammar['name']
        images: list[str] = structure_grammar['images']
        utils: dict = structure_grammar['utils']
        pattern_images =  np.concatenate([np.atleast_3d(load_ldr(os.path.join(folder, image))) for image in images], axis=2)
        white_image = load_ldr(os.path.join(folder, utils['white']))
        black_image = None if 'black' not in utils else load_ldr(os.path.join(folder, utils['black']))
        normalized = normalize_color(color_image=pattern_images,
                                                white_image=white_image,
                                                black_image=black_image)
        np.savez_compressed(os.path.join(folder, f"{table_name}.npz"), pattern=normalized)

    def run(self, depth: bool = False, normalize: bool = False):
        if depth:
            self.find_depth_positions()
        if normalize:
            self.normalize_positions()
        self.stack_results()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look Up Calibration and Reconstruction")
    parser.add_argument('-c', '--calibration_config', type=str, default=None,
                        help="Path to config for Look Up Calibration")
    parser.add_argument('--depth', default=False, action=argparse.BooleanOptionalAction,
                        help="Flag to set if LookUp Calibration should (Re)Calculate Depth")
    parser.add_argument('--normalize', default=False, action=argparse.BooleanOptionalAction,
                        help="Flag to set if LookUp should (Re)Normalize Pattern")
    # parser.add_argument('--gpu', default=False, action=argparse.BooleanOptionalAction,
                        # help="Flag to set if LookUp Reconstruction should use GPU (cuda only)")
    # parser.add_argument('-r', '--reconstruction_config', type=str, default=None,
                    # help='Path to config JSON with parameters for LookUp Reconstruction')

    args = parser.parse_args()

    if args.calibration_config is not None:
        luc = LookUpCalibration(args.calibration_config)
        luc.run(args.depth, args.normalize)

    # if args.reconstruction_config is not None:
    #     lur = LookUpReconstruction(args.reconstruction_config)
    #     lur.run(args.normalize, args.gpu)