import argparse
import cv2
import concurrent.futures
from datetime import datetime
import numpy as np
import os
import torch

from src.utils.three_d_utils import ThreeDUtils
from src.utils.file_io import save_json, load_json, get_all_paths, get_all_folders, get_folder_from_file
from src.utils.image_utils import ImageUtils
from src.utils.numerics import k_smallest_indices, spline_interpolant, calculate_loss, blockLookup, blockLookupNumpy
from src.scanner.camera import Camera
from src.scanner.calibration import Calibration, CheckerBoard, Charuco

def load_low_rank_table(filename: str):
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

    """
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
    start = (index.astype(np.int16)-delta).clip(0, shape[2]-2*delta).astype(np.uint16)
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    k = np.arange(2*delta)
    # collect LUT only +- delta around previous_index
    reduced_lut = lut[i[..., None], j[..., None], start[..., None] + k]
    # in the case of original shape==3, squeeze is necessary here
    return np.squeeze(reduced_lut), np.squeeze(start)

def c2f_lut(lut, normalized_image, ks, deltas, mask=None):
    """
    TODO
    Returns
    -------
    index_map : np.ndarray
        
    loss_map : np.ndarray
    """
    previous_index = np.zeros(shape=(normalized_image.shape[:2]), dtype=np.uint16)
    c2f_mask = np.full(shape=(normalized_image.shape[:2]), fill_value=False)

    if mask is None:
        mask = np.full(shape=(normalized_image.shape[:2]), fill_value=True)

    for iter in range(len(ks) - 1):
        k = ks[iter]
        delta = deltas[iter]
        c2f_mask[::k,::k] = mask[::k, ::k]
        L = lut[c2f_mask,...]
        n = normalized_image[c2f_mask,...]
        p = previous_index[c2f_mask]

        L, start = restrict_lut_depth_range(L, p, delta)
        minD, _ = blockLookupNumpy(L, n, dtype=np.float32, blockSize=1024)
        previous_index[c2f_mask] = minD + start

        jump = k//ks[iter+1]
        previous_index = ImageUtils.replace_with_nearest(previous_index, '=', 0)
        previous_index = ImageUtils.gaussian_blur(previous_index, sigmas=jump)
    
    # FULL RESOLUTION
    loss_map = np.full(shape=normalized_image.shape[:2], fill_value=np.inf, dtype=np.float32)
    index_map = np.zeros(shape=normalized_image.shape[:2], dtype=np.uint16)
    
    L, start = restrict_lut_depth_range(lut, previous_index, delta)
    minD, loss = blockLookupNumpy(L[mask], normalized_image[mask], dtype=np.float32, blockSize=1024)
    loss_map[mask] = loss
    index_map[mask] = minD + start[mask]

    return index_map, loss_map

def tc_lut(lut, normalized_image, delta, previous_index, mask=None):
    """
    TODO

    Returns
    -------
    index_map : np.ndarray
        
    loss_map : np.ndarray
    """
    loss_map = np.full(shape=normalized_image.shape[:2], fill_value=np.inf, dtype=np.float32)
    index_map = np.zeros(shape=normalized_image.shape[:2], dtype=np.uint16)
    if mask is None:
        mask = np.full(shape=(normalized_image.shape[:2]), fill_value=True)
    
    L, start = restrict_lut_depth_range(lut, previous_index, delta)
    minD, loss = blockLookupNumpy(L[mask], normalized_image[mask], dtype=np.float32, blockSize=1024)
    loss_map[mask] = loss
    index_map[mask] = minD + start[mask]
    
    return index_map, loss_map

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
        self.roi = None

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
        
        self.set_camera(config['look_up_calibration']['camera_calibration'])
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
        if isinstance(camera, str) or isinstance(camera, dict):
            self.camera.load_calibration(camera)
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

        if self.roi is not None:
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
        depth_image = ImageUtils.load_ldr(os.path.join(folder, utils['depth'])) if 'depth' in utils else ImageUtils.load_ldr(os.path.join(folder, utils['white']))

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

        C, N = ThreeDUtils.fit_line(tvecs[~np.isnan(tvecs).any(axis=1)])
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

        origin, rays = ThreeDUtils.camera_to_ray_world(campixels,
                                                       R,
                                                       T,
                                                       camera.K,
                                                       camera.dist_coeffs)
        points = ThreeDUtils.intersect_line_with_plane(origin,
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
        pattern_images =  np.concatenate([np.atleast_3d(ImageUtils.load_ldr(os.path.join(folder, image))) for image in images], axis=2)
        white_image = ImageUtils.load_ldr(os.path.join(folder, utils['white']))
        black_image = None if 'black' not in utils else ImageUtils.load_ldr(os.path.join(folder, utils['black']))
        normalized = ImageUtils.normalize_color(color_image=pattern_images,
                                                white_image=white_image,
                                                black_image=black_image)
        np.savez_compressed(os.path.join(folder, f"{table_name}.npz"), pattern=normalized)

    def run(self, depth: bool = False, normalize: bool = False):
        if depth:
            self.find_depth_positions()
        if normalize:
            self.normalize_positions()
        self.stack_results()

class LookUpReconstruction:
    def __init__(self, config: dict | str = None):
        self.camera = Camera()

        self.reconstruction_directory = []
        self.structure_grammar = {}

        # reconstruction utils
        ## TODO: stopgap for GPU for now
        self.lut = None
        self.dep = None
        self.white_image = None
        self.mask_thr = None # threshold for mask
        self.mask = None
        self.pattern_images = None
        self.black_image = None
        self.normalized = None

        # flag for verbose
        self.verbose = False

        # reconstruction outpus
        self.depth_map = None
        self.point_cloud = None
        self.colors = None
        self.normals = None
        self.loss_thr = np.inf # threshold for loss map when creating point cloud
        self.loss_map = None
        self.index_map = None

        if config is not None:
            self.load_config(config)

    def load_config(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        self.set_camera(config['look_up_reconstruction']['camera_calibration'])
        # TODO: consider moving roi and mask_thr to structure_grammar['utils']
        self.set_mask_threshold(config['look_up_reconstruction']['mask_thr'])
        self.set_loss_threshold(config['look_up_reconstruction']['loss_thr'])

        self.set_lookup_table(config['look_up_reconstruction']['lookup_table'])
        self.set_reconstruction_directory(config['look_up_reconstruction']['reconstruction_directory'])
        self.set_structure_grammar(config['look_up_reconstruction']['structure_grammar'])
        self.set_verbose(config['look_up_reconstruction']['verbose'])
        self.set_outputs(config['look_up_reconstruction']['outputs'])

    # setters
    def set_camera(self, camera: str | dict | Camera):
        # if string, load the json file and get the parameters
        # check if dict, get the parameters
        if isinstance(camera, str) or isinstance(camera, dict):
            self.camera.load_calibration(camera)
        else:
            self.camera = camera

    def set_reconstruction_directory(self, path: str):
        """

        Parameters
        ----------
        path : str
            path to directory containing multiple scene folders,
            each with the images that match the "utils" and the "tables" images
            for LookUp reconstruction
        """
        assert os.path.isdir(path), "This is not a directory. This function only works with a folder."
        self.reconstruction_directory = os.path.abspath(path)
        
    def set_lookup_table(self, lookup_table: str | np.ndarray):
        """

        Parameters
        ----------
        lookup_table : str or array_like
            if string, must be path to load lookup table (.npy file)
        """
        if isinstance(lookup_table, str):
            lookup_table = np.load(lookup_table)
        self.lut = lookup_table[:,:,:,:-1]
        self.dep = lookup_table[:,:,:,-1]

    def set_structure_grammar(self, structure_grammar: dict):
        """
        The structure grammar is the configuration to read images, look up tables, 
        and create the depth maps/point clouds accordingly.
        Example below:
            structure_grammar = {
                "name": "gray",
                "images": ["img_02.tiff", "img_04.tiff", "img_06.tiff"],
                "utils": {
                    "white": "white.tiff",
                    "colors: "white.tiff,
                    "black": "black.tiff",
                    "roi": [minX, minY, maxX, maxY]
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

    def set_mask_threshold(self, thr: float):
        """

        Parameters
        ----------
        thr : float
            
        """
        self.mask_thr = min(1., max(0., float(thr)))

    def set_loss_threshold(self, thr: float):
        """

        Parameters
        ----------
        thr : float
            
        """
        self.loss_thr = max(0., float(thr))

    def set_verbose(self, verbose: bool):
        """
        Parameters
        ----------
        verbose : bool
            Whether to print messages while processing.
            Default is False
        """
        self.verbose = verbose

    def set_outputs(self, outputs: dict):
        """
        Parameters
        ----------
        outputs : dict
            Dictionary containing which outputs to save.
            outputs: {
                "depth_map": True,
                "point_cloud": True,
                "loss_map": True,
                "index_map": True,
                "mask": True
            }
            Each can be set to True or False

        Notes: 
        - depth_map is a HxW numpy array (float32) containing depth per pixel of the result.
        - point_cloud is a 3D numpy array of the result.
        - loss_map is a HxW numpy array (float32) containing the loss from the pixel and the first nearest neighbor.
        - index_map is a HxW numpy array (uint16) containing the index of retrieved depth.
        """
        self.outputs = outputs

    # getters
    def mask_already_exists(self):
        return os.path.exists(os.path.join(self.reconstruction_directory, 'mask.npy'))

    # reconstruction functions
    def extract_mask(self):
        """
        """

        if self.mask_already_exists():
            if self.verbose:
                print('-' * 15)
                print("Found mask -- loading it")
            self.mask = np.load(os.path.join(self.reconstruction_directory, 'mask.npy'))
        else:
            self.white_image = ImageUtils.load_ldr(os.path.join(self.reconstruction_directory, self.structure_grammar['utils']['white']))
            if self.verbose:
                print('-' * 15)
                print("Extracting mask from white image")
            self.mask = ImageUtils.extract_mask(np.atleast_3d(self.white_image), self.mask_thr)

    def reconstruct(self, gpu: bool=False):
        """
        """
        if self.verbose:
            print('-' * 15)
            print("Beginning reconstruction")
        table_name = self.structure_grammar['name']

        if self.normalized is None:
            self.normalized = np.load(os.path.join(self.reconstruction_directory, f"{table_name}.npz"))['pattern']
        shape = self.normalized.shape[:2]
        
        if self.mask is None:
            self.extract_mask()
            
        roi = self.structure_grammar['utils']['roi']
        if self.mask.shape != shape:
            self.mask = ImageUtils.crop(self.mask, roi=roi)
        # allocate the memory for depth_map
        self.depth_map = np.full(shape=shape, fill_value=-1., dtype=np.float64)
        self.loss_map = np.full(shape=shape, fill_value=np.inf, dtype=np.float64)
        self.index_map = np.zeros(shape=shape, dtype=np.uint32)

        ## GPU
        if gpu:
            m = torch.from_numpy(self.mask)
            no = torch.from_numpy(self.normalized[self.mask])

            minD, loss_map = blockLookup(self.lut[m], no, dtype=torch.float32, blockSize=65536)
            depth_map = self.dep[m].gather(dim=1, index=minD.unsqueeze(-1)).squeeze(-1)  

            self.depth_map[self.mask] = depth_map.cpu().numpy()
            self.loss_map[self.mask] = loss_map.cpu().numpy()
            self.index_map[self.mask] = minD.cpu().numpy()
        
        ## CPU
        else:
            minD, loss_map = blockLookupNumpy(self.lut[self.mask], self.normalized[self.mask], dtype=np.float32, blockSize=65536)
            depth_map = np.squeeze(np.take_along_axis(self.dep[self.mask],minD[:,None],axis=-1))
            
            self.depth_map[self.mask] = depth_map
            self.loss_map[self.mask] = loss_map
            self.index_map[self.mask] = minD

    def save_outputs(self):
        table_name = self.structure_grammar['name']
        utils: dict = self.structure_grammar['utils']
        roi: tuple = None if 'roi' not in utils else utils['roi']

        if 'depth_map' in self.outputs and self.outputs['depth_map']:
            np.save(os.path.join(self.reconstruction_directory,f"{table_name}_depth_map.npy"), self.depth_map)
            if self.verbose:
                print('-' * 15)
                print("Saved depth map")
        
        if 'loss_map' in self.outputs and self.outputs['loss_map']:
            np.save(os.path.join(self.reconstruction_directory,f"{table_name}_loss_map.npy"), self.loss_map)
            if self.verbose:
                print('-' * 15)
                print("Saved loss map")

        if 'index_map' in self.outputs and self.outputs['index_map']:
            np.save(os.path.join(self.reconstruction_directory,f"{table_name}_index_map.npy"), self.index_map)
            if self.verbose:
                print('-' * 15)
                print("Saved loss map")

        if 'point_cloud' in self.outputs and self.outputs['point_cloud']:
            if self.verbose:
                print('-' * 15)
                print("Constructing point cloud")

            mask = (self.depth_map > 0).flatten() & (self.loss_map < self.loss_thr).flatten()

            self.point_cloud: np.ndarray = ThreeDUtils.point_cloud_from_depth_map(depth_map=self.depth_map,
                                                                            K=self.camera.K,
                                                                            dist_coeffs=self.camera.dist_coeffs,
                                                                            R=self.camera.R,
                                                                            T=self.camera.T,
                                                                            roi=roi)
            if self.verbose:
                print("Extracting normals for point cloud")
            self.normals = ThreeDUtils.normals_from_point_cloud(self.point_cloud)

            if self.verbose:
                print("Extracting colors for point cloud")

            if self.colors is None and ('colors' not in utils or utils['colors'] is None):
                print("No color image set, therefore no color extraction for point cloud")
                ThreeDUtils.save_point_cloud(os.path.join(self.reconstruction_directory,f"{table_name}_point_cloud.ply"),
                    self.point_cloud.reshape((-1,3))[mask],
                    self.normals.reshape((-1,3))[mask])
            elif self.colors is not None:
                ThreeDUtils.save_point_cloud(os.path.join(self.reconstruction_directory,f"{table_name}_point_cloud.ply"),
                                    self.point_cloud.reshape((-1,3))[mask],
                                    self.normals.reshape((-1,3))[mask],
                                    self.colors.reshape((-1,3))[mask])
            else:
                color_image = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(self.reconstruction_directory, utils['colors'])), roi=utils['roi'])
                minimum = np.min(color_image)
                maximum = np.max(color_image)
                self.colors: np.ndarray = ((color_image - minimum) / (maximum - minimum))
                ThreeDUtils.save_point_cloud(os.path.join(self.reconstruction_directory,f"{table_name}_point_cloud.ply"),
                                    self.point_cloud.reshape((-1,3))[mask],
                                    self.normals.reshape((-1,3))[mask],
                                    self.colors.reshape((-1,3))[mask])
            if self.verbose:
                print("Saved point cloud")

        if 'mask' in self.outputs and self.outputs['mask']:
            np.save(os.path.join(self.reconstruction_directory,"mask.npy"), self.mask)
            if self.verbose:
                print('-' * 15)
                print("Saved mask")

    def run(self, normalize: bool=False, gpu: bool=False):
        if normalize:
            self.normalized = self.process_position(self.reconstruction_directory, self.structure_grammar)

        self.reconstruct(gpu=gpu)
        self.save_outputs()

    @staticmethod
    def process_position(folder: str, structure_grammar: dict) -> np.ndarray:
        """
        Processes the #N HxW pattern images and returns the normalized image (HxWxN).
        """
        name = structure_grammar['name']
        images: list[str] = structure_grammar['images']
        utils: dict = structure_grammar['utils']
        roi: tuple = None if 'roi' not in utils else utils['roi']

        print('-' * 15)
        print("Normalizing image")

        pattern_images = ImageUtils.crop(np.concatenate([np.atleast_3d(ImageUtils.load_ldr(os.path.join(folder, image))) for image in images], axis=2), roi=roi)
        white_image = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(folder, utils['white'])), roi=roi)
        black_image = None if (('black' not in utils) or (utils['black'] is None)) else ImageUtils.crop(ImageUtils.load_ldr(os.path.join(folder, utils['black'])), roi=roi)
        
        normalized = ImageUtils.normalize_color(color_image=pattern_images,
                                                white_image=white_image,
                                                black_image=black_image)
        np.savez_compressed(os.path.join(folder, f"{name}.npz"), pattern=normalized)

        return normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look Up Calibration and Reconstruction")
    parser.add_argument('-c', '--calibration_config', type=str, default=None,
                        help="Path to config for Look Up Calibration")
    parser.add_argument('--depth', default=False, action=argparse.BooleanOptionalAction,
                        help="Flag to set if LookUp Calibration should (Re)Calculate Depth")
    parser.add_argument('--normalize', default=False, action=argparse.BooleanOptionalAction,
                        help="Flag to set if LookUp should (Re)Normalize Pattern")
    parser.add_argument('--gpu', default=False, action=argparse.BooleanOptionalAction,
                        help="Flag to set if LookUp Reconstruction should use GPU (cuda only)")
    parser.add_argument('-r', '--reconstruction_config', type=str, default=None,
                    help='Path to config JSON with parameters for LookUp Reconstruction')

    args = parser.parse_args()

    if args.calibration_config is not None:
        luc = LookUpCalibration(args.calibration_config)
        luc.run(args.depth, args.normalize)

    if args.reconstruction_config is not None:
        lur = LookUpReconstruction(args.reconstruction_config)
        lur.run(args.normalize, args.gpu)