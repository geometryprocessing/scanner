import argparse
import cv2
import concurrent.futures
from datetime import datetime
import numpy as np
import os

from src.utils.three_d_utils import ThreeDUtils
from src.utils.file_io import save_json, load_json, get_all_paths, get_all_folders, get_folder_from_file
from src.utils.image_utils import ImageUtils
from src.utils.numerics import k_smallest_indices, spline_interpolant, calculate_loss
from src.scanner.camera import Camera
from src.scanner.calibration import Calibration, CheckerBoard, Charuco


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
            rvec = result['rvec']
            rvec /= np.linalg.norm(rvec)
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
            tvec = np.full(shape=(3,1), fill_value=np.nan) if d['original']['tvec'] is None else d['original']['tvec']
            rvec = np.full(shape=(3,1), fill_value=np.nan) if d['original']['rvec'] is None else d['original']['rvec']
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
        }, os.path.join(get_folder_from_file(get_folder_from_file(calibration_directory[0])), 'board_trajectory.json'))

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
        result3D = np.matmul((points - T.T), R)
        # depth = result3D[:,2]
        # TODO: fix this -- should not be abs, it should simply already be positive
        depth = np.abs(result3D[:,2])
        
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
        self.lookup_table = None
        self.white_image = None
        self.thr = None # threshold for mask
        self.mask = None
        self.pattern_images = None
        self.black_image = None
        self.normalized = None

        # flag for debugging and verbose
        self.debug = False
        self.verbose = False
        # flag for parallelizing pixel processing
        self.parallelize_pixels = False
        self.num_cpus = 1

        # reconstruction outpus
        self.depth_map = None
        self.point_cloud = None
        self.colors = None
        self.normals = None
        # extra outputs if debug=True
        self.loss_map = None
        self.loss_2 = None
        self.loss_12 = None
        self.depth_12 = None

        if config is not None:
            self.load_config(config)

    def load_config(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        self.set_camera(config['look_up_reconstruction']['camera_calibration'])
        # TODO: consider moving roi and mask_thr to structure_grammar['utils']
        self.set_roi(config['look_up_reconstruction']['roi'])
        self.set_mask_threshold(config['look_up_reconstruction']['mask_thr'])

        self.set_lookup_table(config['look_up_reconstruction']['lookup_table'])
        self.set_reconstruction_directory(config['look_up_reconstruction']['reconstruction_directory'])
        self.set_structure_grammar(config['look_up_reconstruction']['structure_grammar'])
        self.set_parallelize_pixels(config['look_up_reconstruction']['parallelize_pixels'])
        self.set_num_cpus(config['look_up_reconstruction']['num_cpus'])
        self.set_debug(config['look_up_reconstruction']['debug'])
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
        self.lookup_table = lookup_table

    def set_structure_grammar(self, structure_grammar: dict):
        """
        The structure grammar is the configuration to read images, look up tables, 
        and create the depth maps/point clouds accordingly.
        Example below:
            structure_grammar = {
                "name": "gray",
                "images": ["img_02.tiff", "img_04.tiff", "img_06.tiff"],
                "interpolant": {
                    "active": true,
                    "knots": [150, 160, 140],
                    "samples": 1000
                },
                "utils": {
                    "white": "white.tiff",
                    "colors: "white.tiff,
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

    def set_mask_threshold(self, thr: float):
        """

        Parameters
        ----------
        thr : float
            
        """
        self.thr = min(1., max(0., float(thr)))

    def set_parallelize_pixels(self, parallelize: bool):
        """
        Parameters
        ----------
        parallelize : bool
            Whether to parallelize the pixel processing.
        """
        self.parallelize_pixels = parallelize

    def set_num_cpus(self, num_cpus: int):
        """
        Parameters
        ----------
        num_cpus : int
            Number of CPUs to use for parallel processing.
        """
        self.num_cpus = int(max(1, num_cpus))

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

    def set_outputs(self, outputs: dict):
        """
        Parameters
        ----------
        outputs : dict
            Dictionary containing which outputs to save.
            outputs: {
                "depth_map": True,
                "point_cloud": True,
                "loss_1": True,
                "loss_2": True,
                "loss_12": True,
                "depth_12": True
            }
            Each can be set to True or False

        Notes: 
        - depth_map is a HxW numpy array containing depth per pixel of the result.
        - point_cloud is a 3D numpy array of the result.
        - loss_1 is a HxW numpy array containing the loss from the pixel and the first nearest neighbor.
        - loss_2 is a HxW numpy array containing the loss from the pixel and the second nearest neighbor.
        - loss_12 is a HxW numpy array containing the loss
            from the first nearest neighbor and the second nearest neighbor.
        - depth_12 is a HxW numpy array containing the depth distance
            from the first nearest neighbor and the second nearest neighbor.
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
            if self.verbose:
                print('-' * 15)
                print("Extracting mask from white image")
            self.mask = ImageUtils.extract_mask(self.white_image, self.thr)
        
    def decode_depth(self,
                     pixel,
                     lookup) -> tuple[float, float, float, float, float] | float:
        """
        TODO: replace slow np.argmin with some form of gradient descent
        Consider scipy.optimize.minimize with Newton-CG, since we can get the first and
        second order derivatives of the splines with scipy.interpolate.BSpline.derivative
        """
        color = lookup[::-1, :-1]
        depth = lookup[::-1, -1]
        if self.structure_grammar['interpolant']['active']:
            knots = self.structure_grammar['interpolant']['knots']
            samples = self.structure_grammar['interpolant']['samples']
            color = spline_interpolant(depth, color, knots, samples)

        loss = calculate_loss(color, pixel, ord=self.structure_grammar['loss']['order'])

        if self.debug:
            k_indices = k_smallest_indices(loss, 2)
            return depth[k_indices[0]], loss[k_indices[0]], loss[k_indices[1]], np.linalg.norm(color[k_indices[0],:]-color[k_indices[1],:]), depth[k_indices[0]] - depth[k_indices[1]]
        
        k_indices = k_smallest_indices(loss, 1)
        return depth[k_indices[0]]

    def process_pixel(self, mask, pixel, lookup):
        if not mask:
            return None
        return self.decode_depth(pixel, lookup)

    def reconstruct(self):
        """
        """
        if self.verbose:
            print('-' * 15)
            print("Beginning reconstruction")
        table_name = self.structure_grammar['name']

        if self.normalized is None:
            self.normalized = np.load(os.path.join(self.reconstruction_directory, f"{table_name}.npz"))['pattern']

        # allocate the memory for depth_map
        shape = self.normalized.shape[:2]
        self.depth_map = np.full(shape=shape, fill_value=-1., dtype=np.float64).flatten()
        if self.debug:
            self.loss_map = np.zeros(shape=shape, dtype=np.float64).flatten()
            self.loss_2 = np.zeros(shape=shape, dtype=np.float64).flatten()
            self.loss_12 = np.zeros(shape=shape, dtype=np.float64).flatten()
            self.depth_12 = np.zeros(shape=shape, dtype=np.float64).flatten()
        
        if self.parallelize_pixels:
            if self.verbose:
                print(f"Parallelizing with {self.num_cpus} CPU cores")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                futures = [executor.submit(self.process_pixel, mask, pixel, lookup)
                           for mask, pixel, lookup in zip(self.mask.flatten(),
                                                          self.normalized.reshape(-1, *self.normalized.shape[2:]),
                                                          self.lookup_table.reshape(-1, *self.lookup_table.shape[2:]))]
                concurrent.futures.wait(futures)  # Wait for all tasks to complete
            for idx, results in enumerate(futures):
                result = results.result()
                if result is None:
                    continue

                if self.debug:
                    self.depth_map[idx] = result[0]
                    self.loss_map[idx] = result[1]
                    self.loss_2[idx] = result[2]
                    self.loss_12[idx] = result[3]
                    self.depth_12[idx] = result[4]
                else:
                    self.depth_map[idx] = result
        else:
            # original sequential processing
            for idx, (mask, pixel, lookup) in enumerate(zip(self.mask.flatten(),
                                                            self.normalized.reshape(-1, *self.normalized.shape[2:]),
                                                            self.lookup_table.reshape(-1, *self.lookup_table.shape[2:]))):
                results = self.process_pixel(mask, pixel, lookup)
                if results is None:
                    continue

                if self.debug:
                    self.depth_map[idx] = results[0]
                    self.loss_map[idx] = results[1]
                    self.loss_2[idx] = results[2]
                    self.loss_12[idx] = results[3]
                    self.depth_12[idx] = results[4]
                else:
                    self.depth_map[idx] = results

        if self.verbose:
            print("Reshaping resulting arrays from 1D back to 2D")
        self.depth_map = self.depth_map.reshape(shape)
        if self.debug:
            self.loss_map = self.loss_map.reshape(shape)
            self.loss_2 = self.loss_2.reshape(shape)
            self.loss_12 = self.loss_12.reshape(shape)
            self.depth_12 = self.depth_12.reshape(shape)

    def save_outputs(self):
        table_name = self.structure_grammar['name']
        utils: dict = self.structure_grammar['utils']
        roi: tuple = None if 'roi' not in utils else utils['roi']

        if ['depth_map'] in self.outputs and self.outputs['depth_map']:
            np.save(os.path.join(self.reconstruction_directory,f"{table_name}_depth_map.npy"), self.depth_map)
            if self.verbose:
                print('-' * 15)
                print("Saved depth map")
        
        if ['loss_map'] in self.outputs and self.outputs['loss_map']:
            np.save(os.path.join(self.reconstruction_directory,f"{table_name}_loss_map.npy"), self.loss_map)
            if self.verbose:
                print('-' * 15)
                print("Saved depth map")

        if ['point_cloud'] in self.outputs and self.outputs['point_cloud']:
            if self.verbose:
                print('-' * 15)
                print("Constructing point cloud")

            mask = (self.depth_map > 0).flatten()

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
            if 'colors' not in utils or utils['colors'] is None:
                print("No color image set, therefore no color extraction for point cloud")
            else:
                color_image = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(self.reconstruction_directory, utils['colors'])), roi=utils['roi'])
                minimum = np.min(color_image)
                maximum = np.max(color_image)
                self.colors: np.ndarray = ((color_image - minimum) / (maximum - minimum)).reshape((-1,3))

            ThreeDUtils.save_ply(os.path.join(self.reconstruction_directory,f"{table_name}_point_cloud.ply"),
                                 self.point_cloud[mask],
                                 self.normals[mask],
                                 self.colors[mask])
            if self.verbose:
                print("Saved point cloud")

        if ['mask'] in self.outputs and self.outputs['mask']:
            np.save(os.path.join(self.reconstruction_directory,"mask.npy"), self.mask)
            if self.verbose:
                print('-' * 15)
                print("Saved mask")

        if self.debug:
            np.savez_compressed(os.path.join(self.reconstruction_directory,f"{table_name}_debug.npz"),
                            loss_1=self.loss_map,
                            loss_2=self.loss_2,
                            loss_12=self.loss_12,
                            depth_12=self.depth_12)
            if self.verbose:
                print('-' * 15)
                print("Saved extra outputs because debug is set to True")

    def run(self, normalize: bool=False):
        if normalize:
            self.normalized = self.process_position(self.reconstruction_directory, self.structure_grammar)

        self.reconstruct()
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

        pattern_images =  ImageUtils.crop(np.concatenate([np.atleast_3d(ImageUtils.load_ldr(os.path.join(folder, image))) for image in images], axis=2), roi=roi)
        white_image = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(folder, utils['white'])), roi=roi)
        
        if 'black' in utils:
            black_image = None if utils['black'] is None else ImageUtils.crop(ImageUtils.load_ldr(os.path.join(folder, utils['black'])), roi=roi)
        
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
    parser.add_argument('-r', '--reconstruction_config', type=str, default=None,
                    help='Path to config JSON with parameters for LookUp Reconstruction')

    args = parser.parse_args()

    if args.calibration_config is not None:
        luc = LookUpCalibration(args.calibration_config)
        luc.run(args.depth, args.normalize)

    if args.reconstruction_config is not None:
        lur = LookUpReconstruction(args.reconstruction_config)
        lur.run(args.normalize)