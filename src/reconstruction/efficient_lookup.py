import cv2
from datetime import datetime
import numpy as np
import os
from scipy import interpolate

from src.utils.three_d_utils import ThreeDUtils
from src.utils.file_io import save_json, load_json, get_all_paths, get_folder_from_file
from src.utils.image_utils import ImageUtils
from src.scanner.camera import Camera
from src.scanner.calibration import Calibration, CheckerBoard, Charuco
# for parallel processing
import concurrent.futures

def concatenate_lookup_tables(lookup_tables: list[str], filename: str):
    """
    Create a lookup table (dictionary) of 2D pixel coordinates, RGB values for many patterns, and depth.
    """
    lookup_tables = [np.load(lut) for lut in lookup_tables]
    result = [lut[:, :, :, :-1] for lut in lookup_tables]
    # append the depth from the last lookup table
    result.append(lookup_tables[-1][:, :, :, -1])

    print("Concatenating...")
    # these are all 4D arrays, so it makes sense we are concatenating on axis=3
    # (hard to visualize, since we only see three dimensions)
    result = np.concatenate(lookup_tables, axis=3)
    np.save(filename, result)

class LookUpCalibration:
    """
    TODO: this should actually take the path to a folder with MANY position folders.
    In each POSITION folder, it contains all of the images with different patterns
    projected on the same depth.
    """
    def __init__(self):
        self.camera = Camera()

        self.plane_pattern         = None
        self.root                  = None
        self.calibration_directory = []
        self.num_directories       = 0
        self.structure_grammar     = {}
        self.roi                   = None

        # flag for parallelizing position processing
        self.parallelize_positions = False

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
            path to directory containing multiple calibration folders,
            each with the same structured light capture of a calibration board
        """
        assert os.path.isdir(path), "This is not a directory. This function only works with a folder."
        self.root = path
        dirs = [f.path for f in os.scandir(path) if f.is_dir()]
        self.num_directories = len(dirs)
        self.calibration_directory = [get_all_paths(dir) for dir in dirs]

    def set_structure_grammar(self, structure_grammar: dict):
        """
        The structure grammar is the configuration to read images
        and save them to pattern/look up tables accordingly.
        Example below:
        structure_grammar = {
            "tables": {
                "gray": ["img_02.tiff", "img_04.tiff", "img_06.tiff"],
                "spiral": ["spiral.tiff"],
                "look_up_name": ["list", "of", "images"]
            },
            "utils": {
                "white": "white.tiff",
                "ambient": "ambient.tiff"

        The list of strings are the images which will be used
        to create a look up table with the key name.
        """
        self.structure_grammar = structure_grammar

    def set_roi(self, roi: list | set):
        """
        Parameters
        ----------
        roi : list or set
            xyxy region of interest to reduce size of look up tables
        """
        self.roi = roi

    # setter for parallelization flag
    def set_parallelize_positions(self, parallelize: bool):
        """
        Parameters
        ----------
        parallelize : bool
            Whether to parallelize the position processing.
        """
        self.parallelize_positions = parallelize

    # getters

    # functions
    def reconstruct_plane(self, white_image: str | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        """
        assert self.camera.K is not None, "No camera defined"
        assert self.plane_pattern is not None, "No Plane Pattern defined"
        
        # detect plane/board markers with camera
        img_points, obj_points, _ = \
            self.plane_pattern.detect_markers(white_image)
        
        # although plane reconstruction requires 3 points,
        # OpenCV extrinsic calibration requires 6 points
        if len(img_points) < 6:
            return None, None
        
        # find relative position of camera and board
        result = Calibration.calibrate_extrinsic(obj_points,
                                                 img_points,
                                                 self.camera.K,
                                                 self.camera.dist_coeffs)
        rvec = result['rvec']
        tvec = result['tvec']

        R, _ = cv2.Rodrigues(rvec)
        T = tvec.reshape((3,1))

        # move markers to world coordinate
        R_combined, T_combined = ThreeDUtils.combine_transformations(self.camera.R, self.camera.T, R, T)
        # NOTE: since obj_points is of shape (Nx3), the matrix multiplication with rotation
        # has to be written as (R @ obj_points.T).T
        # to simplify:
        # np.matmul(R_combined, obj_points.T).T = np.matmul(obj_points, R_combined.T)
        world_points = np.matmul(obj_points, R_combined.T) + T_combined.reshape((1,3))
        
        # fit plane
        return ThreeDUtils.fit_plane(world_points)

    def find_depth(self, white_image: str | np.ndarray) -> np.ndarray:
        """
        TODO: finding depth will happen PER FRAME of calibration, only after it will be glued together
        """
        plane_q, plane_n = self.reconstruct_plane(white_image)
        width, height = self.camera.get_image_shape()
        campixels_x, campixels_y = np.meshgrid(np.arange(width),
                                                    np.arange(height))
        campixels = np.stack([campixels_x, campixels_y], axis=-1).reshape((-1,2))

        origin, rays = ThreeDUtils.camera_to_ray_world(campixels, self.camera.R, self.camera.T, self.camera.K, self.camera.dist_coeffs)
        points = ThreeDUtils.intersect_line_with_plane(origin, rays, plane_q, plane_n)

        depth = np.linalg.norm(points - origin, axis=1, ord=2)
        
        # bs = np.matmul(R.T, (xs - T).T).T
        # idx = (roi[0] < bs[:, 0]) & (bs[:, 0] < roi[2]) & \
        #     (roi[1] < bs[:, 1]) & (bs[:, 1] < roi[3])
        # bs = None

        # depth[~idx] = 0

        depth = depth.reshape((height, width))
        np.savez_compressed(os.path.join(get_folder_from_file(white_image), "depth.npz"), depth=depth)

    def save_lookup_table(self,
                            lookup_table: np.ndarray,
                            filename):
                        #   roi: np.ndarray):
        """
        Save the lookup table as a npy file.
        """
        np.save(os.path.join(self.root, filename), lookup_table)

    def calibrate_positions(self):
        def _calibrate_single_pattern_single_position(pattern_name, image_names, white_image, ambient, dirname):
            """
            Util function to normalize all pattern images with white image (and with black image if available)
            for a single position of calibration.

            This function is meant to be innacessible to user.
            """
            normalized = np.concatenate([ImageUtils.normalize_color(os.path.join(dirname, image), white_image, ambient) for image in image_names], axis=2)
            np.savez_compressed(os.path.join(dirname, f"{pattern_name}.npz"), pattern=normalized)

        def _calibrate_all_patterns_single_positon(position):
            """
            Util function to normalize all pattern images with white image (and with black image if available)
            for a single position of calibration.
            
            This function is meant to be innacessible to user.
            """
            white_image = None
            ambient = None
            for util_name, image_name in self.structure_grammar['utils'].items():
                if util_name == 'white':
                    white_image = os.path.join(position, image_name)
                if util_name == 'black':
                    ambient = os.path.join(position, image_name)
            self.find_depth(white_image)

            for pattern_name, image_names in self.structure_grammar['tables'].items():
                _calibrate_single_pattern_single_position(pattern_name, image_names, white_image, ambient, position)

        # optional parallelization
        if self.parallelize_positions:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                futures = [executor.submit(_calibrate_all_patterns_single_positon, get_folder_from_file(folder[0])) for folder in self.calibration_directory]
                concurrent.futures.wait(futures)  # Wait for all tasks to complete
        else:
        # original sequential processing
            for folder in self.calibration_directory:
                _calibrate_all_patterns_single_positon(get_folder_from_file(folder[0]))

    def stack_results(self):
        """
        TODO: cannot parallelize multiple function calls to the same memory address lookup_table.
        Best thing to do is similar to Yurii's legacy code, where we, at the end, concatenate all single_positions
        into a MEGA lookup table.
        """
        width, height = self.camera.get_image_shape()
        x0, y0 = 0, 0

        if self.roi is not None:
            x0 = self.roi[0]
            y0 = self.roi[1]
            width = self.roi[2] - x0
            height = self.roi[3] - y0

        for pattern_name in self.structure_grammar['tables'].keys():
            # allocate memory for massive, single float precision numpy array
            lookup_table = np.full(shape=(height, width, self.num_directories, 4), fill_value=np.nan, dtype=np.float32)
            for pos, folder in enumerate(self.calibration_directory):
                position_dirname = get_folder_from_file(folder[0])
                depth = np.load(os.path.join(position_dirname, 'depth.npz'))['depth'][y0:y0+height, x0:x0+width]
                pattern = np.load(os.path.join(position_dirname, f'{pattern_name}.npz'))['pattern'][y0:y0+height, x0:x0+width]
                lookup_table[:,:,pos,:] = np.concatenate([pattern, depth[:, :, np.newaxis]], axis=2)
            self.save_lookup_table(lookup_table, pattern_name)

        save_json({'roi': self.roi,
                    'structure_grammar': self.structure_grammar,
                    'date': datetime.now().strftime('%m/%d/%Y, %H:%M:%S')},
                    os.path.join(self.root, 'calibration_info.json'))

    def run(self, config: dict | str):
        if type(config) is str:
            config = load_json(config)
        
        self.set_camera(config['look_up_calibration']['camera_calibration'])
        self.set_plane_pattern(config['look_up_calibration']['plane_pattern'])
        self.set_calibration_directory(config['look_up_calibration']['calibration_directory'])
        self.set_structure_grammar(config['look_up_calibration']['structure_grammar'])
        # check for parallelization
        if 'parallelize_positions' in config['look_up_calibration']:
            self.set_parallelize_positions(config['look_up_calibration']['parallelize_positions'])
        self.calibrate_positions()
        self.stack_results()


class LookUpReconstruction:
    def __init__(self):
        self.camera    = Camera()

        # image paths
        self.white_images = []
        self.color_images = []

        # reconstruction utils
        self.lookup_table = None
        self.knots        = []
        self.samples      = 1000
        # TODO: PROBABLY WILL BE DISCARDED
        # TODO: better ROI, since it's fixed for all frames?
        self.thr         = None
        # TODO: PROBABLY WILL BE DISCARDED, too much memory storing these
        self.mask        = None
        self.point_cloud = None
        self.depth_map   = None
        self.colors      = None
        self.normals     = None

        # flag for parallelizing pixel processing
        self.parallelize_pixels = False

    # setters
    def set_camera(self, camera: str | dict | Camera):
        # if string, load the json file and get the parameters
        # check if dict, get the parameters
        if isinstance(camera, str) or isinstance(camera, dict):
            self.camera.load_calibration(camera)
        else:
            self.camera = camera

    def set_white_images(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for normalizing the color images.
        """
        self.white_images = get_all_paths(image_paths)

    def set_color_images(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be normalized with the white images and then used for finding
            the depth stored in the tables saved from LookUpCalibration.
        """
        self.color_images = get_all_paths(image_paths)

    def set_lookup_table(self, lookup_table_path: str):
        """
        Path to .npy file where LookUp table is stored.
        """
        self.lookup_table = lookup_table_path

    def set_knots(self, knots: int | list[int]):
        """
        Knots will be used to fit a cubic B spline on the LookUp table data.
        Notes
        -----
        If only one integer is passed, it will be used for all color channels.
        If a list is passed, it needs to match the order of the color channels.
        """
        self.knots = knots

    def set_samples(self, samples: int):
        """
        Samples will be used to sample the cubic B spline fit along the depth
        of each color channel in the LookUp table.
        """
        self.samples = max(samples, 0)

    def set_mask(self, mask: np.ndarray):
        """
        """
        self.mask = mask

    # setter for parallelization flag
    def set_parallelize_pixels(self, parallelize: bool):
        """
        Parameters
        ----------
        parallelize : bool
            Whether to parallelize the pixel processing.
        """
        self.parallelize_pixels = parallelize

    # getters
    def get_mask(self):
        return self.mask

    def generate_mask(self, image, ):
        """
        TODO: generating mask will happen PER FRAME of calibration
        """
        pass
        if self.mask is not None:
            print("External mask provided")
            return
        ImageUtils.generate_mask()

    # reconstruction functions
    def extract_depth(self, pixel, lookup):
        """
        TODO: replace slow np.argmin with some form of gradient descent
        Consider scipy.optimize.minimize with Newton-CG, since we can get the first and
        second order derivatives of the splines with scipy.interpolate.BSpline.derivative
        """
        fits = []
        color = lookup[:, :-1]
        depth = lookup[:, -1]
        try:
            for ch in range(color.shape[1]):
                fit = interpolate.splrep(depth, color[:, ch], t=np.linspace(depth[2], depth[-3], self.knots[ch]), k=3)
                fits.append(interpolate.BSpline(*fit))
        except ValueError as e:
            print("Fit failed", e)
            return np.full(shape=(1,3), fill_value=np.nan)

        d_samples = np.linspace(depth[0], depth[-1], self.samples)
        fitted_color = np.array([fits[i](d_samples) for i in range(len(fits))])
        loss = np.linalg.norm(fitted_color - pixel, axis=0) # I think it's axis=1
        argmin = np.argmin(loss)
        return d_samples[argmin]

    def extract_colors(self, white_image: str | np.ndarray) -> np.ndarray:
        """
        Extract colors PER FRAME, using the white pattern image.

        Returns
        -------
        colors
            numpy array of the colors of the scene
        """
        if isinstance(white_image, str):
            white_image = ImageUtils.load_ldr(white_image)
        minimum = np.min(white_image)
        maximum = np.max(white_image)
        colors = (white_image - minimum) / (maximum - minimum)
        return colors

    def extract_normals(self, point_cloud):
        """
        Extract normals PER FRAME, using the reconstructed point cloud.

        Returns
        -------
        normal
            numpy array of the normals of reconstructed point cloud
        """
        normals = ThreeDUtils.normals_from_point_cloud(point_cloud)
        return normals

    def reconstruct(self):
        """
        TODO: reconstruction will happen PER FRAME
        TODO: inside here, it will run PER CAMERA RAY
        """
        lookup_table = np.load(self.lookup_table)

        for frame_number, (white_image, color_image) in enumerate(zip(self.white_images, self.color_images)):
            normalized = ImageUtils.normalize_color(white_image, color_image)

            # allocate the memory for point cloud
            point_cloud = np.zeros(shape=(normalized.shape, 3))

            if self.parallelize_pixels:
                pixels = normalized.reshape(-1, *normalized.shape[2:])
                lookups = lookup_table.reshape(-1, *lookup_table.shape[2:])
                num_pixels = pixels.shape[0]

                point_cloud_temp = np.zeros((num_pixels, 3))

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                    results = list(executor.map(self.extract_depth, pixels, lookups))
                for idx, point3D in enumerate(results):
                    point_cloud_temp[idx, :] = point3D

                point_cloud = point_cloud_temp.reshape(normalized.shape + (3,))

            else:
                # original sequential processing
                for idx, (pixel, lookup) in enumerate(
                    zip(normalized.reshape(-1, *normalized.shape[2:]),
                        lookup_table.reshape(-1, *lookup_table.shape[2:]))):
                    point3D = self.extract_depth(pixel, lookup)
                    point_cloud[idx, :] = point3D

            colors = self.extract_colors(white_image)
            normals = self.extract_normals(point_cloud)

            self.save_point_cloud_as_ply(str(frame_number), point_cloud, normals, colors)

    def save_point_cloud_as_ply(self,
                                filename: str,
                                point_cloud: np.ndarray,
                                normals: np.ndarray=None,
                                colors: np.ndarray=None):
        """
        Save point cloud PER FRAME.
        """
        ThreeDUtils.save_ply(filename, point_cloud, normals, colors)

    def run(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        self.set_camera(config['look_up_reconstruction']['camera_calibration'])
        self.set_white_images(config['look_up_reconstruction']['white_images'])
        self.set_color_images(config['look_up_reconstruction']['color_images'])
        self.set_lookup_table(config['look_up_reconstruction']['lookup_table'])
        self.set_knots(config['look_up_reconstruction']['knots'])
        self.set_samples(config['look_up_reconstruction']['samples'])

        # check for parallelization
        if 'parallelize_pixels' in config['look_up_reconstruction']:
            self.set_parallelize_pixels(config['look_up_reconstruction']['parallelize_pixels'])

        self.reconstruct()
