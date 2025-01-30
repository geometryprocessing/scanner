import cv2
import numpy as np
from scipy import interpolate

from src.utils.three_d_utils import ThreeDUtils
from src.utils.file_io import save_json, load_json, get_all_paths
from src.utils.image_utils import ImageUtils
from src.scanner.camera import Camera
from src.scanner.calibration import Calibration, CheckerBoard, Charuco

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
    def __init__(self):
        self.camera    = Camera()

        # image paths
        self.white_images = []
        self.color_images = []
    
        # calibration utils
        self.plane_pattern = None
        self.num_frames = 0
        self.lookup_table = None

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
            These images will be used for normalizing the color images and for identifying
            the depth for each camera ray.
        """
        self.white_images = get_all_paths(image_paths)
    def set_color_images(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be normalized with the white images and then used to save the
            LookUp table.
        """
        self.color_images = get_all_paths(image_paths)
        self.num_frames = len(self.color_images)
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
    def set_lookup_table(self, lookup_table_path: str):
        """
        Path to .npy file where LookUp table will be stored.
        """
        self.lookup_table = lookup_table_path

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

        # np.savez_compressed(path + "combined/depth/" + name + ".npz", depth=to_16bit(depth, is_depth=True))
        return depth
    
    def stack_results(self):
        pass

    def calibrate(self):
        # allocate memory for massive, single float precision numpy array
        width, height = self.camera.get_image_shape()
        lookup_table = np.full(shape=(height, width, self.num_frames, 4), fill_value=np.nan, dtype=np.float32)

        for idx, (color_image, white_image) in enumerate(zip(self.color_images, self.white_images)):
            normalized = ImageUtils.normalize_color(color_image, white_image)
            depth = self.find_depth(white_image)
            lookup_table[:,:,idx,:] = np.concatenate([normalized, depth[:, :, np.newaxis]], axis=2)

        return lookup_table

    def save_lookup_table(self,
                          lookup_table: np.ndarray,
                          filename):
                        #   roi: np.ndarray):
        """
        Save the lookup table as a npy file.
        """
        np.save(filename, lookup_table)
        # save_json({"roi": roi}, f'{data_path}/lookup_tables/lookup_table_{pattern_name}.json')
    
    def run(self, config: dict | str):
        if type(config) is str:
            config = load_json(config)

        self.set_camera(config['camera_calibration'])
        self.set_white_images(config['white_images'])
        self.set_color_images(config['color_images'])
        lookup_table = self.calibrate()
        self.save_lookup_table(lookup_table, config['lookup_table'])

    
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

            for idx, (pixel, lookup) in enumerate(
                zip(normalized.reshape(-1, *normalized.shape[2:]),
                    lookup_table.reshape(-1, *lookup_table.shape[2:]))):
                point3D = self.extract_depth(pixel, lookup)
                point_cloud[idx, :] = point3D

            colors = self.extract_colors(white_image)
            normals = self.extract_normals(point_cloud)

            self.save_point_cloud_as_ply(frame_number, point_cloud, normals, colors)

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

        self.set_camera(config['camera_calibration'])
        self.set_white_images(config['white_images'])
        self.set_color_images(config['color_images'])
        self.set_lookup_table(config['lookup_table'])
        self.set_knots(config['knots'])
        self.set_samples(config['samples'])
        self.reconstruct()