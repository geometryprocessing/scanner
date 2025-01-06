import cv2
import numpy as np
import structuredlight as sl
import os

from utils.three_d_utils import ThreeDUtils
from utils.file_io import save_json, load_json
from utils.image_utils import ImageUtils
from scanner.camera import Camera
from scanner.calibration import Calibration, CheckerBoard, Charuco

class LookUp:
    def __init__(self):
        self.camera    = Camera()

        self.patterns    = []
        self.concatenate = []

        self.num_frames  = 0

        # image paths
        self.white_images = []
        self.color_images = []

        # directory to store numpy array of lookup tables 
        self.lookup_tables_directory = []

        # calibration utils
        self.plane_pattern = None

        # reconstruction utils
        # TODO: PROBABLY WILL BE DISCARDED
        # TODO: better ROI, since it's fixed for all frames?
        self.thr         = None
        self.mask        = None
        # reconstruction
        # TODO: PROBABLY WILL BE DISCARDED, too much memory storing these
        self.point_cloud = None
        self.depth_map   = None
        self.colors      = None
        self.normals     = None

    # setters
    def set_white_images(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for decoding horizontal structured light patterns.
        """
        if os.path.isdir(image_paths):
            image_paths = [os.path.join(image_paths, f)
                            for f in os.listdir(image_paths)
                            if os.path.isfile(os.path.join(image_paths, f))]
        self.white_images = image_paths
    def set_color_images(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for decoding vertical structure light patterns.
        """
        if os.path.isdir(image_paths):
            image_paths = [os.path.join(image_paths, f)
                            for f in os.listdir(image_paths)
                            if os.path.isfile(os.path.join(image_paths, f))]
        self.colort_images = image_paths
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
    def set_threshold(self, thr: float):
        """
        Set the threshold value for considering a pixel ON or OFF
        based on its intensity.

        Parameters
        ----------
        thr : float
            threshold value for considering a pixel ON or OFF

        Notes
        -----
        If black and white pattern images are set, or if negative/inverse
        patterns are passed, this threshold value will be ignored.
        """
        assert thr > 0, "Incorrect value for threshold, has to be nonnegative"
        self.thr = float(thr)
    def set_camera(self, camera: str | dict | Camera):
        # if string, load the json file and get the parameters
        # check if dict, get the parameters
        if type(camera) is str or dict:
            self.camera.load_calibration(camera)
        else:
            self.camera = camera
    def set_mask(self, mask: np.ndarray):
        """
        """
        self.mask = mask

    # getters
    def get_mask(self):
        return self.mask
    def get_point_cloud(self):
        return self.point_cloud
    def get_depth_map(self):
        return self.depth_map
    def get_colors(self):
        return self.colors
    def get_normals(self):
        return self.normals

    def generate_mask(self):
        pass
        if self.mask is not None:
            print("External mask provided")
            return
        ImageUtils.generate_mask()

    # calibration functions
    def reconstruct_plane(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        TODO: discard this function
        """
        assert self.camera.K is not None, "No camera defined"
        assert self.plane_pattern is not None, "No Plane Pattern defined"

        # detect plane/board markers with camera
        img_points, obj_points, _ = \
            self.plane_pattern.detect_markers(image_path)
        
        # although plane reconstruction requires 3 points,
        # OpenCV extrinsic calibration requires 6 points
        if len(img_points) < max(6, self.min_points):
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
    
    def find_depth(self, image_path: str) -> np.ndarray:
        """
        TODO: finding depth will happen PER FRAME of calibration, only after it will be glued together
        """
        plane_q, plane_n = self.reconstruct_plane(image_path)
        cam_resolution = self.camera.get_image_shape()
        campixels_x, campixels_y = np.meshgrid(np.arange(cam_resolution[1]),
                                               np.arange(cam_resolution[0]))
        
        campixels = np.stack([campixels_x, campixels_y], axis=-1).reshape((-1,2))

        origin, rays = ThreeDUtils.camera_to_ray_world(campixels, self.camera.K, self.camera.dist_coeffs)
        points = ThreeDUtils.intersect_line_with_plane(origin, rays, plane_q, plane_n)

        depth = np.linalg.norm(points - origin, norm=2)
        depth = depth.reshape((cam_resolution[1], cam_resolution[0]))

        # bs = np.matmul(R.T, (xs - T).T).T
        # idx = (roi[0] < bs[:, 0]) & (bs[:, 0] < roi[2]) & \
        #     (roi[1] < bs[:, 1]) & (bs[:, 1] < roi[3])
        # bs = None

        # depth[~idx] = 0
        # depth = d.reshape((h, w))

        # np.savez_compressed(path + "combined/depth/" + name + ".npz", depth=to_16bit(depth, is_depth=True))
        return depth

    def calibrate(self):
        pass

    def save_lookup_table(self,
                          lookup_table: np.ndarray,
                          pattern_name: str):
                        #   roi: np.ndarray):
        """
        Save the lookup table to an npy file.
        """
        np.save(f'{self.lookup_tables_directory}/{pattern_name}.npy', lookup_table)
        # save_json({"roi": roi}, f'{data_path}/lookup_tables/lookup_table_{pattern_name}.json')

    def concatenate_lookup_tables(self):
        """
        Create a lookup table (dictionary) of 3D pixel coordinates, RGB values for many patterns, and depth.
        """
        lookup_tables = [np.load(f'{self.lookup_tables_directory}/{pattern_name}.npy') for pattern_name in self.concatenate]
        lookup_tables = [lookup_table[:, :, :, :-1] for lookup_table in lookup_tables]
        # append the depth from the last lookup table
        lookup_tables.append(lookup_tables[-1][:, :, :, -1])

        print("Concatenating...")
        # these are all 4D arrays, so it makes sense we are concatenating on axis=3
        # (hard to visualize, since we only see three dimensions)
        concatenated_lookup_table = np.concatenate(lookup_tables, axis=3)
        self.save_lookup_table(concatenated_lookup_table, 'concatenated')

    # reconstruction functions
    def reconstruct(self):
        """
        TODO: reconstruction will happen PER FRAME PER PATTERN
        """
        assert self.camera.K is not None, "No camera defined"
    
    def extract_colors(self):
        """
        TODO: extract colors will happen PER FRAME, using the
        white pattern displayed onto the scene.

        For that, we can
        minimum = np.min(white)
        maximum = np.max(white)
        colors = (white - minimum) / (maximum - minimum)

        since we don't have black image
        """
        # assert self.white_image is not None \
        #       and self.black_image is not None, "Need to set both black and white images"
        # img_white = ImageUtils.load_ldr(self.white_image)
        # img_black = ImageUtils.load_ldr(self.black_image)

        # img_clean = img_white - img_black
        # self.colors = img_clean[self.mask]
    
    def extract_normals(self):
        """
        TODO: normals extraction will happen PER FRAME
        """
        assert self.depth_map is not None or self.point_cloud is not None, "No reconstruction yet"
        if self.depth_map:
            self.normals = ThreeDUtils.normals_from_depth_map(self.depth_map)
        elif self.point_cloud:
            self.normals = ThreeDUtils.normals_from_point_cloud(self.point_cloud)

    def save_point_cloud_as_ply(self, filename: str):
        """
        TODO: saving point clouds will happen PER FRAME PER PATTERN
        """
        assert self.point_cloud is not None, "No reconstruction yet"
        ThreeDUtils.save_ply(filename, self.point_cloud, self.normals, self.colors)
        

    def run(self, config: str | dict):
        if type(config) is str:
            config = load_json(config)

        self.set_threshold(config['threshold'])
        self.reconstruct()
        self.extract_colors()
        self.extract_normals()
        self.save_point_cloud_as_ply(config['ply_filename'])