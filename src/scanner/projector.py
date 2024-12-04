import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from scanner.calibration_utils import Charuco, CheckerBoard, Calibration
from scanner.camera import Camera
from scanner.intersection import Plane, fit_plane, plane_line_intersection, undistort_camera_points, camera_to_ray_world


# camera detects all the points and has 2D image points and 3D object points
# use R and T to move those 3D object points into whatever coordinate system
# the camera has
# 
# then, pass those 3D object points with the 2D image points as one volumetric
# number of points to get both the intrinsic and extrinsic parameters of the
# projector

class Projector:
    def __init__(self):
        # resolution
        self.width = None
        self.height = None
        # accompanying camera
        self.camera = Camera()
        # intrinsic
        self.K = None
        self.dist_coeffs = None
        # extrinsic
        self.R = np.identity(3)        # projector is initialized at origin
        self.T = np.zeros(shape=(3,1)) # projector is initialized at origin
        # calibration utils
        self.images = []               # image paths
        self.object_points = []
        self.plane_points = []
        self.planes = []
        self.errors = []
        self.charuco = None
        self.checkerboard = None
        self.error_thr = None
        self.min_points = 0
        self.max_planes = 0

    # setters
    def set_intrinsic_matrix(self, K):
        """
        Parameters
        ----------
        K : array_like
            3x3 Intrinsic Matrix of Projector.
        """
        self.K = K
    def set_distortion(self, dist_coeffs):
        """
        Parameters
        ----------
        dist_coeffs : array_like
            Array of distortion coefficients 
            (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
            of 4, 5, 8, 12 or 14 elements.
        """
        self.dist_coeffs = dist_coeffs
    def set_translation(self, T):
        """
        Parameters
        ----------
        T : array_like
            3x1 translation vector of Projector.
        """
        self.T = T
    def set_rotation(self, r):
        """
        Parameters
        ----------
        r : array_like
            Either 3x1 rotation vector (perform Rodrigues) or
            3x3 rotation matrix of Projector.
            It saves it as 3x3 rotation matrix.
        """
        if r.shape==(3,1) or r.shape==(1,3):
            r, _ = cv2.Rodrigues(r)
        self.R = r

    def load_camera(self, camera: str | dict | Camera):
        if type(camera) is str or dict:
            self.camera.load_calibration(camera)
        else:
            self.camera = camera
    def set_camera_height(self, height: int):
        """
        Set image resolution height in pixels.
        Parameters
        ----------
        height : int
            Image resolution height in pixels.
        """
        self.camera.set_height(height)
    def set_camera_width(self, width: int):
        """
        Set image resolution width in pixels.
        Parameters
        ----------
        width : int
            Image resolution width in pixels.
        """
        self.camera.set_width(width)
    def set_camera_shape(self, shape: tuple):
        """
        Set image resolution in pixels.
        Both numbers have to be integers and nonnegative.
        Parameters
        ----------
        shape : tuple
            Image resolution in (height: int, width: int).
        """
        self.camera.set_height(shape[0])
        self.camera.set_width (shape[1])
    def set_camera_matrix(self, K):
        """
        Parameters
        ----------
        K : array_like
            3x3 Intrinsic Matrix of Camera.
        """
        self.camera.set_intrinsic_matrix(K)
    def set_camera_distortion(self, dist_coeffs):
        """
        Parameters
        ----------
        dist_coeffs : array_like
            Array of distortion coefficients 
            (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
            of 4, 5, 8, 12 or 14 elements.
        """
        self.camera.set_distortion(dist_coeffs)
    def set_camera_translation(self, T):
        """
        Parameters
        ----------
        T : array_like
            3x1 translation vector of Camera.
        """
        self.camera.set_translation(T)
    def set_camera_rotation(self, r):
        """
        Parameters
        ----------
        r : array_like
            Either 3x1 rotation vector (perform Rodrigues) or
            3x3 rotation matrix of Camera.
            It saves it as 3x3 rotation matrix.
        """
        self.camera.set_rotation(r)
    def set_charuco(self, charuco: dict | Charuco):
        """
        Parameters
        ----------
        charuco : dictionary | Charuco object
            ChArUco board used for projector calibration procedures.
        """
        if type(charuco) is dict:
            charuco = Charuco(board_config=charuco)
        self.charuco = charuco
    def set_checkerboard(self, checker: dict | CheckerBoard):
        """
        Parameters
        ----------
        checker : dictionary | CheckerBoard object
            Checker board used for projector calibration procedures.
        """
        if type(checker) is dict:
            checker = CheckerBoard(board_config=checker)
        self.checkerboard = checker
    def set_image_paths(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for calibration of Projector.
        """
        if os.path.isdir(image_paths):
            image_paths = [os.path.join(image_paths, f)
                            for f in os.listdir(image_paths)
                            if os.path.isfile(os.path.join(image_paths, f))]
        self.images = image_paths
    def set_error_threshold(self, error_thr: float):
        self.error_thr = error_thr
    def set_min_points(self, min_points: int):
        self.min_points = min_points

    def set_projector_height(self, height: int):
        self.height = height
    def set_projector_width(self, width: int):
        self.width = width
    def set_projector_shape(self, shape: tuple):
        self.set_projector_height(shape[0])
        self.set_projector_width (shape[1])

    # getters
    def get_intrinsic_matrix(self):
        return self.K
    def get_distortion(self):
        return self.dist_coeffs
    def get_translation(self):
        return self.T
    def get_rotation(self):
        return self.R
    def get_camera(self):
        return self.camera
    def get_charuco(self):
        return self.charuco
    def get_checkerboard(self):
        return self.checkerboard
    def get_image_paths(self):
        return self.images
    def get_error_threshold(self):
        return self.error_thr
    def get_min_points(self):
        return self.min_points
    def get_image_shape(self):
        return (self.camera.height, self.camera.width)
    def get_projector_shape(self):
        return (self.height, self.width)

    # functions
    def reconstruct_plane(self, image_path: str) -> Plane:
        # detect charuco markers with camera
        charuco_img_points, charuco_obj_points, _ = \
            self.charuco.detect_markers(image_path)
        
        # find relative position of camera and charuco board
        result = Calibration.calibrate_extrinsic(charuco_obj_points,
                                                 charuco_img_points,
                                                 self.camera.K,
                                                 self.camera.dist_coeffs)
    
        # undistort pixel coordinates to get normalized coordinates
        charuco_normalized = undistort_camera_points(charuco_img_points,
                                                     self.camera.K,
                                                     self.camera.dist_coeffs)
        
        # project normalized coordinates to world coordinate
        charuco_world_points = np.matmul(np.matmul(self.camera.R,result['R'].T),
                                         charuco_normalized) \
                                            + np.matmul(result['R'].T,self.camera.T) - result['T']

        # fit plane
        return fit_plane(charuco_world_points)

    def calibrate(self):
        """
        Perform projector intrinsic and extrinsic calibration.
        """
        assert self.camera is not None, "No camera defined"
        assert self.charuco is not None, "No ChArUco Board defined"
        assert len(self.images) > 0, "No intrinsic images defined"
        assert self.height is not None or self.width is not None, \
            "Projector resolution has not been defined"

        for image_path in self.images:
            plane = self.reconstruct_plane(image_path)
            self.planes.append(plane)

            # detect checkerboard -> pixel coordinates
            checker_img_points, checker_obj_points, _ = \
                self.checkerboard.detect_markers(image_path)

            # undistort pixel coordinates -> normalized coordinates
            # project normalized coordinates onto Plane - X_3D
            camera_rays = camera_to_ray_world(checker_img_points)

            for camera_ray in camera_rays:
                checker_world_points = plane_line_intersection(plane, camera_ray)

            self.object_points.extend(checkerboard_world_points)

        # run calibrate_intrinsic_extended with projector 2D coordinates and X_3D 
        result = Calibration.calibrate_extended(self.object_points,
                                                self.checkerboard,
                                                self.get_projector_shape())
        rvecs = result['rvecs']
        tvecs = result['tvecs']

        R, _ = cv2.Rodrigues(rvecs[0])
        T = np.matmul(R.T, -tvecs[0])
        
        self.K = result['K']
        self.dist_coeffs = result['dist_coeffs']
        self.R = R
        self.T = T

        self.errors = result['errors']

        # T, (R, _) = tvec.ravel(), cv2.Rodrigues(rvec)
        # origin = np.matmul(R.T, -T)
        # print("\nProjector Origin:", origin)
        # print("\nProjector Basis [ex, ey, ez]:\n", R)
        # extrinsic = origin, R

    def plot_errors(self):
        """
        TODO: implement with matplotlib
        """
        pass
        
    def serialize(self):
        """
        Serialize calibration to write it into JSON dictionary format.
        """
        assert self.K is not None, "Projector has not been calibrated yet"
        assert self.errors is not None, "Projector has not been calibrated yet"
        return {
            "K": self.K.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "height": self.height,
            "width": self.width,
            "error": self.errors.tolist(),
            "error_threshold": self.error_thr
        }

    def save_calibration(self, filename):
        with open(filename, "w") as f:
            json.dump(self.serialize(), f, indent=4)

    def load_calibration(self, calibration: str | dict):
        if type(calibration) is str:
            with open(calibration, "r") as f:
                calibration = json.load(f)
        
        self.set_projector_height(calibration['height'])
        self.set_projector_width(calibration['width'])

        self.set_intrinsic_matrix(calibration['K'])
        self.set_distortion(calibration['dist_coeffs'])
        self.set_rotation(calibration['R'])
        self.set_translation(calibration['T'])

    def run(self, config: str | dict):
        if type(config) is str:
            with open(config, 'r') as f:
                config = json.load(f)

        self.set_image_paths(config['image_folder_path'])
        self.set_projector_height(config['height'])
        self.set_projector_width(config['width'])
        self.load_camera(config['camera_calibration'])
        self.set_charuco(config['aruco_dict'])
        self.set_checkerboard(config['checkerboard_dict'])
        self.set_error_threshold(config['error_thr'])
        self.set_min_points(config['min_points'])

        self.calibrate()
        
        self.save_calibration(config['output_filename'])