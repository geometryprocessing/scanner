import cv2
from enum import Enum
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from scanner.calibration_utils import Calibration, Charuco, CheckerBoard

class CameraModel(Enum):
        """
        TODO: implement different camera models -- each camera model will have a set
        of flags that will be passed down to OpenCV functions.
        
        These camera model names are copied from COLMAP.
        """
        SimplePinholeCameraModel       =  1 # fx=fy=f,cx,cy, dist=None
        PinholeCameraModel             =  2 # fx,fy,cx,cy, dist=None
        SimpleRadialCameraModel        =  3 # fx=fy=f,cx,cy, dist=k
        SimpleRadialFisheyeCameraModel =  4 # fx=fy=f,cx,cy, dist=k
        RadialCameraModel              =  5 # fx=fy=f,cx,cy, dist=k1,k2
        RadialFisheyeCameraModel       =  6 # fx=fy=f,cx,cy, dist=k1,k2
        OpenCVCameraModel              =  7 # fx,fy,cx,cy, dist=k1,k2,p1,p2
        OpenCVFisheyeCameraModel       =  8 # fx,fy,cx,cy, dist=k1,k2,k3,k4
        FullOpenCVCameraModel          =  9 # fx,fy,cx,cy, dist=k1,k2,p1,p2,k3,k4,k5,k6
        FOVCameraModel                 = 10 # fx,fy,cx,cy, dist=omega (not able to do with OpenCV) -- Project Tango, not relevant
        ThinPrismFisheyeCameraModel    = 11 # fx,fy,cx,cy, dist=k1,k2,p1,p2,k3,k4,sx1,sy1
        RadTanThinPrismFisheyeModel    = 12 # fx,fy,cx,cy, dist=k1, k2,k3,k4,k5,k6,p0,p1,s0,s1,s2,s3

class Camera:
    def __init__(self):
        self.model = "to-be-implemented" # this will be the CameraModel ENUM

        # image resolution
        self.image_width = None
        self.image_height = None
        # intrinsic
        self.intrinsic_images = []     # image paths
        self.intrinsic_object_points = []
        self.intrinsic_image_points = []
        self.K = None
        self.scaling_factor = 0
        self.newK = None
        self.roi = None
        self.dist_coeffs = None
        self.rvecs = np.zeros(shape=(3,1))
        self.tvecs = np.zeros(shape=(3,1))
        # extrinsic
        self.extrinsic_image = None     # image paths
        self.extrinsic_object_points =[]
        self.extrinsic_image_points = []
        self.R = np.identity(3)        # camera is initialized at origin
        self.T = np.zeros(shape=(3,1)) # camera is initialized at origin
        # calibration utils
        self.errors = None
        self.charuco = None
        self.checkerboard = None
        self.error_thr = None
        self.min_points = 0

    # setters
    def set_intrinsic_matrix(self, K):
        """
        Parameters
        ----------
        K : array_like
            3x3 Intrinsic Matrix of Camera.
        """
        self.K = np.asarray(K)
    def set_distortion(self, dist_coeffs):
        self.dist_coeffs = np.asarray(dist_coeffs)
    def set_scaling_factor(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Scaling factor for cv2.getOptimalNewCameraMatrix
        """
        self.scaling_factor = alpha
    def set_translation(self, T):
        """
        Parameters
        ----------
        T : array_like
            3x1 translation vector of Camera.
        """
        self.T = np.asarray(T)
    def set_rotation(self, r):
        """
        Parameters
        ----------
        R : array_like
            Either 3x1 rotation vector (perform Rodrigues) or
            3x3 rotation matrix of Camera.
            It saves it as 3x3 rotation matrix.
        """
        r = np.asarray(r)
        if r.shape==(3,1) or r.shape==(1,3):
            r, _ = cv2.Rodrigues(r)
        self.R = r
    def set_charuco(self, charuco: dict | Charuco):
        """
        Parameters
        ----------
        charuco : dictionary | Charuco object
            ChArUco board used for all camera calibration procedures.
        """
        if type(charuco) is dict:
            charuco = Charuco(board_config=charuco)
        self.charuco = charuco
    def set_checkerboard(self, checker: dict | CheckerBoard):
        if type(checker) is dict:
            checker = CheckerBoard(board_config=checker)
        self.checkerboard = checker
    def set_intrinsic_image_paths(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for intrinsic calibration of Camera.
        """
        if os.path.isdir(image_paths):
            image_paths = [os.path.join(image_paths, f)
                            for f in os.listdir(image_paths)
                            if os.path.isfile(os.path.join(image_paths, f))]
        self.intrinsic_images = image_paths
    def set_extrinsic_image_path(self, image_path: str):
        """
        Parameters
        ----------
        image_path : string
            String containing path of image to be used for extrinsic calibration of Camera.
        """
        self.extrinsic_image = image_path
    def set_error_threshold(self, error_thr: float):
        self.error_thr = error_thr
    def set_min_points(self, min_points: int):
        self.min_points = min_points

    def set_height(self, height: int):
        self.image_height = height
    def set_width(self, width: int):
        self.image_width = width
    def set_image_shape(self, shape: tuple):
        """
        Parameters
        ----------
        shape : tuple
            Image resolution in (height: int, width: int).
        """
        self.set_height(shape[0])
        self.set_width (shape[1])

    def add_intrinsic_image_points(self, image_points: np.ndarray):
        """
        Appends more image points to the
        list intrinsic_image_points of Camera.
        Parameters
        ----------
        image_points : array_like
            Array of 2D image points of shape (N,2).
        """
        self.intrinsic_image_points.append(image_points)
    def add_intrinsic_object_points(self, object_points: np.ndarray):
        """
        Appends more object points to the
        list intrinsic_object_points of Camera.
        Parameters
        ----------
        object_points : array_like
            Array of 3D world points of shape (N,3).
        """
        self.intrinsic_object_points.append(object_points)
    def set_extrinsic_image_points(self, image_points: np.ndarray):
        """
        TODO: discard.

        Appends more image points to the
        list extrinsic_image_points of Camera.
        Parameters
        ----------
        image_points : array_like
            Array of 2D image points of shape (N,2).
        """
        self.extrinsic_image_points = image_points
    def set_extrinsic_object_points(self, object_points: np.ndarray):
        """
        TODO: discarn
        Appends more object points to the
        list extrinsic_object_points of Camera.
        Parameters
        ----------
        object_points : array_like
            Array of 3D world points of shape (N,3).
        """
        self.extrinsic_object_points = object_points

    # getters
    def get_intrinsic_matrix(self):
        return self.K
    def get_distortion(self):
        return self.dist_coeffs
    def get_scaling_factor(self):
        return self.scaling_factor
    def get_new_intrinsic_matrix(self):
        return self.newK
    def get_rvecs(self):
        return self.rvecs
    def get_tvecs(self):
        return self.tvecs
    def get_translation(self):
        return self.T
    def get_rotation(self):
        return self.R
    def get_charuco(self):
        return self.charuco
    def get_intrinsic_image_paths(self):
        return self.intrinsic_images
    def get_extrinsic_image_paths(self):
        return self.extrinsic_images
    def get_error_threshold(self):
        return self.error_thr
    def get_min_points(self):
        return self.min_points
    def get_image_shape(self):
        return (self.image_height, self.image_width)
    
    # functions
    def detect_markers_for_intrinsic(self):
        """
        Add markers (2D image coordinates and 3D world coordinates) for intrinsic calibration.
        """
        assert self.charuco is not None, "No ChArUco Board defined"
        assert len(self.intrinsic_images) > 0, "No intrinsic images defined"
        for image_path in self.intrinsic_images:
            img_points, obj_points, _ = self.charuco.detect_markers(image_path)

            self.add_intrinsic_image_points (img_points)
            self.add_intrinsic_object_points(obj_points)
    
    def detect_markers_for_extrinsic(self):
        """
        Add markers (2D image coordinates and 3D world coordinates) for extrinsic calibration.
        """
        assert self.charuco is not None, "No ChArUco Board defined"
        assert self.extrinsic_image is not None, "No extrinsic image defined"

        img_points, obj_points, _ = self.charuco.detect_markers(self.extrinsic_image)

        self.set_extrinsic_image_points (img_points)
        self.set_extrinsic_object_points(obj_points)

    def calibrate_intrinsics(self):
        """
        Perform intrinsic camera calibration and calculate reprojection errors.
        """
        assert len(self.intrinsic_image_points) > 0, "There are no 2D image points"
        assert len(self.intrinsic_object_points) > 0, "There are not 3D object points"
        assert self.image_height is not None or self.image_width is not None, \
            "Image resolution has not been defined"

        result = \
            Calibration.calibrate_intrinsic(self.intrinsic_object_points,
                                            self.intrinsic_image_points,
                                            self.get_image_shape())
                
        self.K = result['K']
        self.dist_coeffs = result['dist_coeffs']
        self.rvecs = result['rvecs']
        self.tvecs = result['tvecs']
        self.errors = result['errors']

    def calibrate_new_optimal_intrinsic_matrix(self):
        result = Calibration.new_intrisic_matrix(self.get_image_shape(),
                                        self.K,
                                        self.dist_coeffs,
                                        self.scaling_factor)
        self.newK = result['newK']
        self.roi = result['roi']

    def refine(self):
        """
        TODO: implement...
        
        Perform intrinsic camera calibration with refinement.
        Refinement can include minimum number of detected markers and maximum error.
        """
        assert self.K is not None, "Camera has not been calibrated yet"

        # Select points with errors below the threshold
        selected_idxs = np.nonzero(self.errors < self.error_thr)[0]
        print(f"Selected {len(selected_idxs)} points for refinement")

        # Filter the object and image points based on selected indexes
        obj_selected = [objs[i] for i in selected_idxs]
        img_selected = [imgs[i] for i in selected_idxs]

        # Filter out images with fewer than the min_points
        valid_idxs = [i for i, img in enumerate(imgs) if img.shape[0] > self.min_points]
        names, imgs, objs = [names[i] for i in valid_idxs], [imgs[i] for i in valid_idxs], [objs[i] for i in valid_idxs]

        # Perform refined calibration using the filtered points
        result = \
            Calibration.calibrate_intrinsic(obj_selected,
                                            img_selected,
                                            self.get_image_shape(),
                                            self.K,
                                            self.dist_coeffs)
                
        self.K = result['K']
        self.dist_coeffs = result['dist_coeffs']
        self.rvecs = result['rvecs']
        self.tvecs = result['tvecs']
        self.errors = result['errors']

    def projection_errors(self):
        """
        TODO: FIX. This needs to do one image at a time.

        Calculate reprojection error of detected markers
        """
        assert len(self.intrinsic_image_points) > 0, "There are no 2D image points"
        assert len(self.intrinsic_object_points) > 0, "There are not 3D object points"
        assert self.K is not None, "Camera has not been calibrated yet"
        errors = Calibration.reprojection_error(self.intrinsic_object_points,
                                                self.intrinsic_image_points,
                                                self.rvecs,
                                                self.tvecs,
                                                self.K,
                                                self.dist_coeffs)
        self.errors = errors

    def calibrate_extrinsics(self):
        """
        Perform extrinsic camera calibration.
        """
        assert len(self.extrinsic_image_points) > 0, "There are no 2D image points"
        assert len(self.extrinsic_object_points) > 0, "There are not 3D object points"
        assert self.K is not None, "Camera has not been calibrated yet"
        
        result = Calibration.calibrate_extrinsic(self.extrinsic_object_points,
                                                 self.extrinsic_image_points,
                                                 self.K,
                                                 self.dist_coeffs)
        
        self.R = result['R']
        self.T = result['T']

    def plot_errors(self):
        """
        TODO: implement with matplotlib
        """
        pass
        
    def serialize(self):
        """
        Serialize calibration to write it into JSON dictionary format.
        """
        assert self.K is not None, "Camera has not been calibrated yet"
        assert self.errors is not None, "Camera has not been calibrated yet"
        return {
            "K": self.K.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "newK": self.newK.tolist(),
            "roi": self.roi,
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "error": self.errors.tolist(),
            "error_threshold": self.error_thr
        }

    def save_calibration(self, filename):
        """
        Save calibration into a JSON file.
        """
        with open(filename, "w") as f:
            json.dump(self.serialize(), f, indent=4)

    def load_calibration(self, filename):
        with open(filename, "r") as f:
            config = json.load(f)
        
        self.set_height(config['height'])
        self.set_width(config['width'])

        self.set_intrinsic_matrix(config['K'])
        self.set_distortion(config['dist_coeffs'])
        self.set_rotation(config['R'])
        self.set_translation(config['T'])

    def run(self, config: str | dict):
        if type(config) is str:
            with open(config, 'r') as f:
                config = json.load(f)

        self.set_intrinsic_image_paths(config['intrinsic_folder_path'])
        self.set_extrinsic_image_path(config['extrinsic_image_path'])

        self.set_height(config['height'])
        self.set_width(config['width'])

        self.set_charuco(config['aruco_dict'])
        self.set_error_threshold(config['error_thr'])
        self.set_min_points(config['min_points'])

        self.detect_markers_for_intrinsic()
        self.calibrate_intrinsics()
        self.projection_errors()
        self.refine()
        self.detect_markers_for_extrinsic()
        self.calibrate_extrinsics()
        
        self.save_calibration(config['output_filename'])
