import cv2
from enum import Enum
import json
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.file_io import save_json, load_json
from utils.plotter import Plotter
from scanner.calibration import Calibration, Charuco, CheckerBoard
from scanner.intersection import undistort_camera_points

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

def get_open_cv_calibration_flags(model):
    """
    TODO: implement function to return specific OpenCV
    calibration flags depending on camera/projector model.
    Example: projector with no tangent distortion should set
    flag cv2.CALIB_FIX_TANGENT_DIST
    """
    flags = 0
    match model:
        case CameraModel('SimplePinholeCameraModel'):
            flags = 0
        case CameraModel('PinholeCameraModel'):
            flags = 0
        case CameraModel('SimpleRadialCameraModel'):
            flags = 0
        case CameraModel('SimpleRadialFisheyeCameraModel'):
            flags = 0
        case CameraModel('RadialCameraModel'):
            flags = cv2.CALIB_ZERO_TANGENT_DIST         
        case CameraModel('RadialFisheyeCameraModel'):
            flags = 0
        case CameraModel('OpenCVCameraModel'):
            flags = 0
        case CameraModel('OpenCVFisheyeCameraModel'):
            flags = 0
        case CameraModel('FullOpenCVCameraModel'):
            flags = cv2.CALIB_RATIONAL_MODEL
        case CameraModel('FOVCameraModel'):
            flags = 0
        case CameraModel('ThinPrismFisheyeCameraModel'):
            flags = 0
        case CameraModel('RadTanThinPrismFisheyeModel'):
            flags = 0
        case _:
            print("Unrecognized camera model, settings flags to zero")

    return flags
    
class Camera:
    def __init__(self):
        self.model = "to-be-implemented" # this will be the CameraModel ENUM

        # image resolution
        self.width = None
        self.height = None
        # intrinsic
        self.intrinsic_images = []     # image paths
        self.discarded_images = set()
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
        self.T = np.zeros(shape=(1,3)) # camera is initialized at origin
        # calibration utils
        self.errors = []
        self.calibration_pattern = None
        self.error_thr = 0
        self.min_points = 4

    # setters
    def set_intrinsic_matrix(self, K):
        """
        Parameters
        ----------
        K : array_like
            3x3 Intrinsic Matrix of Camera.
        """
        self.K = np.array(K, dtype=np.float32).reshape((3,3))
    def set_distortion(self, dist_coeffs):
        """
        Parameters
        ----------
        dist_coeffs : array_like
            Array of distortion coefficients 
            (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
            of 4, 5, 8, 12 or 14 elements.
        """
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
    def set_scaling_factor(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Scaling factor for cv2.getOptimalNewCameraMatrix
        """
        assert alpha > 0, "Incorrect value for alpha, has to be nonnegative"
        self.scaling_factor = float(alpha)
    def set_translation(self, T):
        """
        Parameters
        ----------
        T : array_like
            1x3 translation vector of Camera.
        """
        self.T = np.array(T, dtype=np.float32).reshape((1,3))
    def set_rotation(self, r):
        """
        Parameters
        ----------
        r : array_like
            Either 3x1 rotation vector (perform Rodrigues) or
            3x3 rotation matrix of Camera.
            It saves it as 3x3 rotation matrix.
        """
        r = np.array(r, dtype=np.float32)
        if r.shape!=(3,3):
            r, _ = cv2.Rodrigues(r)
        self.R = r
    def set_calibration_pattern(self, pattern: dict | Charuco | CheckerBoard):
        """
        Parameters
        ----------
        pattern : dictionary | Charuco object | CheckerBoard object
            Calibration pattern used for all camera calibration procedures.
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
        self.calibration_pattern = pattern
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
        """
        Set the reprojection error threshold (in pixels, L2 norm).
        This needs to be set in order to call refine(), since it uses this threshold
        to discard markers.

        Parameters
        ----------
        error_thr : float
            Threshold of reprojection error (in pixels) of identified markers to consider for calibration.
            This number has to be nonnegative.
        """
        assert error_thr > 0, "Incorrect value for threshold, has to be nonnegative"
        self.error_thr = float(error_thr)
    def set_min_points(self, min_points: int):
        """
        Parameters
        ----------
        min_points : int
            Number of minimum identified markers to consider for calibration.
            This number cannot be less than 4, otherwise it fails assertions in OpenCV.
        """
        self.min_points = int(max(4,min_points))

    def set_height(self, height: int):
        """
        Set image resolution height in pixels.

        Parameters
        ----------
        height : int
            Image resolution height in pixels.
        """
        assert height > 0, "Incorrect value for height, has to be nonnegative"
        self.height = int(height)
    def set_width(self, width: int):
        """
        Set image resolution width in pixels.

        Parameters
        ----------
        width : int
            Image resolution width in pixels.
        """
        assert width > 0, "Incorrect value for width, has to be nonnegative"
        self.width = int(width)
    def set_image_shape(self, shape: tuple[int, int]):
        """
        Set image resolution in pixels.
        Both numbers have to be integers and nonnegative.

        Parameters
        ----------
        shape : tuple
            Image resolution in (height: int, width: int).
        """
        self.set_height(shape[0])
        self.set_width (shape[1])
    def discard_intrinsic_images(self):
        self.intrinsic_images = [image for image in self.intrinsic_images if image not in self.discarded_images]
    def add_intrinsic_image_points(self, image_points: np.ndarray):
        """
        Appends more image points to the list intrinsic_image_points of Camera.

        Parameters
        ----------
        image_points : array_like
            Array of 2D image points of shape (N,2).
        """
        self.intrinsic_image_points.append(image_points)
    def add_intrinsic_object_points(self, object_points: np.ndarray):
        """
        Appends more object points to the list intrinsic_object_points of Camera.

        Parameters
        ----------
        object_points : array_like
            Array of 3D world points of shape (N,3).
        """
        self.intrinsic_object_points.append(object_points)
    def set_extrinsic_image_points(self, image_points: np.ndarray):
        """
        Appends more image points to the list extrinsic_image_points of Camera.

        Parameters
        ----------
        image_points : array_like
            Array of 2D image points of shape (N,2).
        """
        self.extrinsic_image_points = image_points
    def set_extrinsic_object_points(self, object_points: np.ndarray):
        """
        Appends more object points to the list extrinsic_object_points of Camera.

        Parameters
        ----------
        object_points : array_like
            Array of 3D world points of shape (N,3).
        """
        self.extrinsic_object_points = object_points

    # getters
    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Returns camera intrinsic matrix (shape 3x3) K.
        """
        return self.K
    def get_distortion(self) -> np.ndarray:
        """
        Returns camera distortion coefficients.
        """
        return self.dist_coeffs
    def get_scaling_factor(self) -> float:
        """
        Returns scaling factor used for new optimal intrinsic matrix.
        """
        return self.scaling_factor
    def get_new_intrinsic_matrix(self) -> np.ndarray:
        """
        Returns new optimal intrinsic matrix (shape 3x3) newK.
        """
        return self.newK
    def get_rvecs(self) -> list[np.ndarray]:
        """
        Returns the list of camera translation rotation vectors (shape 1x3) found during calibration.
        """
        return self.rvecs
    def get_tvecs(self) -> list[np.ndarray]:
        """
        Returns the list of camera translation translation vectors (shape 1x3) found during calibration.
        """
        return self.tvecs
    def get_translation(self) -> np.ndarray:
        """
        Returns the camera translation vector (shape 1x3) T.
        """
        return self.T
    def get_rotation(self) -> np.ndarray:
        """
        Returns the camera rotation matrix (shape 3x3) R.
        """
        return self.R
    def get_calibration_pattern(self) -> Charuco | CheckerBoard:
        return self.calibration_pattern
    def get_intrinsic_image_paths(self) -> list[str]:
        return self.intrinsic_images
    def get_extrinsic_image_paths(self) -> str:
        return self.extrinsic_image
    def get_errors(self) -> list[np.ndarray]:
        """
        Returns a list of the reprojection errors of every marker of every view.
        """
        return self.errors
    def get_per_view_error(self) -> list[float]:        
        """
        Returns a list of the reprojection errors for every view.
        """
        return [float(np.mean(errors)) for errors in self.errors]
    def get_mean_error(self) -> float:
        return float(np.mean(np.concatenate(self.errors)))
    def get_error_threshold(self) -> float:
        """
        Returns the reprojection error threshold (in pixels).
        """
        return self.error_thr
    def get_min_points(self) -> int:
        """
        Returns number of minimum identified markers to consider for calibration.
        """
        return self.min_points
    def get_image_shape(self) -> tuple[int, int]:
        """
        Returns image resolution in pixels as (height, width).
        """
        return (self.height, self.width)
    
    # functions
    def detect_markers_for_intrinsic(self):
        """
        Add markers (2D image coordinates and 3D world coordinates) for intrinsic calibration.
        """
        assert self.calibration_pattern is not None, "No Calibration Pattern defined"
        assert len(self.intrinsic_images) > 0, "No intrinsic images defined"

        # Empty the list of image and object points before going through list of images
        self.intrinsic_image_points = []
        self.intrinsic_object_points = []

        for image_path in self.intrinsic_images:
            img_points, obj_points, _ = self.calibration_pattern.detect_markers(image_path)

            if len(img_points) < self.min_points:
                self.discarded_images.add(image_path)
                continue

            self.add_intrinsic_image_points (img_points)
            self.add_intrinsic_object_points(obj_points)

        self.discard_intrinsic_images()
    
    def detect_markers_for_extrinsic(self):
        """
        Add markers (2D image coordinates and 3D world coordinates) for extrinsic calibration.
        """
        assert self.calibration_pattern is not None, "No Calibration Pattern defined"
        assert self.extrinsic_image is not None, "No extrinsic image defined"

        img_points, obj_points, _ = self.calibration_pattern.detect_markers(self.extrinsic_image)

        assert len(img_points) > self.min_points, "Not enough points for extrinsic calibration"

        self.set_extrinsic_image_points (img_points)
        self.set_extrinsic_object_points(obj_points)

    def calibrate_intrinsics(self):
        """
        Perform intrinsic camera calibration.
        """
        assert len(self.intrinsic_image_points) > 0, "There are no 2D image points"
        assert len(self.intrinsic_object_points) > 0, "There are not 3D object points"
        assert self.height is not None or self.width is not None, \
            "Image resolution has not been defined"

        result = \
            Calibration.calibrate(self.intrinsic_object_points,
                                  self.intrinsic_image_points,
                                  self.get_image_shape())
                
        self.K = result['K']
        self.dist_coeffs = result['dist_coeffs']
        self.rvecs = result['rvecs']
        self.tvecs = result['tvecs']

    def calibrate_new_optimal_intrinsic_matrix(self):
        result = Calibration.new_intrisic_matrix(self.get_image_shape(),
                                        self.K,
                                        self.dist_coeffs,
                                        self.scaling_factor)
        self.newK = result['newK']
        self.roi = result['roi']

    def refine(self):
        """
        TODO: consider saving (how?) pre- and post-refinement errors?

        Perform intrinsic camera calibration with refinement.
        Refinement can include minimum number of detected markers and maximum error.

        Function calculates reprojection errors if not already done. 
        """
        assert self.K is not None, "Camera has not been calibrated yet"
        assert self.error_thr > 0, "Error threshold not defined"
        if len(self.errors) < 1: 
            self.projection_errors()

        img_filtered = []
        obj_filtered = []

        # Select points with errors below the threshold
        for image_path, img_points, obj_points, errors \
            in zip (self.intrinsic_images,
                    self.intrinsic_image_points, 
                    self.intrinsic_object_points,
                    self.errors):
            filtered_idxs = np.nonzero(errors < self.error_thr)[0]

            if len(filtered_idxs) < self.min_points:
                self.discarded_images.add(image_path)
                continue

            # Filter the object and image points based on selected indexes
            img_filtered.append(np.array([img_points[i] for i in filtered_idxs]))
            obj_filtered.append(np.array([obj_points[i] for i in filtered_idxs]))

        # Perform refined calibration using the filtered points
        result = \
            Calibration.calibrate(obj_filtered,
                                  img_filtered,
                                  self.get_image_shape(),
                                  self.K,
                                  self.dist_coeffs)
                
        self.K = result['K']
        self.dist_coeffs = result['dist_coeffs']
        self.rvecs = result['rvecs']
        self.tvecs = result['tvecs']

        # Update list of image and object points before calculating refined errors
        self.discard_intrinsic_images()
        self.intrinsic_image_points = img_filtered
        self.intrinsic_object_points = obj_filtered
        self.projection_errors()

    def projection_errors(self):
        """
        Calculate reprojection error of detected markers.
        The error is calculated per marker per view.
        """
        assert len(self.intrinsic_image_points) > 0, "There are no 2D image points"
        assert len(self.intrinsic_object_points) > 0, "There are not 3D object points"
        assert self.K is not None, "Camera has not been calibrated yet"

        self.errors = []

        for img_points, obj_points, rvec, tvec \
            in zip(self.intrinsic_image_points,
                   self.intrinsic_object_points,
                   self.rvecs,
                   self.tvecs):
            errors = Calibration.reprojection_error(obj_points,
                                                    img_points,
                                                    rvec,
                                                    tvec,
                                                    self.K,
                                                    self.dist_coeffs)
            self.errors.append(errors)

    def calibrate_extrinsics(self):
        """
        Perform extrinsic camera calibration.

        This function saves R and T, where T is the origin of the camera in 
        the world coordinate system and R is the rotation matrix.

        p_w = T + lambda * R @ u, where p_w is point in world coordinate and
        u is normalized image coordinate (u_1, u_2, 1)
        """
        assert len(self.extrinsic_image_points) > 0, "There are no 2D image points"
        assert len(self.extrinsic_object_points) > 0, "There are not 3D object points"
        assert self.K is not None, "Camera has not been calibrated yet"
        
        result = Calibration.calibrate_extrinsic(self.extrinsic_object_points,
                                                 self.extrinsic_image_points,
                                                 self.K,
                                                 self.dist_coeffs)
        
        rvec = result['rvec']
        tvec = result['tvec']
        
        R, _ = cv2.Rodrigues(rvec)
        T = tvec.reshape((1,3))
        
        self.R = R
        self.T = T

    # plotter
    def plot_distortion(self):
        """
        Plot (with matplotlib) an image displaying lens distortion.
        """
        assert self.K is not None, "Camera has not been calibrated yet"

        Plotter.plot_distortion(self.get_image_shape(), self.K, self.dist_coeffs)

    def plot_detected_markers(self):
        """
        Plot (with matplotlib) an image displaying the position
        of detected intrinsic markers used for calibration.
        """
        assert len(self.intrinsic_image_points) > 0, "There are no 2D image points"
        assert self.K is not None, "Camera has not been calibrated yet"

        Plotter.plot_markers(self.intrinsic_image_points, self.get_image_shape(), self.K)

    def plot_errors(self):
        """
        TODO: implement with matplotlib
        """
        pass
        
    def save_calibration(self, filename: str):
        """
        Save calibration into a JSON file.

        Parameters
        ----------
        filename : str
            path to JSON file where calibration will be saved.  
        """
        assert self.K is not None, "Camera has not been calibrated yet"
        assert len(self.errors) > 0, "Reprojection error has not been calculated yet"
        save_json({
            "K": self.K,
            "dist_coeffs": self.dist_coeffs,
            "scaling_factor": self.scaling_factor,
            "newK": self.newK,
            "roi": self.roi,
            "R": self.R,
            "T": self.T,
            "height": self.height,
            "width": self.width,
            "error": self.get_mean_error(),
            "error_threshold": self.error_thr
        }, filename)

    def load_calibration(self, calibration: str | dict):
        if type(calibration) is str:
            calibration = load_json(calibration)
        
        self.set_height(calibration['height'])
        self.set_width(calibration['width'])

        self.set_intrinsic_matrix(calibration['K'])
        self.set_distortion(calibration['dist_coeffs'])
        self.set_rotation(calibration['R'])
        self.set_translation(calibration['T'])

    def run(self, config: str | dict):
        if type(config) is str:
            with open(config, 'r') as f:
                config = json.load(f)

        self.set_intrinsic_image_paths(config['intrinsic_folder_path'])
        self.set_extrinsic_image_path(config['extrinsic_image_path'])

        self.set_height(config['height'])
        self.set_width(config['width'])

        # calibration utils
        self.set_calibration_pattern(config['charuco'])
        self.set_error_threshold(config['error_thr'])
        self.set_min_points(config['min_points'])

        # intrinsic
        self.detect_markers_for_intrinsic()
        self.calibrate_intrinsics()
        self.refine()
        self.set_scaling_factor(config['scaling_factor'])
        self.calibrate_new_optimal_intrinsic_matrix()

        # extrinsic (or skip if camera is origin)
        self.detect_markers_for_extrinsic()
        self.calibrate_extrinsics()
        
        self.save_calibration(config['output_filename'])