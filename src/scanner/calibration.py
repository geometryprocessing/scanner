import cv2
import numpy as np

from src.scanner.camera import Camera, get_cam_config
from src.utils.three_d_utils import combine_transformations, \
    intersect_line_with_plane, fit_plane, camera_to_ray_world
from src.utils.image_utils import load_ldr
from src.utils.plotter import Plotter
from src.utils.file_io import load_json, save_json, get_all_paths

CHARUCO_DICTIONARY_ENUM = {
    "4X4_50"   : cv2.aruco.DICT_4X4_50,
    "4X4_100"  : cv2.aruco.DICT_4X4_100,
    "4X4_250"  : cv2.aruco.DICT_4X4_250,
    "4X4_1000" : cv2.aruco.DICT_4X4_1000,

    "5X5_50"   : cv2.aruco.DICT_5X5_50,
    "5X5_100"  : cv2.aruco.DICT_5X5_100,
    "5X5_250"  : cv2.aruco.DICT_5X5_250,
    "5X5_1000" : cv2.aruco.DICT_5X5_1000,

    "6X6_50"   : cv2.aruco.DICT_6X6_50,
    "6X6_100"  : cv2.aruco.DICT_6X6_100,
    "6X6_250"  : cv2.aruco.DICT_6X6_250,
    "6x6_1000" : cv2.aruco.DICT_6X6_1000,

    "7X7_50"   : cv2.aruco.DICT_7X7_50,
    "7X7_100"  : cv2.aruco.DICT_7X7_100,
    "7X7_250"  : cv2.aruco.DICT_7X7_250,
    "7X7_1000" : cv2.aruco.DICT_7X7_1000,

    "ARUCO_ORIGINAL" : cv2.aruco.DICT_ARUCO_ORIGINAL,

    "APRILTAG_16h5"  : cv2.aruco.DICT_APRILTAG_16h5,
    "APRILTAG_25h9"  : cv2.aruco.DICT_APRILTAG_25h9,
    "APRILTAG_36h10" : cv2.aruco.DICT_APRILTAG_36h10,
    "APRILTAG_36h11" : cv2.aruco.DICT_APRILTAG_36h11
}


class CheckerBoard:
    def __init__(self, rows=None, columns=None, checker_size=None, board_config=None):
        """
        Initialize checkerboard object for corner detection.

        Parameters
        ----------
        rows : int, optional
            Number of rows of board.
        columns : int, optional
            Number of columns of board.
        checker_size : int, optional
            Size (in millimeters) of square.
        board_config : dict, optional
            Dictionary containing m, n, checker_size.
        """
        if board_config:
            rows = board_config["rows"]
            columns = board_config["columns"]
            checker_size = board_config["checker_size"]

        self.rows=rows
        self.columns=columns
        self.checker_size=checker_size

        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        self.flags = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH

        self.object_points = self.create_object_points()
            
        self.ids = np.arange(self.rows*self.columns)
        
    def create_object_points(self):
        obj_points = np.zeros((self.rows * self.columns, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:self.rows, 0:self.columns].T.reshape(-1, 2) * self.checker_size
        return obj_points
    
    def change_criteria(self, criteria: tuple[int, int, float]):
        self.criteria = criteria

    def create_image(self, resolution: tuple):
        pass
    
    def detect_markers(self, image):
        """
        Detect checkerboard markers on grayscale image.

        If image is a string, function opens the image using cv2.imread() and
        converts it to grayscale.

        Parameters
        ----------
        image : array_like or string
            Gray scale image. If string, call OpenCV to read image file.

        Returns
        -------
        img_points
            numpy array of detected checker corner image coordinates
        obj_points
            numpy array of detected checker corner world coordinates
        ids
            numpy array of detected checker corner IDs
        """

        if isinstance(image, str):
            image = load_ldr(image, make_gray=True)
        elif len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # image has to be uint8 for opencv. if not, scaled it to [0, 255]
        dtype = image.dtype
        if dtype != np.uint8:
            m = np.iinfo(dtype).max if dtype.kind in 'iu' else np.finfo(dtype).max
            image = (image / m * 255).astype(np.uint8)

        ret, corners = cv2.findChessboardCorners(image, (self.rows, self.columns), flags=self.flags)

        if ret:
            corners = cv2.cornerSubPix(image, corners, (self.rows, self.columns), (-1, -1), self.criteria)
        else:
            corners = None

        if corners is not None:
            img_points = corners.reshape((self.columns, self.rows, 2))
            obj_points = self.object_points
            ids = self.ids
        else:
            img_points, obj_points, ids = [], [], []

        return img_points, obj_points, ids


class Charuco:
    def __init__(self,
                rows=None,
                columns=None,
                checker_size=None,
                marker_size=None,
                dictionary=None,
                board_config=None):
        """
        Initialize ChArUco object for marker detection.

        Parameters
        ----------
        rows : int, optional
            Number of rows of board.
        columns : int, optional
            Number of columns of board.
        checker_size : int, optional
            Size (in millimeters) of marker.
        dictionary : str, optional
            String containing which ChArUco dictionary to use.
        board_config : dict, optional
            Dictionary containing m, n, checker_size, and dictionary.
        """
        if board_config:
            rows = board_config["rows"]
            columns = board_config["columns"]
            checker_size = board_config["checker_size"]
            marker_size = board_config["marker_size"]
            dictionary = board_config["dictionary"]

        self.rows=rows
        self.columns=columns
        self.checker_size=checker_size
        self.marker_size=marker_size
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICTIONARY_ENUM[dictionary])
        self.charuco_board = cv2.aruco.CharucoBoard((self.columns,self.rows),
                                                    self.checker_size,
                                                    self.marker_size,
                                                    self.aruco_dict)
        self.charuco_board.setLegacyPattern(True)
        self.chessboard_corners = self.charuco_board.getChessboardCorners()
        self.parameters = cv2.aruco.DetectorParameters()

    def adjust_parameters(self, **kwargs):
        """
        Pass keyword arguments from OpenCV aruco library, such as
            adaptiveThreshWinSizeMin,
            adaptiveThreshWinSizeMax,
            adaptiveThreshWinSizeStep,
            adaptiveThreshConstant,
            minMarkerPerimeterRate,
            maxMarkerPerimeterRate,
            minMarkerDistanceRate,
            polygonalApproxAccuracyRate
        with their respective values.

        For more information, visit https://docs.opencv.org/4.5.5/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
        """
        for key, value in kwargs.items():
            setattr(self.parameters, key, value)
    
    def create_image(self, resolution: tuple[int, int]):
        """
        Parameters
        ----------
        resolution : tuple
            Resolution to generate image. Resolution must be passed
            as (width, height).

        Returns
        -------
        image 
            ChArUco board image at the desired resolution.
        """
        return self.charuco_board.generateImage(resolution)
    
    def get_image_points(self, resolution: tuple[int, int], ids: list[int]):
        """
        TODO: discard
        
        Parameters
        ----------
        resolution : tuple
            Resolution to generate image. Resolution must be passed
            as (width, height).
        ids : list
            List of ids that were detected on the image.

        Returns
        -------
        - list of detected image coordinates
        """
        image = self.create_image(resolution)
        img_points, _, _ = self.detect_markers(image)

        return img_points[ids]
        
    def detect_markers(self, image):
        """
        Detect ChArUco markers on image.

        If image is a string, function opens the image using cv2.imread() and
        converts it to grayscale.

        Parameters
        ----------
        image : array_like or string
            Gray scale image. If string, call OpenCV to read image file.
            If not grayscale, convert to grayscale.

        Returns
        -------
        tuple
            - numpy array of detected ChArUco image coordinates
            - numpy array of detected ChArUco world coordinates
            - numpy array of detected ChArUco marker IDs
        """

        if isinstance(image, str):
            image = load_ldr(image, make_gray=True)
        elif len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image has to be uint8 for opencv. if not, scaled it to [0, 255]
        dtype = image.dtype
        if dtype != np.uint8:
            m = np.iinfo(dtype).max if dtype.kind in 'iu' else np.finfo(dtype).max
            image = (image / m * 255).astype(np.uint8)

        m_pos, m_ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.parameters)

        if len(m_pos) > 0:
            count, c_pos, c_ids = cv2.aruco.interpolateCornersCharuco(m_pos, m_ids, image, self.charuco_board)
        else:
            count, c_pos, c_ids = 0, [], []

        if count:
            img_points = np.array(c_pos).reshape((-1, 2))
            obj_points  = self.chessboard_corners[c_ids].reshape((-1, 3))
            ids = c_ids.ravel()
        else:
            img_points, obj_points, ids = [], [], []

        return img_points, obj_points, ids

class Calibration:
    # TODO: implement function calls for OpenCV fisheye

    @staticmethod
    def calibrate_intrinsic(object_points: list,
                            image_points: list,
                            image_shape: tuple
                            ) -> dict:
        """
        TODO: explain function

        Parameters
        ----------
        object_points : list
            List, of length N, of 3D world coordinates, where N is number of images.
        image_points : list
            List, of length N, of 2D image coordinates, where N is number of images.
        image_shape : tuple
            (width, height) of images

        Returns
        -------
        dict
            Dictionary containing
            {
                'rms': root mean squared error,
                'K': 3x3 intrinsic matrix,
                'dist_coeffs': distortion coefficients
            }
        """

        rms, cameraMatrix, distCoeffs = \
            cv2.initCameraMatrix2D(object_points, image_points, image_shape)
        
        if not rms:
            raise RuntimeError("Intrinsic calibration failed")

        return {
            'rms': rms,
            'K': cameraMatrix,
            'dist_coeffs': distCoeffs,
        }


    @staticmethod
    def calibrate(object_points: list,
                  image_points: list,
                  image_shape: tuple,
                  K: np.ndarray=None,
                  dist_coeffs: np.ndarray=None,
                  flags: int=0
                  ) -> dict:
        """
        TODO: explain function

        Parameters
        ----------
        object_points : list
            List, of length N, of 3D world coordinates, where N is number of images.
        image_points : list
            List, of length N, of 2D image coordinates, where N is number of images.
        image_shape : tuple
            (width, height) of images
        K : array_like, optional
            3x3 intrinsic matrix. Default is None, i.e. 
            call to OpenCV will not use any initial guess.
        dist_coeffs : array_like, optional
            Distortion coefficients. Default is None, i.e. 
            call to OpenCV will not use any initial guess.
        flags : int
            flags to pass to OpenCV function. For more information, read here
            https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html
            Default is set to 0

        Returns
        -------
        dict
            Dictionary containing
            {
                'rms': root mean squared error,
                'K': 3x3 intrinsic matrix,
                'dist_coeffs': distortion coefficients,
                'rvecs': list of N rotation vectors, each with shape 1x3, 
                'tvecs': list of N translation vectors, each with shape 1x3
            }
        """
        rms, cameraMatrix, distCoeffs, rvecs, tvecs = \
            cv2.calibrateCamera(object_points, image_points, image_shape, K, dist_coeffs, flags=flags)
        
        if not rms:
            raise RuntimeError("Calibration failed")

        return {
            'rms': rms,
            'K': cameraMatrix,
            'dist_coeffs': distCoeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
        }
    
    @staticmethod
    def calibrate_extended(object_points: list,
                           image_points: list,
                           image_shape: tuple,
                           K: np.ndarray=None,
                           dist_coeffs: np.ndarray=None,
                           flags: int=0
                           ) -> dict:
        """
        TODO: explain function

        Parameters
        ----------
        object_points : list
            List, of length N, of 3D world coordinates, where N is number of images.
        image_points : list
            List, of length N, of 2D image coordinates, where N is number of images.
        image_shape : tuple
            (width, height) of images
        K : array_like, optional
            3x3 intrinsic matrix. Default is None, i.e. 
            call to OpenCV will not use any initial guess.
        dist_coeffs : array_like, optional
            Distortion coefficients. Default is None, i.e. 
            call to OpenCV will not use any initial guess.
        flags : int
            flags to pass to OpenCV function. For more information, read here
            https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html
            Default is set to 0

        Returns
        -------
        dict
            Dictionary containing
            {
                'rms': root mean squared error,
                'K': 3x3 intrinsic matrix,
                'dist_coeffs': distortion coefficients,
                'rvecs': list of N rotation vectors, each with shape 1x3, 
                'tvecs': list of N translation vectors, each with shape 1x3,
                'perViewErrors': list of N root mean squared reprojection errors
            }
        """
        rms, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = \
            cv2.calibrateCameraExtended(object_points, image_points, image_shape, K, dist_coeffs, flags=flags)

        if not rms:
            raise RuntimeError("Extended calibration failed")

        return {
            'rms': rms,
            'K': cameraMatrix,
            'dist_coeffs': distCoeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'perViewErrors': perViewErrors
        }
        
    @staticmethod
    def new_intrisic_matrix(image_shape: tuple,
                            K: np.ndarray,
                            dist_coeffs: np.ndarray,
                            scaling_factor: float) -> dict:
        """
        TODO: explain function

        Parameters
        ----------
        image_shape : tuple
            (width, height) of images
        K : array_like, optional
            3x3 intrinsic matrix.
        dist_coeffs : array_like
            Distortion coefficients.
        scaling_factor : float
            Scaling factor (from 0.0 to 1.0) to exclude/include pixels after undistorting.
            Value of 0.0 includes the whole original image after undistortion.
            Value of 1.0 excludes all black pixels after undistortion.

        Returns
        -------
        dict
            Dictionary containing
            {
                'newK': new 3x3 intrinsic matrix,
                'roi': tuple (x1,y1,x2,y2) of region of interest
            }
        """
        newK, roi = cv2.getOptimalNewCameraMatrix(K,
                                                  dist_coeffs,
                                                  image_shape,
                                                  scaling_factor,
                                                  image_shape)
        return {
            'newK': newK,
            'roi': roi
        }
    
    @staticmethod
    def reprojection_error(object_points: list,
                           image_points: list,
                           rvec: np.ndarray,
                           tvec: np.ndarray,
                           K: np.ndarray, 
                           dist_coeffs: np.ndarray
                           ) -> np.ndarray:
        """
        TODO: explain function

        Parameters
        ----------
        object_points : list
            List, of length N, of 3D world coordinates, where N is number of images.
        image_points : list
            List, of length N, of 2D image coordinates, where N is number of images.
        image_shape : tuple
            (width, height) of images
        rvecs : array_like
            Rotation vectors for each 2D-3D correspondence image.
        tvecs : array_like
            Translation vectors for each 2D-3D correspondence image.
        K : array_like
            3x3 intrinsic matrix.
        dist_coeffs : array_like
            Distortion coefficients.
        
        Returns
        -------
        array_like
            list with length N of float reprojection error for each image
        """
        assert len(object_points)==len(image_points), \
            "Number of 3D world coordinates must equal number of 2D image coordinates"
        reprojected_points, _ = \
            cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
        
        reprojected_points = reprojected_points.reshape((object_points.shape[0], 2))
        errors = np.linalg.norm(image_points - reprojected_points, axis=1)

        return errors
    
    
    @staticmethod
    def calibrate_extrinsic(object_points: list,
                           image_points: list,
                           K: np.ndarray,
                           dist_coeffs: np.ndarray,
                           flags: int=cv2.SOLVEPNP_ITERATIVE
                           ) -> dict:
        """
        TODO: function explanation

        Parameters
        ----------
        object_points : list
            List, of length N, of 3D world coordinates, where N is number of images.
        image_points : list
            List, of length N, of 2D image coordinates, where N is number of images.
        image_shape : tuple
            (width, height) of images
        K : array_like, optional
            3x3 intrinsic matrix. Default is None, i.e. 
            call to OpenCV will not use any initial guess.
        dist_coeffs : array_like, optional
            Distortion coefficients. Default is None, i.e. 
            call to OpenCV will not use any initial guess.
        flags : int
            flags to pass to OpenCV function. For more information, read here
            https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html
            Default is set to cv2.SOLVEPNP_ITERATIVE

        Returns
        -------
        dict
            Dictionary containing
            {
                'rms': root mean squared error,
                'rvec': 3x1 rotation vector, 
                'tvec': 3x1 translation vector
            }

        """
        rms, rvec, tvec = \
            cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags=flags)

        if not rms:
            raise RuntimeError("Extrinsic calibration failed")
        
        return {
            'rms': rms,
            'rvec': rvec,
            'tvec': tvec,
        }
    
    @staticmethod
    def stereo_calibrate(object_points: list,
                         image_points_1: list,
                         image_points_2: list,
                         image_size: tuple,
                         K_1: np.ndarray,
                         dist_coeffs_1: np.ndarray,
                         K_2: np.ndarray,
                         dist_coeffs_2: np.ndarray,
                         R_1: np.ndarray=np.identity(3),
                         T_1: np.ndarray=np.zeros((3,1)),
                         flags: int=cv2.CALIB_FIX_INTRINSIC
                         ) -> dict:
        """
        TODO: function explanation

        Parameters
        ----------
        object_points : list
            List, of length N, of 3D world coordinates, where N is number of images.
        image_points_1 : list
            List, of length N, of 2D image coordinates from camera 1, where N is number of images.
        image_points_2 : list
            List, of length N, of 2D image coordinates from camera 2
            (or projector), where N is number of images.
        image_size : tuple
            Tuple containing (height, width) of image resolution.
            NOTE: OpenCV only accepts one value of image size, so it assumes cameras have
            the same resolution. If they are not the same, pass the intrinsic parameters for each
            and set the flag to cv2.CALIB_FIX_INTRINSIC.
        K_1 : array_like
            3x3 intrinsic matrix from camera 1.
        dist_coeffs_1 : array_like
            Distortion coefficients from camera 1.
        K_2 : array_like
            3x3 intrinsic matrix from camera 2 (or projector).
        dist_coeffs_2 : array_like
            Distortion coefficients from camera 2 (or projector).
        R_1 : array_like, optional
            Rotation matrix from camera 1.
            If not given, assumes identity matrix.
            If given as a 3x1 or 1x3 vector, converts it to
            a 3x3 matrix using Rodrigues.
        T_1 : array_like, optional
            Translation vector from camera 1.
            If not given, assumes origin.
        flags : int
            flags to pass to OpenCV function. For more information, read here
            https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html
            Default is set to cv2.CALIB_FIX_INTRINSIC

        Returns
        -------
        dict
            Dictionary containing
            {
                'rms': root mean squared error,
                'R': 3x3 rotation matrix of camera 2 (or projector),
                'T': 3x1 translation vector of camera 2 (or projector)
            }

        """
        rms, _, _, _, _, R, T, _, _ = \
            cv2.stereoCalibrate(object_points, image_points_1, image_points_2,
                                K_1, dist_coeffs_1, K_2, dist_coeffs_2, image_size,
                                flags=flags)
    
        if not rms:
            raise RuntimeError("Stereo calibration failed")
        
        # as described in the OpenCV documentation,
        # the R and T are "equivalent to the position of 
        # the first camera with respect to the second camera coordinate system."
        # perform matrix multiplication that R and T mean 
        # bringing world points into second camera coordinate system

        # R_combined, T_combined = combine_transformations(R_1, T_1, R.T, -np.matmul(R.T, T.flatten()))
        R_combined, T_combined = combine_transformations(R_1, T_1, R, T)
        # R_combined, T_combined = combine_transformations(R, T, R_1, T_1)

        return {
            'rms': rms,
            'R': R_combined,
            'T': T_combined
        }
    
class CameraCalibration:
    def __init__(self,
                 resx: int = None,
                 resy: int = None,
                 K = None,
                 dist_coeffs = None,
                 R = np.identity(3),
                 T = np.zeros(shape=(3,1)),
                 intrinsic_folder_path: str | list[str] = None,
                 extrinsic_image_path: str | list[str] = None,
                 calibration_pattern: str | Charuco | CheckerBoard = None,
                 scaling_factor: float = 0.,
                 min_points: int = 4,
                 error_thr: float = 0.,
                 output_filename: str = None,
                 config: dict | str = None):

        self.resx  = resx
        self.resy = resy

        # intrinsic
        self.intrinsic_images = intrinsic_folder_path # image paths
        if isinstance(self.intrinsic_images, str):
            self.intrinsic_images = get_all_paths(self.intrinsic_images)
        self.discarded_images = set()
        self.intrinsic_object_points = []
        self.intrinsic_image_points = []
        self.K = K
        self.scaling_factor = scaling_factor
        self.newK = None
        self.roi = ()
        self.dist_coeffs = dist_coeffs
        self.rvecs = np.zeros(shape=(3,1))
        self.tvecs = np.zeros(shape=(3,1))

        # extrinsic
        self.extrinsic_image = extrinsic_image_path    # image paths
        if isinstance(self.extrinsic_image, str):
            self.extrinsic_image = get_all_paths(self.extrinsic_image)
        self.extrinsic_object_points = []
        self.extrinsic_image_points = []
        self.R = R                         # if not passed, camera is initialized at origin
        self.T = T                         # if not passed, camera is initialized at origin

        # calibration utils
        self.errors = []
        self.calibration_pattern = calibration_pattern
        if isinstance(self.calibration_pattern, (str,dict)):
            self.set_calibration_pattern(self.calibration_pattern)
        self.error_thr = error_thr
        self.min_points = min_points

        self.output_filename = output_filename

        if config is not None:
            self.load_config(config)

    def load_config(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        self.set_intrinsic_image_paths(config['camera']['intrinsic_folder_path'])
        self.set_extrinsic_image_path(config['camera']['extrinsic_image_path'])

        self.set_resy(config['camera']['resy'])
        self.set_resx(config['camera']['resx'])

        # calibration utils
        self.set_calibration_pattern(config['camera']['charuco'])
        self.set_error_threshold(config['camera']['error_thr'])
        self.set_min_points(config['camera']['min_points'])
        self.set_scaling_factor(config['camera']['scaling_factor'])
        self.set_output(config['camera']['output_filename'])

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
            3x1 translation vector of Camera.
        """
        self.T = np.array(T, dtype=np.float32).reshape((3,1))
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
        if isinstance(pattern, dict):
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
        self.intrinsic_images = get_all_paths(image_paths)
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

    def set_resy(self, height: int):
        """
        Set image resolution y (height) in pixels.

        Parameters
        ----------
        height : int
            Image resolution y (height) in pixels.
        """
        assert height > 0, "Incorrect value for height, has to be nonnegative"
        self.resy = int(height)
    def set_resx(self, width: int):
        """
        Set image resolution x (width) in pixels.

        Parameters
        ----------
        width : int
            Image resolution x (width) in pixels.
        """
        assert width > 0, "Incorrect value for width, has to be nonnegative"
        self.resx = int(width)
    def set_image_shape(self, shape: tuple[int, int]):
        """
        Set image resolution in pixels.
        Both numbers have to be integers and nonnegative.

        Parameters
        ----------
        shape : tuple
            Image resolution in (width / resx: int, height / resy: int).
        """
        self.set_resx(shape[0])
        self.set_resy(shape[1])
    def discard_intrinsic_images(self):
        self.intrinsic_images = [image for image in self.intrinsic_images if id(image) not in self.discarded_images]
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
    def set_output(self, filename: str):
        """
        
        Parameters
        ----------
        filename : str
            path to where output will be saved as a JSON file 
        """
        self.output_filename = filename

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
        Returns the list of camera translation rotation vectors (shape Nx3,
        where N is the number of images) found during calibration.
        """
        return self.rvecs
    def get_tvecs(self) -> list[np.ndarray]:
        """
        Returns the list of camera translation translation vectors (shape Nx3,
        where N is the number of images) found during calibration.
        """
        return self.tvecs
    def get_translation(self) -> np.ndarray:
        """
        Returns the camera translation vector (shape 3x1) T.
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

        Returns
        -------
        errors
            list of numpy arrays, where each index of the list contains the
            reprojection error (in pixels) for every detected marker in that index view
        """
        return self.errors
    def get_per_view_error(self) -> list[float]:        
        """
        Returns a list of the reprojection errors for every view.

        Returns
        -------
        errors
            list of floats, where each index is the average reprojection
            error (in pixels) of detected markers in that index view
        """
        return [float(np.mean(errors)) for errors in self.errors]
    def get_mean_error(self) -> float:
        """
        Returns
        -------
        error
            average error in pixels of all detected markers in all views
        """
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
        Returns image resolution in pixels as (width, height).

        Returns
        -------
        resx 
            camera resolution x (width) in pixels.
        resy
            camera resolution y (height) in pixels
        """
        return (self.resx, self.resy)
    
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
                self.discarded_images.add(id(image_path))
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
        assert self.resy is not None or self.resx is not None, \
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
                self.discarded_images.add(id(image_path))
                continue

            # Filter the object and image points based on selected indexes
            img_filtered.append(np.array([img_points[i] for i in filtered_idxs]))
            obj_filtered.append(np.array([obj_points[i] for i in filtered_idxs]))

        if len(obj_filtered) == 0:
            print("No points left for calibration, try looser threshold.")
            return

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

        This function saves R and T, where R and T are the rotation and translation
        tuple to bring the world coordinates into the camera coordinate system.

        Notes
        -----
        lambda u = R @ p_w + T, where p_w is point in world coordinate and
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
        T = tvec.reshape((3,1))
        
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
        
    def save_calibration(self):
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
            "resy": self.resy,
            "resx": self.resx,
            "error": self.get_mean_error(),
            "error_threshold": self.error_thr
        }, self.output_filename)

    def run(self):

        # intrinsic
        self.detect_markers_for_intrinsic()
        self.calibrate_intrinsics()
        self.refine()
        self.calibrate_new_optimal_intrinsic_matrix()

        # extrinsic (or skip if camera is origin)
        self.detect_markers_for_extrinsic()
        self.calibrate_extrinsics()

class ProjectorCalibration:
    def __init__(self,
                 resx: int = None,
                 resy: int = None,
                 K = None,
                 dist_coeffs = None,
                 R = np.identity(3),
                 T = np.zeros(shape=(3,1)),
                 camera: str | Camera = None,
                 image_folder_path: str | list[str] = None,
                 plane_pattern = None,
                 calibration_pattern: str | Charuco | CheckerBoard = None,
                 calibration_image: str = None,
                 min_points: int = 4,
                 error_thr: float = 0.,
                 output_filename: str = None,
                 config: dict | str = None):
        

        self.resx  = resx
        self.resy = resy
        # accompanying ALREADY CALIBRATED camera
        self.camera = camera
        if isinstance(self.camera, str):
            self.camera = get_cam_config(self.camera)
        # intrinsic
        self.K = K
        self.dist_coeffs = dist_coeffs
        self.rvecs = np.zeros(shape=(3,1))
        self.tvecs = np.zeros(shape=(3,1))

        # extrinsic
        self.R = R
        self.T = T

        # calibration utils
        self.images = image_folder_path               # image paths
        if isinstance(self.images, str):
            self.images = get_all_paths(self.images)
        self.discarded_images = set()
        self.image_points = []
        self.object_points = []
        self.camera_image_points = []  # camera points will be used for extrinsic calibration
        self.camera_object_points = [] # camera points will be used for extrinsic calibration
        self.planes = []
        self.errors = []
        self.plane_pattern = plane_pattern
        if isinstance(self.plane_pattern, (str, dict)):
            self.set_plane_pattern(self.plane_pattern)
        self.calibration_pattern = calibration_pattern
        if isinstance(self.calibration_pattern, (str, dict)):
            self.set_calibration_pattern(self.calibration_pattern)
        self.calibration_image = calibration_image
        self.error_thr = error_thr
        self.min_points = min_points

        self.output_filename = output_filename

        if config is not None:
            self.load_config(config)

    def load_config(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        self.set_image_paths(config['projector']['image_folder_path'])
        self.set_projector_resy(config['projector']['resy'])
        self.set_projector_resx(config['projector']['resx'])
        self.load_camera(config['projector']['camera_calibration'])
        self.set_plane_pattern(config['projector']['plane_pattern'])
        self.set_calibration_pattern(config['projector']['calibration_pattern'])
        self.set_error_threshold(config['projector']['error_thr'])
        self.set_min_points(config['projector']['min_points'])
        self.set_output(config['projector']['output_filename'])

    # setters
    def set_intrinsic_matrix(self, K):
        """
        Parameters
        ----------
        K : array_like
            3x3 Intrinsic Matrix of Projector.
        """
        self.K = np.array(K, dtype=np.float32)
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
    def set_translation(self, T):
        """
        Parameters
        ----------
        T : array_like
            3x1 translation vector of Projector.
        """
        self.T = np.array(T, dtype=np.float32).reshape((3,1))
    def set_rotation(self, r):
        """
        Parameters
        ----------
        r : array_like
            Either 3x1 rotation vector (perform Rodrigues) or
            3x3 rotation matrix of Projector.
            It saves it as 3x3 rotation matrix.
        """
        r = np.array(r, dtype=np.float32)
        if r.shape!=(3,3):
            r, _ = cv2.Rodrigues(r)
        self.R = r
    def load_camera(self, camera: str | dict | Camera):
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
        if isinstance(pattern, dict):
            try: 
                if pattern["type"] == "charuco":
                    pattern = Charuco(board_config=pattern)
                elif pattern["type"] == "checkerboard":
                    pattern = CheckerBoard(board_config=pattern)
            except:
                pattern = Charuco(board_config=pattern)
        self.plane_pattern = pattern
    def set_calibration_pattern(self, pattern: dict | Charuco | CheckerBoard):
        """
        Parameters
        ----------
        pattern : dictionary | Charuco object | CheckerBoard object
            Calibration pattern used for all projector calibration procedures.
            Function only aceppts ChArUco and Checkerboard/Chessboard.
        """
        if isinstance(pattern, dict):
            try: 
                if pattern["type"] == "charuco":
                    pattern = Charuco(board_config=pattern)
                elif pattern["type"] == "checkerboard":
                    pattern = CheckerBoard(board_config=pattern)
            except:
                pattern = Charuco(board_config=pattern)
        self.calibration_pattern = pattern
    def set_calibration_image_path(self, path: str):
        """
        The projector has projected an image onto the scene for the calibration
        to be possible. The image itself is required in the calibration, since
        the marker coordinates in the projector plane are required.

        Parameters
        ----------
        image_path : string
            Path to calibration image projected by the projector onto a known
            plane.
        """
        self.calibration_image = path
    def set_image_paths(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for calibration of Projector.
        """
        self.images = get_all_paths(image_paths)
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
        self.error_thr = error_thr
    def set_min_points(self, min_points: int):
        """
        Parameters
        ----------
        min_points : int
            Number of minimum identified markers to consider for calibration.
            This number cannot be less than 4, otherwise it fails assertions in OpenCV.
        """
        self.min_points = int(max(4,min_points))
    def set_projector_resy(self, height: int):
        """
        Set projector resolution y (height) in pixels.

        Parameters
        ----------
        height : int
            Image resolution y (height) in pixels.
        """
        assert height > 0, "Incorrect value for height, has to be nonnegative"
        self.resy = int(height)
    def set_projector_resx(self, width: int):
        """
        Set projector image resolution x (width) in pixels.

        Parameters
        ----------
        width : int
            Image resolution x (width) in pixels.
        """
        self.resx = int(width)
    def set_projector_shape(self, shape: tuple[int, int]):
        """
        Set image resolution in pixels.
        Both numbers have to be integers and nonnegative.

        Parameters
        ----------
        shape : tuple
            Image resolution in (width / resx: int, height / resy: int).
        """
        self.set_projector_resx(shape[0])
        self.set_projector_resy(shape[1])
    def set_output(self, filename: str):
        """
        
        Parameters
        ----------
        filename : str
            path to where output will be saved as a JSON file 
        """
        self.output_filename = filename
    def discard_intrinsic_images(self):
        self.images = [image for image in self.images if id(image) not in self.discarded_images]
    

    # getters
    def get_intrinsic_matrix(self):
        """
        Returns projector intrinsic matrix (shape 3x3) K.
        """
        return self.K
    def get_distortion(self):
        """
        Returns projector distortion coefficients.
        """
        return self.dist_coeffs
    def get_translation(self):
        """
        Returns the projector translation vector (shape 3x1) T.
        """
        return self.T
    def get_rotation(self):
        """
        Returns the projector rotation matrix (shape 3x3) R.
        """
        return self.R
    def get_camera(self):
        """
        Returns the accompanying Camera object.
        Check scanner.camera for more information.
        """
        return self.camera
    def get_calibration_pattern(self):
        """
        Returns pattern object used by projector.
        """
        return self.calibration_pattern
    def get_plane_pattern(self):
        """
        Returns pattern object used to identify a plane from camera.
        """
        return self.plane_pattern
    def get_image_paths(self):
        return self.images
    def get_errors(self):
        """
        Returns a list of the reprojection errors of every marker of every view.
        """
        return self.errors
    def get_per_view_error(self):
        """
        Returns a list of the reprojection errors for every view.
        """
        return [np.mean(errors) for errors in self.errors]
    def get_mean_error(self):

        return np.mean(np.concatenate(self.errors))
    def get_error_threshold(self):
        """
        Returns the reprojection error threshold (in pixels).
        """
        return self.error_thr
    def get_min_points(self):
        """
        Returns number of minimum identified markers to consider for calibration.
        """
        return self.min_points
    def get_image_shape(self):
        """
        Returns camera image resolution in pixels as (resx, resy).

        Returns
        -------
        resx
            camera resolution x (width) in pixels.
        resy
            camera resolution y (height) in pixels
        """
        return (self.camera.resx, self.camera.resy)
    def get_projector_shape(self):
        """
        Returns projector image resolution in pixels as (resx, resy).

        Returns
        -------
        resx
            projector resolution x (width) in pixels.
        resy
            projector resolution y (height) in pixels
        """
        return (self.resx, self.resy)

    # functions
    def reconstruct_plane(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        TODO: discard this function
        """
        assert self.camera.K is not None, "No camera defined"
        assert self.plane_pattern is not None, "No Plane Pattern defined"
        assert self.calibration_pattern is not None, "No Calibration Pattern defined"

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
        R_combined, T_combined = combine_transformations(self.camera.R, self.camera.T, R, T)
        # NOTE: since obj_points is of shape (Nx3), the matrix multiplication with rotation 
        # has to be written as (R @ obj_points.T).T
        # to simplify:
        # np.matmul(R_combined, obj_points.T).T = np.matmul(obj_points, R_combined.T)
        world_points = np.matmul(obj_points, R_combined.T) + T_combined.reshape((1,3))

        # fit plane
        return fit_plane(world_points)
    
    def detect_markers(self):
        """
        Add markers (2D image coordinates and 3D world coordinates) for calibration.
        """
        assert self.camera.K is not None, "No camera defined"
        assert len(self.images) > 0, "No intrinsic images defined"
        assert self.plane_pattern is not None, "No Plane Pattern defined"
        assert self.calibration_pattern is not None, "No Calibration Pattern defined"
        
        # Empty the list of image points, object points, and planes before going through list of images 
        self.image_points = []
        self.object_points = []
        self.camera_image_points = []
        self.camera_object_points = []
        self.planes = []
        
        # TODO: the IDs actually matter, since we want to match them with the IDs.
        # It is not always true that the IDs will be a list ranging from 0 to MAX.
        # Rewrite code, but efficiently, to find the correct projector image points
        # given a set of IDs.
        all_projector_image_points, _, _ = self.calibration_pattern.detect_markers(self.calibration_image)

        for image_path in self.images:
            # plane = self.reconstruct_plane(image_path)

            # detect plane/board markers with camera
            cam_img_points, obj_points, _ = \
                self.plane_pattern.detect_markers(image_path)
            
            # although plane reconstruction requires 3 points,
            # OpenCV extrinsic calibration requires 6 points
            if len(cam_img_points) < max(6, self.min_points):
                self.discarded_images.add(id(image_path))
                continue
            
            # find relative position of camera and board
            result = Calibration.calibrate_extrinsic(obj_points,
                                                    cam_img_points,
                                                    self.camera.K,
                                                    self.camera.dist_coeffs)
            
            rvec = result['rvec']
            tvec = result['tvec']
            
            # detect checkerboard -> pixel coordinates
            cam_img_points, _, ids = \
                self.calibration_pattern.detect_markers(image_path)
            
            if len(cam_img_points) < self.min_points:
                self.discarded_images.add(id(image_path))
                continue

            # undistort pixel coordinates -> normalized coordinates
            # project normalized coordinates onto Plane - X_3D
            origin, camera_rays = camera_to_ray_world(cam_img_points,
                                              rvec,
                                              tvec,
                                              self.camera.K,
                                              self.camera.dist_coeffs)
            # opencv calibration only works with PLANAR data, but where we are moving our
            # plane pattern board around and retrieving the 3D world coordinates
            # FIX THIS, OTHERWISE CANNOT RUN PROJECTOR CALIBRATION AS IS
            objs = intersect_line_with_plane(origin,
                                                        camera_rays,
                                                        np.array([[0., 0., 0.]]),
                                                        np.array([[0., 0., 1.]])).astype(np.float32)            
            objs[:,2] = 0. # ENSURE THAT THESE ARE ZERO, OTHERWISE OPENCV WILL NOT ACCEPT THEM
            
            proj_img_points = np.array(all_projector_image_points[ids], dtype=np.float32).reshape((-1,2))

            # save 2D coordinates for projector and camera and 3D points in the board coordinate system
            self.image_points.append(proj_img_points)
            self.object_points.append(objs)
            self.camera_image_points.append(cam_img_points)
                        
            R, _ = cv2.Rodrigues(rvec)
            T = tvec.reshape((3,1))
            
            # move markers to world coordinate
            R_combined, T_combined = combine_transformations(self.camera.R, self.camera.T, R, T)
            # NOTE: since obj_points is of shape (Nx3), the matrix multiplication with rotation 
            # has to be written as (R @ obj_points.T).T
            # to simplify:
            # np.matmul(R_combined, obj_points.T).T = np.matmul(obj_points, R_combined.T)
            camera_obj =  np.matmul(objs, R_combined.T) + T_combined.reshape((1,3))

            # save 3D points in the camera coordinate system
            self.camera_object_points.append(camera_obj)
        
        self.discard_intrinsic_images()

    def calibrate_intrinsic(self):
        """
        Perform projector intrinsic and extrinsic calibration.
        """
        assert len(self.image_points) > 0, "There are no 2D image points"
        assert len(self.object_points) > 0, "There are not 3D object points"
        assert self.resy is not None or self.resx is not None, \
            "Projector resolution has not been defined"

        # run Calibration.calibrate with projector 2D coordinates and X_3D 

        result = Calibration.calibrate(self.object_points,
                                       self.image_points,
                                       self.get_projector_shape())
        
        self.rvecs = result['rvecs']
        self.tvecs = result['tvecs']
        self.K = result['K']
        self.dist_coeffs = result['dist_coeffs']

    def refine(self):
        """
        TODO: implement?
        TODO: consider saving (how?) pre- and post-refinement errors?

        Perform intrinsic camera calibration with refinement.
        Refinement can include minimum number of detected markers and maximum error.

        Function calculates reprojection errors if not already done. 
        """
        assert self.K is not None, "Projector has not been calibrated yet"
        assert self.error_thr > 0, "Error threshold not defined"
        if len(self.errors) < 1: 
            self.projection_errors()

        img_filtered = []
        obj_filtered = []
        cam_img_filtered = []
        cam_obj_filtered = []

        # Select points with errors below the threshold
        for image_path, img_points, obj_points, cam_img_points, cam_obj_points, errors \
            in zip (self.images,
                    self.image_points, 
                    self.object_points,
                    self.camera_image_points,
                    self.camera_object_points,
                    self.errors):
            filtered_idxs = np.nonzero(errors < self.error_thr)[0]

            if len(filtered_idxs) < self.min_points:
                self.discarded_images.add(image_path)
                continue

            # Filter the object and image points based on selected indexes
            img_filtered.append(np.array([img_points[i] for i in filtered_idxs]))
            obj_filtered.append(np.array([obj_points[i] for i in filtered_idxs]))
            cam_img_filtered.append(np.array([cam_img_points[i] for i in filtered_idxs]))
            cam_obj_filtered.append(np.array([cam_obj_points[i] for i in filtered_idxs]))

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
        self.image_points = img_filtered
        self.object_points = obj_filtered
        self.camera_image_points = cam_img_filtered
        self.camera_object_points = cam_obj_filtered
        self.projection_errors()
    
    def projection_errors(self):
        """
        Calculate reprojection error of detected markers.
        The error is calculated per marker per view.
        """
        assert len(self.image_points) > 0, "There are no 2D image points"
        assert len(self.object_points) > 0, "There are not 3D object points"
        assert self.K is not None, "Projector has not been calibrated yet"

        self.errors = []

        for img_points, obj_points, rvec, tvec \
            in zip(self.image_points,
                   self.object_points,
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
        Perform extrinsic projector calibration. This function requires a camera
        paired with the projector because it calls stereo calibration from OpenCV. 

        This function saves R and T, where R and T are the rotation and translation
        tuple to bring the world coordinates into the projector coordinate system.

        Notes
        -----
        lambda u = R @ p_w + T, where p_w is point in world coordinate and
        u is normalized image coordinate (u_1, u_2, 1)
        """
        assert len(self.image_points) > 0, "There are no 2D projector image points"
        assert len(self.object_points) > 0, "There are no 3D object points"
        assert len(self.camera_image_points) > 0, "There are no 2D camera image points "
        assert self.K is not None, "Projector has not been calibrated yet"
        assert self.camera.K is not None, "Camera has not been defined"
        
        result = Calibration.stereo_calibrate(self.camera_object_points,
                                              self.camera_image_points,
                                              self.image_points,
                                              self.camera.get_image_shape(),
                                              self.camera.K,
                                              self.camera.dist_coeffs,
                                              self.K,
                                              self.dist_coeffs,
                                              self.camera.R,
                                              self.camera.T)
        
        self.R = result['R']
        self.T = result['T']

        # result = Calibration.calibrate_extrinsic(np.concatenate(self.camera_object_points),
        #                                          np.concatenate(self.image_points),
        #                                          self.K,
        #                                          self.dist_coeffs)
        
        # rvec = result['rvec']
        # tvec = result['tvec']
        
        # R, _ = cv2.Rodrigues(rvec)
        # T = tvec.reshape((3,1))
        
        # self.R = R
        # self.T = T

    # plotter
    def plot_distortion(self):
        """
        Plot (with matplotlib) an image displaying lens distortion.
        """
        assert self.K is not None, "Projector has not been calibrated yet"

        Plotter.plot_distortion(self.get_projector_shape(), self.K, self.dist_coeffs)
        return 
    
    def plot_detected_markers(self):
        """
        Plot (with matplotlib) an image displaying the position
        of detected intrinsic markers used for calibration.
        """
        assert len(self.image_points) > 0, "There are no 2D image points"
        assert self.K is not None, "Projector has not been calibrated yet"

        Plotter.plot_markers(self.image_points, self.get_projector_shape(), self.K)

    def plot_errors(self):
        """
        TODO: implement with matplotlib
        """
        pass
        
    def save_calibration(self):
        """
        Save calibration into a JSON file.

        Parameters
        ----------
        filename : str
            path to JSON file where calibration will be saved.  
        """
        assert self.K is not None, "Projector has not been calibrated yet"
        assert len(self.errors) > 0, "Reprojection error has not been calculated yet"
        save_json({
            "K": self.K,
            "dist_coeffs": self.dist_coeffs,
            # "scaling_factor": self.scaling_factor,
            # "newK": self.newK,
            # "roi": self.roi,
            "R": self.R,
            "T": self.T,
            "resy": self.resy,
            "resx": self.resx,
            "error": self.get_mean_error(),
            "error_threshold": self.error_thr
        }, self.output_filename)

    def to_dict(self):
        """
        Returns projector config as a dictionary.
        """
        return {
            "resy": self.resy,
            "resx": self.resx,
            "K": self.K,
            "dist_coeffs": self.dist_coeffs,
            "R": self.R,
            "T": self.T,
        }

    def load_config(self, calibration: str | dict):
        if type(calibration) is str:
            calibration = load_json(calibration)
        
        self.set_projector_resy(calibration['resy'])
        self.set_projector_resx(calibration['resx'])

        self.set_intrinsic_matrix(calibration['K'])
        self.set_distortion(calibration['dist_coeffs'])
        self.set_rotation(calibration['R'])
        self.set_translation(calibration['T'])

    def run(self):
        self.calibrate_intrinsic()
        self.calibrate_extrinsics()

if __name__ == "__main__":
    pass
    # TODO: think if it's worth to have a main here