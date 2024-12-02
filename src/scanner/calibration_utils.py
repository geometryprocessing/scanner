import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

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
    def __init__(self, m=None, n=None, checker_size=None, board_config=None):
        """
        Initialize checkerboard object for corner detection.

        Parameters
        ----------
        m : int, optional
            Number of squares horizontally.
        n : int, optional
            Number of squares vertically.
        checker_size : int, optional
            Size (in millimeters) of square.
        board_config : dict, optional
            Dictionary containing m, n, checker_size.
        """
        if board_config:
            m = board_config["m"]
            n = board_config["n"]
            checker_size = board_config["checker_size"]

        self.m=m
        self.n=n
        self.checker_size=checker_size
        
    def detect_markers(self, image):
        """
        Detect checkerboard markers on grayscale image.

        If image is a string, function opens the image using cv2.imread().

        Parameters
        ----------
        image : array_like or string
            Gray scale image. If string, call OpenCV to read image file.

        Returns
        -------
        - list of detected checker corner image coordinates
        - list of detected checker corner world coordinates
        - list of detected checker corner IDs
        """

        if type(image) is str:
            image = cv2.imread(image, flags=cv2.IMREAD_GRAYSCALE)
        elif len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(image, (self.n, self.m))

        if ret:
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
            corners = cv2.cornerSubPix(image, corners, (self.n, self.m), (-1, -1), criteria)
        else:
            corners = None

        if corners is not None:
            img_points = corners.reshape((self.m, self.n, 2))
            obj_points = np.zeros((self.n * self.m, 3), np.float32)
            obj_points[:, :2] = np.mgrid[0:self.n, 0:self.m].T.reshape(-1, 2) * self.checker_size
            ids = np.arange(self.n*self.m)
        else:
            img_points, obj_points, ids = None, None, None

        return img_points, obj_points, ids


class Charuco:
    def __init__(self, m=None, n=None, checker_size=None, dictionary=None, board_config=None):
        """
        Initialize ChArUco object for marker detection.

        Parameters
        ----------
        m : int, optional
            Number of markers horizontally.
        n : int, optional
            Number of markers vertically.
        checker_size : int, optional
            Size (in millimeters) of marker.
        dictionary : str, optional
            String containing which ChArUco dictionary to use.
        board_config : dict, optional
            Dictionary containing m, n, checker_size, and dictionary.
        """
        if board_config:
            m = board_config["m"]
            n = board_config["n"]
            checker_size = board_config["checker_size"]
            dictionary = board_config["dictionary"]

        self.m=m
        self.n=n
        self.checker_size=checker_size
        
        self.aruco_dict = cv2.aruco.Dictionary_get(CHARUCO_DICTIONARY_ENUM[dictionary])
        self.charuco_board = cv2.aruco.CharucoBoard_create(self.n, self.m,
                                              self.checker_size,
                                              self.checker_size * 12 / 15,
                                              self.aruco_dict)
        
    def detect_markers(self, image):
        """
        Detect ChArUco markers on image.

        If image is a string, function opens the image using cv2.imread().

        Parameters
        ----------
        image : array_like or string
            Gray scale image. If string, call OpenCV to read image file.
            If not grayscale, convert to grayscale.

        Returns
        -------
        - list of detected ChArUco image coordinates
        - list of detected ChArUco world coordinates
        - list of detected ChArUco marker IDs
        """

        if type(image) is str:
            image = cv2.imread(image, flags=cv2.IMREAD_GRAYSCALE)
        elif len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        m_pos, m_ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict)

        if len(m_pos) > 0:
            count, c_pos, c_ids = cv2.aruco.interpolateCornersCharuco(m_pos, m_ids, image, self.charuco_board)
        else:
            count, c_pos, c_ids = 0, None, None

        if count:
            img_points = np.array(c_pos).reshape((-1, 2))
            obj_points  = self.charuco_board.chessboardCorners[c_ids].reshape((-1, 3))
            ids = c_ids.ravel()
        else:
            img_points, obj_points, ids = None, None, None

        return img_points, obj_points, ids

class Calibration:
    # TODO: implement function calls for OpenCV fisheye
    @staticmethod
    def calibrate_intrinsic(object_points: list,
                            image_points: list,
                            image_shape: tuple,
                            K=None,
                            dist_coeffs=None
                            ) -> dict:
        """
        TODO: explain function

        Parameters
        ----------
        object_points : array_like
            Array of 3D world coordinates.
        image_points : array_like
            Array of 2D image coordinates.
        image_shape : tuple
            (width, height) of images
        K : array_like, optional
            3x3 intrinsic matrix. Default is None, i.e. 
            call to OpenCV will not use any initial guess.
        dist_coeffs : array_like, optional
            Distortion coefficients. Default is None, i.e. 
            call to OpenCV will not use any initial guess.

        Returns
        -------
        dict
            Dictionary containing
            {
                'K': 3x3 intrinsic matrix,
                'dist_coeffs': distortion coefficients,
                'rvecs': , 
                'tvecs': 
            }
        """
        retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = \
            cv2.calibrateCameraExtended(object_points, image_points, image_shape, K, dist_coeffs)
        
        if not retval:
            raise RuntimeError("Intrinsic calibration failed")

        return {
            'K': cameraMatrix,
            'dist_coeffs': distCoeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'errors': perViewErrors
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
                'roi': region of interest
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
                           rvecs: np.ndarray,
                           tvecs: np.ndarray,
                           K: np.ndarray, 
                           dist_coeffs: np.ndarray
                           ) -> np.ndarray:
        """
        TODO: explain function

        Parameters
        ----------
        object_points : array_like
            Array of 3D world coordinates.
        image_points : array_like
            Array of 2D image coordinates.
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
            float error for each image
        """
        assert len(object_points)==len(image_points), \
            "Number of 3D world coordinates must equal number of 2D image coordinates"
        reprojected_points, _ = \
            cv2.projectPoints(object_points, rvecs, tvecs, K, dist_coeffs)
        
        reprojected_points = reprojected_points.reshape((object_points.shape[0], 2))
        errors = np.linalg.norm(image_points - reprojected_points, axis=1)

        return errors
    
    
    @staticmethod
    def calibrate_extrinsic(object_points: np.ndarray,
                           image_points: np.ndarray,
                           K,
                           dist_coeffs
                           ) -> dict:
        """
        TODO: function explanation

        Parameters
        ----------
        object_points : array_like
            Array of 3D world coordinates.
        image_points : array_like
            Array of 2D image coordinates.
        image_shape : tuple
            (width, height) of images
        K : array_like, optional
            3x3 intrinsic matrix. Default is None, i.e. 
            call to OpenCV will not use any initial guess.
        dist_coeffs : array_like, optional
            Distortion coefficients. Default is None, i.e. 
            call to OpenCV will not use any initial guess.

        Returns
        -------
        dict
            Dictionary containing
            {
                'rvec': 3x1 rotation vector, 
                'R': 3x3 rotation matrix (obtained with Rodrigues formula),
                'tvec': 3x1 translation vector
                'T': 3x1 origin.
            }

        """
        retval, rvec, tvec = \
            cv2.solvePnP(object_points, image_points, K, dist_coeffs)

        if not retval:
            raise RuntimeError("Extrinsic calibration failed")

        R, _ = cv2.Rodrigues(rvec)
        T = np.matmul(R.T, -tvec)
        
        return {
            'rvec': rvec,
            'R': R,
            'tvec': tvec,
            'T': T
        }
    
    @staticmethod
    def stereo_calibrate(object_points: list,
                         image_points_1: list,
                         image_points_2: list,
                         K_1: np.ndarray,
                         dist_coeffs_1: np.ndarray,
                         K_2: np.ndarray,
                         dist_coeffs_2: np.ndarray,
                         R_1: np.ndarray=np.identity(3),
                         T_1: np.ndarray=np.zeros((3,1))
                         ) -> dict:
        """
        TODO: function explanation

        Parameters
        ----------
        object_points : array_like
            Array of 3D world coordinates.
        image_point_1 : array_like
            Array of 2D image coordinates from camera 1.
        image_point_2 : array_like
            Array of 2D image coordinates from camera 2 (or projector).
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
            If given as a 3x1 vector, converts it to
            a 3x3 matrix using Rodrigues.
        T_1 : array_like, optional
            Translation vector from camera 1.
            If not given, assumes origin.

        Returns
        -------
        dict
            Dictionary containing
            {
                'R': 3x3 rotation matrix,
                'T': 3x1 translation vector
            }

        """
        retval, _, _, _, _, R, T, E, F = \
            cv2.stereoCalibrate(object_points, image_points_1, image_points_2,
                                K_1, dist_coeffs_1, K_2, dist_coeffs_2, None)

        if not retval:
            raise RuntimeError("Stereo calibration failed")

        return {
            'R': np.matmul(R_1,R),
            'T': np.matmul(R_1,T) + T_1
        }
        