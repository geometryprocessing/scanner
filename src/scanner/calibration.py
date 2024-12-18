import cv2
import numpy as np

from scanner.intersection import combine_transformations

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

        self.object_points = self.create_object_points()
            
        self.ids = np.arange(self.rows*self.columns)
        
    def create_object_points(self):
        obj_points = np.zeros((self.rows * self.columns, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:self.rows, 0:self.columns].T.reshape(-1, 2) * self.checker_size
        return obj_points

    def create_image(self, resolution: tuple):
        pass
    
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

        ret, corners = cv2.findChessboardCorners(image, (self.rows, self.columns))

        if ret:
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
            corners = cv2.cornerSubPix(image, corners, (self.rows, self.columns), (-1, -1), criteria)
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
        
        self.aruco_dict = cv2.aruco.Dictionary_get(CHARUCO_DICTIONARY_ENUM[dictionary])
        self.charuco_board = cv2.aruco.CharucoBoard_create(self.columns,
                                                           self.rows,
                                                           self.checker_size,
                                                           self.marker_size,
                                                           self.aruco_dict)
    
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
        return self.charuco_board.draw(resolution)
    
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
            count, c_pos, c_ids = 0, [], []

        if count:
            img_points = np.array(c_pos).reshape((-1, 2))
            obj_points  = self.charuco_board.chessboardCorners[c_ids].reshape((-1, 3))
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
        object_points : array_like
            Array of 3D world coordinates.
        image_points : array_like
            Array of 2D image coordinates.
        image_shape : tuple
            (width, height) of images

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

        retval, cameraMatrix, distCoeffs, = \
            cv2.initCameraMatrix2D(object_points, image_points, image_shape)
        
        if not retval:
            raise RuntimeError("Intrinsic calibration failed")

        return {
            'retval': retval,
            'K': cameraMatrix,
            'dist_coeffs': distCoeffs,
        }


    @staticmethod
    def calibrate(object_points: list,
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
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = \
            cv2.calibrateCamera(object_points, image_points, image_shape, K, dist_coeffs)
        
        if not retval:
            raise RuntimeError("Calibration failed")

        return {
            'retval': retval,
            'K': cameraMatrix,
            'dist_coeffs': distCoeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
        }
    
    @staticmethod
    def calibrate_extended(object_points: list,
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
            raise RuntimeError("Extended calibration failed")

        return {
            'retval': retval,
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
                           rvec: np.ndarray,
                           tvec: np.ndarray,
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
            cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
        
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
        
        return {
            'retval': retval,
            'rvec': rvec,
            'tvec': tvec,
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
                         T_1: np.ndarray=np.zeros((1,3))
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
                                K_1, dist_coeffs_1, K_2, dist_coeffs_2)

        if not retval:
            raise RuntimeError("Stereo calibration failed")
        
        R_combined, T_combined = combine_transformations(R_1, T_1, R, T)

        return {
            'retval': retval,
            'R': R_combined,
            'T': T_combined
        }
        