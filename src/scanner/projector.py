import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils.three_d_utils import ThreeDUtils
from src.utils.file_io import save_json, load_json, get_all_paths
from src.utils.plotter import Plotter
from src.scanner.calibration import Charuco, CheckerBoard, Calibration
from src.scanner.camera import Camera


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
        self.width  = None
        self.height = None
        # accompanying camera
        self.camera = Camera()
        # intrinsic
        self.K           = None
        self.dist_coeffs = None
        self.rvecs       = np.zeros(shape=(3,1))
        self.tvecs       = np.zeros(shape=(3,1))
        # extrinsic
        self.R = np.identity(3)        # projector is initialized at origin
        self.T = np.zeros(shape=(3,1)) # projector is initialized at origin
        # calibration utils
        self.images               = []               # image paths
        self.discarded_images     = set()
        self.image_points         = []
        self.object_points        = []
        self.camera_image_points  = []  # camera points will be used for extrinsic calibration
        self.camera_object_points = []  # camera points will be used for extrinsic calibration
        self.planes               = []
        self.errors               = []
        self.plane_pattern        = None
        self.calibration_pattern  = None
        self.calibration_image    = None
        self.error_thr            = None
        self.min_points           = 4
        # self.max_planes = 0 # TODO: discard

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
    def set_projector_height(self, height: int):
        """
        Set projector resolution height in pixels.

        Parameters
        ----------
        height : int
            Image resolution height in pixels.
        """
        assert height > 0, "Incorrect value for height, has to be nonnegative"
        self.height = int(height)
    def set_projector_width(self, width: int):
        """
        Set projector image resolution width in pixels.

        Parameters
        ----------
        width : int
            Image resolution width in pixels.
        """
        self.width = int(width)
    def set_projector_shape(self, shape: tuple[int, int]):
        """
        Set image resolution in pixels.
        Both numbers have to be integers and nonnegative.

        Parameters
        ----------
        shape : tuple
            Image resolution in (width: int, height: int).
        """
        self.set_projector_width (shape[0])
        self.set_projector_height(shape[1])
    def discard_intrinsic_images(self):
        self.images = [image for image in self.images if image not in self.discarded_images]
    

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
        Returns camera image resolution in pixels as (height, width).

        Returns
        -------
        width 
            camera width resolution in pixels.
        height
            camera height resolution in pixels
        """
        return (self.camera.width, self.camera.height)
    def get_projector_shape(self):
        """
        Returns projector image resolution in pixels as (width, height).

        Returns
        -------
        height
            projector height resolution in pixels
        width 
            projector width resolution in pixels.
        """
        return (self.width, self.height)

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
        R_combined, T_combined = ThreeDUtils.combine_transformations(self.camera.R, self.camera.T, R, T)
        # NOTE: since obj_points is of shape (Nx3), the matrix multiplication with rotation 
        # has to be written as (R @ obj_points.T).T
        # to simplify:
        # np.matmul(R_combined, obj_points.T).T = np.matmul(obj_points, R_combined.T)
        world_points = np.matmul(obj_points, R_combined.T) + T_combined.reshape((1,3))

        # fit plane
        return ThreeDUtils.fit_plane(world_points)
    
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
                self.discarded_images.add(image_path)
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
                self.discarded_images.add(image_path)
                continue

            # undistort pixel coordinates -> normalized coordinates
            # project normalized coordinates onto Plane - X_3D
            origin, camera_rays = ThreeDUtils.camera_to_ray_world(cam_img_points,
                                              rvec,
                                              tvec,
                                              self.camera.K,
                                              self.camera.dist_coeffs)
            # opencv calibration only works with PLANAR data, but where we are moving our
            # plane pattern board around and retrieving the 3D world coordinates
            # FIX THIS, OTHERWISE CANNOT RUN PROJECTOR CALIBRATION AS IS
            objs = ThreeDUtils.intersect_line_with_plane(origin,
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
            R_combined, T_combined = ThreeDUtils.combine_transformations(self.camera.R, self.camera.T, R, T)
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
        assert self.height is not None or self.width is not None, \
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
            # "scaling_factor": self.scaling_factor,
            # "newK": self.newK,
            # "roi": self.roi,
            "R": self.R,
            "T": self.T,
            "height": self.height,
            "width": self.width,
            "error": self.get_mean_error(),
            "error_threshold": self.error_thr
        }, filename)

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
        if isinstance(config, str):
            config = load_json(config)

        self.set_image_paths(config['image_folder_path'])
        self.set_projector_height(config['height'])
        self.set_projector_width(config['width'])
        self.load_camera(config['camera_calibration'])
        self.set_plane_pattern(config['aruco_dict'])
        self.set_calibration_pattern(config['checkerboard_dict'])
        self.set_error_threshold(config['error_thr'])
        self.set_min_points(config['min_points'])

        self.calibrate_intrinsic()
        self.calibrate_extrinsics()
        
        self.save_calibration(config['output_filename'])