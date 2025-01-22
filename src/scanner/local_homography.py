import cv2
import numpy as np
import os

from src.utils.file_io import save_json, load_json, get_all_paths
from src.utils.image_utils import ImageUtils
from src.scanner.calibration import Charuco, CheckerBoard, Calibration
from src.reconstruction.structured_light import StructuredLight

class LocalHomographyCalibration:
    def __init__(self):
        self.plane_pattern    = None

        self.structured_light         = StructuredLight()
        self.structured_light.set_pattern('gray')

        self.calibration_directory    = None
        self.num_directories          = 0
        # this is determined from necessary images of gray patterns for the projector resolution
        self.num_vertical_images      = 0 
        self.num_horizontal_images    = 0 
        
        self.window_size = 30

        self.index_x                = []
        self.index_y                = []
        self.object_points          = []
        self.camera_image_points    = []
        self.projector_image_points = []

        self.camera_K              = None
        self.camera_dist_coeffs    = None
        self.camera_width          = None
        self.camera_height         = None
        self.projector_K           = None
        self.projector_dist_coeffs = None
        self.projector_width       = None
        self.projector_height      = None

    # setters
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
    def set_calibration_directory(self, path: str):
        """
        TODO: collect the number of directories

        Parameters
        ----------
        path : str
            path to directory containing multiple calibration folders,
            each with the same structured light capture of a calibration board
        """
        assert os.path.isdir(path), "This is not a directory. This function only works with a folder."
        dirs = [f.path for f in os.scandir(path) if f.is_dir()]
        self.calibration_directory = [get_all_paths(dir) for dir in dirs]
    def set_camera_height(self, height: int):
        """
        Set image resolution height in pixels.

        Parameters
        ----------
        height : int
            Image resolution height in pixels.
        """
        assert height > 0, "Incorrect value for height, has to be nonnegative"
        self.camera_height = int(height)
    def set_camera_width(self, width: int):
        """
        Set image resolution width in pixels.

        Parameters
        ----------
        width : int
            Image resolution width in pixels.
        """
        assert width > 0, "Incorrect value for width, has to be nonnegative"
        self.camera_width = int(width)
    def set_camera_shape(self, shape: tuple[int, int]):
        """
        Set image resolution in pixels.
        Both numbers have to be integers and nonnegative.

        Parameters
        ----------
        shape : tuple
            Image resolution in (width: int, height: int).
        """
        self.set_camera_width (shape[0])
        self.set_camera_height(shape[1])
    def set_projector_height(self, height: int):
        """
        Set projector resolution height in pixels.

        Parameters
        ----------
        height : int
            Image resolution height in pixels.
        """
        assert height > 0, "Incorrect value for height, has to be nonnegative"
        self.projector_height = int(height)
    def set_projector_width(self, width: int):
        """
        Set projector resolution width in pixels.

        Parameters
        ----------
        width : int
            Image resolution width in pixels.
        """
        assert width > 0, "Incorrect value for width, has to be nonnegative"
        self.projector_width = int(width)
    def set_projector_shape(self, shape: tuple[int, int]):
        """
        Set projector resolution in pixels.
        Both numbers have to be integers and nonnegative.

        This will determine the number of images needed from the gray structured light pattern.
        num_images = 4 * (round(log2(max_resolution))) + 2

        Parameters
        ----------
        shape : tuple
            Image resolution in (width: int, height: int).
        """
        self.set_projector_width (shape[0])
        self.set_projector_height(shape[1])
        self.num_vertical_images = 2*int(np.ceil(np.log2(shape[0])))
        self.num_horizontal_images = 2*int(np.ceil(np.log2(shape[1])))
    def set_window_size(self, window: int):
        """
        Set the window (in pixels) of the neighborhood of each calibration corner to calculate the Homography.
        """
        self.window_size = window

    # gettters
    def get_camera_shape(self) -> tuple[int, int]:
        """
        Returns image resolution in pixels as (height, width).

        Returns
        -------
        height
            camera height resolution in pixels
        width 
            camera width resolution in pixels.
        """
        return (self.camera_width, self.camera_height)
    def get_projector_shape(self) -> tuple[int, int]:
        """
        Returns projector resolution in pixels as (height, width).

        Returns
        -------
        height
            projector height resolution in pixels
        width 
            projector width resolution in pixels.
        """
        return (self.projector_width, self.projector_height)

    # functions
    def decode(self):
        """
        
        """
        for folder in self.calibration_directory:
            self.structured_light.set_vertical_pattern_image_paths(folder[slice(2, 2 + self.num_vertical_images, 2)])
            self.structured_light.set_inverse_vertical_image_paths(folder[slice(3, 2 + self.num_vertical_images, 2)])
            self.structured_light.set_horizontal_pattern_image_paths(folder[slice(2 + self.num_vertical_images, 2 + self.num_vertical_images + self.num_horizontal_images, 2)])
            self.structured_light.set_inverse_horizontal_image_paths(folder[slice(2 + self.num_vertical_images + 1, 2 + self.num_vertical_images + self.num_horizontal_images, 2)])
            self.structured_light.decode()
            self.index_x.append(self.structured_light.index_x)
            self.index_y.append(self.structured_light.index_y)

    def detect_markers_and_homographies(self):
        for index_x, index_y, folder in zip(self.index_x, self.index_y, self.calibration_directory):
            white_image = folder[0]
            img_points, obj_points, _ = self.plane_pattern.detect_markers(white_image)
            proj_img_points = np.empty_like(img_points)
        
            # this happens PER NEIGHBORHOOD, hence the for loop
            for idx,  camera_image_point in enumerate(img_points):

                minX = round(camera_image_point[0]-self.window_size)
                maxX = round(camera_image_point[0]+self.window_size)
                minY = round(camera_image_point[1]-self.window_size)
                maxY = round(camera_image_point[1]+self.window_size)

                campixels_x, campixels_y = np.meshgrid(np.arange(minX, maxX),
                                                       np.arange(minY, maxY))
                image_points = np.stack([campixels_x, campixels_y], axis=-1).reshape((-1,2))
                proj_points = np.stack([index_x[minY:maxY, minX:maxX],
                                        index_y[minY:maxY, minX:maxX],
                                        ], axis=-1).reshape((-1,2))
            
                # image_points is the list of points in the neighborhood seen by the camera
                # proj_points is the list of points in the neighborhood that we extract from the index_x index_y after decoding
                H, mask = cv2.findHomography(image_points, proj_points)

                # p is corner in camera image
                p = ImageUtils.homogeneous_coordinates(camera_image_point) # I think it's supposed to be camera_image_point
                Q = np.matmul(p, H.T)
                # q is corner in projector image
                q = Q[:,0:-1] / Q[:,-1]
                proj_img_points[idx] = q

            self.camera_image_points.append(img_points)
            self.object_points.append(obj_points)
            self.projector_image_points.append(proj_img_points)

    def findHomographies(self):
        """
        TODO: discard function
        """
        # clear projector image points
        self.projector_image_points = np.empty_like(self.camera_image_points)
        
        # this happens PER NEIGHBORHOOD, hence the for loop
        for idx, camera_image_point in enumerate(self.camera_image_points):

            minY = round(camera_image_point[1]-self.window_size)
            maxY = round(camera_image_point[1]+self.window_size)
            minX = round(camera_image_point[0]-self.window_size)
            maxX = round(camera_image_point[0]+self.window_size)

            campixels_x, campixels_y = np.meshgrid(np.arange(minX, maxX),
                                                   np.arange(minY, maxY))
            image_points = np.stack([campixels_x, campixels_y], axis=-1).reshape((-1,2))
            proj_points = np.stack([self.index_x[minY:maxY, minX:maxX],
                                    self.index_y[minY:maxY, minX:maxX],
                                    ], axis=-1).reshape((-1,2))
        
            # image_points is the list of points in the neighborhood seen by the camera
            # proj_points is the list of points in the neighborhood that we extract from the index_x index_y after decoding
            H, mask = cv2.findHomography(image_points, proj_points)

            # p is corner in camera image
            p = ImageUtils.homogeneous_coordinates(camera_image_point) # I think it's supposed to be camera_image_point
            Q = np.matmul(p, H.T)
            # q is corner in projector image
            q = Q[:,0:-1] / Q[:,-1]
            self.projector_image_points[idx] = q

    def calibrate_stereo_system(self):
        cam_result = Calibration.calibrate(self.object_points, self.camera_image_points, self.get_camera_shape())
        proj_result = Calibration.calibrate(self.object_points, self.projector_image_points, self.get_projector_shape())

        self.camera_K = cam_result['K']
        self.camera_dist_coeffs = cam_result['dist_coeffs']
        self.camera_error = cam_result['rms']
        self.projector_K = proj_result['K']
        self.projector_dist_coeffs = proj_result['dist_coeffs']
        self.projector_error = proj_result['rms']

        stereo_result = Calibration.stereo_calibrate(self.object_points, self.camera_image_points, self.projector_image_points, self.get_camera_shape(), self.camera_K, self.camera_dist_coeffs, self.projector_K, self.projector_dist_coeffs)

        self.projector_R = stereo_result['R']
        self.projector_T = stereo_result['T']
        self.stereo_error = stereo_result['rms']

    def save_calibration(self):
        """
        Save calibration into a JSON file.

        Parameters
        ----------
        filename : str
            path to JSON file where calibration will be saved.  
        """
        save_json({
            "camera_K": self.camera_K,
            "camera_dist_coeffs": self.camera_dist_coeffs,
            "projector_K": self.projector_K,
            "projector_dist_coeffs": self.projector_dist_coeffs,
            "projector_R": self.projector_R,
            "projector_T": self.projector_T,
            "camera_error": self.camera_error,
            "projector_error": self.projector_error,
            "stereo_error": self.stereo_error
        }, os.path.join(self.calibration_directory, 'calibration.json'))

    def run(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        self.set_camera_shape(config['camera_shape'])
        self.set_projector_shape(config['projector_shape'])
        self.set_plane_pattern(config['plane_pattern'])
        self.set_calibration_directory(config['calibration_directory'])

        self.decode()
        self.detect_markers_and_homographies()

        self.calibrate_stereo_system()
        self.save_calibration()