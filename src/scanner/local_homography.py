import cv2
import numpy as np

from src.utils.file_io import save_json
from src.scanner.calibration import Charuco, CheckerBoard, Calibration
from src.reconstruction.structured_light import StructuredLight

class LocalHomographyCalibration:
    def __init__(self):
        self.structured_light = StructuredLight()
        self.plane_pattern    = None

        self.window_size = 30
        
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
        if type(pattern) is dict:
            try: 
                if pattern["type"] == "charuco":
                    pattern = Charuco(board_config=pattern)
                elif pattern["type"] == "checkerboard":
                    pattern = CheckerBoard(board_config=pattern)
            except:
                pattern = Charuco(board_config=pattern)
        self.plane_pattern = pattern
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
            Image resolution in (height: int, width: int).
        """
        self.set_camera_height(shape[0])
        self.set_camera_width (shape[1])
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

        Parameters
        ----------
        shape : tuple
            Image resolution in (height: int, width: int).
        """
        self.set_projector_height(shape[0])
        self.set_projector_width (shape[1])
    # def set_structured_light_pattern(self, pattern: str):
    #     """
    #     Currently supported structured light patterns: Gray, Binary, and XOR.
    #     """
    #     self.structured_light.set_pattern(pattern)
    # def set_horizontal_pattern_image_paths(self, image_paths: list | str):
    #     """
    #     Parameters
    #     ----------
    #     image_paths : list | string
    #         List of strings containing image paths (or string for folder containing images).
    #         These images will be used for decoding horizontal structured light patterns.
    #     """
    #     self.structured_light.set_horizontal_pattern_image_paths(image_paths)
    # def set_vertical_pattern_image_paths(self, image_paths: list | str):
    #     """
    #     Parameters
    #     ----------
    #     image_paths : list | string
    #         List of strings containing image paths (or string for folder containing images).
    #         These images will be used for decoding vertical structure light patterns.
    #     """
    #     self.structured_light.set_vertical_pattern_image_paths(image_paths)
    # def set_inverse_horizontal_image_paths(self, image_paths: list | str):
    #     """
    #     Inverse horizontal structured light pattern images will be used for 
    #     setting a threshold value for pixel intensity when decoding horizontal patterns.
    #     Read the self.structured_light docs for more information:
    #     https://github.com/elerac/self.structured_light/wiki#how-to-binarize-a-grayscale-image

    #     Parameters
    #     ----------
    #     image_paths : list | string
    #         List of strings containing image paths (or string for folder containing images).
    #     """
    #     self.structured_light.set_inverse_horizontal_image_paths(image_paths)
    # def set_inverse_vertical_image_paths(self, image_paths: list | str):
    #     """
    #     Inverse vertical structured light pattern images will be used for 
    #     setting a threshold value for pixel intensity when decoding vertical patterns.
    #     Read the self.structured_light docs for more information:
    #     https://github.com/elerac/self.structured_light/wiki#how-to-binarize-a-grayscale-image

    #     Parameters
    #     ----------
    #     image_paths : list | string
    #         List of strings containing image paths (or string for folder containing images).
    #     """
    #     self.structured_light.set_inverse_vertical_image_paths(image_paths)
    def set_white_pattern_image(self, image_path: str):
        """
        Set the path of captured image of scene with an all-white pattern projected.
        The white image will be used to extract the calibration pattern corners.

        Parameters
        ----------
        image_path : str
            path to white pattern image
        """
        self.white_image = image_path
    # def set_black_pattern_image(self, image_path: str):
    #     """
    #     Set the path of captured image of scene with an all-black pattern projected.
    #     The black image can be used to extract colors for the reconstructed point cloud.
    #     The black image can also be used for setting a threshold value when decoding
    #     structured light patterns.
    #     Read the self.structured_light docs for more information:
    #     https://github.com/elerac/self.structured_light/wiki#how-to-binarize-a-grayscale-image

    #     Parameters
    #     ----------
    #     image_path : str
    #         path to black pattern image

    #     Notes
    #     -----
    #     This has to be used in combination with set_white_pattern_image, since the 
    #     threshold for ON or OFF will be set per pixel as:
    #     thr = 0.5 * white_pattern_image + 0.5 * black_pattern_image.
    #     If negative/inverse structured light patterns are passed,
    #     black and white pattern images will be ignored for threshold setting.
    #     """
    #     self.structured_light.set_black_pattern_image(image_path)
    # def set_threshold(self, thr: float):
    #     """
    #     Set the threshold value for considering a pixel ON or OFF
    #     based on its intensity.

    #     Parameters
    #     ----------
    #     thr : float
    #         threshold value for considering a pixel ON or OFF

    #     Notes
    #     -----
    #     If black and white pattern images are set, or if negative/inverse
    #     patterns are passed, this threshold value will be ignored.
    #     """
    #     self.structured_light.set_threshold(thr)
    # def set_minimum_contrast(self, contrast: float):
    #     """
    #     Set minimum contrast float value which can be used to generate a mask.
    #     This will be used with the black and white pattern images to create a
    #     per-pixel mask of pixels that pass the test
    #     white-black > max_pixel * minimum_contrast,
    #     where max_pixel is the maximum intensity a pixel can be assigned (that's 255 for uint8).
    #     If the pixel does not pass that test, it will get masked out.
    #     """
    #     self.structured_light.set_minimum_contrast(contrast)
    # def set_mask(self, mask: np.ndarray):
    #     """
    #     """
    #     self.mask = mask

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
        return (self.camera_height, self.camera_width)
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
        return (self.projector_height, self.projector_width)

    # functions
    def decode(self):
        """
        
        """
        self.structured_light.decode()
        self.index_x = self.structured_light.index_x
        self.index_y = self.structured_light.index_y

    def detect_markers(self):
        img_points, obj_points, _ = self.calibration_pattern.detect_markers(self.white_image)
        self.camera_image_points = img_points
        self.object_points = obj_points

    def findHomography(self):
        # this happens PER NEIGHBORHOOD, hence the for loop
        for i in range(count):
        
            # image_points is the list of points in the neighborhood seen by the camera
            # proj_points is the list of points in the neighborhood that we extract from the index_x index_y after decoding
            H = cv2.findHomography(image_points, proj_points)
            
            # p is corner in camera image
            Q = np.matmul(H, [p[0], p[1], 1.0])
            # q is corner in projector image
            q = Q[0:2] / Q[2]
            self.projector_image_points.append(q)
        

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

    def save_calibration(self, filename: str):
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
        }, filename)

    def run(self):
        pass