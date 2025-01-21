import cv2
import numpy as np
import structuredlight as sl
import os

from src.utils.three_d_utils import ThreeDUtils
from src.utils.file_io import save_json, load_json, get_all_paths
from src.utils.image_utils import ImageUtils
from src.utils.plotter import Plotter
from src.scanner.camera import Camera
from src.scanner.projector import Projector

class StructuredLight:
    def __init__(self):
        self.projector = Projector()
        self.camera    = Camera()
        self.pattern   = None

        # image paths
        self.horizontal_images         = []
        self.vertical_images           = []
        self.inverse_horizontal_images = []
        self.inverse_vertical_images   = []

        self.black_image = None
        self.white_image = None

        # reconstruction utils
        self.thr              = None
        self.index_x          = None
        self.index_y          = None
        self.minimum_contrast = 0.1
        self.mask             = None
        # reconstruction
        self.point_cloud = None
        self.depth_map   = None
        self.colors      = None
        self.normals     = None

    # setters
    def set_pattern(self, pattern: str):
        """
        Currently supported structured light patterns: Gray, Binary, and XOR.
        """
        match pattern.lower():
            case 'gray':
                self.pattern = sl.Gray()
            case 'binary':
                self.pattern = sl.Binary()
            case 'bin':
                self.pattern = sl.Binary()
            case 'xor':
                self.pattern = sl.XOR()
            case _:
                print("Unrecognized structured light pattern type, defaulting to gray...\n")
                self.pattern = sl.Gray()
    def set_horizontal_pattern_image_paths(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for decoding horizontal structured light patterns.
        """
        self.horizontal_images = get_all_paths(image_paths)
    def set_vertical_pattern_image_paths(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for decoding vertical structure light patterns.
        """
        self.vertical_images = get_all_paths(image_paths)
    def set_inverse_horizontal_image_paths(self, image_paths: list | str):
        """
        Inverse horizontal structured light pattern images will be used for 
        setting a threshold value for pixel intensity when decoding horizontal patterns.
        Read the structuredlight docs for more information:
        https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image

        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
        """
        self.inverse_horizontal_images = get_all_paths(image_paths)
    def set_inverse_vertical_image_paths(self, image_paths: list | str):
        """
        Inverse vertical structured light pattern images will be used for 
        setting a threshold value for pixel intensity when decoding vertical patterns.
        Read the structuredlight docs for more information:
        https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image

        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
        """
        self.inverse_vertical_images = get_all_paths(image_paths)
    def set_white_pattern_image(self, image_path: str):
        """
        Set the path of captured image of scene with an all-white pattern projected.
        The white image can be used to extract colors for the reconstructed point cloud.
        The white image can also be used for setting a threshold value when decoding
        structured light patterns.
        Read the structuredlight docs for more information:
        https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image

        Parameters
        ----------
        image_path : str
            path to white pattern image

        Notes
        -----
        This has to be used in combination with set_black_pattern_image, since the 
        threshold for ON or OFF will be set per pixel as:
        thr = 0.5 * white_pattern_image + 0.5 * black_pattern_image.
        If negative/inverse structured light patterns are passed,
        black and white pattern images will be ignored for threshold setting.
        """
        self.white_image = image_path
    def set_black_pattern_image(self, image_path: str):
        """
        Set the path of captured image of scene with an all-black pattern projected.
        The black image can be used to extract colors for the reconstructed point cloud.
        The black image can also be used for setting a threshold value when decoding
        structured light patterns.
        Read the structuredlight docs for more information:
        https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image

        Parameters
        ----------
        image_path : str
            path to black pattern image

        Notes
        -----
        This has to be used in combination with set_white_pattern_image, since the 
        threshold for ON or OFF will be set per pixel as:
        thr = 0.5 * white_pattern_image + 0.5 * black_pattern_image.
        If negative/inverse structured light patterns are passed,
        black and white pattern images will be ignored for threshold setting.
        """
        self.black_image = image_path
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
    def set_projector(self, projector: str | dict |Projector):
        # if string, load the json file and get the parameters
        # check if dict, get the parameters
        if isinstance(projector, str) or isinstance(projector, dict):
            self.projector.load_calibration(projector)
        else:
            self.projector = projector
    def set_camera(self, camera: str | dict | Camera):
        # if string, load the json file and get the parameters
        # check if dict, get the parameters
        if isinstance(camera, str) or isinstance(camera, dict):
            self.camera.load_calibration(camera)
        else:
            self.camera = camera
    def set_minimum_contrast(self, contrast: float):
        """
        Set minimum contrast float value which can be used to generate a mask.
        This will be used with the black and white pattern images to create a
        per-pixel mask of pixels that pass the test
        white-black > max_pixel * minimum_contrast,
        where max_pixel is the maximum intensity a pixel can be assigned (that's 255 for uint8).
        If the pixel does not pass that test, it will get masked out.
        """
        self.minimum_contrast = max(min(contrast, 1.), 0.)
    def set_mask(self, mask: np.ndarray):
        """
        """
        self.mask = mask

    # getters
    def get_minimum_contrast(self):
        return self.minimum_contrast
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

    # function
    def generate_mask(self):
        if self.mask is not None:
            print("External mask provided")
            return
        
        if self.white_image is None or self.black_image is None:
            cam_resolution = self.camera.get_image_shape()
            self.mask = np.full(cam_resolution, True)
            return

        white = ImageUtils.load_ldr(self.white_image, make_gray=True)
        black = ImageUtils.load_ldr(self.black_image, make_gray=True)
        data_type = white.dtype
        m = np.iinfo(data_type).max if data_type.kind in 'iu' else np.finfo(data_type).max
        self.mask = (abs(white-black) > m*self.minimum_contrast)

    def decode(self):
        """
        
        """
        assert self.pattern is not None, "Structured light pattern not specified"

        thresh = self.thr
        if self.white_image and self.black_image:
            img_white = ImageUtils.load_ldr(self.white_image, make_gray=True)
            img_black = ImageUtils.load_ldr(self.black_image, make_gray=True)
            thresh = 0.5*img_white + 0.5*img_black

        if len(self.horizontal_images) > 0:
            gray_horizontal = [ImageUtils.load_ldr(img, make_gray=True) for img in self.horizontal_images]
            if self.inverse_horizontal_images:
                assert len(self.inverse_horizontal_images) == len(self.horizontal_images), \
                    "Mismatch between number of horizontal patterns \
                        and inverse horizontal patterns. Must be the same"
                horizontal_second_argument = [ImageUtils.load_ldr(img, make_gray=True) 
                                              for img in self.inverse_horizontal_images]
            else:
                horizontal_second_argument = thresh
            self.index_y = self.pattern.decode(gray_horizontal, horizontal_second_argument)

        if len(self.vertical_images) > 0:
            gray_vertical = [ImageUtils.load_ldr(img, make_gray=True) for img in self.vertical_images]
            if self.inverse_vertical_images:
                assert len(self.inverse_vertical_images) == len(self.vertical_images), \
                    "Mismatch between number of horizontal patterns \
                        and inverse horizontal patterns. Must be the same"
                vertical_second_argument = [ImageUtils.load_ldr(img, make_gray=True)
                                            for img in self.inverse_vertical_images]
            else:
                vertical_second_argument = thresh
            self.index_x = self.pattern.decode(gray_vertical, vertical_second_argument)

    def plot_decoding(self):
        """
        
        """
        Plotter.plot_decoding(self.camera.get_image_shape(),
                              self.index_x,
                              self.index_y)

    def reconstruct(self):
        """
        Take correspondence indices between camera and projector and use
        ray intersection to triangulate points in 3D.

        If decoding was done for both x and y coordinates, this functions does a
        direct triangulation of the projector pixels and camera pixels.
        Otherwise, it does a plane-line intersection to find the 3D points in 
        world coordinates. 
        """
        assert self.camera.K is not None, "No camera defined"
        assert self.projector.K is not None, "No projector defined"
        assert self.index_x is not None or self.index_y is not None, \
            "Decoding function call missing"
        
        if self.mask is None:
            self.generate_mask()

        cam_resolution = self.camera.get_image_shape()
        campixels_x, campixels_y = np.meshgrid(np.arange(cam_resolution[1]),
                            np.arange(cam_resolution[0]))
        
        campixels = np.stack([campixels_x, campixels_y], axis=-1)[self.mask].reshape((-1,2))

        if self.index_x is not None and self.index_y is not None:
            projpixels = np.stack([self.index_x, self.index_y], axis=-1)[self.mask].reshape((-1,2))
            point_cloud = ThreeDUtils.triangulate_pixels(
                campixels,
                self.camera.K,
                self.camera.dist_coeffs,
                self.camera.R,
                self.camera.T,
                projpixels,
                self.projector.K,
                self.projector.dist_coeffs,
                self.projector.R,
                self.projector.T
            )
        elif self.index_x is not None:
            point_cloud = ThreeDUtils.intersect_pixels(
                campixels,
                self.camera.K,
                self.camera.dist_coeffs,
                self.camera.R,
                self.camera.T,
                self.index_x[self.mask],
                self.projector.get_projector_shape(),
                self.projector.K,
                self.projector.dist_coeffs,
                self.projector.R,
                self.projector.T,
                index = 'x'
            )
        elif self.index_y is not None:
            point_cloud = ThreeDUtils.intersect_pixels(
                campixels,
                self.camera.K,
                self.camera.dist_coeffs,
                self.camera.R,
                self.camera.T,
                self.index_y[self.mask],
                self.projector.get_projector_shape(),
                self.projector.K,
                self.projector.dist_coeffs,
                self.projector.R,
                self.projector.T,
                index = 'y'
            )

        self.point_cloud = point_cloud
    
    def extract_colors(self):
        """
        Use black and white pattern images to extract the RGB values from the scene.
        If save_point_cloud_as_ply is called and colors have been extracted, the
        point cloud will be saved with the color values.
        
        Notes
        -----
        colors will be stored in [0., 1.[ range as a numpy array of type np.float32
        """
        assert self.white_image is not None \
              and self.black_image is not None, "Need to set both black and white images"
        
        if self.mask is None:
            self.generate_mask()

        img = ImageUtils.load_ldr(self.white_image)
        # clip RGB range to [0., 1.[
        data_type = img.dtype
        m = np.iinfo(data_type).max if data_type.kind in 'iu' else np.finfo(data_type).max

        self.colors = img[self.mask] / m

    def extract_depth_map(self):
        assert self.point_cloud is not None, "No reconstruction yet"
        self.depth_map = \
            ThreeDUtils.depth_map_from_point_cloud(self.point_cloud,
                                                   self.mask,
                                                   ThreeDUtils.get_origin(self.camera.R, self.camera.T))
    
    def extract_normals(self):
        assert self.depth_map is not None or self.point_cloud is not None, "No reconstruction yet"
        if self.depth_map is not None:
            self.normals = ThreeDUtils.normals_from_depth_map(self.depth_map)
        elif self.point_cloud is not None:
            self.normals = ThreeDUtils.normals_from_point_cloud(self.point_cloud)

    def save_point_cloud_as_ply(self, filename: str):
        assert self.point_cloud is not None, "No reconstruction yet"
        ThreeDUtils.save_ply(filename, self.point_cloud, self.normals, self.colors.reshape((-1,3)))

    def plot_normal_map(self, figsize: tuple=(12,16), filename: str=None):
        Plotter.plot_normal_map(self.normals, self.mask, figsize=figsize, filename=filename)

    def plot_depth_map(self,
                       cmap: str='turbo',
                       max_percentile: int=95,
                       min_percentile: int=5,
                       figsize: tuple=(12,16),
                       filename: str=None):
        Plotter.plot_depth_map(self.depth_map,
                               cmap=cmap,
                               max_percentile=max_percentile,
                               min_percentile=min_percentile,
                               figsize=figsize,
                               filename=filename)

    def run(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        self.set_black_pattern_image(config['black_image'])
        self.set_white_pattern_image(config['white_image'])
        self.set_threshold(config['threshold'])
        self.decode()
        self.reconstruct()
        self.extract_colors()
        self.extract_normals()
        self.save_point_cloud_as_ply(config['ply_filename'])