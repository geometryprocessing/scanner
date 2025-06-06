import argparse
import cv2
import numpy as np
import structuredlight
import hilbert
import scipy
import os

from src.utils.three_d_utils import ThreeDUtils
from src.utils.file_io import save_json, load_json, get_all_paths
from src.utils.image_utils import ImageUtils
from src.utils.plotter import Plotter
from src.scanner.camera import Camera
from src.scanner.projector import Projector

class StructuredLight:
    def __init__(self, config: dict | str = None):
        self.projector = Projector()
        self.camera = Camera()

        self.structure_grammar = {}

        # reconstruction utils
        self.index_x = None
        self.index_y = None
        self.mask = None
        # reconstruction outpus
        self.point_cloud = None
        self.depth_map = None
        self.colors = None
        self.normals = None

        if config is not None:
            self.load_config(config)

    def load_config(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)
        self.set_reconstruction_directory(config['structured_light']['reconstruction_directory'])
        self.set_structure_grammar(config['structured_light']['structure_grammar'])
        self.set_minimum_contrast(config['structured_light']['minimum_contrast'])
        self.set_camera(config['structured_light']['camera_calibration'])
        self.set_projector(config['structured_light']['projector_calibration'])
        self.set_outputs(config['structured_light']['outputs'])

    def set_reconstruction_directory(self, path: str):
        """

        Parameters
        ----------
        path : str
            path to directory containing multiple scene folders,
            each with the images that match the "utils" and the "tables" images
            for LookUp reconstruction
        """
        assert os.path.isdir(path), "This is not a directory. This function only works with a folder."
        self.reconstruction_directory = os.path.abspath(path)

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

    def set_structure_grammar(self, structure_grammar: dict):
        """
        The structure grammar is the configuration to read images
        and save them to pattern/look up tables accordingly.
        Example below:
            structure_grammar = {
                "name": "gray",
                "images": ["img_02.tiff", "img_04.tiff", "img_06.tiff"],
                "utils": {
                    "white": "white.tiff", (or "green.tiff" if monochormatic, for instance)
                    "black": "black.tiff",
                }
            }
        The list of strings are the images which will be used
        to create a look up table with the key name.
        """
        self.structure_grammar = structure_grammar

    def set_outputs(self, outputs: dict):
        """
        Parameters
        ----------
        outputs : dict
            Dictionary containing which outputs to save.
            outputs: {
                "depth_map": True,
                "point_cloud": True
            }
            Each can be set to True or False

        Notes: 
        - depth_map is a HxW numpy array containing depth per pixel of the result.
        - point_cloud is a 3D numpy array of the result.
        """
        self.outputs = outputs

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
        
        white = None if 'white' not in self.structure_grammar else self.structure_grammar['white']
        black = None if 'black' not in self.structure_grammar else self.structure_grammar['black']
        
        if white is None or black is None:
            cam_width, cam_height = self.camera.get_image_shape()
            self.mask = np.full(shape=(cam_height, cam_width), fill_value=True)
            return

        white = ImageUtils.load_ldr(os.path.join(self.reconstruction_directory, white), make_gray=True)
        black = ImageUtils.load_ldr(os.path.join(self.reconstruction_directory, black), make_gray=True)
        data_type = white.dtype
        m = np.iinfo(data_type).max if data_type.kind in 'iu' else np.finfo(data_type).max
        self.mask = (abs(white-black) > m*self.minimum_contrast)

    def decode(self):
        """
        
        """
        pattern = self.structure_grammar['pattern']
        if pattern == 'gray' or pattern == 'binary' or pattern == 'xor':
            # handle structure grammar for Gray, Binary, and XOR types
            vert = None if 'vertical_images' not in self.structure_grammar \
                else [os.path.join(self.reconstruction_directory, img) \
                      for img in self.structure_grammar['vertical_images']]
            
            horz = None if 'horizontal_images' not in self.structure_grammar \
                else [os.path.join(self.reconstruction_directory, img) \
                      for img in self.structure_grammar['horizontal_images']]
            
            inv_vert = None if 'inverse_vertical_images' not in self.structure_grammar \
                else [os.path.join(self.reconstruction_directory, img) \
                      for img in self.structure_grammar['inverse_vertical_images']]
            
            inv_horz = None if 'inverse_horizontal_images' not in self.structure_grammar \
                else [os.path.join(self.reconstruction_directory, img) \
                      for img in self.structure_grammar['inverse_horizontal_images']]
            
            white = None if 'white' not in self.structure_grammar \
                else os.path.join(self.reconstruction_directory, self.structure_grammar['white'])
            black = None if 'black' not in self.structure_grammar \
                else os.path.join(self.reconstruction_directory, self.structure_grammar['black'])
            thr = None if 'threshold' not in self.structure_grammar \
                else self.structure_grammar['threshold']
            
            self.index_x, self.index_y = StructuredLight.decode_gray_binary_xor(pattern,
                                                                                vert,
                                                                                inv_vert,
                                                                                horz,
                                                                                inv_horz,
                                                                                white,
                                                                                black,
                                                                                thr)
        elif pattern == 'hilbert':
            # handle structure grammar for Hilbert
            self.index_x = StructuredLight.decode_hilbert([os.path.join(self.reconstruction_directory, img) 
                                            for img in self.structure_grammar['images']],
                                            self.structure_grammar['num_bits'])
        elif pattern == 'phaseshift':
            # handle structure grammar for Phase Shift
            F = 1.0 if 'F' not in self.structure_grammar else self.structure_grammar['F']
            self.index_x = StructuredLight.decode_phaseshift(self.projector.width,
                                              [os.path.join(self.reconstruction_directory, img) 
                                               for img in self.structure_grammar['images']], 
                                               F)
        elif pattern == 'microphaseshift' or pattern == 'mps':
            # handle structure grammar for Micro Phase Shift
            self.index_x = StructuredLight.decode_mps(
                                              [os.path.join(self.reconstruction_directory, img) 
                                               for img in self.structure_grammar['images']], 
                                               self.structure_grammar['frequency_vector'],
                                               self.camera.get_image_shape(),
                                               self.projector.get_projector_shape(),
                                               self.structure_grammar['median_filter'])
        else:
            raise RuntimeError("Unrecognized pattern, cannot decode structured light")
        
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
        campixels_x, campixels_y = np.meshgrid(np.arange(cam_resolution[0]),
                                               np.arange(cam_resolution[1]))
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
        if 'colors' not in self.structure_grammar:
            print("No color image set, therefore no color extraction for point cloud")
            return
        
        color_image = self.structure_grammar['colors']
        if self.mask is None:
            self.generate_mask()

        img = ImageUtils.load_ldr(os.path.join(self.reconstruction_directory, color_image))
        # clip RGB range to [0., 1.[
        minimum = np.min(img)
        maximum = np.max(img)
        self.colors: np.ndarray = ((img - minimum) / (maximum - minimum))[self.mask].reshape((-1,3))

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
    
    def save_outputs(self):
        name = self.structure_grammar['pattern']

        if self.outputs['depth_map']:
            self.extract_depth_map()
            np.save(os.path.join(self.reconstruction_directory,f"structured_light_{name}_depth_map.npy"), self.depth_map)

        if self.outputs['point_cloud']:
            # self.extract_normals()
            self.extract_colors()
            ThreeDUtils.save_point_cloud(os.path.join(self.reconstruction_directory,f"structured_light_{name}_point_cloud.ply"),
                                 self.point_cloud,
                                 self.normals,
                                 self.colors)
            
    def plot_decoding(self):
        """
        
        """
        Plotter.plot_decoding(self.camera.get_image_shape(),
                              self.index_x,
                              self.index_y)

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

    def run(self):
        self.decode()
        self.reconstruct()
        self.save_outputs()

    @staticmethod
    def decode_hilbert(images, num_bits):
        if isinstance(images, list):
            images = [ImageUtils.load_ldr(img, make_gray=True) 
                        if isinstance(img, str) else np.atleast_3d(img)
                        for img in images]
            images = np.concatenate(images, axis=2)
        
        num_dims = images.shape[2]

        decoded = hilbert.encode(images, num_dims=num_dims, num_bits=num_bits)
        return decoded.reshape((images.shape[0],images.shape[1]))

    @staticmethod
    def decode_gray_binary_xor(pattern,
                               vertical_images=None,
                               inverse_vertical_images=None,
                               horizontal_images=None,
                               inverse_horizontal_images=None,
                               white_image=None,
                               black_image=None,
                               thresh=None):
        """

        Parameters
        ----------
        vertical_images : list
            List of strings containing image paths (or string for folder containing images).
            These images will be used for decoding vertical structured light patterns.
        inverse_vertical_images : list, optional
            List of strings containing image paths (or string for folder containing images).
            Inverse vertical structured light pattern images will be used for 
            setting a threshold value for pixel intensity when decoding vertical patterns.
            Read the structuredlight docs for more information:
            https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image
        horizontal_images : list
            List of strings containing image paths (or string for folder containing images).
            These images will be used for decoding horizontal structured light patterns.
        inverse_horizontal_images : list, optional
            List of strings containing image paths (or string for folder containing images).
            Inverse horizontal structured light pattern images will be used for 
            setting a threshold value for pixel intensity when decoding horizontal patterns.
            Read the structuredlight docs for more information:
            https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image
        white_image : str | np.ndarray, optional
            (if str, path to) captured image of scene with an all-white pattern projected.
            The white image can be used for setting a threshold value when decoding
            structured light patterns. This has to be used in combination with black_image,
            since the  threshold for ON or OFF will be set per pixel as:
            thr = 0.5 * white_image + 0.5 * black_image.
            If negative/inverse structured light patterns are passed,
            black and white pattern images will be ignored for threshold setting.
            Read the structuredlight docs for more information:
            https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image

        
        black_image : str | np.ndarray, optional
            (if str, path to) captured image of scene with an all-black pattern projected.
            The black image can be used to extract colors for the reconstructed point cloud.
            The black image can also be used for setting a threshold value when decoding
            structured light patterns. This has to be used in combination with white_image,
            since the  threshold for ON or OFF will be set per pixel as:
            thr = 0.5 * white_image + 0.5 * black_image.
            If negative/inverse structured light patterns are passed,
            black and white pattern images will be ignored for threshold setting.
            Read the structuredlight docs for more information:
            https://github.com/elerac/structuredlight/wiki#how-to-binarize-a-grayscale-image
        thresh : float, optional
            threshold value for considering a pixel ON or OFF based on its intensity.
            If black and white pattern images are set, or if negative/inverse
            patterns are passed, this threshold value will be ignored.

        Returns
        -------
        index_x, index_y
        """
        
        if isinstance(pattern, str):
            match pattern.lower():
                case 'gray':
                    pattern = structuredlight.Gray()
                case 'binary':
                    pattern = structuredlight.Binary()
                case 'bin':
                    pattern = structuredlight.Binary()
                case 'xor':
                    pattern = structuredlight.XOR()
                case _:
                    print("Unrecognized structured light pattern type, defaulting to gray...\n")
                    pattern = structuredlight.Gray()
        
        index_x, index_y = None, None
        if white_image and black_image:
            white_image = ImageUtils.load_ldr(white_image, make_gray=True) \
                  if isinstance(white_image, str) else ImageUtils.convert_to_gray(white_image) 
            black_image = ImageUtils.load_ldr(black_image, make_gray=True) \
                  if isinstance(black_image, str) else ImageUtils.convert_to_gray(black_image) 
            thresh = 0.5*white_image + 0.5*black_image

        if horizontal_images:
            gray_horizontal = [ImageUtils.load_ldr(img, make_gray=True) 
                               if isinstance(img, str) else ImageUtils.convert_to_gray(img) 
                               for img in horizontal_images]
            if inverse_horizontal_images:
                assert len(inverse_horizontal_images) == len(horizontal_images), \
                    "Mismatch between number of horizontal patterns \
                        and inverse horizontal patterns. Must be the same"
                horizontal_second_argument = [ImageUtils.load_ldr(img, make_gray=True) 
                                              if isinstance(img, str) else ImageUtils.convert_to_gray(img) 
                                              for img in inverse_horizontal_images]
            else:
                horizontal_second_argument = thresh
            index_y = pattern.decode(gray_horizontal, horizontal_second_argument)

        if vertical_images:
            gray_vertical = [ImageUtils.load_ldr(img, make_gray=True) 
                               if isinstance(img, str) else ImageUtils.convert_to_gray(img) 
                               for img in vertical_images]
            if inverse_vertical_images:
                assert len(inverse_vertical_images) == len(vertical_images), \
                    "Mismatch between number of vertical patterns \
                        and inverse vertical patterns. Must be the same"
                vertical_second_argument = [ImageUtils.load_ldr(img, make_gray=True) 
                                              if isinstance(img, str) else ImageUtils.convert_to_gray(img) 
                                              for img in inverse_vertical_images]
            else:
                vertical_second_argument = thresh
            index_x = pattern.decode(gray_vertical, vertical_second_argument)

        return index_x, index_y
    
    @staticmethod
    def decode_phaseshift(width, images, F=1.0):
        pattern = structuredlight.PhaseShifting(num=len(images), F=F)
        pattern.width = width

        images = [ImageUtils.load_ldr(img, make_gray=True) 
                    if isinstance(img, str) else ImageUtils.convert_to_gray(img) 
                    for img in images]
        result = pattern.decode(images)

        return result
    
    @staticmethod
    def decode_mps(images: list,
                   frequency_vec: list,
                   cam: tuple,
                   pro: tuple,
                   medfilt_param: int=5):
        num_frequency = len(frequency_vec)

        # Making the measurement matrix M (see paper for definition)
        M = np.zeros((num_frequency+2, num_frequency+2))
        
        # Filling the first three rows -- correpsonding to the first frequency
        M[0,:3] = [1, np.cos(2*np.pi*0/3), -np.sin(2*np.pi*0/3)]
        M[1,:3] = [1, np.cos(2*np.pi*1/3), -np.sin(2*np.pi*1/3)]
        M[2,:3] = [1, np.cos(2*np.pi*2/3), -np.sin(2*np.pi*2/3)]
        
        # Filling the remaining rows - one for each subsequent frequency
        for f in range(1, num_frequency):
            #print([1, np.zeros(f), 1, np.zeros(numFrequency-f)], M[f+1, :])
            line = [1.0]
            line.extend([0.0]*(f+1))
            line.extend([1.0])
            line.extend([0.0]*(num_frequency-f-1))
            M[f+2, :] = line

        #%%%%%%%%%%%% Making the observation matrix (captured images) %%%%%%%%%%%%%
        R = np.zeros((num_frequency+2, cam[0]*cam[1]))


        # Filling the observation matrix (image intensities)
        for i, img in enumerate(images):
            img = ImageUtils.load_ldr(img, make_gray=True) if isinstance(img, str) else ImageUtils.convert_to_gray(img)
            # img = cv2.imread(img_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)   # reads an image in the BGR format
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float64")
            img = img / 255
            R[i,:]  = img.T.reshape(-1)
            
        #%%%%%%%%%%%%%%%%%% Solving the linear system %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # The unknowns are [Offset, Amp*cos(phi_1), Amp*sin(phi_1), Amp*cos(phi_2),
        # ..., Amp*cos(phi_F)], where F = numFrequency. See paper for details. 
        U = np.linalg.solve(M, R)

        # Computing the amplitude 
        Amp = np.sqrt(U[1,:]**2 + U[2,:]**2)

        # Dividing the amplitude to get the CosSinMat --- matrix containing the sin
        # and cos of the phases corresponding to different frequencies. For the
        # phase of the first frequency, we have both sin and cos. For the phases of
        # the remaining frequencies, we have cos. 

        CosSinMat = U[1:, :] / np.tile(Amp, (num_frequency+1, 1))  

        #%%%%%%%%%%%%%% Converting the CosSinMat into column indices %%%%%%%%%%%%%%
        # IC            -- correspondence map (corresponding projector column (sub-pixel) for each camera pixel. Size of IC is the same as input captured imgaes.
        IC = StructuredLight.phase_unwrap_cos_sin_to_column_index(CosSinMat, frequency_vec, pro[0], cam[1], cam[0])
        IC = scipy.signal.medfilt2d(IC, medfilt_param) # Applying median filtering

        return IC



    # This function converts the CosSinMat into column-correspondence.  
    #
    # CosSinMat is the matrix containing the sin and cos of the phases 
    # corresponding to different frequencies for each camera pixel. For the
    # phase of the first frequency, we have both sin and cos. For the phases of
    # the remaining frequencies, we have cos. 
    #
    # The function first performs a linear search on the projector column
    # indices. Then, it adds the sub-pixel component. 

    def phase_unwrap_cos_sin_to_column_index(CosSinMat, frequencyVec, numProjColumns, nr, nc):
        x0 = np.array([list(range(0, numProjColumns))]) # Projector column indices

        
        # Coomputing the cos and sin values for each projector column. The format 
        # is the same as in CosSinMat - for the phase of the first frequency, we 
        # have both sin and cos. For the phases of the remaining frequencies, we 
        # have cos. These will be compared against the values in CosSinMat to find 
        # the closest match. 

        TestMat = np.tile(x0, (CosSinMat.shape[0], 1)).astype("float64")
        
        TestMat[0,:] = np.cos((np.mod(TestMat[0,:], frequencyVec[0]) / frequencyVec[0]) * 2 * np.pi) # cos of the phase for the first frequency
        TestMat[1,:] = np.sin((np.mod(TestMat[1,:], frequencyVec[0]) / frequencyVec[0]) * 2 * np.pi) # sin of the phase for the first frequency

        for i in range(2, CosSinMat.shape[0]):
            TestMat[i,:] = np.cos((np.mod(TestMat[i,:], frequencyVec[i-1]) / frequencyVec[i-1]) * 2 * np.pi) # cos of the phases of the remaining frequency
            
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        IC = np.zeros((1, nr*nc), dtype="float64") # Vector of column-values
        
        # For each camera pixel, find the closest match
        # TODO parpool(4);% This loop can be run in parallel using MATLAB parallel toolbox. The number here is the number of cores on your machine. 
        for i in range(0, CosSinMat.shape[1]):
            CosSinVec = CosSinMat[:,i]
            CosSinVec = CosSinVec.reshape((CosSinVec.shape[0], 1))
            ErrorVec = np.sum(np.abs(np.tile(CosSinVec, (1, numProjColumns)) - TestMat)**2, axis=0)
            #print(ErrorVec.shape, ErrorVec)
            Ind = np.argmin(ErrorVec)
            #print(Ind)
            IC[0, i] = Ind

        # Computing the fractional value using phase values of the first frequency 
        # since it has both cos and sin values. 

        PhaseFirstFrequency = np.arccos(CosSinMat[0,:]) # acos returns values in [0, pi] range. There is a 2 way ambiguity.
        PhaseFirstFrequency[CosSinMat[1,:]<0] = 2 * np.pi - PhaseFirstFrequency[CosSinMat[1,:]<0] # Using the sin value to resolve the ambiguity
        ColumnFirstFrequency = PhaseFirstFrequency * frequencyVec[0] / (2 * np.pi) # The phase for the first frequency, in pixel units. This is equal to mod(trueColumn, frequencyVec(1)). 

        NumCompletePeriodsFirstFreq = np.floor(IC / frequencyVec[0]) # The number of complete periods for the first frequency 
        ICFrac = NumCompletePeriodsFirstFreq * frequencyVec[0] + ColumnFirstFrequency # The final correspondence, with the fractional component

        # If the difference after fractional correction is large (because of noise), keep the original value. 
        ICFrac[np.abs(ICFrac-IC)>=1] = IC[np.abs(ICFrac-IC)>=1]
        IC = ICFrac

        IC = np.reshape(IC, [nr, nc], order='F')
        return IC



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Light Reconstruction")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to config for Structured Light Reconstruction")

    args = parser.parse_args()

    struc = StructuredLight(args.config)
    struc.run()