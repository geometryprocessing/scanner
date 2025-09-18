import inspect
import os
import sys

import cv2
import numpy as np

from src.utils.file_io import save_json, load_json

class Projector:
    def __init__(self,
                 resx: int = None, resy: int = None,
                 K = None,
                 dist_coeffs = None,
                 R = np.identity(3),
                 T = np.zeros(shape=(3,1)),
                 config: dict | str = None):
        
        self.pretty_name = 'Empty Projector'
        self.name = 'emptyprojector'

        self.resx  = resx
        self.resy = resy
        self.K = K                         # if not passed, None
        self.dist_coeffs = dist_coeffs     # if not passed, None
        self.R = R                         # if not passed, projector is initialized at origin
        self.T = T                         # if not passed, projector is initialized at origin

        if config is not None:
            self.load_config(config)

    def load_config(self, calibration: str | dict):
        if type(calibration) is str:
            calibration = load_json(calibration)
        
        self.set_projector_resy(calibration['resy'])
        self.set_projector_resx(calibration['resx'])

        self.set_intrinsic_matrix(calibration['K'])
        self.set_distortion(calibration['dist_coeffs'])
        self.set_rotation(calibration['R'])
        self.set_translation(calibration['T'])
        
    # setters to guarantee numpy arrays
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
        
class TestProjector(Projector):
    def __init__(self):
        super.__init__(resx=200, resy=200,
                       K = np.asarray([[100, 0, 100], 
                             [0, 100, 100],
                             [0,0,1]], dtype=np.float32),
                       dist_coeffs = None)
        self.pretty_name = 'Test Projector'
        self.name = 'test'

class DLPProjector(Projector):
    def __init__(self):
        super.__init__(resx=1920, resy=1080,
                       K = np.asarray([[2.86463085e+03, 0.00000000e+00, 9.46935454e+02],
                                        [0.00000000e+00, 2.87024447e+03, 1.10580621e+03],
                                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32),
                       dist_coeffs = np.asarray([-0.02902245, -0.30154025,  0.00232378, -0.00189861,  1.03203233], dtype=np.float32),
                       R = np.asarray([[ 0.878584  , -0.01702959,  0.4772841 ],
                                        [ 0.01292312,  0.99984586,  0.01188583],
                                        [-0.47741294, -0.0042747 ,  0.87866867]], dtype=np.float32),
                       T = np.asarray([-395.79297, -81.8893, -250.9899 ], dtype=np.float32))
        self.pretty_name = 'TexasInstruments DLP Projector'
        self.pretty_name_short_short = 'DLP Projector'
        self.name = 'dlp'
        
class LCDProjector(Projector):
    def __init__(self):
        super.__init__(resx=1920, resy=1080,
                        K = np.asarray([[2.36118237e+03, 0.00000000e+00, 1.03067073e+03],
                                        [0.00000000e+00, 2.35098328e+03, 9.65838667e+02],
                                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32),
                        dist_coeffs = np.asarray([ 0.02788432, -0.05765782,  0.00803185,  0.0074843 ,  0.10213349], dtype=np.float32),
                        R = np.asarray([[ 0.9827554 , -0.05533259,  0.17643735],
                                        [ 0.06008505,  0.9979573 , -0.02170372],
                                        [-0.17487602,  0.03193069,  0.98407257]], dtype=np.float32),
                        T = np.asarray([-301.79178, -23.397247, -108.09623 ], dtype=np.float32))
        self.pretty_name = 'Full HD LCD VOPPLS Projector'
        self.pretty_name_short = 'LCD Projector'
        self.name = 'lcd'
        
CONFIGS = {name.lower(): obj
    for name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(obj)}

def get_proj_config(config):
    """
    Function checks first if projector config already exists,
    then check if path to a JSON file exists,
    finally throws an error.
    """
    if isinstance(config, str):
        if config.lower() in CONFIGS:
            return CONFIGS[config.lower()]()
        elif os.path.exists(config):
            return Projector(config=config)
    elif isinstance(config, dict):
        return Projector(config=config)
    else:
        raise ValueError(f"Could not find projector config {config}!")