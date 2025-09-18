import inspect
import os
import sys

import cv2
import numpy as np

from src.utils.file_io import save_json, load_json

class Camera:
    def __init__(self,
                 resx: int = None,
                 resy: int = None,
                 K = None,
                 dist_coeffs = None,
                 R = np.identity(3),
                 T = np.zeros(shape=(3,1)),
                 config: dict | str = None):
        
        self.pretty_name = 'Empty Camera'
        self.name = 'emptycamera'
        # filename is used, for now, only in multiview, 
        # where a data folder contains multiple camera folders
        self.filename = '' 

        self.resx  = resx
        self.resy = resy
        self.K = K                         # if not passed, None
        self.dist_coeffs = dist_coeffs     # if not passed, None
        self.R = R                         # if not passed, camera is initialized at origin
        self.T = T                         # if not passed, camera is initialized at origin

        if config is not None:
            self.load_config(config)

    def load_config(self, calibration: str | dict):
        if isinstance(calibration, str):
            calibration = load_json(calibration)

        self.set_resy(calibration['resy'])
        self.set_resx(calibration['resx'])
        self.set_intrinsic_matrix(calibration['K'])
        self.set_distortion(calibration['dist_coeffs'])
        if 'R' in calibration:
            self.set_rotation(calibration['R'])
        if 'T' in calibration:
            self.set_translation(calibration['T'])

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

    def to_dict(self):
        """
        Returns camera config as a dictionary.
        """
        return {
            "name": self.name,
            "resy": self.resy,
            "resx": self.resx,
            "K": self.K,
            "dist_coeffs": self.dist_coeffs,
            "R": self.R,
            "T": self.T,
        }

class TestCamera(Camera):
    def __init__(self):
        super().__init__(resx=200, resy=200,
                       K = np.asarray([[100, 0, 100], 
                             [0, 100, 100],
                             [0,0,1]], dtype=np.float32),
                       dist_coeffs = None)
        self.pretty_name = 'Test Camera'
        self.name = 'test'

class AtlasCamera(Camera):
    def __init__(self):
        super().__init__(resx=6464, resy=4852,
                       K = np.asarray([[1.4852e04, 0, 3.1818e03], 
                             [0, 1.48677e04, 2.46895e03],
                             [0,0,1]], dtype=np.float32),
                       dist_coeffs = np.asarray([-1.0079e-01,
                                       -9.6801e-01,
                                       4.155e-04,
                                       -4.9249e-04,
                                       8.8487e00], dtype=np.float32))
        self.pretty_name = 'Atlas 32Mpx Camera'
        self.name = 'atlas'

class ChronosCamera(Camera):
    def __init__(self):
        super().__init__(resx=1920, resy=1080,
                       K = np.asarray([[3580.915139167154,0.0,967.7411991347725],
                            [0.0,3578.1633141223797,509.395929381727],
                            [0.0,0.0,1.0]], dtype=np.float32),
                       dist_coeffs = np.asarray([-0.05202839890657609,
                                       -0.05977211653771139,
                                       0.0014034145040185428,
                                       -0.00031193552659360146,
                                       0.3585615042719389], dtype=np.float32))
        self.pretty_name = 'Chronos HD Camera'
        self.name = 'chronos'

class Triton1Camera(Camera):
    def __init__(self):
        super().__init__(resx=2448, resy=2048,
                       K = np.asarray([[4398.865805444324, 0.0, 1224.998494272802],
                             [0.0,4395.209485484965,1010.5491943764957],
                             [0.0,0.0,1.0]], dtype=np.float32),
                       dist_coeffs = np.asarray([-0.18346018257897984,
                                        0.31643440753823476,
                                        0.0003134068085886509,
                                        0.00020178521826665733,
                                        0.11827598690686253], dtype=np.float32))
        self.pretty_name = 'Triton10 5Mpx Camera No.1'
        self.name = 'triton1'
        self.filename = 'Triton1'
        
class Triton2Camera(Camera):
    def __init__(self):
        super().__init__(resx=2448, resy=2048,
                       K = np.asarray([[4415.772826208225,0.0,1214.1772432727691],
                             [0.0,4412.163698025551,1041.2152001109412],
                             [0.0,0.0,1.0]], dtype=np.float32),
                       dist_coeffs = np.asarray([-0.1845207964010922,
                                        0.3764315875260389,
                                        0.0006885791954024719,
                                        0.00010983578916095759,
                                       -0.04955677781648061], dtype=np.float32),
                       R = np.asarray([   0.02470863, -2.22986647,    0.10327809], dtype=np.float32),
                       T = np.asarray([ 562.73912308, 30.61416842, 1021.49771131], dtype=np.float32))
        self.pretty_name = 'Triton10 5Mpx Camera No.2'
        self.name = 'triton2'
        self.filename = 'Triton2'


CONFIGS = {name.lower(): obj
    for name, obj in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(obj)}

def get_cam_config(config):
    """
    Function checks first if camera config already exists,
    then check if path to a JSON file exists,
    finally throws an error.
    """
    if isinstance(config, str):
        if config.lower() in CONFIGS:
            return CONFIGS[config.lower()]()
        elif os.path.exists(config):
            return Camera(config=config)
        else:
            raise ValueError(f"Could not find camera config {config}!")
    elif isinstance(config, dict):
        return Camera(config=config)
    else:
        raise ValueError(f"Could not find camera config {config}!")