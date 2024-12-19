import cv2
import numpy
import structuredlight as sl

from scanner.camera import Camera
from scanner.projector import Projector

class StructuredLight:
    def __init__(self):
        self.projector = Projector()
        self.camera = Camera()
        self.pattern = None

    # setters
    def set_pattern(self, pattern: str):
        """
        
        """
        match pattern:
            case 'gray':
                self.pattern = sl.Gray()
            case 'grey':
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

    def set_image_paths(self, image_paths: list | str):
        pass
    def set_white_pattern_image(self, image_path: str):
        """
        Set the path of captured image of scene with an all-white pattern projected.
        
        This has to be used in combination with set_black_pattern_image, since the 
        threshold for ON or OFF will be set per pixel as:
        thr = 0.5 * white_pattern_image + 0.5 * black_pattern_image
        """        
        pass
    def set_black_pattern_image(self, image_path: str):
        """
        Set the path of captured image of scene with an all-black pattern projected.

        This has to be used in combination with set_black_pattern_image, since the 
        threshold for ON or OFF will be set per pixel as:
        thr = 0.5 * white_pattern_image + 0.5 * black_pattern_image
        """        
        pass
    def set_projector(self, projector: str | dict |Projector):
        # if string, load the json file and get the parameters
        # check if dict, get the parameters
        if type(projector) is str or dict:
            self.projector.load_calibration(projector)
        else:
            self.projector = projector
    def set_camera(self, camera: str | dict | Camera):
        # if string, load the json file and get the parameters
        # check if dict, get the parameters
        if type(camera) is str or dict:
            self.camera.load_calibration(camera)
        else:
            self.camera = camera
    def decode(self):
        """
        
        """
        self.pattern.decode(self.images)
        
    def reconstruct(self):
        """
        
        """
        pass