import argparse
import cv2
import numpy as np
import os

from src.utils.file_io import save_json, load_json, get_all_paths, get_folder_from_file
from src.utils.image_utils import homogeneous_coordinates
from src.scanner.calibration import Charuco, CheckerBoard, Calibration, CameraCalibration, ProjectorCalibration
from src.reconstruction.configs import StructuredLightConfig
from src.reconstruction.structured_light import decode

class LocalHomographyCalibration:
    def __init__(self, config: dict | str = None):
        self.plane_pattern = None

        self.calibration_directory = None
        self.num_directories = 0
        # this is determined from necessary images of gray patterns for the projector resolution
        self.num_vertical_images = 0 
        self.num_horizontal_images = 0 
        
        self.window_size = 30

        self.index_x = []
        self.index_y = []
        self.object_points = []
        self.camera_image_points = []
        self.projector_image_points = []

        self.camera = CameraCalibration()
        self.projector = ProjectorCalibration()

        if config is not None:
            self.load_config(config)

    def load_config(self, config: str | dict):
        if isinstance(config, str):
            config = load_json(config)

        self.camera.set_image_shape(config['local_homography']['camera_shape'])
        self.projector.set_projector_shape(config['local_homography']['projector_shape'])

        self.camera.set_error_threshold(config['local_homography']['camera_error_thr'])
        self.projector.set_error_threshold(config['local_homography']['projector_error_thr'])

        self.set_plane_pattern(config['local_homography']['plane_pattern'])
        self.set_calibration_directory(config['local_homography']['calibration_directory'])

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
        # TODO: change the get_all_paths to get_all_folders?

        Parameters
        ----------
        path : str
            path to directory containing multiple calibration folders,
            each with the same structured light capture of a calibration board
        """
        assert os.path.isdir(path), "This is not a directory. This function only works with a folder."
        dirs = [f.path for f in os.scandir(path) if f.is_dir()]
        self.num_directories = len(dirs)
        self.calibration_directory = [get_all_paths(dir) for dir in dirs]

        self.num_vertical_images      = int(2*np.ceil(np.log2(self.projector.resx)))
        self.num_horizontal_images    = int(2*np.ceil(np.log2(self.projector.resy)))
    def set_window_size(self, window: int):
        """
        Set the window (in pixels) of the neighborhood of each calibration corner to calculate the Homography.
        """
        self.window_size = window
    def load_camera(self, camera: str | dict | CameraCalibration):
        if isinstance(camera, (str, dict)):
            self.camera.load_config(camera)
        else:
            self.camera = camera

    # functions
    def decode(self):
        """
        """
        for folder in self.calibration_directory:
            config = StructuredLightConfig(pattern='gray',
                                           vertical_images=[f"gray_{d:02d}.tiff" for d in range(1, self.num_vertical_images, 2)],
                                           inverse_vertical_images=[f"gray_{d:02d}.tiff" for d in range(2, self.num_vertical_images + 1, 2)],
                                           horizontal_images= [f"gray_{d:02d}.tiff" for d in range(self.num_vertical_images + 1, self.num_vertical_images + self.num_horizontal_images, 2)],
                                           inverse_horizontal_images=[f"gray_{d:02d}.tiff" for d in range(self.num_vertical_images + 2, self.num_vertical_images + self.num_horizontal_images + 1, 2)])
            _index_x, _index_y = decode(get_folder_from_file(folder[0]), config)
            self.index_x.append(_index_x)
            self.index_y.append(_index_y)

    def detect_markers_and_homographies(self):
        """
        
        """
        for i, folder in enumerate(self.calibration_directory):
            white_image = self.calibration_directory[i][-1]
            self.camera.intrinsic_images.append(white_image)
            self.projector.images.append(white_image)
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
                proj_points = np.stack([self.index_x[i][minY:maxY, minX:maxX],
                                        self.index_y[i][minY:maxY, minX:maxX],
                                        ], axis=-1).reshape((-1,2))
            
                # image_points is the list of points in the neighborhood seen by the camera
                # proj_points is the list of points in the neighborhood that we extract from the index_x index_y after decoding
                H, mask = cv2.findHomography(image_points, proj_points)

                # p is corner in camera image
                p = homogeneous_coordinates(camera_image_point) # I think it's supposed to be camera_image_point
                Q = np.matmul(p, H.T)
                # q is corner in projector image
                q = Q[:,0:-1] / Q[:,-1]
                proj_img_points[idx] = q

            self.camera.intrinsic_image_points.append(img_points)
            self.camera.intrinsic_object_points.append(obj_points)
            self.projector.image_points.append(proj_img_points)
            self.projector.object_points.append(obj_points)

    def calibrate_stereo_system(self):
        # if camera is already calibrated, retrieve only the root mean squared reprojection error
        if self.camera.K is not None:
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
        else:
            flags = 0

        cam_result = Calibration.calibrate(self.camera.intrinsic_object_points,
                                           self.camera.intrinsic_image_points,
                                           self.camera.get_image_shape(),
                                           K=self.camera.K,
                                           dist_coeffs=self.camera.dist_coeffs,
                                           flags=flags)

        self.camera.K = cam_result['K']
        self.camera.dist_coeffs = cam_result['dist_coeffs']
        self.camera.tvecs = cam_result['tvecs']
        self.camera.rvecs = cam_result['rvecs']
        self.camera.projection_errors()
        # self.camera.refine()
        self.camera_error = cam_result['rms']

        proj_result = Calibration.calibrate(self.projector.object_points,
                                            self.projector.image_points,
                                            self.projector.get_projector_shape())

        self.projector.K = proj_result['K']
        self.projector.dist_coeffs = proj_result['dist_coeffs']
        self.projector.tvecs = proj_result['tvecs']
        self.projector.rvecs = proj_result['rvecs']
        self.projector.projection_errors()
        # self.projector.refine() # this is not yet implemented
        self.projector_error = proj_result['rms']

        stereo_result = Calibration.stereo_calibrate(self.camera.intrinsic_object_points,
                                                     self.camera.intrinsic_image_points,
                                                     self.projector.image_points,
                                                     self.camera.get_image_shape(),
                                                     self.camera.K,
                                                     self.camera.dist_coeffs,
                                                     self.projector.K,
                                                     self.projector.dist_coeffs)

        self.projector.R = stereo_result['R']
        self.projector.T = stereo_result['T']
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
            "camera_K": self.camera.K,
            "camera_dist_coeffs": self.camera.dist_coeffs,
            "projector_K": self.projector.K,
            "projector_dist_coeffs": self.projector.dist_coeffs,
            "projector_R": self.projector.R,
            "projector_T": self.projector.T,
            "camera_error": self.camera_error,
            "projector_error": self.projector_error,
            "stereo_error": self.stereo_error
        }, os.path.join(self.calibration_directory, 'calibration.json'))

    def run(self):
        self.decode()
        self.detect_markers_and_homographies()

        self.calibrate_stereo_system()
        self.save_calibration()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Homography Calibration")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to config for Local Homography Calibration")

    args = parser.parse_args()

    lhc = LocalHomographyCalibration(config=args.config)
    lhc.run()