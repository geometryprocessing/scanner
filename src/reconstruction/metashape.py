import cv2
import Metashape
import numpy as np
import os
import xml.etree.ElementTree as ET

from scanner.calibration import Calibration, Charuco, CheckerBoard
from utils.file_io import ensure_exists, load_json, save_json, get_all_paths

class MetashapeReconstruction:
    def __init__(self):
        self.doc = Metashape.Document()

        self.doc.addChunk()
        self.chunk = self.doc.chunks[-1]
        
        # image paths
        self.images = []

        # calibration utils
        self.fisheye = False
        self.calibration_pattern = None
        self.downscale = 1

    # setters
    def set_downscale(self, downscale: int):
        self.downscale = max(1, int(downscale))
    def set_fisheye(self, fisheye: bool):
        self.fisheye = fisheye
    def set_image_paths(self, image_paths: list | str):
        """
        Parameters
        ----------
        image_paths : list | string
            List of strings containing image paths (or string for folder containing images).
            These images will be used for decoding horizontal structured light patterns.
        """
        self.images = get_all_paths(image_paths)
    def set_calibration_pattern(self, pattern: dict | Charuco | CheckerBoard):
        """
        Parameters
        ----------
        pattern : dictionary | Charuco object | CheckerBoard object
            Calibration pattern used for all camera calibration procedures.
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
        self.calibration_pattern = pattern

    # functions
    def load_images(self):
        """
        Load photos from data folder onto Agisoft Metashape.
        """
        self.chunk.addPhotos(self.images)

    def add_markers(self):
        """
        Detect ChAruCo markers and add them to Metashape project.
        """
        for image_path in self.images:
            img_points, obj_points, ids = self.calibration_pattern.detect_markers(image_path)

            camera_label, _  = os.path.splitext(image_path)
            camera = None
            for c in self.chunk.cameras:
                if c.label == camera_label:
                    camera = c

            if camera is None or img_points is None:
                continue

            for img_point, position, id in zip(img_points, obj_points, ids):
                id = str(id)
                duplicate_found = False
                for marker in self.chunk.markers:
                    if marker.label == id:
                            m = marker
                            duplicate_found = True
        
                if not duplicate_found:
                    self.chunk.addMarker()
                    m = self.chunk.markers[-1]
                    m.label = id
        
                m.projections[camera] = Metashape.Marker.Projection(Metashape.Vector(img_point))
                m.projections[camera].pinned = True
                m.reference.location = Metashape.Vector(position)

        self.chunk.updateTransform()

    def align_cameras(self):
        """
        Align cameras/images.
        """
        if self.fisheye:
            for sensor in self.chunk.sensors:
                sensor.type = Metashape.Sensor.Type.Fisheye
        
        self.chunk.matchPhotos(downscale=self.downscale)
        self.chunk.alignCameras(reset_alignment=True)

    def build_depth_maps(self):
        """
        Build depth maps from images.
        """
        self.chunk.buildDepthMaps(downscale=self.downscale)

    def build_dense_cloud(self):
        """
        Build dense point cloud from depth maps.
        """
        self.chunk.buildPointCloud(source_data=Metashape.DepthMapsData,
                                   point_confidence=True)

    def build_mesh(self):
        """
        Build triangular mesh from depth maps.
        """
        self.chunk.buildModel(source_data=Metashape.DepthMapsData)

    def build_texture(self):
        """
        Build texture of mesh.
        """
        self.chunk.buildUV()
        self.chunk.buildTexture()

    def save_undistorted_images(self, folder: str):
        """
        Save undistorted images.
        """
        for camera in self.chunk.cameras:
            currentCameraImage = camera.photo.image()
            calibration = camera.sensor.calibration
            undistortedCameraImage = currentCameraImage.undistort(calibration)

            undistortedCameraImage.save(f"{folder}/{camera.label}.JPG")

    def save_camera_calibration(self, filename: str):
        """
        Save JSON file with camera calibration.
        """
        sensor = self.chunk.sensors[-1]
        cam_calib = sensor.calibration

        # create tmp folder
        cam_calib.save("tmp/cam_calib.xml",
            format=Metashape.CalibrationFormatOpenCV
            )

        tree = ET.parse("tmp/cam_calib.xml")
        root = tree.getroot()

        width = int(root.find("image_Width").text)
        height = int(root.find("image_Height").text)

        cam_mtx_element = root.find("Camera_Matrix")
        cam_mtx_list = cam_mtx_element.find("data").text.split()

        cam_mtx_rows = int(cam_mtx_element.find("rows").text)
        cam_mtx_cols = int(cam_mtx_element.find("cols").text)

        cam_mtx = np.array(cam_mtx_list, dtype=float).reshape(
            (cam_mtx_rows, cam_mtx_cols))
        
        dist_coeffs_element = root.find("Distortion_Coefficients")
        dist_coeffs_list = dist_coeffs_element.find("data").text.split()

        dist_coeffs_rows = int(dist_coeffs_element.find("rows").text)
        dist_coeffs_cols = int(dist_coeffs_element.find("cols").text)

        dist_coeffs = np.array(dist_coeffs_list, dtype=float).reshape(
            (dist_coeffs_rows, dist_coeffs_cols))
        
        # detele tmp folder
        os.remove("tmp/cam_calib.xml")
        
        save_json({'width': width,
                   'height': height,
                   'K': cam_mtx,
                   'dist_coeffs': dist_coeffs},
                   filename)
        
        return width, height, cam_mtx, dist_coeffs

    def save_camera_locations(self, filename: str):
        """
        Save JSON file with camera locations.
        """
        cameras = self.chunk.cameras
        camera_positions = {}
        
        for camera in cameras:
            pose = self.chunk.transform.matrix * camera.transform

            camera_positions[camera.label] = [pose.row(0)[3],pose.row(1)[3],pose.row(2)[3]]
        
        save_json(camera_positions, filename)

    def save_camera_rotations(self, filename: str):
        """
        Save JSON file with camera rotations.
        """
        cameras = self.chunk.cameras
        camera_rotations = {}
        
        for camera in cameras:
            pose = self.chunk.transform.matrix * camera.transform

            rotation_matrix_np = np.eye(3, dtype=np.float32)

            for row_idx in range(3):
                row = pose.row(row_idx)
                rotation_matrix_np[row_idx, :] = np.array([row.x, row.y, row.z])

            camera_rotations[camera.label] = rotation_matrix_np

        save_json(camera_rotations, filename)

    def save_dense_cloud(self, filename: str):
        """
        Save point cloud of reconstruction.
        """
        self.chunk.exportPointCloud(filename)

    def save_mesh(self, filename: str):
        """
        Save mesh of reconstruction.
        """
        self.chunk.exportModel(filename)

    def save_depth_maps(self, folder: str):
        """
        Save depth maps of project.
        """
        # Data management
        ensure_exists(folder)
        for camera in self.chunk.cameras:
            depth = self.chunk.depth_maps[camera].image()
            depth *= self.chunk.transform.scale
            depth.save(f'{folder}/{camera.label}.TIFF')


    def run(self, config: str | dict):
        if type(config) is str:
            config = load_json(config)

        # paths
        self.set_image_paths(config['image_paths'])

        # calibration utils
        self.set_downscale(config['downscale'])
        self.set_calibration_pattern(config['charuco'])

        # functions
        self.load_images()
        self.add_markers()
        self.align_cameras()
        self.build_depth_maps()
        self.build_mesh()

        self.save_mesh(config['output']['mesh'])
        self.save_depth_maps(config['output']['depth_maps'])
        self.save_camera_calibration(config['output']['camera_calibration'])
        self.save_camera_locations(config['output']['camera_locations'])
        self.save_camera_rotations(config['output']['camera_rotations'])