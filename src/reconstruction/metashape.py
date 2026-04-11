import Metashape

from paleo_pipeline.assets.utils import *

import cv2
import glob
import json
from natsort import natsorted
import numpy as np
import os
import shutil
import tifffile
import trimesh
import xml.etree.ElementTree as ET


# TODO: move this to script
# def initialize_metashape():
#     '''
#     Create a new project in Agisoft Metashape.
#     '''
#     scanID = scanID
    
#     doc = Metashape.Document()
#     doc.addChunk()
    
#     # Data management
#     os.makedirs(OUTPUT_PATH.format(scanID, '/'), exist_ok=True)

#     doc.save(METASHAPE_PATH.format(scanID))



def load_images(metashape_doc: Metashape.Document, image_paths: list[str]):
    '''
    Load photos from data folder onto Agisoft Metashape.
    '''
    chunk = metashape_doc.chunks[-1]
    chunk.addPhotos(image_paths)


def align_cameras(metashape_doc: Metashape.Document,
                  downscale: int = 0,
                  reset_alignment: bool = True,
                  fisheye: bool = False):
    '''
    Align cameras/images.
    '''
    assert downscale in [0, 1, 2, 3], "Downscale must be (0,1,2,3), where 0 is no downscaling and 3 is the most downscaling."
    chunk = metashape_doc.chunks[-1]
    if fisheye:
        for sensor in chunk.sensors:
            sensor.type = Metashape.Sensor.Type.Fisheye
    
    chunk.matchPhotos(downscale=2**downscale)
    chunk.alignCameras(reset_alignment=reset_alignment)

def build_depth_maps(metashape_doc: Metashape.Document,
                     downscale: int = 0):
    '''
    Build depth maps from images.
    '''
    assert downscale in [0, 1, 2, 3], "Downscale must be (0,1,2,3), where 0 is no downscaling and 3 is the most downscaling."
    chunk = metashape_doc.chunks[-1]
    chunk.buildDepthMaps(downscale=2**downscale)


def opencv_calibration(scanID: str, charuco_board: dict) -> tuple[int, int, np.ndarray, np.ndarray]:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard_create(charuco_board['n'],
                                          charuco_board['m'],
                                          charuco_board['checker_size'],
                                          charuco_board['checker_size'] * 12 / 15,
                                          aruco_dict)
    charuco_objs = board.chessboardCorners.reshape((-1, 3))

    wb_paths = natsorted(glob.glob(FOLDER_WB_PATH.format(scanID, "*.TIFF")))

    img_points, obj_points = [], []

    for wb_path in wb_paths:

        image = cv2.imread(wb_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        width, height = gray.shape

        imgp, ids = charuco_detector(gray, aruco_dict, board)

        if ids is None:
            continue

        objp = charuco_objs[ids]

        img_points.append(imgp.reshape((1,-1,2)))
        obj_points.append(objp.reshape((1,-1,3)))

    flags = cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_FIX_K3
    ret, cam_mtx, dist_coeffs, rvecs, tvecs = \
        cv2.calibrateCamera(np.array(obj_points, dtype=np.float32), 
                            np.array(img_points, dtype=np.float32),
                            (width, height),
                            None,
                            None,
                            flags=flags)
    
    return width, height, cam_mtx, dist_coeffs


def export_calibration(scanID: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    doc = Metashape.Document()
    doc.open(METASHAPE_PATH.format(scanID))
    chunk = doc.chunks[-1]
    
    sensor = chunk.sensors[-1]
    cam_calib = sensor.calibration

    cam_calib.save(
        path=FOLDER_UNDISTORTED_PATH.format(scanID, "cam_calib.xml"),
        format=Metashape.CalibrationFormatOpenCV
        )

    tree = ET.parse(FOLDER_UNDISTORTED_PATH.format(scanID, "cam_calib.xml"))
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
    
    return width, height, cam_mtx, dist_coeffs

def calibrate_camera():
    '''
    Save JSON file with camera calibration parameters.
    '''
    scanID = scanID
    
    # Data management
    os.makedirs(FOLDER_UNDISTORTED_PATH.format(scanID, '/'), exist_ok=True)

    if fisheye:
        width, height, cam_mtx, dist_coeffs = \
            opencv_calibration(scanID=scanID,
                               charuco_board=charuco_board)
    else:
        width, height, cam_mtx, dist_coeffs = metashape_calibration(scanID=scanID)

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx,
                                                 dist_coeffs,
                                                 (width, height),
                                                 1,
                                                 (width, height),
                                                 centerPrincipalPoint=True)

    with open(OUTPUT_PATH.format(scanID, "cam_calib.json"), "w") as f:
        json.dump({
            "mtx": cam_mtx.tolist(),
            "dist": dist_coeffs.tolist(),
            "new_mtx": new_mtx.tolist()
            }, f)


def opencv_undistort( scanID: str):
    '''
    Undistort white balanced images and save as .TIFF files using OpenCV.
    '''

    with open(OUTPUT_PATH.format(scanID, "cam_calib.json"), "r") as f:
        cam_calib = json.load(f)

    wb_paths = natsorted(glob.glob(FOLDER_WB_PATH.format(scanID, "*.TIFF")))

    for wb_path in wb_paths:
        wb_image = cv2.imread(wb_path)

        undistorted_image = cv2.undistort(
            src=wb_image,
            cameraMatrix=np.array(cam_calib["mtx"]),
            distCoeffs=np.array(cam_calib["dist"]),
            newCameraMatrix=np.array(cam_calib["new_mtx"]))

        filename, ext = os.path.splitext(
            FOLDER_UNDISTORTED_PATH.format(
                scanID,
                os.path.basename(wb_path)
                ))
        
                
        .log.info(
            f"Undistorted {filename} from scan {scanID}"
        )

        # save as .TIFF
        cv2.imwrite(
            filename + ".TIFF",
            (undistorted_image).astype(np.uint16)
            )
        

def metashape_undistort( scanID: str):
    doc = Metashape.Document()
    doc.open(METASHAPE_PATH.format(scanID))
    chunk = doc.chunks[-1]

    for camera in chunk.cameras:
        currentCameraImage = camera.photo.image()
        calibration = camera.sensor.calibration
        undistortedCameraImage = currentCameraImage.undistort(calibration)

        .log.info(
            f"Undistorted {camera.label} from scan {scanID}"
        )

        undistortedCameraImage.save(
            FOLDER_UNDISTORTED_PATH.format(scanID, f"{camera.label}.JPG")
            )

def undistort_images():
    '''
    Save all images after undistortion.
    '''
    scanID = scanID
    
    # Data management
    os.makedirs(FOLDER_UNDISTORTED_PATH.format(scanID, ''), exist_ok=True)

    if fisheye:
        .log.info("Will not use Metashape to undistort fisheye images. \n\
                         Moving to OpenCV function.")
        opencv_undistort( scanID=scanID)
    else:
        .log.info("Using Metashape to undistort images.")
        metashape_undistort( scanID=scanID)


def add_markers():
    '''
    Detect ChAruCo markers on undistorted images with OpenCV.
    '''
    scanID = scanID
    charuco_board = charuco_board

    doc = Metashape.Document()
    doc.open(METASHAPE_PATH.format(scanID))
    chunk = doc.chunks[-1] 
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard_create(charuco_board['n'],
                                          charuco_board['m'],
                                          charuco_board['checker_size'],
                                          charuco_board['checker_size'] * 12 / 15,
                                          aruco_dict)
    charuco_objs = board.chessboardCorners.reshape((-1, 3))

    wb_paths = natsorted(glob.glob(FOLDER_WB_PATH.format(scanID, "*.TIFF")))

    for wb_path in wb_paths:

        image = cv2.imread(wb_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_points, ids = charuco_detector(gray, aruco_dict, board)

        camera_label, ext  = os.path.splitext(os.path.basename(wb_path))
        camera = None
        for c in chunk.cameras:
            if c.label == camera_label:
                camera = c

        if camera is None or img_points is None:
            continue

        .log.info(
            f"Adding markers for {camera_label} from scan {scanID}"
        )

        for img_point, id in zip(img_points, ids):
            position = charuco_objs[id]
            id = str(id)
            duplicate_found = False
            for marker in chunk.markers:
                if marker.label == id:
                        m = marker
                        duplicate_found = True
       
            if not duplicate_found:
                chunk.addMarker()
                m = chunk.markers[-1]
                m.label = id
    
            m.projections[camera] = Metashape.Marker.Projection(Metashape.Vector(img_point))
            m.projections[camera].pinned = True
            m.reference.location = Metashape.Vector(position * 1e-3) # convert from millimeters to meters

    chunk.updateTransform()
    
    doc.save()


def clean_sparse_cloud():
    '''
    Using the position of charuco board, delete tie points outside of board.
    '''
    scanID = scanID
    charuco_board = charuco_board

    doc = Metashape.Document()
    doc.open(METASHAPE_PATH.format(scanID))
    chunk = doc.chunks[-1]

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard_create(charuco_board['n'],
                                          charuco_board['m'],
                                          charuco_board['checker_size'],
                                          charuco_board['checker_size'] * 12 / 15,
                                          aruco_dict)
    
    charuco_objs = board.chessboardCorners.reshape((-1, 3))

    x_min, x_max = np.min(charuco_objs[:, 0])*1e-3, np.max(charuco_objs[:, 0])*1e-3
    y_min, y_max = np.min(charuco_objs[:, 1])*1e-3, np.max(charuco_objs[:, 1])*1e-3

    width = x_max - x_min
    height = y_max - y_min

    for point in chunk.tie_points.points:
        world_coordinate: Metashape.Vector = chunk.transform.matrix * point.coord
        if world_coordinate.z > -.05 and \
            x_min - width * 0.05 < world_coordinate.x < x_max + width * 0.05 and \
               y_min - height * 0.05 < world_coordinate.y < y_max + height * 0.05:
            point.selected = False
        else:
            point.selected = True
    chunk.tie_points.removeSelectedPoints()

    doc.save()

# @asset(
#     deps=[add_markers],
#     required_resource_keys={"paleo_config"}          
# )
# def add_scale_bars():
#     '''
#     From added ChAruCo markers, add scale bars on Metashape to accurately resize project.
#     '''
#     scanID = scanID
#     charuco_board = charuco_board

#     doc = Metashape.Document()
#     doc.open(METASHAPE_PATH.format(scanID))
#     chunk = doc.chunks[-1] 
    
#     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
#     board = cv2.aruco.CharucoBoard_create(charuco_board['n'],
#                                           charuco_board['m'],
#                                           charuco_board['checker_size'],
#                                           charuco_board['checker_size'] * 12 / 15,
#                                           aruco_dict)
    
#     charuco_objs = board.chessboardCorners.reshape((-1, 3))

#     markers_pairs = list(itertools.combinations(chunk.markers, 2))
    
#     for marker1, marker2 in markers_pairs:
#         chunk.addScalebar(marker1, marker2)
#         scalebar = chunk.scalebars[-1]
    
#         marker1_pos = charuco_objs[int(marker1.label)]
#         marker2_pos = charuco_objs[int(marker2.label)]

#         distance = np.linalg.norm(marker1_pos - marker2_pos, ord=2)
    
#         scalebar.reference.distance = distance * 1e-3 #convert from millimeters to meters
    
#     chunk.updateTransform()
    
#     doc.save()


def export_camera_locations():
    '''
    Save JSON file with camera locations (in meters).
    '''
    scanID = scanID

    doc = Metashape.Document()
    doc.open(METASHAPE_PATH.format(scanID))
    chunk = doc.chunks[-1]
    
    cameras = chunk.cameras
    camera_positions = {}
    
    for camera in cameras:
        pose = chunk.transform.matrix * camera.transform

        camera_positions[camera.label] = {
            "x": pose.row(0)[3], 
            "y": pose.row(1)[3],
            "z": pose.row(2)[3]}
        
    .log.info("Saving camera translations vectors...")

    with open(
        OUTPUT_PATH.format(scanID, "cam_positions.json"),
        'w') as f:
        json.dump(camera_positions, f)


def upload_camera_locations():
    '''
    Upload camera location JSON to database.
    '''
    scanID = scanID

    pass


def export_camera_rotations(metashape_doc: Metashape.Document):
    '''
    Save JSON file with camera rotations.
    '''
    chunk = metashape_doc.chunks[-1]
    
    cameras = chunk.cameras
    camera_rotations = {}
    
    for camera in cameras:
        pose = chunk.transform.matrix * camera.transform

        rotation_matrix_np = np.zeros(shape=(3, 3), dtype=np.float32)

        for row_idx in range(3):
            row = pose.row(row_idx)
            rotation_matrix_np[row_idx, :] = np.array([row.x, row.y, row.z])

        rotation_vector, _ = cv2.Rodrigues(rotation_matrix_np)
        camera_rotations[camera.label] = rotation_vector.tolist()
    
    .log.info("Saving camera rotation vectors...")

    with open(
        OUTPUT_PATH.format(scanID, "cam_rotations.json"),
        'w') as f:
        json.dump(camera_rotations, f)



def export_depth_maps(metashape_doc: Metashape.Document):
    '''
    Save depth maps (in ?) of project locally.
    '''
    # Data management
    os.makedirs(OUTPUT_PATH.format(scanID, "depthmaps"), exist_ok=True)

    chunk = metashape_doc.chunks[-1]
            
    for camera in chunk.cameras:
        depth = chunk.depth_maps[camera].image()
        depth *= chunk.transform.scale

        .log.info(f"Saving depth map (in ?) for {camera.label}...")
            
        depth.save(OUTPUT_PATH.format(scanID, f'depthmaps/{camera.label}.TIFF'))


# @asset(
#     deps=[add_markers,
#           clean_sparse_cloud],
#     required_resource_keys={"paleo_config"}
# )
# def build_dense_cloud():
#     '''
#     Build dense point cloud from depth maps.
#     '''
#     scanID = scanID

#     doc = Metashape.Document()
#     doc.open(METASHAPE_PATH.format(scanID))
#     chunk = doc.chunks[-1]

#     chunk.buildPointCloud(source_data=Metashape.DepthMapsData,
#                           point_confidence=True)

#     doc.save()


def build_mesh(metashape_doc: Metashape.Document):
    '''
    Build triangular mesh from depth maps.
    '''
    chunk = metashape_doc.chunks[-1]
    chunk.buildModel(source_data=Metashape.DepthMapsData)

def build_texture(metashape_doc: Metashape.Document):
    '''
    Build texture of mesh.
    '''
    chunk = metashape_doc.chunks[-1]

    chunk.buildUV()
    chunk.buildTexture()



# @asset(
#     deps=[build_dense_cloud],
#     required_resource_keys={"paleo_config"}
# )
# def export_dense_cloud():
#     '''
#     Save point cloud of reconstruction locally.
#     '''
#     scanID = scanID

#     doc = Metashape.Document()
#     doc.open(METASHAPE_PATH.format(scanID))
#     chunk = doc.chunks[-1]

#     chunk.exportPointCloud(OUTPUT_PATH.format(scanID, 'pointcloud.ply'))


def export_mesh(metashape_doc: Metashape.Document):
    '''
    Save mesh of reconstruction locally.
    '''
    chunk = metashape_doc.chunks[-1]
    chunk.exportModel(OUTPUT_PATH.format(scanID, 'mesh.obj'))


def clean_mesh():
    '''
    Delete anything outside of ChArUco board from triangular mesh .
    '''
    scanID = scanID
    charuco_board = charuco_board

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard_create(charuco_board['n'],
                                          charuco_board['m'],
                                          charuco_board['checker_size'],
                                          charuco_board['checker_size'] * 12 / 15,
                                          aruco_dict)
    charuco_objs = board.chessboardCorners.reshape((-1, 3))

    x_min, x_max = np.min(charuco_objs[:, 0])*1e-3, np.max(charuco_objs[:, 0])*1e-3
    y_min, y_max = np.min(charuco_objs[:, 1])*1e-3, np.max(charuco_objs[:, 1])*1e-3

    width = x_max - x_min
    height = y_max - y_min

    x_min -= width * 0.05 
    x_max += width * 0.05
    y_min -= height * 0.05
    y_max += height * 0.05

    width = x_max - x_min
    height = y_max - y_min

    z = 5e-3

    boxOrigin = np.array([x_min+width/2, y_min+height/2, z])

    planeOrigins = np.array([[x_min, y_min+height/2, z],
                            [x_min+width/2, y_min, z],
                            [x_max, y_min+height/2, z],
                            [x_min+width/2, y_max, z]])
    
    planeNormals = np.array([boxOrigin - origin for origin in planeOrigins])

    boxOriginNormal = np.cross( planeNormals[0], planeNormals[1] )

    planeNormals = np.append(planeNormals, boxOriginNormal.reshape(1,-1), axis=0)
    planeOrigins = np.append(planeOrigins, boxOrigin.reshape(1,-1), axis=0)

    for idx in range(planeNormals.shape[0]):
        planeNormals[idx, :] /= np.linalg.norm(planeNormals[idx, :])

    mesh = trimesh.load(OUTPUT_PATH.format(scanID, "mesh.obj"))

    result = mesh.slice_plane(planeOrigins, planeNormals)

    result.export(OUTPUT_PATH.format(scanID, "mesh.obj"))


# @asset(
#     deps=[build_texture],
#     required_resource_keys={"paleo_config"}
# )
# def export_texture():
#     '''
#     Save texture of reconstruction locally.
#     '''
#     scanID = scanID

#     doc = Metashape.Document()
#     doc.open(METASHAPE_PATH.format(scanID))
#     chunk = doc.chunks[-1]

#     chunk.exportTexture(OUTPUT_PATH.format(scanID, 'texture.jpg'))

#     doc.save()


# @asset(
#     deps=[export_dense_cloud],
#     required_resource_keys={"paleo_config", "minio_api"}
# )
# def upload_dense_cloud():
#     '''
#     Upload point cloud of reconstruction to database.
#     '''
#     scanID = scanID

#     minioClient = minio_api.get_client()
#     bucket_name = 'fossils'
#     minioClient.fput_object(bucket_name=bucket_name,
#                             object_name=scanID+'/pointcloud.ply',
#                             file_path=OUTPUT_PATH.format(scanID, 'pointcloud.ply'),)
#                             # content_type='image/jpeg')