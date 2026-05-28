###################################
# 
# A convenient, reduced wrapper library of functions from Agisoft Metashape.
# Author: Giancarlo Pereira (NYU)
# Last Updated: 2026-05-01
# Compatible Metashape Version: 2.3
# 
###################################

compatible_major_version = "2.3"
try:
    import Metashape
    print("Metashape successfully found and imported.")
except ImportError:
    raise ImportError("Metashape module not found. Please ensure Agisoft Metashape is installed, " \
                        "its license key is active, and Python wheels are properly built.\n" \
                        "Otherwise, COLMAP is an open-source structure-from-motion and " \
                        "multi-view stereo software and we provide some wrapper functions for it.") 

found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

DEFAULTS = {
    'MATCH_PHOTOS_DEFAULTS': {
        'downscale': 0, 
        'generic_preselection': True, 
        'reference_preselection': False,
        'keypoint_limit': 40000,
        'tiepoint_limit': 4000,
        'filter_stationary_points': False,
        'filter_mask': False,
        'mask_tiepoints': False,
        'reset_matches': False,

        # only play with these if you know what you're doing
        'subdivide_task': True,
        'workitem_size_cameras': 20,
        'max_workgroup_size': 100
    },

    'ALIGN_CAMERAS_DEFAULTS': {
        'reset_alignment': False,
        'min_image': 2,
        'adaptive_fitting': False,
        'subdivide_task': True
    },

    'BUILD_DEPTH_MAPS_DEFAULTS': {
        'downscale': 1,
        'filter_mode': Metashape.ModerateFiltering,
        'max_neighbors': 16,
        'reuse_depth': False,
        'subdivide_task': True,
        'workitem_size_cameras': 20,
        'max_workgroup_size': 100
    },

    'BUILD_DENSE_CLOUD_DEFAULTS': {
        'replace_asset': False,
        'source_data': Metashape.DepthMapsData,
        'point_confidence': True,
        'point_colors': True,
        'keep_depth': True,
        'max_neighbors': 100,
        'uniform_sampling': True,
        'points_spacing': 0.1,

        # only play with these if you know what you're doing
        'subdivide_task': True,
        'workitem_size_cameras': 20,
        'max_workgroup_size': 100
    },

    'BUILD_MODEL_DEFAULTS': {
        'replace_asset': False,
        'source_data': Metashape.DepthMapsData,
        'surface_type': Metashape.SurfaceType.Arbitrary,
        'interpolation': Metashape.EnabledInterpolation,
        'vertex_confidence': True,
        'face_count': Metashape.FaceCount.HighFaceCount,
        'face_count_custom': 5_000_000,
        'vertex_colors': True,
        'build_texture': False,
        'keep_depth': True,

        # only play with these if you know what you're doing
        'volumetric_masks': False,
        'split_in_blocks': False,
        'blocks_size': 250,
        'clip_to_boundary': False,
        'export_blocks': False,
        'trimming_radius': 10,
        'subdivide_task': True,
        'workitem_size_cameras': 20,
        'max_workgroup_size': 100
    },

    'CLEAN_POINT_CLOUD_DEFAULTS': {
        'criterion': Metashape.PointCloud.Criterion.Confidence,
        'threshold': 5
    },

    'CLEAN_MESH_DEFAULTS': {
        'criterion': Metashape.Model.Criterion.VertexConfidence,
        'threshold': 5
    },

    'EXPORT_POINT_CLOUD_DEFAULTS': {
        'source_data': Metashape.DataSource.PointCloudData,
        'binary': True,
        'save_point_color': True,
        'save_point_normal': True,
        'save_point_intensity': True,
        'save_point_classification': True,
        'save_point_confidence': True,
        'save_point_return_number': True,
        'save_point_scan_angle': True,
        'save_point_source_id': True,
        'save_point_timestamp': True,
        'save_point_index': True,

        # only play with these if you know what you're doing
        'comment': 'point cloud generated and saved by Agisoft Metashape in Skelevision pipeline',
        'save_comment': True,
        'raster_transform': Metashape.RasterTransformNone,
        'colors_rgb_8bit': True,
        'image_format': Metashape.ImageFormatPNG,
        'clip_to_boundary': True,
        'clip_to_region': False,
        'block_width': 1000,
        'block_height': 1000,
        'split_in_blocks': False,
        'save_images': False,   
        'subdivide_task': True,
        'no_double_precision': True
    },

    'EXPORT_MESH_DEFAULTS': {
        'binary': True,
        'precision': 6,
        'save_normals': True,
        'save_colors': True,
        'save_confidence': False,
        'save_texture': True,
        'texture_format': Metashape.ImageFormatPNG,
        'save_uv': True,
        'save_cameras': True,
        'save_markers': True,

        # only play with these if you know what you're doing
        'save_udim': False,
        'save_alpha': False,
        'embed_texture': False,
        'strip_extensions': False,
        'raster_transform': Metashape.RasterTransformNone,
        'colors_rgb_8bit': True,
        'gltf_y_up': True,
        'comment': 'mesh generated and saved by Agisoft Metashape in Skelevision pipeline',
        'save_comment': True,
        'clip_to_boundary': True,
        'clip_to_region': False,
        'clip_to_block': False,
        'block_margin': 0.5,
        'save_metadata_xml': False
}}

import numpy as np
def save_opencv_extrinsics_for_metashape(filename: str, labels: list[str], rvecs: list[np.ndarray], tvecs: list[np.ndarray],
         delimiter : str = ';', precision: int = 6):
    """
    Util function to convert list of camera labels, OpenCV rvecs and tvecs 
    to CVS format for Metashape in order 'nxyzabc'.
    """
    with open(filename, "w") as f:
        f.write(f"#n{delimiter}x{delimiter}y{delimiter}z{delimiter}a{delimiter}b{delimiter}c\n")
        for label, rvec, tvec in zip(labels, rvecs, tvecs):
            f.write(label + delimiter + print_opencv_to_metashape_reference(rvec,tvec,delimiter,precision)+ "\n")

def print_opencv_to_metashape_reference(rvec, tvec,
                                        delimiter: str = ';',
                                        precision: int = 6) -> str:
    """
    Util function to convert OpenCV rvec and tvec to location (xyz) 
    and euler angles of rotation (abc) in degrees for Metashape in order 'xyzabc'.

    Parameters
    ----------
        rvec : array_like
            rvec from OpenCV
        tvec : array_like
            tvec from OpenCV
        delimiter : char, optional
            default value is ';' (semicolon)
        precision : int, optional
            default value is 6

    Returns
    -------
        fmt : str
            string formated as 'xyzabc'
    """
    from src.utils.three_d_utils import get_origin
    from src.utils.file_io import print_vector
    xyz = np.asarray(get_origin(rvec, tvec)).flatten() # (3,)
    abc = np.rad2deg(rotation_matrix_to_metashape_opk(rvec)).flatten() # (3,)
    return print_vector(xyz,delimiter,precision) + delimiter + print_vector(abc,delimiter,precision)

def intrinsics_matrix_to_metashape_dictionary(resx,resy,K) -> dict:
    """
    Based on this forum question https://www.agisoft.com/forum/index.php?topic=7523.0
    
    Returns a dictionary with focal length f, affinity b1, non-orthogonality b2, and 
    principal points cx cy from a camera matrix K.
    
    Parameters
    ----------
    resx : int
        width of sensor in pixels
    resy : int
        height of sensor in pixels
    K : array_like
        3x3 camera matrix
    """
    K_dict = {}
    K = np.asarray(K).reshape(3,3)
    fx, fy = K[0,0], K[1,1]
    skew   = K[0,1]
    cx, cy = K[0,2], K[1,2]
    K_dict['f'] = fy
    K_dict['b1'] = fx - fy
    K_dict['b2'] = skew
    K_dict['cx'] = cx - resx/2
    K_dict['cy'] = cy - resy/2
    return K_dict

def rotation_matrix_to_metashape_opk(R):
    """
    Converts rotation matrix R of shape (3,3) to omega, phi, and kappa Euler angles.

    If rvec of shape (3,) is passed instead of R, it converts
    to R using OpenCV Rodrigues.

    Parameters
    ----------
    R : array_like

    Returns
    -------
    tuple containing omega, phi, kappa (all in radians)

    Notes
    -----
    We assume omega is rotation around X, phi is rotation arouynd Y, kappa is rotation around Z.
    In that order, they form the rotation matrix R = Z_kappa Y_phi X_omega

    Check this page https://en.wikipedia.org/wiki/Euler_angles to see how to get these angles from a rotation matrix R.
    """
    if R.shape != (3,3):
        import cv2
        R, _ = cv2.Rodrigues(R)
    
    omega = np.arctan2(R[2, 1], R[2, 2]) # rotation around X
    phi   = np.arcsin(-R[2, 0])          # rotation around Y
    kappa = np.arctan2(R[1, 0], R[0, 0]) # rotation around Z

    # metashape strangely always makes omega be mirrored (i.e. if omega=40, metashape says 140; if omega=-10, metashape says -170)
    omega = np.sign(omega) * np.abs(np.pi - np.abs(omega)) 

    return omega, phi, kappa

def load_images(chunk: Metashape.Chunk,
                image_paths: list[str],
                filegroups: list[int] = None):
    """
    Load list of images onto Agisoft Metashape.
    """
    if filegroups is not None:
        assert len(image_paths) == len(filegroups), "Length of image paths and filegroups do not match."
        chunk.addPhotos(filenames=image_paths, filegroups=filegroups)
    else:
        chunk.addPhotos(image_paths)

def load_sensor_calibration(sensor: Metashape.Sensor,
                            calibration_path: str = None,
                            fixed: bool = False,
                            format = Metashape.CalibrationFormatOpenCV,
                            **kwargs):
    """
    Load camera intrinsics.

    Can pass a path to JSON calibration in OpenCV format or pass the
    key-value pairs as kwargs.
    """
    calib = Metashape.Calibration()
    calib.width = sensor.width
    calib.height = sensor.height
    if calibration_path is not None:
        calib.load(calibration_path, format = format)
    
    for k, v in kwargs.items():
        if hasattr(calib, k):
            setattr(calib, k, v)

    # allow user calibration to either be INITIAL GUESS or FIXED
    if fixed:
        sensor.fixed = True
    sensor.user_calib = calib

def load_image_extrinsics(chunk: Metashape.Chunk,
                          extrinsics_path: str,
                          delimiter=';'):
    """
    Load camera extrinsics.

    Needs to be in CSV format (.txt file is fine) with semicolon (;) delimiter
    with the following columns:
    #n;x;y;z;a;b;c
    where n is camera label
    x;y;z are camera location
    a;b;c are the Euler angles (in degrees) in OPK format, i.e.
    omega is rotation around x axis, phi around y, kappa around z. 

    """
    chunk.importReference(path=extrinsics_path,
                          format=Metashape.ReferenceFormatCSV,
                          items=Metashape.ReferenceItemsCameras,
                          rotation_angles=Metashape.EulerAnglesOPK,
                          load_location=True,
                          load_rotation=True,
                          delimiter=delimiter)

def match_photos(chunk: Metashape.Chunk, **kwargs):
    """
    Match photos in the chunk.

    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing the point cloud to filter.

    **kwargs : dict
        Additional keyword arguments to customize the matchPhotos process. Possible keys include:
        - downscale: options are 0,1,2,4,8 where 0 is no downscaling and 8 is the most downscaling (default: 1)
        - generic_preselection: whether to use generic preselection (default: True).
        - reference_preselection: whether to use reference preselection (default: False).
        - reference_preselection_mode: which reference to use, with options Source, Estimated, and ) (default: Metashape.ReferencePreselectionSource)
        - keypoint_limit: maximum number of keypoints to detect per image (default: 40000).
        - tiepoint_limit: maximum number of tie points to keep per image (default: 4000).
        - filter_mask: whether to use a mask to filter keypoints (default: False).
        - mask_tiepoints: whether to mask tie points (default: False).
        - reset_matches: whether to reset existing matches before matching (default: False).
    """
    chunk.matchPhotos(**kwargs)
    
def align_cameras(chunk: Metashape.Chunk, **kwargs):
    """
    Align cameras/images in the chunk.

    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing images to align / generate sparse point cloud.
    **kwargs : dict
        Additional keyword arguments to customize the alignCameras process. Possible keys include:
        - reset_alignment: Whether to reset existing camera alignment before aligning (default: False).
        - min_image: Minimum number of images that must observe a point for it to be used in alignment (default: 2).
        - adaptive_fitting: Whether to use adaptive fitting for distortion coefficients (default: False). This is useful if you
    """
    chunk.alignCameras(**kwargs)

# def metashape_undistort(chunk: Metashape.Chunk):
#     for camera in chunk.cameras:
#         currentCameraImage = camera.photo.image()
#         calibration = camera.sensor.calibration
#         undistortedCameraImage = currentCameraImage.undistort(calibration)

#         .log.info(
#             f"Undistorted {camera.label} from scan {scanID}"
#         )

#         undistortedCameraImage.save(
#             FOLDER_UNDISTORTED_PATH.format(scanID, f"{camera.label}.JPG")
#             )

# def add_markers():
#     """
#     Detect ChAruCo markers on undistorted images with OpenCV.
#     """
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

#     wb_paths = natsorted(glob.glob(FOLDER_WB_PATH.format(scanID, "*.TIFF")))

#     for wb_path in wb_paths:

#         image = cv2.imread(wb_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         img_points, ids = charuco_detector(gray, aruco_dict, board)

#         camera_label, ext  = os.path.splitext(os.path.basename(wb_path))
#         camera = None
#         for c in chunk.cameras:
#             if c.label == camera_label:
#                 camera = c

#         if camera is None or img_points is None:
#             continue

#         .log.info(
#             f"Adding markers for {camera_label} from scan {scanID}"
#         )

#         for img_point, id in zip(img_points, ids):
#             position = charuco_objs[id]
#             id = str(id)
#             duplicate_found = False
#             for marker in chunk.markers:
#                 if marker.label == id:
#                         m = marker
#                         duplicate_found = True
       
#             if not duplicate_found:
#                 chunk.addMarker()
#                 m = chunk.markers[-1]
#                 m.label = id
    
#             m.projections[camera] = Metashape.Marker.Projection(Metashape.Vector(img_point))
#             m.projections[camera].pinned = True
#             m.reference.location = Metashape.Vector(position * 1e-3) # convert from millimeters to meters

#     chunk.updateTransform()
    
#     doc.save()


# def add_scale_bars():
#     """
#     From added ChAruCo markers, add scale bars on Metashape to accurately resize project.
#     """
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

def build_depth_maps(chunk: Metashape.Chunk, **kwargs):
    """
    Build depth maps.

    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing the point cloud to filter.

    **kwargs : dict
        Additional keyword arguments to customize the buildDepthMaps process. Possible keys include:
        - downscale: options are 1,2,4,8,16 where 1 is no downscaling and 16 is the most downscaling (default: 1)
        - filter_mode (default: Metashape.DepthFilterMode.Moderate)
        - max_neighbors (int): 16
    """
    chunk.buildDepthMaps(**kwargs)   


def build_dense_cloud(chunk: Metashape.Chunk, **kwargs):
    """
    Build dense point cloud.

    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing the point cloud to filter.

    **kwargs : dict
        Additional keyword arguments to customize the buildPointCloud process. Possible keys include:
        - source_data: (default: Metashape.DepthMapsData)
        - uniform_sampling: (default: True)
        - points_spacing: (default: 0.1)
        - max_neighbors: (default: 100)
        - point_confidence: (default: True)
        - point_colors: (default: True)
        - replace_asset: (default: False)
    """
    chunk.buildPointCloud(**kwargs)   


def build_mesh(chunk: Metashape.Chunk, **kwargs):
    """
    Build triangular mesh.

    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing the point cloud to filter.

    **kwargs : dict
        Additional keyword arguments to customize the buildModel process. Possible keys include:
        - source_data: (default: Metashape.DepthMapsData)
        - face_count: (default: Metashape.FaceCount.HighFaceCount)
        - interpolation: (default: Metashape.ModelInterpolation.Enabled)
        - vertex_confidence: (default: True)
        - vertex_colors: (default: True)
        - replace_asset: (default: False)
    """
    chunk.buildModel(**kwargs)   

def build_texture(chunk: Metashape.Chunk):
    """
    Build texture of mesh.

    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing the point cloud to filter.
    """
    chunk.buildUV()
    chunk.buildTexture()


def clean_dense_cloud(chunk: Metashape.Chunk, **kwargs):
    """
    Filter points of point cloud in chunk.
    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing the point cloud to filter.

    **kwargs : dict
        Additional keyword arguments to customize the cleanPointCloud process. Possible keys include:
        - criterion: which criterion to use for filtering (default: Metashape.PointCloud.Criterion.Confidence; other option is Metashape.PointCloud.Criterion.ScanAngle).
        - threshold: The threshold value for the chosen criterion (default: 5 for Confidence).
        """
    chunk.cleanPointCloud(**kwargs)

def clean_mesh(chunk: Metashape.Chunk, **kwargs):
    """
    Filter faces of mesh in chunk.
    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing the mesh to filter.

    **kwargs : dict
        Additional keyword arguments to customize the cleanModel process. Possible keys include:
        - criterion: which criterion to use for filtering (default: Metashape.Model.Criterion.VertexConfidence; other options are Metashape.Model.Criterion.ComponentSize and Metashape.Model.Criterion.PolygonSize).
        - level: The threshold value for the chosen criterion (default: 5 for VertexConfidence).
        """
    chunk.cleanModel()

def export_dense_cloud(chunk: Metashape.Chunk, output_path: str, **kwargs):
    """
    Save point cloud of reconstruction locally.
    Parameters
    ----------
    chunk : Metashape.Chunk
        The chunk containing the point cloud to export.
    output_path : str
        The path to save the exported point cloud file.
    **kwargs : dict
        Additional keyword arguments to customize the exportPointCloud process. Possible keys include:
        - source_data: The source data to export (default: Metashape.DataSource.PointCloudData; other option is Metashape.DataSource.TiePointsData).
        - binary: Whether to save the point cloud in binary format (default: True).
        - save_point_color: Whether to save point colors (default: True).
        - save_point_normal: Whether to save point normals (default: True).
        - save_point_confidence: Whether to save point confidence (default: True).
    """
    import os
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()
    if ext not in ['.ply', 'obj']:
        raise ValueError("Can only export point cloud as either .ply or .obj, but received unsupported {}".format(ext))
    
    format = Metashape.PointCloudFormatNone
    if ext == '.ply':
        format = Metashape.PointCloudFormatPLY
    elif ext == '.obj':
        format = Metashape.PointCloudFormatOBJ

    chunk.exportPointCloud(
        path=output_path,
        format=format,
        )


def export_mesh(chunk: Metashape.Chunk, output_path: str, **kwargs):
    """
    Save mesh of reconstruction locally.
    """

    chunk.exportModel(
        path=output_path,
        **kwargs
        )

# def export_camera_locations(chunk: Metashape.Chunk):
#     """
#     Save JSON file with camera locations (in meters).
#     """ 
#     cameras = chunk.cameras
#     camera_positions = {}
    
#     for camera in cameras:
#         pose = chunk.transform.matrix * camera.transform

#         camera_positions[camera.label] = {
#             "x": pose.row(0)[3], 
#             "y": pose.row(1)[3],
#             "z": pose.row(2)[3]}
        
#     .log.info("Saving camera translations vectors...")

#     with open(
#         OUTPUT_PATH.format(scanID, "cam_positions.json"),
#         'w') as f:
#         json.dump(camera_positions, f)


# def export_camera_rotations(chunk: Metashape.Chunk):
#     """
#     Save JSON file with camera rotations.
#     """
#     cameras = chunk.cameras
#     camera_rotations = {}
    
#     for camera in cameras:
#         pose = chunk.transform.matrix * camera.transform

#         rotation_matrix_np = np.zeros(shape=(3, 3), dtype=np.float32)

#         for row_idx in range(3):
#             row = pose.row(row_idx)
#             rotation_matrix_np[row_idx, :] = np.array([row.x, row.y, row.z])

#         rotation_vector, _ = cv2.Rodrigues(rotation_matrix_np)
#         camera_rotations[camera.label] = rotation_vector.tolist()
    
#     .log.info("Saving camera rotation vectors...")

#     with open(
#         OUTPUT_PATH.format(scanID, "cam_rotations.json"),
#         'w') as f:
#         json.dump(camera_rotations, f)



# def export_depth_maps(chunk: Metashape.Chunk):
#     """
#     Save depth maps (in ?) of project locally.
#     """
#     # Data management
#     os.makedirs(OUTPUT_PATH.format(scanID, "depthmaps"), exist_ok=True)
            
#     for camera in chunk.cameras:
#         depth = chunk.depth_maps[camera].image()
#         depth *= chunk.transform.scale

#         .log.info(f"Saving depth map (in ?) for {camera.label}...")
            
#         depth.save(OUTPUT_PATH.format(scanID, f'depthmaps/{camera.label}.TIFF'))

# def export_texture():
#     """
#     Save texture of reconstruction locally.
#     """
#     scanID = scanID

#     doc = Metashape.Document()
#     doc.open(METASHAPE_PATH.format(scanID))
#     chunk = doc.chunks[-1]

#     chunk.exportTexture(OUTPUT_PATH.format(scanID, 'texture.jpg'))

#     doc.save()