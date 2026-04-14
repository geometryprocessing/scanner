###################################
# 
# A convenient, reduced wrapper library of functions from Agisoft Metashape.
# Author: Giancarlo Pereira (NYU)
# Last Updated: 2026-04-11
# 
###################################
try:
    import Metashape
    print("Metashape successfully found and imported.")
except ImportError:
    raise ImportError("Metashape module not found. Please ensure Agisoft Metashape is installed, " \
                        "its license key is active, and Python wheels are properly built. " \
                        "Otherwise, COLMAP is an open-source structure-from-motion and " \
                        "multi-view stereo software and we provide some wrapper functions for it.") 


import glob
import json
import os
import xml.etree.ElementTree as ET

def load_images(chunk: Metashape.Chunk,
                image_paths: list[str],
                filegroups: list[int] = None):
    '''
    Load list of images onto Agisoft Metashape.
    '''
    if filegroups is not None:
        assert len(image_paths) == len(filegroups), "Length of image paths and filegroups do not match."
        chunk.addPhotos(filenames=image_paths, filegroups=filegroups)
    else:
        chunk.addPhotos(image_paths)

def match_photos(chunk: Metashape.Chunk, **kwargs):
    '''
    Match photos in the chunk.
    '''
    chunk.matchPhotos(
        downscale=kwargs.get("downscale", 1),
        generic_preselection=kwargs.get("generic_preselection", True),
        reference_preselection=kwargs.get("reference_preselection", False),
        keypoint_limit=kwargs.get("keypoint_limit", 40000),
        tiepoint_limit=kwargs.get("tiepoint_limit", 4000),
        reuse_matches=kwargs.get("reuse_matches", False),
        subdivide_task=kwargs.get("subdivide_task", True),
        workitem_size_cameras=kwargs.get("workitem_size_cameras", 20),
        max_workgroup_size=kwargs.get("max_workgroup_size", 100)
        )
    
def align_cameras(chunk: Metashape.Chunk, **kwargs):
    '''
    Align cameras/images in the chunk.
    '''
    chunk.alignCameras(
        reset_alignment=kwargs.get("reset_alignment", False),
        min_image=kwargs.get("min_image", 2),
        adaptive_fitting=kwargs.get("adaptive_fitting", False),
        reset_alignment=kwargs.get("reset_alignment", False),
        subdivide_task=True
        )

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
#     '''
#     Detect ChAruCo markers on undistorted images with OpenCV.
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

def build_depth_maps(chunk: Metashape.Chunk, **kwargs):
    '''
    Build depth maps.

    Defaults but can be overridden by kwargs.
    Most relevant ones are:
        downscale (int): 1 (options are 1,2,4,8,16 where 1 is no downscaling and 16 is the most downscaling)
        filter_mode (Metashape.DepthFilterMode): Metashape.DepthFilterMode.Moderate 
        max_neighbors (int): 16
    '''
    chunk.buildDepthMaps(
        # important ones
        downscale=kwargs.get("downscale", 1),
        filter_mode=kwargs.get("filter_mode", Metashape.DepthFilterMode.Moderate),
        max_neighbors=kwargs.get("max_neighbors", 16),
        reuse_depth=kwargs.get("reuse_depth", False),
        subdivide_task=True,
        workitem_size_cameras=kwargs.get("workitem_size_cameras", 20),
        max_workgroup_size=kwargs.get("max_workgroup_size", 100)
        )   


def build_dense_cloud(chunk: Metashape.Chunk, **kwargs):
    '''
    Build dense point cloud.
    '''
    chunk.buildPointCloud(
        # important ones
        replace_asset=kwargs.get("replace_asset", False),
        source_data=kwargs.get("source_data", Metashape.DepthMapsData),
        point_confidence=kwargs.get("point_confidence", True),
        point_colors=kwargs.get("point_colors", True),
        keep_depth=kwargs.get("keep_depth", True),
        max_neighbors=kwargs.get("max_neighbors", 100),
        uniform_sampling=kwargs.get("uniform_sampling", True),
        points_spacing=kwargs.get("points_spacing", 0.1),

        # only play with these if you know what you're doing
        subdivide_task=kwargs.get("subdivide_task", True),
        workitem_size_cameras=kwargs.get("workitem_size_cameras", 20),
        max_workgroup_size=kwargs.get("max_workgroup_size", 100)
        )   


def build_mesh(chunk: Metashape.Chunk, **kwargs):
    '''
    Build triangular mesh.

    Defaults but can be overridden by kwargs.
    Most relevant ones are:
        source_data: Metashape.DepthMapsData
        face_count: Metashape.FaceCount.HighFaceCount
        interpolation: Metashape.ModelInterpolation.Enabled
        vertex_confidence: True
        vertex_colors: True
        replace_asset: False
    '''
    chunk.buildModel(
        # important ones
        replace_asset=kwargs.get("replace_asset", False),
        source_data=kwargs.get("source_data", Metashape.DepthMapsData),
        surface_type=kwargs.get("surface_type", Metashape.SurfaceType.Arbitrary),
        interpolation=kwargs.get("interpolation", Metashape.ModelInterpolation.Enabled),
        vertex_confidence=kwargs.get("vertex_confidence", True),
        face_count=kwargs.get("face_count", Metashape.FaceCount.HighFaceCount),
        face_count_custom=kwargs.get("face_count_custom", 200_000),
        vertex_colors=kwargs.get("vertex_colors", True),
        build_texture=kwargs.get("build_texture", False),
        keep_depth=kwargs.get("keep_depth", True),


        # only play with these if you know what you're doing
        volumetric_masks=kwargs.get("volumetric_masks", False),
        split_in_blocks=kwargs.get("split_in_blocks", False),
        blocks_size=kwargs.get("blocks_size", 250),
        clip_to_boundary=kwargs.get("clip_to_boundary", False),
        export_blocks=kwargs.get("export_blocks", False),
        trimming_radius=kwargs.get("trimming_radius", 10),
        subdivide_task=kwargs.get("subdivide_task", True),
        workitem_size_cameras=kwargs.get("workitem_size_cameras", 20),
        max_workgroup_size=kwargs.get("max_workgroup_size", 100)
        )   

def build_texture(chunk: Metashape.Chunk):
    '''
    Build texture of mesh.
    '''
    chunk.buildUV()
    chunk.buildTexture()


def export_dense_cloud(chunk: Metashape.Chunk, output_path: str, **kwargs):
    '''
    Save point cloud of reconstruction locally.
    '''
    chunk.exportPointCloud(
        path=output_path,
        source_data=kwargs.get("source_data", Metashape.DataSource.PointCloudData),
        binary=kwargs.get("binary", True),
        save_point_color=kwargs.get("save_point_color", True),
        save_point_normal=kwargs.get("save_point_normal", True),
        save_point_intensity=kwargs.get("save_point_intensity", True),
        save_point_classification=kwargs.get("save_point_classification", True),
        save_point_confidence=kwargs.get("save_point_confidence", True),
        save_point_return_number=kwargs.get("save_point_return_number", True),
        save_point_scan_angle=kwargs.get("save_point_scan_angle", True),
        save_point_source_id=kwargs.get("save_point_source_id", True),
        save_point_timestamp=kwargs.get("save_point_timestamp", True),
        save_point_index=kwargs.get("save_point_index", True),
        raster_transform=RasterTransformNone,
        colors_rgb_8bit=True,
        comment='',
        save_comment=True,
        format=PointCloudFormatNone,
        image_format=ImageFormatJPEG,
        clip_to_boundary=True,
        clip_to_region=False,
        block_width=1000,
        block_height=1000,
        split_in_blocks=False,
        save_images=False,
        compression=True,
        tileset_version='1.0',
        screen_space_error=16,
        folder_depth=5,
        subdivide_task=True,
        no_double_precision=False)


def export_mesh(chunk: Metashape.Chunk, output_path: str, **kwargs):
    '''
    Save mesh of reconstruction locally.
    '''

    chunk.exportModel(
        path=output_path,
        format=ModelFormatNone,
        binary=True,
        precision=6,
        save_normals=True,
        save_colors=True,
        save_confidence=False,
        save_texture=True,
        texture_format=ImageFormatJPEG,
        save_uv=True,
        save_cameras=True,
        save_markers=True,
        save_udim=False,
        save_alpha=False,
        embed_texture=False,
        strip_extensions=False,
        raster_transform=RasterTransformNone,
        colors_rgb_8bit=True,
        gltf_y_up=True,
        comment='',
        save_comment=True,
        clip_to_boundary=True,
        clip_to_region=False,
        clip_to_block=False,
        block_margin=0.5,
        save_metadata_xml=False)

# def export_camera_locations(chunk: Metashape.Chunk):
#     '''
#     Save JSON file with camera locations (in meters).
#     ''' 
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
#     '''
#     Save JSON file with camera rotations.
#     '''
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
#     '''
#     Save depth maps (in ?) of project locally.
#     '''
#     # Data management
#     os.makedirs(OUTPUT_PATH.format(scanID, "depthmaps"), exist_ok=True)
            
#     for camera in chunk.cameras:
#         depth = chunk.depth_maps[camera].image()
#         depth *= chunk.transform.scale

#         .log.info(f"Saving depth map (in ?) for {camera.label}...")
            
#         depth.save(OUTPUT_PATH.format(scanID, f'depthmaps/{camera.label}.TIFF'))

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