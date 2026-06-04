import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from src.utils.file_io import (
    load_yaml, load_json, get_all_paths, 
    opencv_distortion_coefficients_to_dictionary,
    get_all_folder_names
    )

def run_metashape_pipeline(data_path: str, calibration_path: str = None, extension: str = 'png'):
    """
    Parameters
    ----------
    config: dict
        A dictionary containing configuration parameters for the Metashape pipeline.

    intrinsics: dict, optional
        If passed, this should be a dictionary mapping 
        sensor names (str) to their intrinsic parameters (K and dist_coeffs).
        If intrinsics is passed, it will not use Metashape to estimate them.
    extrinsics: dict, optional
        If passed, this should be a dictionary mapping
        image names (str) to their extrinsic parameters (R and T, 
        following OpenCV convention (https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)).

    """
    try:
        import Metashape
        print("[INFO] Metashape successfully found and imported.")
    except ImportError:
        raise ImportError("Metashape module not found. Please ensure Agisoft Metashape is installed, " \
                            "its license key is active, and Python wheels are properly built.\n" \
                            "Otherwise, COLMAP is an open-source structure-from-motion and " \
                            "multi-view stereo software and we provide some wrapper functions for it.") 
    import src.reconstruction.metashape as metashape 
    
    # create metashape document
    doc = Metashape.Document()

    doc.save(os.path.join(data_path, 'project.psx'))

    sensor_names = get_all_folder_names(data_path)
    ### HACKy
    sensor_names = [s for s in sensor_names if '.files' not in s]

    print("[INFO] Found {} sensors: {}".format(len(sensor_names), sensor_names))

    chunks = []
    chunk_labels = []
    # should it be a dictionary instead?
    for i, sensor_name in enumerate(sensor_names): # NUM CAMERAS from config
        print("[INFO] Working on sensor {}".format(sensor_name))
        chunk = doc.addChunk(); chunk.label = f"{sensor_name}"
        chunks.append(chunk)
        chunk_labels.append(chunk.label)
        # collect all images from that sensor
        sensor_path = os.path.join(data_path, sensor_name)
        image_paths = get_all_paths(sensor_path, extensions=extension)
        print("[INFO] Found {} images from camera ID {}, loading all into Metashape".format(len(image_paths), sensor_name))
        metashape.load_images(chunk, image_paths=image_paths)
        doc.save()
        ## there is a possibility of using RIG configuration for Metashape, but it hasn't worked well for me
        
        if calibration_path is not None:
            print("[INFO] Passing precomputed intrinsics to Metashape")
            data = load_json(os.path.join(calibration_path, sensor_name,f'{sensor_name}_camera_intrinsics.json'))
            metashape.load_sensor_intrinsics(chunk.sensors[0], fixed=True,
                        **opencv_distortion_coefficients_to_dictionary(data['dist_coeffs'], swap_p1_p2=True), 
                        **metashape.intrinsics_matrix_to_metashape_dictionary(resx=data['resx'],resy=data['resy'],K=data['K']))
            doc.save()

            print("[INFO] Passing precomputed extrinsics to Metashape")
            metashape.load_image_extrinsics(chunk, 
                                            extrinsics_path=os.path.join(calibration_path, sensor_name,
                                                                            f'{sensor_name}_camera_extrinsics_metashape.txt'))
            doc.save()

        # print("[INFO] Maching photos...")
        # d = metashape.DEFAULTS['MATCH_PHOTOS_DEFAULTS']
        # d['reference_preselection'] = True
        # metashape.match_photos(chunk, **d)
        # doc.save()

        # print("[INFO] ... and aligning cameras")
        # metashape.align_cameras(chunk, **metashape.DEFAULTS['ALIGN_CAMERAS_DEFAULTS'])
        # doc.save()


    # if multiple cameras, align chunks and merge chunks
    if len(chunks) > 1:
        print("[INFO] Multiple chunks ({}) detected with different cameras, merging them now".format(chunk_labels))
        # method = 0 means tie points (1 is marker based, 2 is camera based)
        # doc.alignChunks(chunks=chunks,method=0)
        # doc.save()
    
        doc.mergeChunks(chunks=chunks,
                        copy_laser_scans=True, copy_masks=True, copy_depth_maps=True,
                        copy_point_clouds=True, copy_models=True, copy_tiled_models=True,
                        copy_elevations=True, copy_orthomosaics=True, merge_markers=True,
                        merge_tiepoints=True, merge_assets=True)
        doc.save()

    chunk = doc.chunks[-1]
    print("[INFO] Now working on chunk {}".format(chunk.label))

    print("[INFO] Maching photos...")
    d = metashape.DEFAULTS['MATCH_PHOTOS_DEFAULTS']
    d['reference_preselection'] = True
    metashape.match_photos(chunk, **d)
    doc.save()

    print("[INFO] ... and aligning cameras")
    metashape.align_cameras(chunk, **metashape.DEFAULTS['ALIGN_CAMERAS_DEFAULTS'])
    doc.save()

    d = metashape.DEFAULTS['BUILD_DEPTH_MAPS_DEFAULTS']
    d['filter_mode'] = Metashape.MildFiltering
    metashape.build_depth_maps(chunk, **d)

    metashape.build_dense_cloud(chunk, **metashape.DEFAULTS['BUILD_DENSE_CLOUD_DEFAULTS'])
    doc.save()

    metashape.build_mesh(chunk, **metashape.DEFAULTS['BUILD_MODEL_DEFAULTS'])
    doc.save()

    # save results
    # if config.export.
    chunk.exportReport(os.path.join(data_path, 'metashape_report.pdf'))
    metashape.export_dense_cloud(chunk, output_path=os.path.join(data_path, 'metashape_dense_cloud.ply'))
    metashape.export_mesh(chunk, output_path=os.path.join(data_path, 'metashape_mesh.obj'))

    metashape.clean_dense_cloud(chunk,
                                criterion=Metashape.PointCloud.Criterion.Confidence,
                                threshold=5)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d',
                        required=True,
                        help='path to data folder')
    parser.add_argument('--calibration_path', '-c',
                        required=False, default=None,
                        help='path to data folder')
    parser.add_argument('--extension', '-e',
                        required=False, default='png',
                        help='path to data folder')

    args, extras = parser.parse_known_args()


    import time

    tic = time.time()

    print("[INFO] Running Metashape pipeline")
    run_metashape_pipeline(args.data_path, args.calibration_path, args.extension)

    elapsed = time.time() - tic
    print("[INFO] The Metashape process took {:2f} seconds".format(elapsed))

if __name__ == '__main__':
    main()