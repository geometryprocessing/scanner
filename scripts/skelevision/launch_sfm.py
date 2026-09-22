import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from src.utils.file_io import (
    load_json, load_yaml, get_all_paths, 
    opencv_distortion_coefficients_to_dictionary
    )

def run_colmap_pipeline(config: dict):
    pass

def run_metashape_pipeline(config: dict):
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

    resume = config['metashape'].get('resume', None)
    if resume is None:
        doc.save(os.path.join(config['dataset']['path'], 'project.psx'))
    elif not os.path.isfile(resume):
        raise ValueError(f"Resume file {resume} does not exist.")
    else:
        print(f"Resuming from existing Metashape document at {resume}")
        doc.open(resume, ignore_lock=True)

    chunks = []
    chunk_labels = []
    # should it be a dictionary instead?
    for sensor_name in enumerate(config['dataset']['sensor_names']): # NUM CAMERAS from config
        chunk = doc.addChunk(); chunk.label = f"{sensor_name}"
        chunks.append(chunk)
        chunk_labels.append(chunk.label)
        # collect all images from that sensor
        sensor_path = os.path.join(config['dataset']['path'], sensor_name)
        image_paths = get_all_paths(sensor_path, extensions=config['dataset']['image_format'])
        print("[INFO] Found {} images from camera ID {}, loading all into Metashape".format(len(image_paths), sensor_name))
        metashape.load_images(chunk, image_paths=image_paths)
        doc.save()
        ## there is a possibility of using RIG configuration for Metashape, but it hasn't worked well for me
        
        if config['load']['camera_intrinsics']:
            print("[INFO] Passing precomputed intrinsics to Metashape")
            data = load_json(os.path.join(config['calibration_path'], sensor_name,f'{sensor_name}_camera_intrinsics.json'))
            metashape.load_sensor_intrinsics(chunk.sensors[0], fixed=True,
                        **opencv_distortion_coefficients_to_dictionary(data['dist_coeffs']), 
                        **metashape.intrinsics_matrix_to_metashape_dictionary(resx=data['resx'],resy=data['resy'],K=data['K']))
            doc.save()

        if config['load']['camera_extrinsics']:
            print("[INFO] Passing precomputed extrinsics to Metashape")
            metashape.load_image_extrinsics(chunk, 
                                            extrinsics_path=os.path.join(config['calibration_path'], sensor_name,
                                                                         f'{sensor_name}_camera_extrinsics_metashape.txt'))
            doc.save()

        metashape.match_photos(chunk, **config['metashape']['match_photos'])
        doc.save()
        
        metashape.align_cameras(chunk, **config['metashape']['align_cameras'])
        doc.save()

        # metashape.build_dense_cloud(chunk, )
        # doc.save()


    # if multiple cameras, align chunks and merge chunks
    if len(chunks) > 1:
        # method = 0 means tie points (1 is marker based, 2 is camera based)
        doc.alignChunks(method=0)
        doc.save()
    
        doc.mergeChunks(copy_laser_scans=True, copy_masks=True, copy_depth_maps=True,
                        copy_point_clouds=True, copy_models=True, copy_tiled_models=True,
                        copy_elevations=True, copy_orthomosaics=True, merge_markers=True,
                        merge_tiepoints=True, merge_assets=True)
        doc.save()

    metashape.build_dense_cloud()
    doc.save()

    metashape.build_mesh()

    
    # save results
    # if config.export.
    chunk.exportReport(os.path.join(config.dataset.path, 'metashape_report.pdf'))
    metashape.export_dense_cloud(chunk, os.path.join(config.dataset.path, 'metashape_dense_cloud.ply'))
    metashape.export_mesh(chunk, os.path.join(config.dataset.path, 'metashape_mesh.obj'))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/sfm_config.yaml', required=False,
                        help='path to config file')

    args, extras = parser.parse_known_args()

    config = load_yaml(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    import time

    # save config in the dataset folder for reproducibility (save it as YAML? JSON?)
    # save_config(path=)

    # should we create a logger? pass it to the functions?

    assert config['dataset']['num_cameras'] == len(config['dataset']['sensor_names']), "Number of cameras must match the number of sensor names provided in the config."
    
    if config['camera_calibration']:
        print("[INFO] Running camera calibration pipeline")
        # run_camera_calibration_pipeline(config) 

    if config['pose_registration']:
        print("[INFO] Running pose registration pipeline")
        # run_pose_registration_pipeline(config)


    if config['run_metashape']:
        print("[INFO] Running Metashape pipeline")
        run_metashape_pipeline(config)

    if config['run_colmap']:
        print("[INFO] Running COLMAP pipeline")
        run_colmap_pipeline(config)


if __name__ == '__main__':
    main()