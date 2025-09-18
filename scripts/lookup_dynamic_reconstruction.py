import argparse
import os

import numpy as np
from natsort import natsorted

import sys
sys.path.append('../')
from src.reconstruction.lookup import process_position,  save_reconstruction_outputs, tc_lut, c2f_lut
from src.utils.image_utils import ImageUtils
from src.utils.configs import LookUp3DConfig, apply_cmdline_args, get_config, is_valid_lookup_config
from src.scanner.camera import get_cam_config, Camera
from src.utils.numerics import blockLookupNumpy


def reconstruct(base_path: str, config: LookUp3DConfig, cam: Camera):

    lut, dep = config.load_lut()

    # swap this guy for the base_path?
    # reconstruction_folder = config['reconstruction_directory']
    reconstruction_folder = base_path
    frames = natsorted([os.path.join(reconstruction_folder, name) for name in os.listdir(reconstruction_folder) if os.path.isdir(os.path.join(reconstruction_folder, name))])
    
    # handle c2f and tc
    c2f_frames = []
    tc_frames = []
    naive_frames = []
    if config.use_coarse_to_fine and config.use_temporal_consistency:
        c2f_frames = [frames[0]]
        tc_frames = frames[1:]
    elif config.use_coarse_to_fine:
        c2f_frames = frames
    elif config.use_temporal_consistency:
        naive_frames = [frames[0]]
        tc_frames = frames[1:]
    else:
        naive_frames = frames


    # COARSE-TO-FINE
    for frame_path in c2f_frames:
        normalized, mask, colors = process_position(frame_path, config)

        minD, L = c2f_lut(lut, normalized, config.c2f_ks, config.c2f_deltas, mask)
        depth_map = np.full(shape=(normalized.shape[:2]), fill_value=-1., dtype=np.float32)
        index_map = np.zeros(shape=(normalized.shape[:2]), dtype=np.uint16)
        loss_map = np.full(shape=(normalized.shape[:2]), fill_value=np.inf, dtype=np.float32)
        depth_map[mask] = np.squeeze(np.take_along_axis(dep[mask],minD[:,None],axis=-1))
        loss_map[mask] = L
        index_map[mask] = minD
        save_reconstruction_outputs(folder=frame_path,
                                    mask=mask,
                                    depth_map=depth_map,
                                    loss_map=loss_map,
                                    index_map=index_map,
                                    colors=colors,
                                    config=config)

    # NAIVE
    for frame_path in naive_frames:
        normalized, mask, colors = process_position(frame_path, config)

        minD, L = blockLookupNumpy(lut[mask], normalized[mask], dtype=np.float32, blockSize=65536)
        depth_map = np.full(shape=(normalized.shape[:2]), fill_value=-1., dtype=np.float32)
        index_map = np.zeros(shape=(normalized.shape[:2]), dtype=np.uint16)
        loss_map = np.full(shape=(normalized.shape[:2]), fill_value=np.inf, dtype=np.float32)
        depth_map[mask] = np.squeeze(np.take_along_axis(dep[mask],minD[:,None],axis=-1))
        loss_map[mask] = L
        index_map[mask] = minD
        save_reconstruction_outputs(folder=frame_path,
                                    mask=mask,
                                    depth_map=depth_map,
                                    loss_map=loss_map,
                                    index_map=index_map,
                                    colors=colors,
                                    config=config)

    # TEMPORAL CONSISTENCY
    for frame_path in tc_frames:
        normalized, mask, colors = process_position(frame_path, config)

        prior_index_map = ImageUtils.gaussian_blur(ImageUtils.replace_with_nearest(index_map, '=', 0), sigmas=config.tc_blur_sigma)
        prior_index_map = ImageUtils.replace_with_nearest(loss_map, '<', config.loss_thr, prior_index_map)

        minD, L = tc_lut(lut, normalized, config.tc_deltas[-1], (prior_index_map).astype(np.uint16), mask)
        depth_map[mask] = np.squeeze(np.take_along_axis(dep[mask],minD[:,None],axis=-1))
        loss_map[mask] = L
        index_map[mask] = minD
        save_reconstruction_outputs(folder=frame_path,
                                    mask=mask,
                                    depth_map=depth_map,
                                    loss_map=loss_map,
                                    index_map=index_map,
                                    colors=colors,
                                    config=config)


def main(args):
    parser = argparse.ArgumentParser(description="Reconstructs scenes with LookUp3D")
    parser.add_argument('-i', '--input', type=str, default=None, required=True,
                        help='Path to input folder containing camera folders')
    parser.add_argument('--camconfigs', nargs='+', type=str,
                        help='Camera configuration -- can either be a path to JSON file ' \
                        'or a known camera config name. Check src/scanner/camera.py file.')
    parser.add_argument('--configs', nargs='+', type=str,
                        help='LookUp3D Reconstruction configuration -- can either be a path to JSON file ' \
                        'or a known lookup3d config name. Check src/utils/configs.py file.' \
                        'NOTE: should be in the same order as camconfigs')

    args, uargs = parser.parse_known_args(args)
    
    if args.camconfigs is None:
        raise ValueError('Must at least specify one camera config!')

    if any(not is_valid_lookup_config(config) for config in args.configs):
        raise ValueError(f'Unknown lookup config detected: {args.configs}')

    assert len(args.camconfigs) == len(args.configs), "Configs and CameraConfigs should match"

    for cam_config, config_name in zip(args.camconfigs, args.configs):
        cam = get_cam_config(cam_config)
        config = get_config(config_name)
        base_path = os.path.join(args.input, cam.filename)
        
        remaining_args = apply_cmdline_args(config, uargs, return_dict=True)
        # print(f"Starting {base_path} folder with config {config_path}")
        reconstruct(base_path, config, cam)

if __name__ == '__main__':
    main(sys.argv[1:])
