import argparse
import os

import numpy as np

import sys
sys.path.append('../')
from src.reconstruction.lookup import process_position,  save_reconstruction_outputs, naive_lut, tc_lut, c2f_lut
from src.utils.image_utils import ImageUtils
from src.reconstruction.configs import LookUp3DConfig, apply_cmdline_args, get_config, is_valid_lookup_config
from src.utils.file_io import save_json, get_all_folders

def reconstruct(lut, dep, base_path: str, config: LookUp3DConfig):

    frames = get_all_folders(base_path)
    
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
        depth_map, loss_map, index_map = c2f_lut(lut, dep, normalized, config.c2f_ks, config.c2f_deltas, mask=mask)
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
        depth_map, loss_map, index_map = naive_lut(lut, dep, normalized, config.block_size, config.use_gpu, mask=mask)
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
        depth_map, loss_map, index_map = tc_lut(lut, dep, normalized, config.tc_deltas[-1], (prior_index_map).astype(np.uint16), mask=mask)
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
                        help='Path to input folder to run reconstruction on. It should have' \
                        'multiple frame position folders or, in the case of multiview, ' \
                        'multiple camera folders')
    parser.add_argument('--configs', nargs='+', type=str,
                        help='LookUp3D Reconstruction configuration -- can either be a path to JSON file ' \
                        'or a known lookup3d config name. Check src/reconstruction/configs.py file.')
    # print params good for debugging
    parser.add_argument('--print_params', '-pp', action='store_true', help='Print the parameters of the provided scene and exit.')
    args, uargs = parser.parse_known_args(args)
    
    if args.camconfigs is None:
        raise ValueError('Must at least specify one camera config!')

    if any(not is_valid_lookup_config(config) for config in args.configs):
        raise ValueError(f'Unknown lookup config detected: {args.configs}')

    assert len(args.camconfigs) == len(args.configs), "Configs and CameraConfigs should match"

    for config_name in args.configs:
        config: LookUp3DConfig = get_config(config_name)
        lut, dep = config.load_lut()
        base_path = os.path.join(args.input, config.cam.filename)
        
        # TODO: with multiview, command line arguments will apply to all cameras
        # how would I like for the ability to change each separately?
        remaining_args = apply_cmdline_args(config, uargs, return_dict=True)
        if config.verbose:
            print(f"Starting {base_path} folder with config {config_name}")
        
        if args.print_params:
            print(config.to_dict())
            continue

        reconstruct(lut, dep, base_path, config)
        save_json(config.to_dict(), os.path.join(base_path), f'{config_name}_lookup_reconstruction_config.json')

if __name__ == '__main__':
    main(sys.argv[1:])
