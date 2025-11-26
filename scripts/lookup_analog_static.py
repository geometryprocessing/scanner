import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.reconstruction.lookup import load_lut, save_reconstruction_outputs
from src.reconstruction.configs import get_config, is_valid_config
from src.utils.file_io import get_all_folder_names, get_all_folders
from src.utils.image_utils import (crop, normalize_color, load_ldr,
      denoise_fft, denoise_lowrank, gaussian_blur, generate_mask_binary_structure,
      extract_mask, convert_to_gray, replace_with_nearest)
from src.reconstruction.lookup import naive_lut, c2f_lut

from copy import deepcopy
import numpy as np

def reconstruct(lut, dep, base_path, config, frames):
    original_config_name = config.name

    frame_folders = get_all_folders(base_path)

    # prepare all the data
    # TODO: this is heavy to carry around, but otherwise too repetitive to
    # keep accessing and loading the same data
    roi = config.roi
    images: list[str] = config.images
    pattern_images = []
    white_images = []
    black_images = []
    for folder in frame_folders:
        pattern_images += [crop(np.concatenate([np.atleast_3d(load_ldr(os.path.join(folder, image))) for image in images], axis=2), roi=roi)]
        white_images += [crop(load_ldr(os.path.join(folder, config.white_image)), roi=roi)]
        if config.black_image is not None:
                black_images += [crop(load_ldr(os.path.join(folder, config.black_image)), roi=roi)]
    
    pattern_images = np.array(pattern_images)
    white_images = np.array(white_images)
    black_images = np.array(black_images)

    for frame in frames:  
        fname = str(frame)
        if frame == -1:
            fname = 'all'  
        if config.verbose:
            print('-' * 15)
            print(f"Using {fname} frames")
        summed_pattern = np.sum(pattern_images[:frame].astype(np.int32), axis=0)
        summed_white = np.sum(white_images[:frame].astype(np.int32), axis=0)

        summed_black = None
        if len(black_images) > 0:
            summed_black = np.sum(black_images[:frame].astype(np.int32), axis=0)

        mask_thr: float = config.mask_thr
        image_for_mask = pattern_images if config.use_pattern_for_mask else summed_white
        # if mask_thr close to zero or negative, don't calculate mask
        if np.isclose(mask_thr, 0.) or mask_thr < 0:
            mask = None
        else:
            mask = generate_mask_binary_structure(convert_to_gray(image_for_mask), mask_thr) \
                if config.use_binary_mask else extract_mask(np.atleast_3d(image_for_mask), mask_thr)
            
        normalized = normalize_color(color_image=summed_pattern,
                                    white_image=summed_white,
                                    mask=mask,
                                    black_image=summed_black)
        if config.denoise_input:
            if config.denoise_input_type == 'fft':
                normalized = denoise_fft(normalized, int(config.denoise_input_value))
            elif config.denoise_input_type == 'lowrank':
                normalized = denoise_lowrank(normalized, int(config.denoise_input_value))
            if mask is not None:
                normalized[~mask] = 0.

        if config.blur_input:
            # to avoid blurring background
            # TODO: a bit dangerous of an operation with floating point
            normalized = replace_with_nearest(normalized, '=', 0.)
            normalized = gaussian_blur(normalized, sigmas=int(config.blur_input_sigma))
            # if there is a mask, this won't actually matter
            if mask is not None:
                normalized[~mask] = 0

        if config.use_coarse_to_fine:
            depth_map, index_map, loss_map = c2f_lut(lut,
                                                    dep,
                                                    normalized,
                                                    config.c2f_ks,
                                                    config.c2f_deltas,
                                                    mask=mask,
                                                    use_gpu=config.use_gpu)
        else:
            depth_map, index_map, loss_map = naive_lut(lut, 
                                                    dep,
                                                    normalized,
                                                    config.block_size,
                                                    mask=mask,
                                                    use_gpu=config.use_gpu)
        
        config = deepcopy(config)
        config.name = original_config_name + f'_{fname}_frames'
        save_reconstruction_outputs(folder=base_path,
                                    mask=mask,
                                    depth_map=depth_map,
                                    loss_map=loss_map,
                                    index_map=index_map,
                                    # colors=colors,
                                    config=config)


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description="Reconstructs static scenes with Analog LookUp3D")
    parser.add_argument('-i', '--input', type=str, default=None, required=True,
                        help='Path to input folder to run reconstruction on. It should have' \
                        'multiple scene folders.')
    parser.add_argument('-c', '--configs', nargs='+', type=str,
                        help='LookUp3D Reconstruction configuration -- can either be a path to JSON file ' \
                        'or a known lookup3d config name. Check src/reconstruction/configs.py file.')
    parser.add_argument('-s', '--scenes', nargs='*', type=str,
                        help='Name of scenes inside of input folder. If none passed, script' \
                        'will run reconstruction on *every* scene inside the folder.')
    parser.add_argument('-f', '--frames', nargs='*', type=int,
                        help='Number of frames to averege out; needs to be integer. If none passed, \
                            script will use all frames available.')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='LookUp3D Reconstruction device name (if any) -- if passed,'
                        'it gets added to base_paths to get the correct device.')

    # print params good for debugging
    parser.add_argument('--print_params', '-pp', action='store_true', help='Print the parameters of the provided scene and exit.')
    args, uargs = parser.parse_known_args(args)

    if any(not is_valid_config(config) for config in args.configs):
        raise ValueError(f'Unknown lookup config detected: {args.configs}')

    for config_name in args.configs:
        config, remaining_args = get_config(config_name, uargs)

        scenes = args.scenes
        print(scenes)
        if scenes is None:
            scenes = ['']
        elif len(scenes) == 1 and scenes[0] == 'all': 
            scenes = get_all_folder_names(args.input)
        if args.print_params:
            print(config.to_dict())
            continue
        
        lut, dep = load_lut(config.lut_path, config.is_lowrank, use_gpu=False)
        
        for scene in scenes:
            if config.verbose:
                print('=' * 15)
                print(f"Starting scene {scene}")
            base_path = os.path.join(args.input, scene, args.device)
            reconstruct(lut, dep, base_path, config, args.frames)
            config.dump_json(os.path.join(base_path, f'{config.name}_reconstruction_config.json'))


if __name__ == '__main__':
    main(sys.argv[1:])