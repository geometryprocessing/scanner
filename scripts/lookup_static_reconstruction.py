import argparse
import os

import sys
sys.path.append('../')
from src.reconstruction.lookup import load_lut, process_position, save_reconstruction_outputs, naive_lut, c2f_lut
from src.reconstruction.configs import LookUp3DConfig, apply_cmdline_args, get_config, is_valid_lookup_config
from src.utils.file_io import get_all_folder_names


def reconstruct(lut, dep, base_path: str, config: LookUp3DConfig):

    normalized, mask, colors = process_position(base_path, config)
    if config.use_coarse_to_fine:
        depth_map, loss_map, index_map = c2f_lut(lut, dep, normalized, config.c2f_ks, config.c2f_deltas, mask=mask)
    else:
        depth_map, loss_map, index_map = naive_lut(lut, dep, normalized, config.block_size, config.use_gpu, mask=mask)

    save_reconstruction_outputs(folder=base_path,
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
                        'multiple scene folders.')
    parser.add_argument('--configs', nargs='+', type=str,
                        help='LookUp3D Reconstruction configuration -- can either be a path to JSON file ' \
                        'or a known lookup3d config name. Check src/reconstruction/configs.py file.')
    parser.add_argument('--scenes', nargs='*', type=str,
                        help='Name of scenes inside of input folder. If none passed, script' \
                        'will run reconstruction on *every* scene inside the folder.')
    # print params good for debugging
    parser.add_argument('--print_params', '-pp', action='store_true', help='Print the parameters of the provided scene and exit.')
    args, uargs = parser.parse_known_args(args)

    if any(not is_valid_lookup_config(config) for config in args.configs):
        raise ValueError(f'Unknown lookup config detected: {args.configs}')

    for config_name in args.configs:
        config: LookUp3DConfig = get_config(config_name)
        lut, dep = load_lut(config.lut_path, config.is_lowrank, config.use_gpu, config.gpu_device)

        scenes = args.scenes
        if len(scenes) == 0: 
            scenes = get_all_folder_names(args.input)
        
        for scene in scenes:
            base_path = os.path.join(args.input, scene)

            if not os.path.isdir(base_path):
                print(f'Did not find scene {scene}, skipping...')
                continue

            remaining_args = apply_cmdline_args(config, uargs, return_dict=True)
            if config.verbose:
                print(f"Starting {base_path} folder with config {config_name}")
            
            if args.print_params:
                print(config.to_dict())
                continue

            reconstruct(lut, dep, base_path, config)
            config.dump_json(os.path.join(base_path), f'{config_name}_lookup_reconstruction_config.json')

if __name__ == "__main__":
    main(sys.argv[1:])
