import argparse
import numpy as np
import os
import sys

sys.path.append('../')
import src.reconstruction.lookup as lu
from src.utils.file_io import load_json

def reconstruct_many_objects_single_pattern(config):
    lookup_table = np.load(config['look_up_reconstruction'].pop('lookup_table'))

    config['look_up_reconstruction']['lookup_table'] = lookup_table

    reconstruction_directory = config['look_up_reconstruction'].pop('reconstruction_directory')
    scenes = config['look_up_reconstruction'].pop('scenes')

    for scene in scenes:
        print('-' * 15)
        print(f"Reconstructing {scene} folder now")
        print('-' * 15)
        config['look_up_reconstruction']['reconstruction_directory'] = os.path.join(reconstruction_directory, scene)

        lur = lu.LookUpReconstruction(config)
        lur.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look Up Reconstruction for Many Scenes with a single LookUp Table")
    parser.add_argument('-r', '--reconstruction_config', type=str, default=None,
                    help='Path to config JSON with parameters for LookUp Reconstruction')

    args = parser.parse_args()

    reconstruct_many_objects_single_pattern(load_json(args.reconstruction_config))
