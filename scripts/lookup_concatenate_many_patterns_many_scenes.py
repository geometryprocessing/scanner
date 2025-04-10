import argparse
import numpy as np
import os
import sys

sys.path.append('../')
import src.reconstruction.lookup as lu
from src.utils.image_utils import ImageUtils

def concatenate_many_patterns_into_one(reconstruction_directory: str,
                                       scenes: list[str],
                                       patterns: list[str],
                                       concatenated_pattern: str):

    print('-' * 15)
    print(f"Will concatenate {patterns} into a single one named {concatenated_pattern}")
    for scene in scenes:
        print('-' * 15)
        print(f"Concatenating {scene} folder now")
        print('-' * 15)
        folder = os.path.join(reconstruction_directory, scene)
        concatenated = [np.load(os.path.join(folder, f'{pattern}.npz'))['pattern'] for pattern in patterns]
        np.savez_compressed(os.path.join(folder, f'{concatenated_pattern}'.npz), pattern=concatenated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate many Look Up Tables into a single LookUp Table")
    parser.add_argument('-c', '--concatenation_config', type=str, default=None,
                    help='Path to config JSON with parameters for LookUp Reconstruction')

    args = parser.parse_args()

    config = args.concatenation_config
    reconstruction_directory = config['look_up_concatenation']['reconstruction_directory']
    patterns = config['look_up_concatenation']['patterns']
    concatenated_pattern = config['look_up_concatenation']['concatenated_pattern']
    scenes = config['look_up_concatenation']['scenes']

    concatenate_many_patterns_into_one(reconstruction_directory, scenes, patterns, concatenated_pattern)