import argparse
import numpy as np
import os
import sys

sys.path.append('../')
import src.reconstruction.lookup as lu

def concatenate_many_table_into_one(config):
    reconstruction_directory = config['look_up_concatenation']['reconstruction_directory']
    tables = config['look_up_reconstruction']['tables']
    concatenated_table = config['look_up_reconstruction']['concatenated_table']
    
    print('-' * 15)
    print(f"Will concatenate {tables} into a single one named {concatenated_table}")

    tables = [os.path.join(reconstruction_directory, f'{table}.npy') for table in tables]
    concatenated_table = os.path.join(reconstruction_directory, f'{concatenated_table}.npy')
    lu.concatenate_lookup_tables(tables,concatenated_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate many Look Up Tables into a single LookUp Table")
    parser.add_argument('-c', '--concatenation_config', type=str, default=None,
                    help='Path to config JSON with parameters for LookUp Reconstruction')

    args = parser.parse_args()

    concatenate_many_table_into_one(args.concatenation_config)
