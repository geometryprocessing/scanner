import argparse
from datetime import datetime
import numpy as np
import os
import sys

sys.path.append('../')
import src.reconstruction.lookup as lu
from src.utils.file_io import load_json, save_json

def concatenate_many_table_into_one(config):
    reconstruction_directory = config['look_up_concatenation']['reconstruction_directory']
    tables = config['look_up_reconstruction']['tables']
    concatenated_table = config['look_up_reconstruction']['concatenated_table']

    print('-' * 15)
    print(f"Will concatenate {tables} into a single one named {concatenated_table}")

    tables = [os.path.join(reconstruction_directory, f'{table}.npy') for table in tables]
    concatenated_table = os.path.join(reconstruction_directory, f'{concatenated_table}.npy')
    lu.concatenate_lookup_tables(tables,concatenated_table)

    infos = [load_json(os.path.join(reconstruction_directory, f'{table}_calibration_info.json')) for table in tables]

    save_json({'roi': infos[0]['roi'],
               'tables': infos,
               'date': datetime.now().strftime('%m/%d/%Y, %H:%M:%S')},
                os.path.join(reconstruction_directory, f'{concatenated_table}_calibration_info.json'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate many Look Up Tables into a single LookUp Table")
    parser.add_argument('-c', '--concatenation_config', type=str, default=None,
                    help='Path to config JSON with parameters for LookUp Reconstruction')
    
    parser.add_argument('-tables', nargs='*')
    parser.add_argument('-o', '--output', type=str, default=None,
            help='Path to save the concatenated LookUp table')

    args = parser.parse_args()

    if args.concatenation_config:
        concatenate_many_table_into_one(args.concatenation_config)
    else:
        lu.concatenate_lookup_tables(args.tables,args.output) 
