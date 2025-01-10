import json
from natsort import natsorted
import numpy as np
import os

def ensure_exists(path: str):
    """
    Check if path to folder exists. If not, create it.

    Parameters
    ----------
    path : str 
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

def get_all_paths(paths: str | list[str]) -> list[str]:
    """
    Check if the given path(s) is a directory or file, and retrieve all files inside (recursively).

    Parameters
    ----------
    paths : str or list of str
        A path or list of paths to files and/or directories.

    Returns
    -------
    list of str
        A list of naturally sorted full paths to all files found in the input path(s).

    Notes
    -----
    Naturally sorted means that the file
    'hello_world_01.txt' will come before 'hello_world_10.txt', because 01 < 10.
    For more information, read
    https://github.com/SethMMorton/natsort/wiki/How-Does-Natsort-Work%3F-(1-%E2%80%90-Basics)
    """
    if isinstance(paths, str):
        paths = [paths]  # Convert a single path to a list for uniform processing

    all_files = []

    for path in paths:
        if os.path.isdir(path):  # If it's a directory, walk through it recursively
            for root, _, files in os.walk(path):
                for file in files:
                    all_files.append(os.path.join(root, file))
        elif os.path.isfile(path):  # If it's a file, add it to the list
            all_files.append(os.path.abspath(path))
        else:
            print(f"The path '{path}' is neither a valid file nor a directory.")

    return natsorted(all_files)

class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder class for numpy types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def numpinize(data: dict) -> dict:
    return {k: (np.array(v) if (type(v) is list or type(v) is tuple) else
               (numpinize(v) if type(v) is dict else v)) for k, v in data.items()}

def load_json(filename: str) -> dict:
    """
    Function to load JSON file.

    Parameters
    ----------
    filename : str
        path to file where JSON data is stored.  
    """
    with open(filename, 'r') as f:
        return numpinize(json.load(f))

def save_json(data: dict, filename: str):
    """
    Function to save data as JSON file.

    Parameters
    ----------
    data : dict
        dictionary containing data
    filename : str
        path to file where JSON data will be saved.  
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)