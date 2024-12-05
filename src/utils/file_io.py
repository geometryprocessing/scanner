import json
import numpy as np
import os

def ensure_exists(path: str):
    '''
    Check if path to folder exists. If not, create it.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    '''
    JSON encoder class for numpy types
    '''
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

def load_json(path: str) -> dict:
    '''
    Function to load JSON file.
    '''
    with open(path, 'r') as f:
        return numpinize(json.load(f))

def save_json(data: dict, path: str):
    '''
    Function to save data as JSON file.
    '''
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)