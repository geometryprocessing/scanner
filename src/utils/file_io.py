import json
from natsort import natsorted
import numpy as np
import os

def parse_value(dest_type, value):
    if value == 'None':
        return None
    if isinstance(value, str) and dest_type == bool:
        return value.lower() in ['true', '1']
    if dest_type == list and not isinstance(value, list):
        return [value]
    return dest_type(value)

def is_int(elem) -> bool:
    """
    Hacky way to check if a given element can be converted to integer.
    """
    if elem is None:
        return False
    try:
        int(elem)
        return True
    except ValueError:
        return False

def is_float(elem) -> bool:
    """
    Hacky way to check if a given element can be converted to float.
    """
    if elem is None:
        return False
    try:
        float(elem)
        return True
    except ValueError:
        return False
    
def is_json(filename) -> bool:
    """
    Check if a given path to file is JSON object.

    NOTE: This function does not check if file exists.
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() == '.json'

def is_yaml(filename) -> bool:
    """
    Check if a given path to file is YAML object.

    NOTE: This function does not check if file exists.
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() in ['.yml', '.yaml']

def ensure_exists(path: str):
    """
    Check if path to folder exists. If not, create it.

    Parameters
    ----------
    path : str 
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

def get_folder_from_file(path: str) -> str:
    """
    Returns the path to the folder containing the file.

    Parameters
    ----------
    path : str
        A path to file. If path is already a directory, returns the same path.

    Returns
    -------
    str
        The string containing the path to folder.

    """
    return os.path.dirname(path)

def get_all_folder_names(path: str) -> list[str]:
    """
    Retrieves all directories' names within one level of path given.

    Parameters
    ----------
    path : str
        A path to a directory.

    Returns
    -------
    list of str
        A list of naturally sorted of names of all directories found in the input directory.

    Notes
    -----
    Naturally sorted means that the file
    'hello_world_01.txt' will come before 'hello_world_10.txt', because 01 < 10.
    For more information, read
    https://github.com/SethMMorton/natsort/wiki/How-Does-Natsort-Work%3F-(1-%E2%80%90-Basics)
    """

    return natsorted([f.name for f in os.scandir(path) if f.is_dir()])

def get_all_folders(path: str) -> list[str]:
    """
    Retrieves all directories within one level of path given.

    Parameters
    ----------
    path : str
        A path to a directory.

    Returns
    -------
    list of str
        A list of naturally sorted full paths to all directories found in the input directory.

    Notes
    -----
    Naturally sorted means that the file
    'hello_world_01.txt' will come before 'hello_world_10.txt', because 01 < 10.
    For more information, read
    https://github.com/SethMMorton/natsort/wiki/How-Does-Natsort-Work%3F-(1-%E2%80%90-Basics)
    """

    return natsorted([f.path for f in os.scandir(path) if f.is_dir()])


def get_all_paths(paths: str | list[str], extensions: list[str] = None) -> list[str]:
    """
    Check if the given path(s) is a directory or file, and retrieve all files inside (recursively).

    Parameters
    ----------
    paths : str or list of str
        A path or list of paths to files and/or directories.
    extensions : list of str, optional
        A list of file extensions to filter for. If None, all files are returned.

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
    if extensions is not None:
        if isinstance(extensions, str):
            extensions = [extensions]
        extensions = [ext.lower() for ext in extensions]  # Normalize extensions to lowercase
        extensions = [ext if ext[0] == '.' else '.' + ext for ext in extensions]  # add dot if not there
    if isinstance(paths, str):
        paths = [paths]  # Convert a single path to a list for uniform processing

    all_files = []
    for path in paths:
        # If it's a directory, walk inside it
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    p = os.path.join(root, file)
                    if extensions is None or os.path.splitext(p)[1].lower() in extensions:
                        all_files.append(p)
        # if it's a file
        elif os.path.isfile(path):
            # add file to the list (if extensions passed, see if extension matches)
            if extensions is None or os.path.splitext(path)[1].lower() in extensions:
                all_files.append(os.path.abspath(path))
        else:
            print(f"The path '{path}' is neither a valid file nor a directory. Returning empty list")

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
    if not os.path.isfile(filename):
        raise ValueError(f'{filename} does not exist')
    if not is_json(filename):
        raise ValueError(f'{filename} is not JSON')
    with open(filename, 'r') as f:
        return numpinize(json.load(f))

def save_json(data: dict,
              filename: str):
    """
    Function to save data as JSON file. If path to file
    does not exist, this function creates the necessary directories.

    Parameters
    ----------
    data : dict
        dictionary containing data
    filename : str
        path to file where JSON data will be saved.  
    """
    ensure_exists(filename)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)

def update_yaml_value(yaml_data: dict, key: str, data):
    keys = key.split('.')
    accessable = yaml_data
    for k in keys[:-1]:
        accessable = accessable[k]
    accessable[keys[-1]] = data
    return yaml_data


def load_yaml(filename: str,
              cli_args: dict = None) -> dict:
    """
    Function to load YAML file.

    Parameters
    ----------
    filename : str
        path to file where YAML data is stored.  
    """
    import yaml
    if not os.path.isfile(filename):
        raise ValueError(f'{filename} does not exist')
    if not is_yaml(filename):
        raise ValueError(f'{filename} is not a YAML file')
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    
    # TODO: FIX THIS! is cli_args dict? I think it might be list
    if cli_args is not None:
        for k, v in cli_args.items():
            if k in data:
                dest_type = type(data[k])
                data[k] = parse_value(dest_type, v) 
    return data

def write_opencv_calibration_to_xml(filename,
                                    resx,
                                    resy,
                                    K,
                                    dist_coeffs):
    """
    Write OpenCV camera calibration into XML file.
    This can be ingested by Agisoft Metashape for loading camera calibration.

    Parameters
    ----------
    filename : str
        Path to XML file where calibration will be saved.
    resx : int
        Image width in pixels.
    resy : int
        Image height in pixels.
    K : np.ndarray
        3x3 camera intrinsic matrix.
    dist_coeffs : np.ndarray or None
        Distortion coefficients. If None, will convert it first to 5*[0]
    
    """
    # TODO: NEED TO SAVE R AND T AS WELL
    # IMPORTANT TO NOTE THAT IN METASHAPE, origin is LOCATION
    # origin is DIFFERENT from T
    # origin is calculated as -R^T @ T 
    import xml.etree.ElementTree as ET
    opencv_storage = ET.Element("opencv_storage")
    ET.SubElement(opencv_storage, "image_Width").text = str(resx)
    ET.SubElement(opencv_storage, "image_Height").text = str(resy)
    camera_matrix = ET.SubElement(opencv_storage, "Camera_Matrix", type_id="opencv-matrix")
    ET.SubElement(camera_matrix, "rows").text = "3"
    ET.SubElement(camera_matrix, "cols").text = "3"
    ET.SubElement(camera_matrix, "dt").text = "d"
    ET.SubElement(camera_matrix, "data").text = ' '.join(map(str, K.flatten()))
    dist_coeffs = ET.SubElement(opencv_storage, "Distortion_Coefficients", type_id="opencv-matrix")
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    ET.SubElement(dist_coeffs, "rows").text = str(len(dist_coeffs))
    ET.SubElement(dist_coeffs, "cols").text = "1"
    ET.SubElement(dist_coeffs, "dt").text = "d"
    ET.SubElement(dist_coeffs, "data").text = ' '.join(map(str, dist_coeffs.flatten()))
    tree = ET.ElementTree(opencv_storage)
    tree.write(filename)