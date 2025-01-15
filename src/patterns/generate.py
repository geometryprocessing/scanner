import hilbert
import numpy as np
import structuredlight as sl

from src.scanner.calibration import Charuco

def generate_pattern(dsize: tuple[int, int], type: str, dims: int=3) -> np.ndarray | list:
    """
    Parameters
    ----------
    dsize : tuple
        width, height (both int) of projector resolution
    type : str
        ('gray', 'binary', 'xor', 'hilbert') structured light patterns
    dims : int
        Number of dimensions for Hilbert pattern generation
        Default is 3

    Returns
    -------
    pattern
        list of patterns
    """
    width, height = dsize

    match type.lower():
        case 'gray':
            pattern = sl.Gray().generate(dsize)
        case 'binary':
            pattern = sl.Binary().generate(dsize)
        case 'bin':
            pattern = sl.Binary().generate(dsize)
        case 'xor':
            pattern = sl.XOR().generate(dsize)
        case 'hilbert':
            max_dim = max(dsize)
            num_bits = int(np.ceil(np.log2(max_dim) / dims))
            locs = hilbert.decode(np.arange(max_dim, dtype=np.uint16), dims, num_bits)
            pattern = np.broadcast_to(locs + 1, shape=(height,width,dims)) * (2**(8-num_bits)) - 1
        case _:
            print("Unrecognized structured light pattern type, defaulting to gray...\n")
            pattern = sl.Gray().generate(dsize)

    return pattern

def generate_charuco(dsize : tuple[int, int], 
                     rows : int,
                     columns : int,
                     checker_size : int,
                     marker_size : int,
                     dictionary : dict) -> np.ndarray:
    """
    Parameters
    ----------
    dsize : tuple
        width, height (both int) of projector resolution

    Returns
    -------
    charuco image
    """
    return Charuco(rows, columns, checker_size, marker_size, dictionary).create_image(dsize)