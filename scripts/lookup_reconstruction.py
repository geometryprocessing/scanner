import argparse
import os

import numpy as np
from natsort import natsorted

import sys
sys.path.append('/home/vida/Workspace/scanner/')
from src.reconstruction.lookup import LookUpReconstruction, tc_lut, c2f_lut, restrict_lut_depth_range
from src.utils.image_utils import ImageUtils
from src.utils.configs import apply_cmdline_args, get_config
from src.utils.numerics import blockLookupNumpy
from src.utils.file_io import load_json

# TODO: check out if the repetition can be
# swapped for lur.process_position(folder, structure_grammar)


def reconstruct(base_path, config, c2f=False, tc=False):
    
    # TODO: deal with config BEFORE we get here
    # and then config is just a CLASS with all this stuff in there
    if 'look_up_reconstruction' in config:
        config = config['look_up_reconstruction']


    ## TODO: c2f and tc
    ## bring these in from config
    # ks = config['ks']
    # deltas = config['deltas']
    # sigma = config['blur_sigma']


    lur = LookUpReconstruction()
    lur.set_camera(config['camera_calibration'])
    lur.roi = config['roi']
    lur.structure_grammar = config['structure_grammar']
    lur.loss_thr = config['loss_thr']

    lur.verbose = config['verbose']
    lur.outputs = config['outputs']

    mask_thr = config['mask_thr']

    lookup_table = np.load(config['lookup_table'])
    # TODO: handle is table is original size and we want to just use a crop of it
    # lut = ImageUtils.crop(lookup_table[...,:-1], roi=lur.roi)
    # dep = ImageUtils.crop(lookup_table[...,-1], roi=lur.roi)
    lut = lookup_table[...,:-1]
    dep = lookup_table[...,-1]

    # swap this guy for the base_path?
    # reconstruction_folder = config['reconstruction_directory']
    reconstruction_folder = base_path
    frames = natsorted([os.path.join(reconstruction_folder, name) for name in os.listdir(reconstruction_folder) if os.path.isdir(os.path.join(reconstruction_folder, name))])
    
    # handle c2f and tc
    c2f_frames = []
    tc_frames = []
    naive_frames = []
    if c2f and tc:
        c2f_frames = [frames[0]]
        tc_frames = frames[1:]
    elif c2f:
        c2f_frames = frames
    elif tc:
        naive_frames = [frames[0]]
        tc_frames = frames[1:]
    else:
        naive_frames = frames


    # COARSE-TO-FINE
    for frame_path in c2f_frames:
        lur.reconstruction_directory = frame_path
        pattern = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(frame_path,'pattern.tiff')), lur.roi)
        white = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(frame_path,'white.tiff')), lur.roi)
        normalized = ImageUtils.normalize_color(pattern, white)
        lur.colors = np.sqrt(white / np.max(white)).reshape(-1,3).astype(np.float32)
        mask = ImageUtils.extract_mask(pattern, mask_thr)
        lur.mask = mask

        minD, loss_map = c2f_lut(lookup_table, normalized, ks, deltas, mask)
        depth_map = np.full(shape=(normalized.shape[:2]), fill_value=-1., dtype=np.float32)
        depth_map[mask] = np.squeeze(np.take_along_axis(dep[mask],minD[:,None],axis=-1))
        lur.depth_map = depth_map
        lur.loss_map = loss_map
        lur.save_outputs()

    # NAIVE
    for frame_path in naive_frames:
        lur.reconstruction_directory = frame_path
        pattern = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(frame_path,'pattern.tiff')), lur.roi)
        white = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(frame_path,'white.tiff')), lur.roi)
        normalized = ImageUtils.normalize_color(pattern, white)
        lur.normalized = normalized
        lur.colors = np.sqrt(white / np.max(white)).reshape(-1,3).astype(np.float32)
        mask = ImageUtils.extract_mask(pattern, mask_thr)
        lur.mask = mask

        minD, L = blockLookupNumpy(lut[mask], normalized[mask], dtype=np.float32, blockSize=65536)
        depth_map = np.full(shape=(normalized.shape[:2]), fill_value=-1., dtype=np.float32)
        index_map = np.zeros(shape=(normalized.shape[:2]), dtype=np.uint16)
        loss_map = np.full(shape=(normalized.shape[:2]), fill_value=np.inf, dtype=np.float32)
        depth_map[mask] = np.squeeze(np.take_along_axis(dep[mask],minD[:,None],axis=-1))
        loss_map[mask] = L
        index_map[mask] = minD
        lur.depth_map = depth_map
        lur.loss_map = loss_map
        lur.save_outputs()

    # TEMPORAL CONSISTENCY
    for frame_path in tc_frames:
        lur.reconstruction_directory = frame_path
        pattern = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(frame_path,'pattern.tiff')), lur.roi)
        white = ImageUtils.crop(ImageUtils.load_ldr(os.path.join(frame_path,'white.tiff')), lur.roi)
        normalized = ImageUtils.normalize_color(pattern, white)
        lur.normalized = normalized
        lur.colors = np.sqrt(white / np.max(white)).reshape(-1,3).astype(np.float32)
        mask = ImageUtils.extract_mask(pattern, mask_thr)
        lur.mask = mask

        prior_index_map = ImageUtils.gaussian_blur(ImageUtils.replace_with_nearest(index_map, '=', 0), sigmas=sigma)
        prior_index_map = ImageUtils.replace_with_nearest(loss_map, '<', lur.loss_thr, prior_index_map)

        index_map, loss_map = tc_lut(lut, normalized, deltas[-1], (prior_index_map).astype(np.uint16), mask)
        depth_map = np.full(shape=(normalized.shape[:2]), fill_value=-1., dtype=np.float32)
        depth_map[mask] = np.squeeze(np.take_along_axis(dep[mask],index_map[mask,None],axis=-1))
        lur.depth_map = depth_map
        lur.loss_map = loss_map
        lur.save_outputs()

def main(args):
    parser = argparse.ArgumentParser(description="Reconstructs scenes with LookUp3D")
    parser.add_argument('-i', '--input', type=str, default=None, required=True,
                    help='Path to input folder containing camera folders')
    parser.add_argument('--camconfigs', nargs='*', type=str, default=[''],
                        help='In case of multiview, name of camera folders inside input folder,' \
                        'otherwise leave blank.' \
                        'Default is blank.')
    parser.add_argument('--configs', nargs='*', type=str, required=True,
                    help="Path to all config json files"
                    "(pass it in order that matches the cameras)")
    parser.add_argument('--c2f', action='store_true',
                    help="Flag to set if coarse-to-fine (C2F) should be used" \
                    "(NOTE: if used with tc, only first frame will use c2f)")
    parser.add_argument('--tc' ,action='store_true',
                    help="Flag to set if temporal consistency (TC) should be used" \
                    "(NOTE: if used with static scene, nothing happens)")

    args, uargs = parser.parse_known_args(args)
    
    assert len(args.camconfigs) == len(args.configs), "Configs and CameraConfigs should match"

    for cam, config_name in zip(args.cameras, args.configs):
        config = get_config(config_name)
        remaining_args = apply_cmdline_args(config, uargs, return_dict=True)
        config = load_json(config_path)
        base_path = os.path.join(args.input, cam)
        print(f"Starting {base_path} folder with config {config_path}")
        reconstruct( base_path, config, args.c2f, args.tc)

if __name__ == '__main__':
    main(sys.argv[1:])
