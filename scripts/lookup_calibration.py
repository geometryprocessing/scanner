import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from src.reconstruction.lookup import LookUpCalibration
from src.utils.file_io import load_json

def main(args):
    import argparse
    parser = argparse.ArgumentParser(description="Look Up Calibration and Reconstruction")
    parser.add_argument('-c', '--calibration_config', type=str, required=True,
                        help="Path to config for Look Up Calibration")
    parser.add_argument('-i', '--input', type=str, default=None,
                        help="Path to calibration directory containing position folders.")
    parser.add_argument('--depth', default=False, action=argparse.BooleanOptionalAction,
                        help="Flag to set if LookUp Calibration should (Re)Calculate Depth")
    parser.add_argument('--normalize', default=False, action=argparse.BooleanOptionalAction,
                        help="Flag to set if LookUp should (Re)Normalize Pattern")

    args, uargs = parser.parse_known_args(args)

    config_dict = load_json(args.calibration_config)
    if args.input is not None:
        config_dict['look_up_calibration']['calibration_directory'] = args.input

    luc = LookUpCalibration(args.calibration_config)
    luc.run(args.depth, args.normalize)

if __name__ == "__main__":
    main(sys.argv[1:])