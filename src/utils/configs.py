import inspect
import sys
import numpy as np

# TODO: what can we pass here?
# need to know the camera, so it is connected to the camera configs 
# need to pass the "name" (half gray, GSF_21, etc) and this has the images attached to it
# here, can set if gpu, if low rank, if c2f, if tc

class LookUp3DBaseConfig:
    def __init__(self):
        self.learning_rate = 4e-2
        self.n_iter = 512
        self.spp = 64
        self.integrator = 'sdf_direct_reparam'
        self.use_autodiff = True
        self.primal_spp_mult = 4

        self.edge_epsilon = 0.01
        self.refined_intersection = False
        self.pretty_name = 'baseconfig'
        self.name = 'default'
        self.use_finite_differences = False
        self.mask_optimizer = False

        # Clamp the geometry terms used in the reparameterization to avoid extreme outliers
        self.geom_clamp_threshold = 0.05
        self.warp_weight_strategy = 6


        self.use_gpu = False

        # Mitsuba's parallel scene loading can cause issues in combination with our SDFs. 
        # We therefore disable it by default.
        self.use_temporal_consistency = False


        self.use_coarse_to_fine = False 


CONFIGS = {name.lower(): obj
           for name, obj in inspect.getmembers(sys.modules[__name__])
           if inspect.isclass(obj)}

def get_config(config):
    config = config.lower()
    if config in CONFIGS:
        return CONFIGS[config]()
    else:
        raise ValueError(f"Could not find config {config}!")


def apply_cmdline_args(config, unknown_args, return_dict=False):
    """Update flat dictionnary or object from unparsed argpase arguments"""
    return_dict |= isinstance(unknown_args, dict)  # Always return a dict if input is a dict
    unused_args = dict() if return_dict else list()
    if unknown_args is None:
        return unused_args

    def parse_value(dest_type, value):
        if value == 'None':
            return None
        if dest_type == bool:
            return value.lower() in ['true', '1']
        return dest_type(value)

    # Parse input list of strings key=value
    input_args = {}
    if isinstance(unknown_args, list):
        for s in unknown_args:
            if '=' in s:
                k = s[2:s.index('=')]
                v = s[s.index('=') + 1:]
            else:
                k = s[2:]
                v = True
            input_args[k] = v
    else:
        input_args = unknown_args

    for k, v in input_args.items():
        if isinstance(config, dict) and k in config:
            old_v = config[k]
            config[k] = parse_value(type(old_v), v)
            print(f"Overriden parameter: {k} = {old_v} -> {config[k]}")
        elif hasattr(config, k):
            old_v = getattr(config, k)
            setattr(config, k, parse_value(type(old_v), v))
            print(f"Overriden parameter: {k} = {old_v} -> {getattr(config, k)}")
        else:
            if return_dict:
                unused_args[k] = v
            else:
                unused_args.append('--' + k + '=' + v)
    return unused_args


#     "lookup_calibration": {
#         "camera_calibration": {
#             "width": 6464,
#             "height": 4852,
#             "K": [[1.48519991e+04, 0.00000000e+00, 3.18182681e+03],
#                    [0.00000000e+00, 1.48677088e+04, 2.46895383e+03],
#                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
#             "dist_coeffs": [[-1.00797656e-01, -9.68012384e-01,  4.15507574e-04,
#                     -4.92491262e-04,  8.84867983e+00]],
#             "R": [0,0,0],
#             "T": [0,0,0]
#         },
#         "plane_pattern": {
#             "type": "charuco",
#             "rows": 18,
#             "columns": 25,
#             "marker_size": 11,
#             "checker_size": 15,
#             "dictionary": "5X5_250"
#         },
#         "calibration_directory": "REQUIRED",
#         "structure_grammar": {
#             "name": "spiral",
#             "images": ["spiral.tiff"],
#             "num_channels": 3,
#             "utils": {
#                 "white": "white.tiff",
#                 "colors": "white.tiff",
#                 "black": null
#             }
#         },
#         "verbose": true,        
#         "num_cpus": 1,
#         "parallelize_positions": false
#     },
#     "look_up_reconstruction": {
#         "camera_calibration": {
#             "height": 4852,
#             "width": 6464,
#             "dist_coeffs": [-1.0079e-01, -9.6801e-01, 4.155e-04, -4.9249e-04, 8.8487e00],
#             "K": [[1.4852e04, 0, 3.1818e03], [0, 1.48677e04, 2.46895e03], [0,0,1]],
#             "R": [0,0,0],
#             "T": [0,0,0]
#         },
#         "mask_thr": 0.01,
#         "structure_grammar": {
#             "name": "spiral",
#             "images": ["spiral.tiff"],
#             "interpolant": {
#                 "active": false,
#                 "type": "cubic_bspline",
#                 "knots": [100, 100, 100],
#                 "samples": 500
#             },
#             "loss": {
#                 "order": "inf"
#             },
#             "utils": {
#                 "white": "white.tiff",
#                 "black": null,
#                 "black_scale": 0.1,
#                 "roi": [2500, 500, 4500, 2000]
#             }
#         },
#         "reconstruction_directory": "/Users/pereira/Workspace/scanner/data/pawn_1/",
#         "lookup_table": "/Users/pereira/Workspace/scanner/spiral.npy",
#         "debug": true,
#         "verbose": true,
#         "parallelize_pixels": true,
#         "num_cpus": 4,
#         "outputs": {
#             "depth_map": true,
#             "point_cloud": true,
#             "mask": true
#         }
#     }
# }