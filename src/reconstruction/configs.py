"""
MOST OF THIS IS COPIED AND INSPIRED BY DELIO VICINI AND WENZEL JAKOB
"""
import os

from src.scanner.camera import Camera, get_cam_config
from src.scanner.projector import Projector, get_proj_config
from src.utils.file_io import is_int, is_float, save_json, load_json, parse_value

class ReconstructionConfig:
    def __init__(self,
                name: str = '', camera: str | Camera = '',
                verbose: bool = True,
                roi: tuple = (),
                images: list[str] = [],
                white_image: str = '',
                colors_image: str = '',
                black_image: str = '',
                use_binary_mask: bool = False, mask_thr: float = 0.2,
                save_depth_map: bool = True, save_point_cloud: bool = True,
                save_index_map: bool = True):

        self.name = name
        self.camera = camera
        if isinstance(self.camera, str):
            self.camera = get_cam_config(self.camera)
        self.verbose=verbose
        self.roi = roi

        self.use_binary_mask = use_binary_mask
        self.mask_thr = mask_thr

        self.images = images
        self.white_image = white_image
        self.colors_image = colors_image
        self.black_image = black_image

        # outputs
        self.save_depth_map = save_depth_map
        self.save_point_cloud = save_point_cloud
        self.save_index_map = save_index_map

    def load_config(self, filename):
        data = load_json(filename)
        for k,v in data.items():
            if k == 'camera':
                v = get_cam_config(v)
            if k == 'projector':
                v = get_proj_config(v)
            setattr(self, k, v)
    
    def to_dict(self):
        """
        Returns reconstruction config as a dictionary.
        """
        config = {}
        for k,v in self.__dict__.items():
            if isinstance(v, (Camera, Projector)):
                v = v.to_dict()
            config[k] = v
        return config

    def dump_json(self, filename):
        save_json(self.to_dict(), filename)
    
class StructuredLightConfig(ReconstructionConfig):
    def __init__(self,
                name: str = '', camera: str | Camera = '', projector: str | Projector = '',
                pattern: str = '',
                verbose: bool = True,
                roi: tuple = (),
                images: list = [],
                white_image: str = '',
                colors_image: str = '',
                black_image: str = '',
                vertical_images: list[str] = [], inverse_vertical_images: list[str] = [],
                horizontal_images: list[str] = [], inverse_horizontal_images: list[str] = [], 
                binary_threshold: float = 0, num_bits: int = 3,
                phaseshift_frequency: float | int = 1.0,
                frequency_vector: list[float|int] = [], median_filter: int = 5,
                use_binary_mask: bool = False, mask_thr: float = 0.2,
                save_depth_map: bool = True, save_point_cloud: bool = True,
                save_index_map: bool = True):
        super().__init__(name=name, camera=camera, verbose=verbose, roi=roi, images=images,
                         white_image=white_image, colors_image=colors_image, black_image=black_image,
                         mask_thr=mask_thr, use_binary_mask=use_binary_mask,
                         save_depth_map=save_depth_map, save_point_cloud=save_point_cloud,
                         save_index_map=save_index_map)
        
        self.pattern = pattern
        self.projector = projector
        if isinstance(self.projector, str):
            self.projector = get_proj_config(self.projector)

        # images exclusive to structured light processing
        self.vertical_images = vertical_images
        self.horizontal_images = horizontal_images
        self.inverse_vertical_images = inverse_vertical_images
        self.inverse_horizontal_images = inverse_horizontal_images
        # for XOR, Gray, Binary
        self.binary_threshold = binary_threshold
        # for Hilbert
        self.num_bits = num_bits
        # for Phaseshift
        self.phaseshift_frequency = phaseshift_frequency
        # for MPS
        self.frequency_vector = frequency_vector
        self.median_filter = median_filter

class LookUp3DConfig(ReconstructionConfig):
    def __init__(self,
                name: str = '', camera: str | Camera = '', lut_path: str = '/path/is/required/',
                images: str | list[str] = [],
                roi: tuple = (),
                white_image: str = '',
                colors_image: str = '',
                black_image: str = '',
                verbose: bool = True,
                use_binary_mask: bool = False, use_pattern_for_mask: bool = False, mask_thr: float = 0.2,
                denoise_input: bool = False, denoise_cutoff: int = 0,
                blur_input: bool = False, blur_input_sigma: bool = 0,
                loss_thr: float = 0.2,
                is_lowrank: bool = False,
                use_gpu: bool = False, gpu_device: int = 0, block_size: int = 65536,
                use_coarse_to_fine: bool = False, c2f_ks: list[int] = [], c2f_deltas: list[int] = [],
                use_temporal_consistency: bool = False, tc_deltas: list[int] = [], tc_blur_sigma: int = 5,
                save_depth_map: bool = True, save_point_cloud: bool = True,
                save_loss_map: bool = True, save_index_map: bool = False):

        super().__init__(name=name, camera=camera, verbose=verbose, roi=roi, images=images, 
                         white_image=white_image, colors_image=colors_image, black_image=black_image,
                         mask_thr=mask_thr, use_binary_mask=use_binary_mask,
                         save_depth_map=save_depth_map, save_point_cloud=save_point_cloud,
                         save_index_map=save_index_map)

        self.lut_path = lut_path
        self.loss_thr = loss_thr

        # input manipulation exclusive to lookup
        self.use_pattern_for_mask = use_pattern_for_mask
        self.denoise_input = denoise_input
        self.denoise_cutoff = denoise_cutoff
        self.blur_input = blur_input
        self.blur_input_sigma = blur_input_sigma

        self.is_lowrank = is_lowrank
        
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        # block size works for both GPU and CPU
        self.block_size = block_size

        # coarse-to-fine
        self.use_coarse_to_fine = use_coarse_to_fine
        self.c2f_ks = c2f_ks
        self.c2f_deltas = c2f_deltas

        # temporal consistency
        self.use_temporal_consistency = use_temporal_consistency
        self.tc_deltas = tc_deltas
        self.tc_blur_sigma = tc_blur_sigma # TODO: consider if blur sigma is necessary here

        # output exclusive to lookup
        self.save_loss_map = save_loss_map

CONFIG_DICTS = [
    {
        'name': 'sl-gray',
        'config_class': StructuredLightConfig,
        'camera': 'atlascamera',
        'projector': 'dlpprojector',
        'pattern': 'gray',
        'white_image': 'green.tiff',
        'colors_image': 'white.tiff',
        'black_image': 'black.tiff',
        'save_index_map': True,
        'save_point_cloud': True,
        'save_depth_map': True,
        "vertical_images": ['gray_01.tiff', 'gray_03.tiff', 'gray_05.tiff', 'gray_07.tiff', 'gray_09.tiff', 'gray_11.tiff', 'gray_13.tiff', 'gray_15.tiff', 'gray_17.tiff', 'gray_19.tiff', 'gray_21.tiff'],
        "inverse_vertical_images": ['gray_02.tiff', 'gray_04.tiff', 'gray_06.tiff', 'gray_08.tiff', 'gray_10.tiff', 'gray_12.tiff', 'gray_14.tiff', 'gray_16.tiff', 'gray_18.tiff', 'gray_20.tiff', 'gray_22.tiff'],
        "horizontal_images": ['gray_23.tiff', 'gray_25.tiff', 'gray_27.tiff', 'gray_29.tiff', 'gray_31.tiff', 'gray_33.tiff', 'gray_35.tiff', 'gray_37.tiff', 'gray_39.tiff', 'gray_41.tiff', 'gray_43.tiff'],
        "inverse_horizontal_images": ['gray_24.tiff', 'gray_26.tiff', 'gray_28.tiff', 'gray_30.tiff', 'gray_32.tiff', 'gray_34.tiff', 'gray_36.tiff', 'gray_38.tiff', 'gray_40.tiff', 'gray_42.tiff', 'gray_44.tiff'],        
    },
    {
        'name': 'cheap-sl-gray',
        'parent': 'sl-gray',
        'projector': 'lcdprojector',
    },
    {
        'name': 'lookup-static-base',
        'config_class': LookUp3DConfig,
        'camera': 'atlascamera',
        'white_image': 'green.tiff',
        'colors_image': 'white.tiff',
        'black_image': 'black.tiff',
        'roi': (),
        'mask_thr': 0.1,
        'loss_thr': 0.1,
        'verbose': True,
        'use_gpu': False,
        'block_size': 65536,
        'use_pattern_for_mask': False,
        'use_binary_mask': False,
        'denoise_input': False,
        'blur_input': False,
        'is_lowrank': False,
        'save_point_cloud': True,
        'save_depth_map': True,
        'save_loss_map': True,
        'save_index_map': False
    },
    {
        'name': 'lookup-half-gray',
        'parent': 'lookup-static-base',
        'roi': [2500, 500, 4500, 2000],
        'images': ['gray_01.tiff', 'gray_03.tiff', 'gray_05.tiff', 'gray_07.tiff',
                    'gray_09.tiff', 'gray_11.tiff', 'gray_13.tiff', 'gray_15.tiff', 
                    'gray_17.tiff',  'gray_19.tiff', 'gray_21.tiff'],
    },
]


def apply_cmdline_args(config, unknown_args, return_dict=False):
    """Update flat dictionnary or object from unparsed argpase arguments"""
    return_dict |= isinstance(unknown_args, dict)  # Always return a dict if input is a dict
    unused_args = dict() if return_dict else list()
    if unknown_args is None:
        return unused_args
    
    # Parse input list of strings key=value
    input_args = {}
    if isinstance(unknown_args, list):
        for s in unknown_args:
            print(s)
            if '=' in s:
                k = s[2:s.index('=')]
                v = s[s.index('=') + 1:]
                # Handle commas with no spaces, assume list
                if ',' in s:
                    v = v.split(',')
                    # Handle ints and floats next
                    if all([is_int(elem) for elem in v]):
                        v = [int(elem) for elem in v]
                    elif all([is_float(elem) for elem in v]):
                        v = [float(elem) for elem in v]
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
            # to override camera and projector
            if isinstance(old_v, (Camera, Projector)):
                old_v = old_v.name
            setattr(config, k, parse_value(type(old_v), v))
            print(f"Overriden parameter: {k} = {old_v} -> {getattr(config, k)}")
        else:
            if return_dict:
                unused_args[k] = v
            else:
                unused_args.append('--' + k + '=' + v)
    return unused_args

SCENE_CONFIGS = {}

def create_scene_config_init_fn(name, config_class, **kwargs):
    return (lambda: config_class(name, **kwargs), name)

def process_config_dicts(configs):
    """Takes a list of config dictionary, resolves parent-child dependencies
        and adds them to the config list"""
    assert len({c['name'] for c in configs}) == len(configs), "Each config name has to be unique!"
    name_map = {c['name']: c for c in configs}
    output_dicts = []
    for c in configs:
        current = c
        children = []
        while 'parent' in current:
            children.append(current)
            current = name_map[current['parent']]
            assert not current in children, "Circular dependency is not allowed!"

        final = dict(current)
        for child in reversed(children):
            for k in child:
                final[k] = child[k]
        if 'parent' in final:
            final.pop('parent')
        output_dicts.append(final)
    return output_dicts


PROCESSED_SCENE_CONFIG_DICTS = process_config_dicts(CONFIG_DICTS)
for processed in PROCESSED_SCENE_CONFIG_DICTS:
    fn, name = create_scene_config_init_fn(**processed)
    SCENE_CONFIGS[name] = fn
del fn, name


def is_valid_config(scene):
    return scene in SCENE_CONFIGS


def get_config(scene, cmd_args=None):
    """Retrieve configuration options associated with a given scene"""
    if scene in SCENE_CONFIGS:
        if cmd_args is None:
            return SCENE_CONFIGS[scene]()
        else:
            # Somewhat involved logic to allow for command line arguments to override parameters
            # of the original config dict *and* the processed config object, plus returns any remaining args

            # 1. obtain the dict with the right config name
            d = [d for d in PROCESSED_SCENE_CONFIG_DICTS if d['name'] == scene][0]

            # 2. apply args to the dict
            cmd_args = apply_cmdline_args(d, cmd_args)

            # 3. Instantiate the actual config
            config = create_scene_config_init_fn(**d)[0]()

            # 4. Potentially apply args to the config after instantiation too (might be redundant)
            cmd_args = apply_cmdline_args(config, cmd_args)
            return config, cmd_args
    elif os.path.isfile(scene):
        # TODO: is this allowing too much mess?
        config = ReconstructionConfig()
        config.load_config(scene)
        return config
    else:
        raise ValueError("Invalid scene config name!")