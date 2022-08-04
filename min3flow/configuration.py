'''Model configurations.'''
import os
from pathlib import Path

import torch

ROOT_PATH = Path(__file__).parent.parent
MODEL_ROOT =  ROOT_PATH.joinpath('pretrained') 
OUTPUT_ROOT =  ROOT_PATH.joinpath('output') 


class _BaseConfigAlt:
    '''Alternative BaseConfig for a single, non-local pretrained directory.'''
    def __init__(self, pretrained_root=MODEL_ROOT, output_root=OUTPUT_ROOT, device=None):
        self.pretrained_root = pretrained_root
        self.output_root = output_root
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __repr__(self) -> str:
        
        name = self.__class__.__name__
        repr_str = ''
        # hackery to alias model paths in repr
        for k,v in self.to_dict().items():
            if isinstance(v,Path) and hasattr(self, 'base_config'):
                v = str(Path('{base_config.pretrained_root}')/v.relative_to(self.base_config.pretrained_root))
            
            repr_str += '{}={!r}, '.format(k,v)
        # ref: https://stackoverflow.com/a/44595303
        #return f'{name}({", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items())})'
        return f'{name}({repr_str[:-2]})'

    def to_dict(self) -> dict:
        return {k:v for k,v in self.__dict__.items() if not k=='base_config'}

class BaseConfig:
    def __init__(self, seed=-1, device=None):
        '''Base configuration class for all models
        
        Args:
            seed: (int): random seed
            device: (str, torch.device): device to use
        '''

        self.seed = seed
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __repr__(self) -> str:
        name = self.__class__.__name__
        # ref: https://stackoverflow.com/a/44595303
        return f'{name}({", ".join("{}={!r}".format(k, v) for k, v in self.to_dict().items())})'
        #return f'{name}({repr_str[:-2]})'

    def to_dict(self) -> dict:
        return {k:v for k,v in self.__dict__.items() if not k=='base_config'}



class MinDalleConfig(BaseConfig):
    def __init__(self, dtype:torch.dtype=torch.float16, is_mega:bool=True, is_reusable:bool=True, is_verbose=True, models_root:str=None, base_config:BaseConfig=None):
        '''Configuration for MinDalle

        Args:
            dtype (torch.dtype): controls model precision, using float16 reduces memory usage by ~4gb over f32 (default: torch.float16)
            is_mega (bool): if True, uses the Mega-dalle model (default: True)
            is_reusable (bool): if True, keeps model in memory. If destroys and frees memory after each stage (default: True)
            is_verbose (bool): if True, prints out model info (default: True)
            models_root (str): path to pretrained models. If None, defaults to min_dalle/pretrained (default: None)
            base_config (BaseConfig): base configuration for the model (default: None)
        '''

        self.base_config = BaseConfig() if base_config is None else base_config
        #self.models_root = self.base_config.pretrained_root.joinpath(models_root)
        self.models_root = models_root #if models_root is not None else rel_model_root_dalle()
        self.dtype = dtype
        self.is_mega = is_mega
        self.is_reusable = is_reusable
        self.is_verbose = is_verbose
        self.device = self.base_config.device

class MinDalleExtConfig(BaseConfig):
    def __init__(self, dtype:torch.dtype=torch.float16, model_variant:str='mega', is_reusable:bool=True, is_verbose=True, models_root:str=None, base_config:BaseConfig=None):
        '''Configuration for MinDalleExt (extension of MinDalle)

        Requires weights converted to pytorch with https://github.com/kuprel/min-dalle-flax
        (Not yet supported)

        Args:
            dtype (torch.dtype): controls model precision, using float16 reduces memory usage by ~4gb over f32 (default: torch.float16)
            model_variant (str): the dalle variant to use. Can be {'mega','mini', 'mega_beta', 'mega_bf16', 'mega_full'} (default: 'mega')
            is_reusable (bool): if True, keeps model in memory. If destroys and frees memory after each stage (default: True)
            is_verbose (bool): if True, prints out model info (default: True)
            models_root (str): path to pretrained models. If None, defaults to min_dalle/pretrained (default: None)
            base_config (BaseConfig): base configuration for the model (default: None)
        '''

        self.base_config = BaseConfig() if base_config is None else base_config
        
        #self.models_root = self.base_config.pretrained_root.joinpath(models_root)
        self.models_root = models_root #if models_root is not None else rel_model_root_dalle()
        self.model_variant = model_variant
        self.dtype = dtype
        self.is_reusable = is_reusable
        
        self.is_verbose = is_verbose
        self.seed = self.base_config.seed

class Glid3XLConfig(BaseConfig):
    def __init__(self, guidance_scale=3.0, batch_size=16, steps=100, sample_method='plms', imout_size=(256,256), 
                 diffusion_weight='finetune.pt', kl_weight='kl-f8.pt', bert_weight='bert.pt', 
                 weight_root = None, base_config:BaseConfig=None):
        '''Configuration for Glid3XL

        Args:
            guidance_scale (float): classifier-free guidance scale. Values higher that ~5.0 have diminishing effects. (default: 3.0)
            batch_size (int): batch size (default: 16)
            steps (int): number of steps per epoch (default: 100)
            sample_method (str): diffusion sampling method. Can be {'plms','ddim','ddpm'} (default: 'plms')
            imout_size (tuple(int,int)): output image size. Must be a multiple of 8 (default: (256,256))
            model_path (str): path to diffusion model model weights. If None, defaults to min_glid3xl/pretrained/finetune.pt (default: None)
            kl_path (str): path to LDM first stage model weights. If None, defaults to min_glid3xl/pretrained/kl-f8.pt (default: None)
            bert_path (str): path to bert model weights. If None, defaults to min_glid3xl/pretrained/bert.pt (default: None)
        '''

        self.base_config = BaseConfig() if base_config is None else base_config
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.steps = steps
        self.sample_method = sample_method
        self.imout_size = imout_size
        self.diffusion_weight = diffusion_weight #if model_path is not None else rel_model_root_glid3xl('finetune.pt')
        #self.base_config.pretrained_root.joinpath(model_path)
        self.kl_weight = kl_weight #if kl_path is not None else rel_model_root_glid3xl('kl-f8.pt')
        #self.base_config.pretrained_root.joinpath(kl_path)
        self.bert_weight = bert_weight #if bert_path is not None else rel_model_root_glid3xl('bert.pt')
        self.weight_root = weight_root
        #self.base_config.pretrained_root.joinpath(bert_path)
        self.seed = self.base_config.seed
        

class Glid3XLClipConfig(Glid3XLConfig):
    def __init__(self, clip_guidance_scale=500, cutn=16, guidance_scale=3.0, batch_size=1, steps=100, sample_method='plms', imout_size=(256,256), 
                 diffusion_weight='finetune.pt', kl_weight='kl-f8.pt', bert_weight='bert.pt', 
                 weight_root = None, base_config:BaseConfig=None):
        '''Configuration for Glid3XLClip

        Args:
            clip_guidance_scale (float): clip guidance scale. Controls how much the image should match the prompt. Typical range: 100-750. (default: 500)
            guidance_scale (float): classifier-free guidance scale. Values higher that ~5.0 have diminishing effects. (default: 3.0)
            batch_size (int): batch size. Note: values > 1 not yet supported for clip guided Glid3Xl. (default: 1)
            steps (int): number of steps per epoch (default: 100)
            sample_method (str): diffusion sampling method. Can be {'plms','ddim','ddpm'} (default: 'plms')
            imout_size (tuple(int,int)): output image size. Must be a multiple of 8 (default: (256,256))
            model_path (str): path to diffusion model model weights. If None, defaults to min_glid3xl/pretrained/finetune.pt (default: None)
            kl_path (str): path to LDM first stage model weights. If None, defaults to min_glid3xl/pretrained/kl-f8.pt (default: None)
            bert_path (str): path to bert model weights. If None, defaults to min_glid3xl/pretrained/bert.pt (default: None)
            base_config (BaseConfig): base configuration for the model (default: None)
        '''
        assert batch_size==1, "Clip guided model currently only supports batch_size=1"
        super().__init__(guidance_scale, batch_size, steps, sample_method, imout_size, diffusion_weight, kl_weight, bert_weight, weight_root, base_config)

        self.clip_guidance_scale = clip_guidance_scale
        self.cutn = cutn



class SwinIRConfig(BaseConfig):
    def __init__(self, task='real_sr', scale=4, large_model=True, training_patch_size=None, noise=None, jpeg=None, weight_root=None, base_config:BaseConfig=None):
        '''Configuration for SwinIR

        In contrast to other configurations, the primary purpose of the arguments is to determine which model weights to use.
        Only certain combinations of arguments are valid. For instance, when
            task='real_sr' scale can be {2,4} if large_model=False, otherwise only scale=4 is valid.
            task='classical_sr', scale can be {2, 3, 4, 8} and training_patch_size must be {48, 64}.
            task='lightweight_sr', scale can be {2, 3, 4}

        Args:
            task (str): task to perform. Can be {'classical_sr', 'lightweight_sr', 'real_sr', 'gray_dn', 'color_dn', 'jpeg_car'} (default: 'real_sr')
            scale (int): scale factor if performing upsamping. Can be {1, 2, 3, 4, 8} (default: 4)
            large_model (bool): if True, and task='real_sr' uses a large model trained on more data (default: True)
            training_patch_size (int): Ignored unless task='classical_sr'. If using classical_sr, can be {48, 64} (default: None)
            noise (int): noise strength level. Can be {15, 25, 50}. Ignored unless task={'gray_dn' | 'color_dn'}.  (default: None)
            jpeg (int): jpeg artifact level. Ignored unless task='jpeg_car' (default: None)
            model_dir (str): path to model weights. If None, defaults to min_swinir/pretrained/ (default: None)
            base_config (BaseConfig): base configuration for the model (default: None)

        '''

        self.base_config = BaseConfig() if base_config is None else base_config
        
        self.task = task
        self.scale = scale
        self.large_model = large_model
        self.training_patch_size = training_patch_size
        self.noise = noise
        self.jpeg = jpeg
        self.weight_root = weight_root #if model_dir is not None else rel_model_root_swinir()
        #self.base_config.pretrained_root.joinpath(model_dir)
        
