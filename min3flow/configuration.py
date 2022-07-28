import os
from pathlib import Path

import torch
import inspect

ROOT_PATH = Path(__file__).parent.parent
MODEL_ROOT =  ROOT_PATH.joinpath('pretrained') 
OUTPUT_ROOT =  ROOT_PATH.joinpath('output') 

from .min_dalle.min_dalle import _rel_model_root as rel_model_root_dalle
from .min_glid3xl.min_glid3xl import _rel_model_root as rel_model_root_glid3xl
from .min_swinir.min_swinir import _rel_model_root as rel_model_root_swinir


# class BaseConfig:
#     def __init__(self, pretrained_root=MODEL_ROOT, output_root=OUTPUT_ROOT, device=None):
#         self.pretrained_root = pretrained_root
#         self.output_root = output_root
#         self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#     def __repr__(self) -> str:
        
#         name = self.__class__.__name__
#         repr_str = ''
#         # hackery to alias model paths in repr
#         for k,v in self.to_dict().items():
#             if isinstance(v,Path) and hasattr(self, 'base_config'):
#                 v = str(Path('{base_config.pretrained_root}')/v.relative_to(self.base_config.pretrained_root))
            
#             repr_str += '{}={!r}, '.format(k,v)
#         # ref: https://stackoverflow.com/a/44595303
#         #return f'{name}({", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items())})'
#         return f'{name}({repr_str[:-2]})'

#     def to_dict(self) -> dict:
#         return {k:v for k,v in self.__dict__.items() if not k=='base_config'}

class BaseConfig:
    def __init__(self, seed=-1, device=None):
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
        self.base_config = BaseConfig() if base_config is None else base_config
        #self.models_root = self.base_config.pretrained_root.joinpath(models_root)
        self.models_root = models_root if models_root is not None else rel_model_root_dalle()
        self.dtype = dtype
        self.is_mega = is_mega
        self.is_reusable = is_reusable
        self.is_verbose = is_verbose
        self.device = self.base_config.device

class MinDalleExtConfig(BaseConfig):
    def __init__(self, dtype:torch.dtype=torch.float16, model_variant:str='mega', is_reusable:bool=True, is_verbose=True, models_root:str=None, base_config:BaseConfig=None):
        self.base_config = BaseConfig() if base_config is None else base_config
        
        #self.models_root = self.base_config.pretrained_root.joinpath(models_root)
        self.models_root = models_root if models_root is not None else rel_model_root_dalle()
        self.model_variant = model_variant
        self.dtype = dtype
        self.is_reusable = is_reusable
        
        self.is_verbose = is_verbose
        self.seed = self.base_config.seed

class Glid3XLConfig(BaseConfig):
    def __init__(self, guidance_scale=3.0, batch_size=16, steps=100, sample_method='plms', imout_size=(256,256), 
                 model_path=None, kl_path=None, bert_path=None, base_config:BaseConfig=None):

        self.base_config = BaseConfig() if base_config is None else base_config
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.steps = steps
        self.sample_method = sample_method
        self.imout_size = imout_size
        self.model_path = model_path if model_path is not None else rel_model_root_glid3xl('finetune.pt')
        #self.base_config.pretrained_root.joinpath(model_path)
        self.kl_path = kl_path if kl_path is not None else rel_model_root_glid3xl('kl-f8.pt')
        #self.base_config.pretrained_root.joinpath(kl_path)
        self.bert_path = bert_path if bert_path is not None else rel_model_root_glid3xl('bert.pt')
        #self.base_config.pretrained_root.joinpath(bert_path)
        self.seed = self.base_config.seed
        

class Glid3XLClipConfig(Glid3XLConfig):
    def __init__(self, clip_guidance_scale=500, cutn=16, guidance_scale=3.0, batch_size=16, steps=100, sample_method='plms', imout_size=(256,256), 
                 model_path=None, kl_path=None, bert_path=None, base_config:BaseConfig=None):
        
        super().__init__(guidance_scale, batch_size, steps, sample_method, imout_size, model_path, kl_path, bert_path, base_config)

        self.clip_guidance_scale = clip_guidance_scale
        self.cutn = cutn



class SwinIRConfig(BaseConfig):
    def __init__(self, task='real_sr', scale=4, large_model=True, training_patch_size=None, noise=None, jpeg=None, model_dir=None, base_config:BaseConfig=None):
        self.base_config = BaseConfig() if base_config is None else base_config
        
        self.task = task
        self.scale = scale
        self.large_model = large_model
        self.training_patch_size = training_patch_size
        self.noise = noise
        self.jpeg = jpeg
        self.model_dir = model_dir if model_dir is not None else rel_model_root_swinir()
        #self.base_config.pretrained_root.joinpath(model_dir)
        
