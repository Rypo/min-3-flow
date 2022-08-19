'''Model configurations.'''
import torch


class BaseConfig:
    def __init__(self, device=None):
        '''Base configuration class for all models'''
        
        #self.seed = seed
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __repr__(self) -> str:
        name = self.__class__.__name__
        # ref: https://stackoverflow.com/a/44595303
        return f'{name}({", ".join("{}={!r}".format(k, v) for k, v in self.to_dict().items())})'
        #return f'{name}({repr_str[:-2]})'

    def to_dict(self) -> dict:
        return self.__dict__#{k:v for k,v in self.__dict__.items() if not k=='base_config'}



class MinDalleConfig(BaseConfig):
    def __init__(self, dtype:torch.dtype=torch.float32, is_mega:bool=True, is_reusable:bool=True, is_verbose=True, models_root:str=None):
        '''Configuration for MinDalle

        Args:
            dtype (torch.dtype): controls model precision, using float16 reduces memory usage by ~4gb over f32 (default: torch.float16)
            is_mega (bool): if True, uses the Mega-dalle model (default: True)
            is_reusable (bool): if True, keeps model in memory. If destroys and frees memory after each stage (default: True)
            is_verbose (bool): if True, prints out model info (default: True)
            models_root (str): path to pretrained models. If None, defaults to ~/.cache/min3flow/min_dalle (default: None)
            
        '''
        super().__init__()
        self.models_root = models_root
        self.dtype = dtype
        self.is_mega = is_mega
        self.is_reusable = is_reusable
        self.is_verbose = is_verbose
        

class MinDalleExtConfig(BaseConfig):
    def __init__(self, dtype:torch.dtype=torch.float16, model_variant:str='mega', is_reusable:bool=True, is_verbose=True, models_root:str=None):
        '''Configuration for MinDalleExt (extension of MinDalle)

        model_variant!='mega' requires weights converted to pytorch with https://github.com/kuprel/min-dalle-flax
        (Not yet supported)

        Args:
            dtype (torch.dtype): controls model precision, using float16 reduces memory usage by ~4gb over f32 (default: torch.float16)
            model_variant (str): the dalle variant to use. Can be {'mega','mini', 'mega_beta', 'mega_bf16', 'mega_full'} (default: 'mega')
            is_reusable (bool): if True, keeps model in memory. If destroys and frees memory after each stage (default: True)
            is_verbose (bool): if True, prints out model info (default: True)
            models_root (str): path to pretrained models. If None, defaults to ~/.cache/min3flow/min_dalle (default: None)
            
        '''
        super().__init__()
        self.models_root = models_root
        self.dtype = dtype
        self.model_variant = model_variant
        self.is_reusable: is_reusable
        self.is_verbose = is_verbose
        
        


class Glid3XLConfig(BaseConfig):
    def __init__(self, guidance_scale=5.0, batch_size=16, steps=100, sample_method='plms', imout_size=(256,256), 
                 diffusion_weight='finetune.pt', kl_weight='kl-f8.pt', bert_weight='bert.pt', weight_root = None):
        '''Configuration for Glid3XL

        Args:
            guidance_scale (float): classifier-free guidance scale. Values higher that ~5.0 have diminishing effects. (default: 3.0)
            batch_size (int): batch size (default: 16)
            steps (int): total number of diffusion steps. Negligible difference higher than 250 (default: 100)
            sample_method (str): diffusion sampling method. Can be {'plms','ddim','ddpm'} (default: 'plms')
            imout_size (tuple(int,int)): output image size. Must be a multiple of 8 (default: (256,256))
            diffusion_weight (str): alias or path to diffusion model weights. 
                If not an existing path or in `weight_root`, treat as alias and attempt to download.
            kl_weight (str): kl-f8.pt or path to LDM first stage model weights. (default: 'kl-f8.pt')
                If not an existing path or in `weight_root` treat as alias and attempt to download.
            bert_weight (str): bert.pt or path to bert model weights. (default: 'bert.pt')
                If not an existing path or in `weight_root` treat as alias and attempt to download.
            weight_root (str): path to pretrained models. (default: None)
                If None, defaults to ~/.cache/min3flow/glid3_xl
        '''

        super().__init__()
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
        #self.seed = self.base_config.seed
        

class Glid3XLClipConfig(Glid3XLConfig):
    def __init__(self, clip_guidance_scale=500, cutn=16, guidance_scale=5.0, batch_size=1, steps=100, sample_method='plms', imout_size=(256,256), 
                 diffusion_weight='finetune.pt', kl_weight='kl-f8.pt', bert_weight='bert.pt', weight_root = None):
        '''Configuration for Glid3XLClip

        Args:
            clip_guidance_scale (float): clip guidance scale. Controls how much the image should match the prompt. Typical range: 100-750. (default: 500)
            guidance_scale (float): classifier-free guidance scale. Values higher that ~5.0 have diminishing effects. (default: 3.0)
            batch_size (int): batch size. Note: values > 1 not yet supported for clip guided Glid3Xl. (default: 1)
            steps (int): number of steps per epoch (default: 100)
            sample_method (str): diffusion sampling method. Can be {'plms','ddim','ddpm'} (default: 'plms')
            imout_size (tuple(int,int)): output image size. Must be a multiple of 8 (default: (256,256))
            diffusion_weight (str): alias or path to diffusion model weights. 
                If not an existing path or in `weight_root`, treat as alias and attempt to download.
            kl_weight (str): kl-f8.pt or path to LDM first stage model weights. (default: 'kl-f8.pt')
                If not an existing path or in `weight_root` treat as alias and attempt to download.
            bert_weight (str): bert.pt or path to bert model weights. (default: 'bert.pt')
                If not an existing path or in `weight_root` treat as alias and attempt to download.
            weight_root (str): path to pretrained models. (default: None)
                If None, defaults to ~/.cache/min3flow/glid3_xl
        '''
        assert batch_size==1, "Clip guided model currently only supports batch_size=1"
        super().__init__(guidance_scale, batch_size, steps, sample_method, imout_size, diffusion_weight, kl_weight, bert_weight, weight_root)

        self.clip_guidance_scale = clip_guidance_scale
        self.cutn = cutn



class SwinIRConfig(BaseConfig):
    def __init__(self, task='real_sr', scale=4, large_model=True, training_patch_size=None, noise=None, jpeg=None, weight_root=None):
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
        super().__init__()
        self.task = task
        self.scale = scale
        self.large_model = large_model
        self.training_patch_size = training_patch_size
        self.noise = noise
        self.jpeg = jpeg
        self.weight_root = weight_root #if model_dir is not None else rel_model_root_swinir()
        #self.base_config.pretrained_root.joinpath(model_dir)
        
