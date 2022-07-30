import gc
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from PIL import Image, ImageDraw, ImageFont

from .min_dalle import  MinDalle, MinDalleExt
from .min_glid3xl import Glid3XL, Glid3XLClip, utils as gutils
from .min_swinir import SwinIR, utils as sutils

from .configuration import BaseConfig, MinDalleConfig, MinDalleExtConfig, Glid3XLConfig, Glid3XLClipConfig, SwinIRConfig




class Min3Flow:
    '''Control flow for 3 pipeline stages.

        Typical usage:
            1. Min3Flow.generate: Generate a set of (256,256) images given a text prompt (MinDalle).
            2. Min3Flow.diffuse: Perform diffusion sampling given the generated image(s) and prompt to produce modified outputs (Glid3XL). 
            3. Min3Flow.upscale: upscale the final (256,256) image(s) to 1024x1024 (SwinIR).

        Args:
            dalle_config (MinDalleConfig): Configuration for the MinDalle model. (default: None)
            glid3xl_config (Glid3XLConfig|Glid3XLClipConfig): Configuration for the Glid3XL model. (default: None)
            swinir_config (SwinIRConfig): Configuration for the SwinIR model. (default: None)
            base_config (BaseConfig): Configuration for the base model.  (default: None)
            persist (bool|'cpu'): Whether to persist the prior stages models in memory after a stage is complete. (default: False)
                If False, at the begining of each stage, unload non-active stage models and free cached memory. 
                If 'cpu', at the begining of each stage, move all non-active stage models to cpu and free cached memory. 
                If True, the model will be persisted in GPU memory. Warning: With f16 will use >16gb VRAM, with f32 VRAM usage > 19gb .
            seed (int): Random seed for the models. Active when > 0 (default: -1)
            device (str): Device to use for the models. Defaults to cuda if available. (default: None)
    '''
    def __init__(self, dalle_config=None, glid3xl_config=None, swinir_config=None, base_config:BaseConfig=None, persist:(bool|str)=False, seed: int = -1, device=None) -> None:
        self.base_config = base_config if base_config is not None else BaseConfig(seed=seed, device=device)
        self.dalle_config = dalle_config if dalle_config is not None else MinDalleConfig(base_config=self.base_config)
        self.glid3xl_config = glid3xl_config if glid3xl_config is not None else Glid3XLConfig(base_config=self.base_config)
        self.swinir_config = swinir_config if swinir_config is not None else SwinIRConfig(base_config=self.base_config)

        self.persist = persist
        self.seed = seed

        self.model_dalle = None
        self.model_glid3xl = None
        self.model_swinir = None

        self._cache = {}

    def __repr__(self) -> str:
        config= 'Min3Flow(\n {}, \n {}, \n {}, \n {}\n)'.format(self.dalle_config, self.glid3xl_config, self.swinir_config, self.base_config)
        return config


    def _begin_stage(self, stage: str) -> None:
        '''Unload or transfer non-active stage models and free cached memory.'''
        if self.seed > 0:
            torch.manual_seed(self.seed)
        if not self.persist:
            if stage == 'generate':
                #self.model_dalle = None
                self.model_glid3xl = None
                self.model_swinir = None
            elif stage == 'diffuse':
                self.model_dalle = None
                #self.model_glid3xl = None
                self.model_swinir = None
            elif stage == 'upscale':
                self.model_dalle = None
                self.model_glid3xl = None
                #self.model_swinir = None
        elif self.persist == 'cpu':
            if stage == 'generate':
                if self.model_dalle is not None: self.model_dalle._to(self.base_config.device)
                if self.model_glid3xl is not None: self.model_glid3xl._to('cpu')
                if self.model_swinir is not None: self.model_swinir._to('cpu')    
            elif stage == 'diffuse':
                if self.model_dalle is not None: self.model_dalle._to('cpu')
                if self.model_glid3xl is not None: self.model_glid3xl._to(self.base_config.device)
                if self.model_swinir is not None: self.model_swinir._to('cpu')  
            elif stage == 'upscale':
                if self.model_dalle is not None: self.model_dalle._to('cpu') 
                if self.model_glid3xl is not None: self.model_glid3xl._to('cpu')
                if self.model_swinir is not None: self.model_swinir._to(self.base_config.device)  

        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(self, text: str, grid_size: int = 4, supercondition_factor: int = 16, temperature: float = 1.0, top_k: int = 256) -> Image.Image:
        '''Generate a set of (256,256) images given a text prompt (MinDalle).
        
        Args:
            text (str): Text prompt to generate images from.
            grid_size (int): Size of image output grid in x,y. E.g. grid_size=4 produces 16 (256,256) images. (default: 4)
            supercondition_factor (int): Higher values better match text prompt, but narrow image out variety. (default: 16)
            temperature (float): Values > 1 supress the influence of the most probable tokens in top_k, providing more diverse sampling. (default: 1.0)
            top_k (int): The number of most probable tokens to use when sampling each image token. (default: 256)

        Returns:
            Image.Image: A set of (256,256) images arranged in a grid of (grid_size x grid_size).
        '''
        self._begin_stage('generate')

        if self.model_dalle is None:
            if isinstance(self.dalle_config, MinDalleExtConfig):
                self.model_dalle = MinDalleExt(**self.dalle_config.to_dict())
            else:
                self.model_dalle = MinDalle(**self.dalle_config.to_dict())

        
        image = self.model_dalle.generate_image(
            text, 
            self.seed, 
            grid_size, 
            temperature=temperature,
            top_k = top_k,
            supercondition_factor=supercondition_factor,
            is_verbose=True
        )
        self._cache['text'] = text
        self._cache['grid_size'] = grid_size
        return image

    def show_grid(self, image: Image.Image, grid_size=None, cell_h=256, cell_w=256) -> Image.Image:
        '''Show a grid of images with index annotations.

        Args:
            image (Image.Image): Image grid to show.
            grid_size (int): The number of images in the grid in both x and y. (default: None)
                If None, use last grid_size value if available else infer from the image size.
            cell_h (int): The height of each image in the grid. (default: 256)
            cell_w (int): The width of each image in the grid. (default: 256)

        Returns:
            Image.Image: A grid of images with index annotations.
        '''

        if grid_size is None:
            grid_size = self._cache.get('grid_size', 
                int(image.shape[-1]/cell_w) if isinstance(image, torch.Tensor) else int(imgc.width / cell_w)
            )

        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(vutils.make_grid(image, nrow=grid_size , padding=0, normalize=False))
        
        imgc = image.copy()
        
        draw = ImageDraw.Draw(imgc)
        fnt = ImageFont.truetype("DejaVuSans.ttf", 20)

        
        for i,(x,y) in enumerate(np.mgrid[:grid_size,:grid_size].T.reshape(-1,2)*[cell_w,cell_h]):
            draw.text((x, y), str(i), (255, 255, 255), font=fnt, stroke_width=1, stroke_fill=(0,0,0))

        return imgc


    @torch.inference_mode()
    def diffuse(self, init_image, grid_idx=None, skip_rate=0.5, text: str=None, negative='', num_batches=1) -> Image.Image:
        '''Perform diffusion sampling on 1 or more images using Glid3XL.
        
        Args:
            init_image (Image.Image): Initial image to seed the sampling.
            grid_idx (int, list): Index or list of indices of the image in the grid to sample (can display indices with Min3Flow.show_grid). If None, use all images. (default: None)
            skip_rate (float): Percent of diffusion steps to skip. Values near 1 will minimally change the init_image, near 0 will significantly change the input image. (default: 0.5)
            text (str): Text prompt to generate images from. If None, use the last text prompt used. (default: None)
            negative (str): Negative Text prompt to oppose text prompt. (default: '')
            num_batches (int): Number of batches of size batch_size to run. (default: 1)
        
        Returns:
            Image.Image: Diffused image as a grid of batch_size images.
        '''

        self._begin_stage('diffuse')

        if text is None:
            text = self._cache['text']

        if self.model_glid3xl is None:
            if isinstance(self.glid3xl_config, Glid3XLClipConfig):
                self.model_glid3xl = Glid3XLClip(**self.glid3xl_config.to_dict())
            else:
                self.model_glid3xl = Glid3XL(**self.glid3xl_config.to_dict())
                
        
        image = self.model_glid3xl.gen_samples(
            text=text, 
            init_image=init_image, 
            negative=negative, 
            num_batches=num_batches,
            grid_idx=grid_idx,
            skip_rate=skip_rate,
            outdir=None
        )

        return image

    @torch.inference_mode()
    def upscale(self, init_image: Image.Image, tile:(int|bool)=None, tile_overlap: int=0) -> Image.Image:
        '''Upscale an image using SwinIR.

        Args:
            init_image (Image.Image): image to upsample.
            tile (int): Patch size for sliding window upsampling. (default: None). 
                If integer, will partition init_image into (tile x tile) size patches, 
                upsample each patch, then construct the final image from the patches. 
                If None, images larger than (and a multiple of) 256 will use tile=256. 
                If False, pass the entire image with partioning regardless of dimensions.
            tile_overlap (int): Overlap between image patch tiles. (default: 0)
                Useful to blend patch boundaries in large, non-grid images.
        '''

        self._begin_stage('upscale')

        if self.model_swinir is None:
            self.model_swinir = SwinIR(**self.swinir_config.to_dict())
        

        init_image = np.array(init_image)
        
        if tile is None:
            d,rem = divmod(init_image.width,256)
            if d > 1 and rem == 0:
                tile, tile_overlap = 256, 0
            else: 
                tile = False
                
        if tile is False: # strict object False to throw error if value falsey
            image = self.model_swinir.upscale(init_image)
        else:
            image = self.model_swinir.upscale_patchwise(init_image, slice_dim=tile, slice_overlap=tile_overlap)

        return Image.fromarray(image)