import gc
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from PIL import Image, ImageDraw, ImageFont

from .min_dalle import  MinDalle, MinDalleExt
from .min_glid3xl import Glid3XL, Glid3XLClip
from .min_swinir import SwinIR
from .utils import tensor_ops as tops

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
        config = 'Min3Flow(\n {}, \n {}, \n {}, \n {}\n)'.format(self.dalle_config, self.glid3xl_config, self.swinir_config, self.base_config)
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

        
        image = self.model_dalle.generate_images_tensor(
            text=text, 
            seed=self.seed, 
            grid_size=grid_size, 
            temperature=temperature,
            top_k = top_k,
            supercondition_factor=supercondition_factor,
            is_verbose=True
        )
        self._cache['text'] = text
        self._cache['grid_size'] = grid_size
        return image

    def clip_sort(self, grid_image, text):
        import clip
        clip_model, clip_preprocess = clip.load('ViT-L/14', device='cuda', jit=True)
        clip_model = clip_model.eval()
        toks = clip.tokenize([text], truncate=True)
        
        image_batch = tops.ungrid(grid_image)
        imgs = torch.stack([clip_preprocess(TF.to_pil_image(i)) for i in image_batch],dim=0)
        scos = clip_model(imgs.to('cuda'),toks.to('cuda'))[0].squeeze().sort(descending=True)
        return image_batch[scos.indices]

    def to_image(self, tensor):
        if tensor.ndim == 4:
            grid_size = int(image.shape[0]**0.5) #; cell_w = image.shape[-1]
            image = vutils.make_grid(image, nrow=grid_size, padding=0, normalize=False)
        
        # will fail if (H,W,C) from min_dalle
        image = TF.to_pil_image(image)
        return image

    def to_tensor(self, image):
        return TF.to_tensor(image)

    def show_grid(self, image: Image.Image, grid_size=None) -> Image.Image:
        '''Show a grid of images with index annotations.

        Args:
            image (Image.Image): Image grid to show.
            grid_size (int): The number of images in the grid in both x and y. (default: None)
                If None, use last grid_size value if available else infer from the image size.
            cell_hw tuple(int,int): The height of each image in the grid. (default: 256)
            cell_w (int): The width of each image in the grid. (default: 256)

        Returns:
            Image.Image: A grid of images with index annotations.
        '''
        # case 1.: image is a tensor
        #   case 1A: batched tensor (N,C,H,W) 
        #       case 1A.I: no upsampling (G*G,3,H,W) =                              (16,3,256,256)
        #       case 1A.II: upsampled (G*G, 3, H*4, W*4) = (4*4,3,256*4,256*4) =  (16,3,1024,1024)
        #   case 1B: unbatched tensor (C,H,W) 
        #       case 1B.I: single (3,H,W) =                                            (3,256,256)
        #       case 1B.II: grid (3, H*4, W*4) = (3,256*4,256*4) =                   (3,1024,1024)
        # case 2: image is a PIL image
        #   case 2A: single 
        #       case 2A.I no upsampling                                                 (256,256,3)
        #       case 2A.II upsampled  (256*4, 256*4, 3) =                             (1024,1024,3)
        #   case 2B: grid
        #       case 2B.I no upsampling                                               (1024,1024,3)
        #       case 2B.II upsampled  (256*4*4, 256*4*4, 3) =                         (4096,4096,3)

        # case 1B.I, 2A.* 
        #   OOS - function expects a grid, if not given, okay if unexpected results
        # case 1A.*
        #   grid_size: sqrt(N), cell_width=image.shape[-1], total_width=cell_width*grid_size
        # case 1B.II
        #   shouldn't happen unless use vutils.make_grid w/o converting to tensor
        #   or use wrong function from min_dalle which returns (H, W, C) tensors by default
        # case 2B.*
        #   try to get grid_size from cache, otherwise, throw error if not found
        #   cell_width=image.width/grid_size, total_width=image.width


        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                # case 1A.*
                grid_size = int(image.shape[0]**0.5) #; cell_w = image.shape[-1]
                
                image = vutils.make_grid(image, nrow=grid_size, padding=0, normalize=False)
            
            # will fail if (H,W,C) from min_dalle
            image = TF.to_pil_image(image)
            
        
        if grid_size is None:
            if 'grid_size' in self._cache:
                grid_size = self._cache['grid_size']
            else:
                raise ValueError('grid_size not provided and not previously set.')

        cell_h, cell_w = int(image.height/grid_size), int(image.width/grid_size)
        
        imgc = image.copy()
        draw = ImageDraw.Draw(imgc)
        try:
            fnt = ImageFont.truetype("DejaVuSans.ttf", 16*(cell_w//256))
        except OSError as e:
            fnt = ImageFont.truetype("arial.ttf", 16*(cell_w//256))

        
        for i,(x,y) in enumerate(np.mgrid[:grid_size,:grid_size].T.reshape(-1,2)*[cell_w,cell_h]):
            draw.text((x, y), str(i), (255, 255, 255), font=fnt, stroke_width=1, stroke_fill=(0,0,0))

        return imgc


    
    def diffuse(self, init_image, grid_idx:(int|list)=None, skip_rate:float=0.5, text:str=None, negative:str='', num_batches:int=1) -> Image.Image:
        '''Perform diffusion sampling on 1 or more images using Glid3XL or Glid3XLClip.
        
        Args:
            init_image (Image.Image): Initial image to seed the sampling.
            grid_idx (int, list): Index or list of indices of the image in the grid to sample. (default: None)
                Use Min3Flow.show_grid to display grid indices overlay. If None, use all images. 
            skip_rate (float): Percent of diffusion steps to skip. (default: 0.5)
                Values near 1 will minimally change the init_image, near 0 will significantly change the input image. 
            text (str): Text prompt to generate images from. (default: None)
                If None, use the last text prompt used. 
            negative (str): Negative Text prompt to oppose text prompt. (default: '')
            num_batches (int): Number of batches of size batch_size to run. (default: 1)
        
        Returns:
            Image.Image: Diffused image as a grid of batch_size images.
        '''

        self._begin_stage('diffuse')

        if text is None:
            text = self._cache['text']
        
        inference_safe = True
        if self.model_glid3xl is None:
            if isinstance(self.glid3xl_config, Glid3XLClipConfig):
                self.model_glid3xl = Glid3XLClip(**self.glid3xl_config.to_dict())
                inference_safe = False
            else:
                self.model_glid3xl = Glid3XL(**self.glid3xl_config.to_dict())
                
                
        with torch.inference_mode(inference_safe):
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
        

        
        if tile is None:
            W = init_image.width if isinstance(init_image, Image.Image) else init_image.shape[-1]
            d,rem = divmod(W,256)
            if d > 1 and rem == 0:
                tile, tile_overlap = 256, 0
            else: 
                tile = False
                
        if tile is False: # strict object False to throw error if value falsey
            image = self.model_swinir.upscale(init_image)
        elif tile_overlap == 0:
            image = self.model_swinir.upscale_prebatched(init_image)
        else:
            image = self.model_swinir.upscale_patchwise(init_image, tile=tile, tile_overlap=tile_overlap)

        return image