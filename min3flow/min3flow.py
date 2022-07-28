import gc
from pathlib import Path

import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont

from .min_dalle import  MinDalle, MinDalleExt
from .min_glid3xl import Glid3XL, Glid3XLClip, utils as glid3xl_utils
from .min_swinir import SwinIR, utils as swinir_utils

from .configuration import BaseConfig, MinDalleConfig, MinDalleExtConfig, Glid3XLConfig, Glid3XLClipConfig, SwinIRConfig

# ROOT_PATH = Path(__file__).parent.parent#'../../'  #Path(__file__).parent.parent.parent
# MODEL_ROOT =  ROOT_PATH.joinpath('pretrained') #ROOT_PATH.joinpath('pretrained', 'glid-3-xl')


class Min3Flow:
    def __init__(self, dalle_config=None, glid3xl_config=None, swinir_config=None, base_config=None, persist=False, seed: int = None, device=None) -> None:

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

    # def _end_stage(self, *stages: str) -> None:
    #     if not self.persist:
    #         for stage in stages:
    #             if stage == 'generate':
    #                 del self.model_dalle
    #                 self.model_dalle = None
    #             elif stage == 'diffuse':
    #                 del self.model_glid3xl
    #                 self.model_glid3xl = None
    #             elif stage == 'upscale':
    #                 del self.model_swinir
    #                 self.model_swinir = None

    #     gc.collect()
    #     torch.cuda.empty_cache()

    def _begin_stage(self, stage: str) -> None:
        if not self.persist:
            if stage == 'generate':
                #self.model_dalle = None
                self.model_glid3xl = None
                self.model_swinir = None
                #del self.model_dalle
                
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

    def generate(self, text: str, grid_size: int = 4, supercondition_factor: int = 32, temperature: float = 1.0, top_k: int = 256) -> Image.Image:
        #self._end_stage('diffuse','upscale')
        self._begin_stage('generate')

        if self.model_dalle is None:
            if isinstance(self.dalle_config, MinDalleExtConfig):
                self.model_dalle = MinDalleExt(**self.dalle_config.to_dict())
            else:
                self.model_dalle = MinDalle(**self.dalle_config.to_dict())
            #self.model_dalle =  MDalle(**self.dalle_config.to_dict())
        
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

    def show_grid(self, image: Image.Image, grid_size=4, cell_h=256, cell_w=256) -> Image.Image:
        imgc = image.copy()
        
        draw = ImageDraw.Draw(imgc)
        fnt = ImageFont.truetype("DejaVuSans.ttf", 20)
        
        for i,(x,y) in enumerate(np.mgrid[:grid_size,:grid_size].T.reshape(-1,2)*[cell_w,cell_h]):
            draw.text((x, y), str(i), (255, 255, 255), font=fnt, stroke_width=1, stroke_fill=(0,0,0))

        return imgc



    def diffuse(self, init_image, grid_idx=None, skip_rate=0.5, text: str=None, negative='', num_batches=1) -> Image.Image:
        #self._end_stage('generate','upscale')
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

    def upscale(self, init_image: Image.Image, tile: int=None, tile_overlap: int=0):
        #self._end_stage('diffuse','generate')
        self._begin_stage('upscale')

        if self.model_swinir is None:
            self.model_swinir = SwinIR(**self.swinir_config.to_dict())
        

        #filename = Path(args.init_img).name
        #outpath = Path(args.output_dir).joinpath(filename).as_posix()

        init_image = np.array(init_image)
        
        if tile is None:
            image = self.model_swinir.upscale(init_image)
        else:
            image = self.model_swinir.upscale_patchwise(init_image, slice_dim=tile, slice_overlap=tile_overlap)

        return Image.fromarray(image)