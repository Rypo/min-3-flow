import os
import gc
import warnings

from typing import  Union

import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

import clip

from PIL import Image, ImageDraw, ImageFont

from .min_dalle import  MinDalle, MinDalleExt
from .min_glid3xl import Glid3XL, Glid3XLClip
from .min_swinir import SwinIR
from .utils import tensor_ops as tops

from .configuration import MinDalleConfig, MinDalleExtConfig, Glid3XLConfig, Glid3XLClipConfig, SwinIRConfig




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
            persist (bool|'cpu'): Whether to persist the prior stages models in memory after a stage is complete. (default: False)
                If False, at the begining of each stage, unload non-active stage models and free cached memory. 
                If 'cpu', at the begining of each stage, move all non-active stage models to cpu and free cached memory. 
                If True, the model will be persisted in GPU memory. Warning: With f16 will use >16gb VRAM, with f32 VRAM usage > 19gb .
            global_seed (int): Random seed shared for all models. Active when > 0 (default: -1)
            device (str): Device to use for the models. Defaults to cuda if available. (default: None)
    '''
    def __init__(self, dalle_config=None, glid3xl_config=None, swinir_config=None,  persist:Union[bool,str]=False, global_seed: int = -1, device=None) -> None:
        
        self.dalle_config = dalle_config if dalle_config is not None else MinDalleConfig()
        self.glid3xl_config = glid3xl_config if glid3xl_config is not None else Glid3XLConfig()
        self.swinir_config = swinir_config if swinir_config is not None else SwinIRConfig()

        self.persist = persist
        self.global_seed = global_seed
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_dalle = None
        self.model_glid3xl = None
        self.model_swinir = None

        self._clip_model = None
        self._clip_preprocess = None

        self._cache = {}

    def __repr__(self) -> str:
        config = 'Min3Flow(\n {}, \n {}, \n {}, \n)'.format(self.dalle_config, self.glid3xl_config, self.swinir_config)
        return config


    def _begin_stage(self, stage: str) -> None:
        '''Unload or transfer non-active stage models and free cached memory.'''
        if self.global_seed > 0:
            torch.manual_seed(self.global_seed)
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
                if self.model_dalle is not None: self.model_dalle._to(self.device)
                if self.model_glid3xl is not None: self.model_glid3xl._to('cpu')
                if self.model_swinir is not None: self.model_swinir._to('cpu')    
            elif stage == 'diffuse':
                if self.model_dalle is not None: self.model_dalle._to('cpu')
                if self.model_glid3xl is not None: self.model_glid3xl._to(self.device)
                if self.model_swinir is not None: self.model_swinir._to('cpu')  
            elif stage == 'upscale':
                if self.model_dalle is not None: self.model_dalle._to('cpu') 
                if self.model_glid3xl is not None: self.model_glid3xl._to('cpu')
                if self.model_swinir is not None: self.model_swinir._to(self.device)  
        
        self._cache['active_stage'] = stage
         
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(self, text: str, grid_size: int = 4, supercondition_factor: int = 16, temperature: float = 1.0, top_k: int = 256, seed=None) -> Image.Image:
        '''Generate a set of (256,256) images given a text prompt (MinDalle).
        
        Args:
            text (str): Text prompt to generate images from.
            grid_size (int): Size of image output grid in x,y. E.g. grid_size=4 produces 16 (256,256) images. (default: 4)
            supercondition_factor (int): Higher values better match text prompt, but narrow image out variety. (default: 16)
            temperature (float): Values > 1 supress the influence of the most probable tokens in top_k, providing more diverse sampling. (default: 1.0)
            top_k (int): The number of most probable tokens to use when sampling each image token. (default: 256)
            seed (int): Random seed for reproducibility. (default: None)
                If None, use global_seed. If negative, use no seed.

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
            seed=(seed if seed is not None else self.global_seed), 
            grid_size=grid_size, 
            temperature=temperature,
            top_k = top_k,
            supercondition_factor=supercondition_factor,
            is_verbose=True
        )
        self._cache['gen_text'] = text
        self._cache['grid_size'] = grid_size
        return image

    
    def diffuse(self, text:str=None, init_image:torch.FloatTensor=None, grid_idx:Union[int,list]=None, skip_rate:float=0.5,  negative:str='', num_batches:int=1, seed=None) -> Image.Image:
        '''Perform diffusion sampling on 1 or more images using Glid3XL or Glid3XLClip.
        
        Args:
            text (str): Text prompt to generate images from. (default: None)
                If None, use the last text prompt used. 
            init_image (FloatTensor): Initial image to seed the sampling. (default: None)
                If None, will generate images from scratch using the text prompt.
            grid_idx (int, list): Index or list of indices of the image in the grid to sample. (default: None)
                Use Min3Flow.show_grid to display grid indices overlay. If None, use all images. 
            skip_rate (float): Percent of diffusion steps to skip. (default: 0.5)
                Values near 1 will minimally change the init_image, near 0 will significantly change the input image. 
            negative (str): Negative Text prompt to oppose text prompt. (default: '')
            num_batches (int): Number of batches of size batch_size to run. (default: 1)
            seed (int): Random seed for reproducibility. (default: None)
                If None, use global_seed. If negative, no seed is used.
        
        Returns:
            Image.Image: Diffused image as a grid of batch_size images.
        '''

        self._begin_stage('diffuse')

        if text is None:
            text = self._cache['gen_text']
            print(f'Using last text prompt: "{text}"')
            
        self._cache['dif_text'] = text

        
        inference_safe = self._cache.get('inference_safe', None)
        if self.model_glid3xl is None:
            if isinstance(self.glid3xl_config, Glid3XLClipConfig):
                self.model_glid3xl = Glid3XLClip(**self.glid3xl_config.to_dict())
                inference_safe = False
            else:
                self.model_glid3xl = Glid3XL(**self.glid3xl_config.to_dict())
                inference_safe = True
            self._cache['inference_safe'] = inference_safe
        
                
                
        with torch.inference_mode(inference_safe):
            image = self.model_glid3xl.gen_samples(
                text=text, 
                init_image=init_image, 
                negative=negative, 
                num_batches=num_batches,
                grid_idx=grid_idx,
                skip_rate=skip_rate,
                outdir=None,
                seed=(seed if seed is not None else self.global_seed)
            )

        return image

    @torch.inference_mode()
    def upscale(self, init_image: torch.FloatTensor, tile:Union[int,bool]=None, tile_overlap: int=0) -> Image.Image:
        '''Upscale an image using SwinIR.

        Args:
            init_image (FloatTensor): image to upsample.
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


    @torch.inference_mode()
    def clip_sort(self, img_batch, text) -> torch.return_types.sort:
        '''Sort image batch by cosine similarity to CLIP text embeddings.
        
        Args:
            img_batch (tensor): float tensor of shape (N, C, H, W)
            text (str): text to encode and use for similarity

        Returns:
            torch.sort: torch namedtuple(values,indices) of desc-sorted clip scores
        '''
        if self._clip_model is None:
            _name = 'ViT-L/14'+'@336px'
            self._clip_model, self._clip_preprocess = clip.load(_name, device=self.device, jit=True)
            self._clip_model = self._clip_model.eval()

        toks = clip.tokenize([text], truncate=True)
        imgs = torch.stack([self._clip_preprocess(TF.to_pil_image(i)) for i in img_batch],dim=0)

        
        self._clip_model=self._clip_model.to(self.device)
        scos = self._clip_model(imgs.to(self.device),toks.to(self.device))[0].squeeze().sort(descending=True)
        self._clip_model.to('cpu')
        
        gc.collect()
        torch.cuda.empty_cache()

        return scos



    def show_grid(self, image: Union[torch.FloatTensor,Image.Image], cell_hw:Union[int,tuple]=None, plot_index=True, clip_sort_text:str=None) -> Image.Image:
        '''Show a grid of images with index annotations.

        Args:
            image (FloatTensor | Image): Image grid to show.
            cell_hw (int | tuple(int,int)): The height,width of each image in the grid. (default: None)
                Must specify if image is not a batched FloatTensor of shape (N,C,H,W).
            plot_index (bool): Whether to plot the index of each image in the grid. (default: True)
            clip_sort_text (str): Text to use for sorting images by clip score. (default: None)
                Will load clip model if not None, using additional GPU memory.

        Returns:
            Image.Image: A grid of images with index annotations.
        '''

        if isinstance(image, Image.Image):
            assert cell_hw is not None, 'cell_hw must be specified when passing a PIL image.'
            image = tops.ungrid(image, hw_out=cell_hw)
        elif image.ndim == 3:
            if cell_hw is None:
                # TRADEOFF: If image is a pre-grid tensor (e.g. from vutils.make_grid)) this will not give the desired result.
                # But, this is less likely to occur than having a single image passed to this function.
                # Both swinir and Glid3XLClip can/will return 1 image. From a useablity standpoint, just returning the image
                # is better than throwing an error and require a different function to handle the single image case. 
                warnings.warn('Note: Unbatched tensors are assumed to be a single image. To treat as a grid, explictly set `cell_hw`')
                return tops.to_image(image)
            else:
                image = tops.ungrid(image, hw_out=cell_hw)
            

        N,C,H,W = image.shape
        
        n_rows = int(N**0.5)
        n_cols = int(np.ceil(N / n_rows))

        idx = np.arange(N)
        if clip_sort_text is not None:
            scores = self.clip_sort(image, clip_sort_text)
            image = image[scores.indices]
            idx = scores.indices.cpu().numpy()
            
         
        image = tops.to_image(image, n_rows=n_rows)
        
        if not plot_index:
            return image

        imgc = image.copy()
        draw = ImageDraw.Draw(imgc)
        try:
            fnt = ImageFont.truetype("DejaVuSans.ttf", 16*(W//256))
        except OSError as e:
            fnt = ImageFont.truetype("arial.ttf", 16*(W//256))

        
        for i,(x,y) in enumerate(np.mgrid[:n_rows,:n_cols].T.reshape(-1,2)*[W,H]):
            if i < N:
                draw.text((x, y), str(idx[i]), (255, 255, 255), font=fnt, stroke_width=1, stroke_fill=(0,0,0))


        return imgc

    def save_images(self, image:Union[torch.FloatTensor,Image.Image], as_grid=True, cell_hw=None, outdir='output/results'):
        '''Save images to disk.
        
        Args:
            image (FloatTensor | Image): Grid of images to save.
            as_grid (bool): Whether to save as a grid. (default: True)
            cell_hw (int | tuple(int,int)): The height,width of each image in the grid. (default: None)
                Must specify if image is not a batched FloatTensor of shape (N,C,H,W).
            outdir (str): output dir path. (default: None)
                If None, will save to output/results/<last_used_text_prompt>.png if as_grid=True
                else output/results/<last_used_text_prompt>__<idx>.png
        '''
        os.makedirs(outdir, exist_ok=True)
        
        gen_text = self._cache.get('gen_text',None)
        dif_text = self._cache.get('dif_text',None)
        if self._cache.get('active_stage') == 'diffuse' and dif_text is not None:
            text = dif_text
        else:
            text = gen_text
        #text = (gen_text if gen_text is not None else dif_text)
        if text is None:
            text = 'result'
            warnings.warn('no prompt text found, saving to result.png')
        text = text.replace(' ','_')

        if as_grid:
            image = self.show_grid(image, cell_hw, plot_index=False)
            path = os.path.join(outdir, '{}.png'.format(text))
            image.save(path)
            print('saved to {}'.format(path))
        else:
            path = os.path.join(outdir, '{}__{:03d}.png')
            for i,img in enumerate(image):
                
                img = TF.to_pil_image(img)
                img.save(path.format(text,i))

       
       
        
