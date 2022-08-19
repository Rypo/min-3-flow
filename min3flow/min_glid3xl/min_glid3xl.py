import os
import gc
from pathlib import Path
from typing import  Union
from contextlib import contextmanager

from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import torchvision.utils as vutils

#from einops import rearrange

import clip
from ldm.models.autoencoder import AutoencoderKL
#from dalle_pytorch import DiscreteVAE, VQGanVAE

#from .models.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from .models.encoders.modules import BERTEmbedder

from . import utils
from ..utils.io import download_weights
from ..utils import tensor_ops as tops

#from .models.guided_diffusion import gaussian_diffusion as gd
from .models.guided_diffusion.gaussian_diffusion import get_named_beta_schedule
from .models.guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .models.guided_diffusion.unet import UNetModel
#from .unet import UNetModel

_DEFAULT_WEIGHT_ROOT = "~/.cache/min3flow/glid3xl"

# https://github.com/Jack000/glid-3-xl
# https://github.com/LAION-AI/ldm-finetune

_WEIGHT_DOWNLOAD_URLS = {
    'kl-f8.pt': 'https://dall-3.com/models/glid-3-xl/kl-f8.pt',
    # ldm first stage (required)
    'bert.pt': 'https://dall-3.com/models/glid-3-xl/bert.pt',
    # text encoder (required)
    
    'base.pt': 'https://dall-3.com/models/glid-3-xl/diffusion.pt', 
    # Model Name: base.pt
    # Base Model: N/A
    # Data: LAION-400M
    # Desc: Original diffusion model from CompVis
    # Tips: Works for 2d illustrations, but may not follow prompts as well as others. May generate watermarks/blurry images. 

    'finetune.pt': 'https://dall-3.com/models/glid-3-xl/finetune.pt', 
    # Model Name: finetune.pt 
    # Base Model: base.pt
    # Finetuning data: Cleaned dataset.
    # Desc: Finetuned model by jack000 that will not generate watermarks or blurry images.
    # Tips: Best for realistic photography, follows prompts better than others.

    'inpaint.pt': 'https://dall-3.com/models/glid-3-xl/inpaint.pt', 
    # Model Name: inpainting.pt
    # Base Model: base.pt (?)
    # Finetuning data: Cleaned dataset (?).
    # Desc: The second finetune from jack000's glid-3-xl adds support for inpainting. 
    # Can be used for unconditional output by setting the inpaint image_embed to zeros. 
    # Additionally finetuned to use the CLIP text embed via cross-attention (similar to unCLIP).

    'erlich.pt': 'https://huggingface.co/laion/erlich/resolve/main/model/ema_0.9999_120000.pt',
    # Model Name: erlich.pt
    # Base Model: inpaint.pt 
    # Finetuning data: Large Logo Dataset
    # Desc: erlich is inpaint.pt finetuned on LAION-5B's Large Logo Dataset. 
    # A dataset consisting of ~100K images of logos with captions generated via BLIP using aggressive re-ranking and filtering.
    # Tips: Best for logos, nothing else. Adding "logo" to the end of prompt can sometimes help.

    'ongo.pt': 'https://huggingface.co/laion/ongo/resolve/main/ongo.pt',
    # Model Name: ongo.pt
    # Base Model: inpaint.pt
    # Finetuning data: Wikiart dataset 
    # Ongo is inpaint.pt finetuned on the Wikiart dataset. 
    # A dataset consisting of ~100K paintings with captions generated via BLIP using aggressive re-ranking and filtering.
    # Original captions containing the author name and the painting title also included.
    # Tips: Best for paintings, nothing else. Adding "painting" to the end of prompt can sometimes help.

    'puck.pt': 'https://huggingface.co/laion/puck/resolve/main/puck.pt'
    # Model Name: puck.pt
    # Base Model: inpaint.pt
    # Finetuning data: pixel art
    # Desc: puck has been trained on pixel art. 
    # While the underlying kl-f8 encoder seems to struggle somewhat with pixel art, results are still interesting.
} 

def _available_weights(stage='diffusion'):
    all_weights = list(_WEIGHT_DOWNLOAD_URLS.keys())
    if stage == 'diffusion':
        return [w for w in all_weights if w not in ['bert.pt', 'kl-f8.pt']]
    elif stage == 'encoder':
        return ['bert.pt']
    elif stage == 'ldm':
        return ['kl-f8.pt']
    

class Glid3XL:
    def __init__(self, guidance_scale=5.0, batch_size=16, steps=100, sample_method='plms', imout_size=(256,256), 
                 diffusion_weight='finetune.pt', kl_weight='kl-f8.pt', bert_weight='bert.pt', weight_root = None, dtype = torch.float32, device=None) -> None:
        
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.steps = steps
        #self.skip_rate = skip_rate
        self.sample_method = sample_method
        self.H, self.W = int(imout_size[0]), int(imout_size[1])
        self.dtype = dtype
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self._weight_root = weight_root if weight_root is not None else os.path.expanduser(_DEFAULT_WEIGHT_ROOT)

        
        self.cond_fn = None
        self.cur_t = None
        self._LDM_SCALE_FACTOR = 0.18215

        self._cache = {}
        self._model_kwargs = {}
        self._inference_safe = True
        self._setup_init(diffusion_weight=diffusion_weight, kl_weight=kl_weight, bert_weight=bert_weight, sample_method=sample_method, steps=steps)


    def _setup_init(self, diffusion_weight, kl_weight, bert_weight, sample_method, steps):
        wpaths = []
        for stage,weight in zip(['diffusion','ldm','encoder'],[diffusion_weight, kl_weight, bert_weight]):
            if os.path.exists(weight):
                wpath = weight
            elif weight in _WEIGHT_DOWNLOAD_URLS:
                dl_url = _WEIGHT_DOWNLOAD_URLS[weight]
                wpath = os.path.join(self._weight_root, weight)
                wpath = download_weights(wpath, dl_url)
            else:
                raise ValueError(f"'{weight}' is not a path or known {stage} weight. Available weights aliases are {_available_weights()}")
            
            wpaths.append(wpath)

            
        diffusion_path, kl_path, bert_path = wpaths

        self._bert_path = bert_path # save for lazy loading bert

        self.model, self.diffusion, self.model_config = utils.timed(self.load_models, model_path=diffusion_path, sample_method=sample_method, steps=steps)
        self.clip_model, self.clip_preprocess = utils.timed(self.load_clip)
        self.ldm = utils.timed(self.load_ldm, kl_path=kl_path)
        self.bert = None

        # TODO: ddpm doesn't exist? verify and remove option if intentional
        self.sample_fn = self.diffusion.plms_sample_loop_progressive if sample_method=='plms' else  self.diffusion.ddim_sample_loop_progressive 


    @property
    def _models_list(self):
        return [self.model, self.clip_model, self.ldm]
    def _to(self, device):
        for model in self._models_list:
            model.to(device)
        
    def load_models(self, model_path, sample_method, steps):
        use_fp16 = (self.dtype==torch.float16 and self.device.type!='cpu')
        model_state_dict = torch.load(model_path, map_location='cpu')
        model_config = {
            # 768 for all known pretrained models.
            'clip_embed_dim':768, # (768 if 'clip_proj.weight' in model_state_dict else None)
            # False for ['base.pt','finetune.pt'] else True
            'image_condition':(model_state_dict['input_blocks.0.0.weight'].shape[1] == 8),
            # False for all known pretrained models.
            'super_res_condition':False, #(True if 'external_block.0.0.weight' in model_state_dict else False,)
        }
        
        model = UNetModel(
            image_size=32,
            in_channels=4,
            model_channels=320,
            out_channels=4,
            num_res_blocks=2,
            attention_resolutions=(1, 2, 4),
            dropout=0.0,
            channel_mult=(1,2,4,4),
            num_classes=None,
            use_checkpoint=False,
            use_fp16=use_fp16,
            num_heads=8,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_spatial_transformer=True,
            context_dim=1280,
            clip_embed_dim=model_config['clip_embed_dim'], 
            image_condition=model_config['image_condition'],
            super_res_condition=model_config['super_res_condition'],
        ).eval().requires_grad_(False)

        model.convert_to_dtype(use_fp16)
        model.load_state_dict(model_state_dict, strict=False)
        #del model_state_dict
        model=model.to(self.device)

        diffusion_steps = 1000
        timestep_respacing = {'ddpm': 1000, 'ddim': f'ddim{steps}'}.get(sample_method, str(steps)) #'27'
        
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=get_named_beta_schedule('linear', diffusion_steps),
            rescale_timesteps=True,
        )

        
        return model, diffusion, model_config


    def load_ldm(self, kl_path): # vae
        ldm: AutoencoderKL = torch.load(kl_path, map_location=self.device)
        # TODO: verify that LDM can be frozen while using clip guidance. 
        # It does not throw an error, but need to verify that the output doesn't change.
        #ldm.requires_grad_(self.args.clip_guidance); #set_requires_grad(ldm, self.args.clip_guidance)
        ldm.freeze() 
        ldm.eval()

        return ldm

    def load_clip(self): 
        clip_model, clip_preprocess = clip.load('ViT-L/14@336px', device=self.device, jit=False)
        clip_model = clip_model.eval().requires_grad_(False)

        return clip_model, clip_preprocess

    def load_bert(self, bert_path):
        bert = BERTEmbedder(1280, 32).eval().requires_grad_(False).to(self.device)#.half()
        #bert = bert.requires_grad_(False)
        bert.load_state_dict(torch.load(bert_path, map_location=self.device), strict=False)
        bert = bert.half()#.to(self.device)

        return bert

    @contextmanager
    def offloading_bert(self, bert_path, text, negative='', destroy=False):
        '''Offload bert object to CPU after use. Second use add ~0.50 seconds.'''
        with torch.inference_mode():
            if self.bert is None:
                self.bert = utils.timed(self.load_bert,bert_path=bert_path)
            self.bert = self.bert.to(self.device)

            #text_emb = bert.encode([self.args.text]).to(device=self.device, dtype=torch.float).expand(self.batch_size,-1,-1)
            #text_blank = bert.encode([self.args.negative]).to(device=self.device, dtype=torch.float).expand(self.batch_size,-1,-1)
            
            text_emb = self.bert.encode([text]*self.batch_size).to(device=self.device)#, dtype=torch.float)
            text_blank = self.bert.encode([negative]*self.batch_size).to(device=self.device)#, dtype=torch.float)

            try:
                yield text_emb, text_blank
            finally:
                self.bert = None if destroy else self.bert.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()

    
    def encode_text(self, text, negative=''):
        with self.offloading_bert(self._bert_path, text, negative) as bert_output:
            text_emb, text_blank = bert_output

        toktext = clip.tokenize([text], truncate=True).expand(self.batch_size, -1).to(self.device)
        text_clip_blank = clip.tokenize([negative], truncate=True).expand(self.batch_size, -1).to(self.device)

        # clip context
        text_emb_clip = self.clip_model.encode_text(toktext)
        text_emb_clip_blank = self.clip_model.encode_text(text_clip_blank)

        return text_emb, text_blank, text_emb_clip, text_emb_clip_blank

    #@torch.inference_mode()
    def encode_image_grid(self, images, grid_idx=None):
        # assume grid for now, will break if 512x512 or similar. TODO: add argument later.
        if images is None:
            return None

        if isinstance(grid_idx, int):
            init = images[[grid_idx]]
            #init = init.resize((self.W, self.H), Image.Resampling.LANCZOS)
            #init = TF.to_tensor(init).to(self.device).unsqueeze(0).clamp(0,1)
        elif isinstance(grid_idx, list):
            init = images[grid_idx]
        elif grid_idx is None:
            init = images

        
        #init = init.to(self.device, dtype=torch.float).div(255.).clamp(0,1)
        h = self.ldm.encode(init.mul(2).sub(1)).sample() * self._LDM_SCALE_FACTOR 
        #init = torch.cat(self.batch_size*2*[h], dim=0)
        #init = torch.repeat_interleave(h, 2*(self.batch_size//h.size(0)), dim=0, output_size=2*self.batch_size) # (2*BS, 4, H/8, W/8)
    
        osize=2*max(1,self.batch_size//h.size(0))
        init = h.tile(osize,1,1,1) # (2*BS, 4, H/8, W/8)
    
        if init.shape[0] < 2*self.batch_size:
            # Case when number of grid_idx samples does not evenly divide 2*batch_size
            # For now, just repeat encoding until 2*batch_size is reached
            # TODO: assess the impact of this on image diversity
            diff = 2*self.batch_size-init.shape[0]

            init = torch.cat([init, init[-diff:]], dim=0)
                

        return init

    @torch.inference_mode()
    def decode_sample(self, samples):
        if isinstance(samples, dict):
            samples = samples['pred_xstart'][:self.batch_size]
        
        # images = samples / self._LDM_SCALE_FACTOR
        # out =  self.ldm.decode(images).add(1).div(2).clamp(0, 1)#.detach()
        # return out

        decoded_images = []
        samples /= self._LDM_SCALE_FACTOR
       
        # shape: (num_batches, batch_size, 3, height=256, width=256)

        for sample in samples:
            #images = sample / self._LDM_SCALE_FACTOR
            images = self.ldm.decode(sample).add(1).div(2).clamp(0, 1).detach()
            decoded_images.append(images) 

        return torch.cat(decoded_images,dim=0).squeeze() # shape: (batch_size*num_batches, 3, height=256, width=256)


    def clf_free_sampling(self, x_t, ts, **kwargs):
        # Create a classifier-free guidance sampling function
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = torch.lerp(uncond_eps, cond_eps, self.guidance_scale)
        #half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


    def sample_batches(self, num_batches, init_image_embed, skip_ts, model_kwargs):
        # TODO: the clip model can be unloaded in a similar fashion to Bert. 
        # after the initial image is encoded and the text is encoded.
        # In fact, the same is true for the LDM.
        # The only dependency for generation is the diffusion model and the sampler.
        # BUT, the LDM is needed for decoding the samples afterwards, so may not make sense to unload it.
        # TODO: look into have a "slow but low memory" mode and a "fast high memory" mode.
        
        batch_samples = []
        for i in range(num_batches):
            self.cur_t = self.diffusion.num_timesteps - 1

            samples = self.sample_fn(
                self.clf_free_sampling,
                (self.batch_size*2, 4, int(self.H/8), int(self.W/8)),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                cond_fn=self.cond_fn,
                device=self.device,
                progress=True,
                init_image=init_image_embed,
                skip_timesteps=skip_ts
            )

            for j, sample in enumerate(samples):
                self.cur_t -= 1
                #if j % 5 == 0 and j != self.diffusion.num_timesteps - 1:
                    #self.save_sample(sample, i, fname)
                    #save_sample(i, sample)
            #print(sample['pred_xstart'].shape)
            batch_samples.append(sample['pred_xstart'][:self.batch_size].detach())
            # Cut at batch_size because data is noisy/corrupted afterwards ~~duplicated afterwards (?)~~

           
        bsample = torch.stack(batch_samples, dim=0).detach_()
        #bsample = torch.cat(batch_samples, dim=0).detach_()
        
        return bsample


    #@functools.cache
    def cache_model_kwargs(self, text: str, negative: str='', image_embed=None):
        if self.model_config['image_condition'] and image_embed is None:
             # using inpaint model but no image is provided
            image_embed = torch.zeros(self.batch_size*2, 4, self.H//8, self.W//8, device=self.device)
        if (text, negative) != self._cache.get('texts', None):
            
            text_emb, text_blank, text_emb_clip, text_emb_clip_blank = self.encode_text(text=text, negative=negative)
            self._text_emb_clip = text_emb_clip

            self._model_kwargs = {
                "context": torch.cat([text_emb, text_blank], dim=0).float(),
                "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float() if self.model_config['clip_embed_dim'] else None,
                "image_embed": image_embed 
            }

            self._cache['texts'] = (text, negative)

    
    def make_model_kwargs(self, text: str, negative: str='', image_embed=None):
        
        if self.model_config['image_condition'] and image_embed is None:
            # using inpaint model but no image is provided
            image_embed = torch.zeros(self.batch_size*2, 4, self.H//8, self.W//8, device=self.device, dtype=self.dtype)

            
        text_emb, text_blank, text_emb_clip, text_emb_clip_blank = self.encode_text(text=text, negative=negative)
        self._text_emb_clip = text_emb_clip

        model_kwargs =  {
            "context": torch.cat([text_emb, text_blank], dim=0).to(dtype=self.dtype),
            "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).to(dtype=self.dtype), 
            "image_embed": image_embed,
        }
        return model_kwargs



    def process_inputs(self,  text: str, init_image: Union[Image.Image,str]):
        if text =='' and init_image != '':
            text = Path(init_image).stem.replace('_', ' ')
        
        if isinstance(init_image, str):
            init_image = Image.open(init_image).convert('RGB')
            
        if isinstance(init_image, Image.Image):
            init_image = tops.ungrid(init_image, hw_out=256).to(self.device)
            

        return text, init_image


    def gen_samples(self, text: str, init_image:Union[torch.FloatTensor,Image.Image], negative: str='', num_batches: int=1, grid_idx:Union[int,list]=None, skip_rate=0.5,  outdir: str=None, seed=-1,) -> torch.Tensor:
        if seed > 0:
            torch.manual_seed(seed)


        text, init_image = self.process_inputs(text, init_image)

        with torch.no_grad(): # inference_mode breaks clip guidance
            init_image_embed = self.encode_image_grid(init_image, grid_idx=grid_idx)#.to(dtype=self.dtype)
            model_kwargs = self.make_model_kwargs(text, negative)

        #print('init_image_embed, context, clip_embed, image_embed')
        #print(init_image_embed.dtype, model_kwargs['context'].dtype, model_kwargs['clip_embed'].dtype, model_kwargs['image_embed'].dtype)
            
        with torch.inference_mode(self._inference_safe):
            bsample = self.sample_batches(num_batches, init_image_embed, skip_ts=int(self.steps*skip_rate), model_kwargs = model_kwargs) 
        
        #with torch.inference_mode():
        output_images = self.decode_sample(bsample)
        
        if outdir is None:
            return output_images

        # else

        fname = Path(init_image).name if init_image else 'GEN__'+text.replace(' ', '_')+'.png'
        outpath = os.path.join(outdir,  f'{num_batches:05d}_{fname}')
        vutils.save_image(output_images, outpath, nrow=int((self.batch_size*max(1,num_batches)**0.5)), padding=0)




class Glid3XLClip(Glid3XL):
    # May be cleaner to use a single class for both Glid3XL and Glid3XLClip. 
    # However, seperation allows for future optimization over non-clip model default. 
    # The non-clip guided model could temporarily unload clip and ldm while generating samples, and reload for final output.
    # The clip guided model needs to keep both models loaded for generating samples.
    def __init__(self, clip_guidance_scale=500, cutn=16, guidance_scale=3.0, batch_size=1, steps=100, sample_method='plms', imout_size=(256,256), 
                 diffusion_weight='finetune.pt', kl_weight='kl-f8.pt', bert_weight='bert.pt', weight_root=None, device=None) -> None:
        
        assert batch_size==1, "Clip guided model currently only supports batch_size=1"
        super().__init__(guidance_scale=guidance_scale, batch_size=batch_size, steps=steps, sample_method=sample_method, imout_size=imout_size, 
                        diffusion_weight=diffusion_weight, kl_weight=kl_weight, bert_weight=bert_weight, weight_root=weight_root, device=device)

        self.model = self.model.requires_grad_(True) # required for conditioning_fn, superclass sets False
        self._inference_safe = False
        self.cond_fn = self.conditioning_fn

        self.clip_guidance_scale = clip_guidance_scale
        self.cutn = cutn
    
        N_PX = self.clip_model.visual.input_resolution # 224
        self.cutnorm = T.Compose([utils.MakeCutouts(N_PX, self.cutn), T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    
    def conditioning_fn(self, x, t, context=None, clip_embed=None, image_embed=None):
        
        kw = {
            'context': context[:self.batch_size],
            'clip_embed': clip_embed[:self.batch_size] if self.model_config['clip_embed_dim'] else None,
            'image_embed': image_embed[:self.batch_size] if image_embed is not None else None
        }
        n = self.batch_size
        my_t = torch.ones([n], device=self.device, dtype=torch.long) * self.cur_t
        #out = self.diffusion.p_mean_variance(self.model, x, my_t, clip_denoised=False, model_kwargs=kw)
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[self.cur_t]
        with torch.enable_grad():
            x = x[:self.batch_size].detach().requires_grad_()
            #n = x.shape[0]
            #my_t = torch.ones([n], device=self.device, dtype=torch.long) * self.cur_t
            out = self.diffusion.p_mean_variance(self.model, x, my_t, clip_denoised=False, model_kwargs=kw)
            #fac = self.diffusion.sqrt_one_minus_alphas_cumprod[self.cur_t]
            x_in = out['pred_xstart']*fac + x*(1 - fac)

            x_in /= self._LDM_SCALE_FACTOR

            x_img = self.ldm.decode(x_in)

            clip_in = self.cutnorm(x_img.add(1).div(2))
            clip_embeds = self.clip_model.encode_image(clip_in).float()
            dists = utils.spherical_dist_loss(clip_embeds.unsqueeze(1), self._text_emb_clip.unsqueeze(0))
            dists = dists.view([self.cutn, n, -1])

            losses = dists.sum(2).mean(0)
            #loss = losses * self.clip_guidance_scale
            loss = losses.sum() * self.clip_guidance_scale

            return -torch.autograd.grad(loss, x)[0]


    def encode_image_grid(self, images, grid_idx=None):
        if grid_idx is None:
            print("WARNING: Clip guided model requires a single grid_idx, defaulting to grid_idx = 0")
            grid_idx = 0
        elif (isinstance(grid_idx,list) and len(grid_idx)>1):
            print(f"WARNING: Clip guided model requires a single grid_idx, defaulting to first (grid_idx = {grid_idx[0]})")
            grid_idx = grid_idx[0]
        return super().encode_image_grid(images, grid_idx)



            
