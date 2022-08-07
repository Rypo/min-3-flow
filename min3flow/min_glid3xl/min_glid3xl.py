import os
import gc

import functools
from pathlib import Path
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
import requests

import clip
#from dalle_pytorch import DiscreteVAE, VQGanVAE

from .models.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from .models.encoders.modules import BERTEmbedder

from . import utils
from ..utils.io import download_weights
from ..utils import tensor_ops as tops


_DEFAULT_WEIGHT_ROOT = "~/.cache/min3flow/glid3xl"

# https://github.com/Jack000/glid-3-xl
# https://github.com/LAION-AI/ldm-finetune

_WEIGHT_DOWNLOAD_URLS = {
        # ldm first stage (required)
        'kl-f8.pt': 'https://dall-3.com/models/glid-3-xl/kl-f8.pt',
        # text encoder (required)
        'bert.pt': 'https://dall-3.com/models/glid-3-xl/bert.pt',
        
        # original diffusion model from CompVis
        'base.pt': 'https://dall-3.com/models/glid-3-xl/diffusion.pt', 
        # new model fine tuned on a cleaner dataset (will not generate watermarks, split images or blurry images
        'finetune.pt': 'https://dall-3.com/models/glid-3-xl/finetune.pt', 
        # The second finetune from jack000's glid-3-xl adds support for inpainting and can be used for unconditional output as well 
        # by setting the inpaint image_embed to zeros. Additionally finetuned to use the CLIP text embed via cross-attention (similar to unCLIP).
        'inpaint.pt': 'https://dall-3.com/models/glid-3-xl/inpaint.pt', 
        # erlich is inpaint.pt finetuned on a dataset collected from LAION-5B named Large Logo Dataset. 
        # It consists of roughly 100K images of logos with captions generated via BLIP using aggressive re-ranking and filtering.
        'erlich.pt': 'https://huggingface.co/laion/erlich/resolve/main/model/ema_0.9999_120000.pt',
        # Ongo is inpaint.pt finetuned on the Wikiart dataset consisting of about 100K paintings with captions generated via BLIP 
        # using aggressive re-ranking and filtering. We also make use of the original captions which contain the author name and the painting title.
        'ongo.pt': 'https://huggingface.co/laion/ongo/resolve/main/ongo.pt',
        # puck has been trained on pixel art. While the underlying kl-f8 encoder seems to struggle somewhat with pixel art, results are still interesting.
        'puck.pt': 'https://huggingface.co/laion/puck/resolve/main/puck.pt'
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
    def __init__(self, guidance_scale=3.0, batch_size=16, steps=100, sample_method='plms', imout_size=(256,256), 
                 diffusion_weight='finetune.pt', kl_weight='kl-f8.pt', bert_weight='bert.pt', weight_root = None, device=None) -> None:
        
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.steps = steps
        #self.skip_rate = skip_rate
        self.sample_method = sample_method
        self.H, self.W = int(imout_size[0]), int(imout_size[1])
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            else:
                try: 
                    dl_url = _WEIGHT_DOWNLOAD_URLS[weight]
                    wpath = os.path.join(self._weight_root, weight)
                    wpath = download_weights(wpath, dl_url)
                except KeyError:
                    raise ValueError(f"'{weight}' is not a path or known {stage} weight. Available weights aliases are {_available_weights()}")
            
            wpaths.append(wpath)

            
        diffusion_path, kl_path, bert_path = wpaths

        self._bert_path = bert_path # save for lazy loading bert

        self.model, self.diffusion, self.model_config = utils.timed(self.load_models, model_path=diffusion_path, sample_method=sample_method, steps=steps)
        self.clip_model, self.clip_preprocess = utils.timed(self.load_clip)
        self.ldm = utils.timed(self.load_ldm, kl_path=kl_path)
        #self.bert = utils.timed(self.load_bert, bert_path=self._bert_path)

        # TODO: ddpm doesn't exist? verify and remove option if intentional
        self.sample_fn = self.diffusion.plms_sample_loop_progressive if sample_method=='plms' else  self.diffusion.ddim_sample_loop_progressive 


    @property
    def _models_list(self):
        return [self.model, self.clip_model, self.ldm]
    def _to(self, device):
        for model in self._models_list:
            model.to(device)
        

    def init_model_config(self, sample_method, steps, model_state_dict):
        model_params = {
            'attention_resolutions': '32,16,8',
            'class_cond': False,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': '27',  # Modify this value to decrease the number of  # timesteps.               
            'image_size': 32,
            'learn_sigma': False,
            'noise_schedule': 'linear',
            'num_channels': 320,
            'num_heads': 8,
            'num_res_blocks': 2,
            'resblock_updown': False,
            'use_fp16': False,
            'use_scale_shift_norm': False,
            'clip_embed_dim': 768 if 'clip_proj.weight' in model_state_dict else None,
            'image_condition': True if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8 else False,
            'super_res_condition': True if 'external_block.0.0.weight' in model_state_dict else False,
        }
        
        if sample_method=='ddpm':
            model_params['timestep_respacing'] = 1000
        elif sample_method=='ddim':
            model_params['timestep_respacing'] = 'ddim'+ (str(steps) if steps else '50')
        elif steps:
            model_params['timestep_respacing'] = str(steps)

        model_config = model_and_diffusion_defaults()
        model_config.update(model_params)
        model_config['use_fp16'] &= (self.device.type!='cpu')

        return model_config


    def load_models(self, model_path, sample_method, steps):
        model_state_dict = torch.load(model_path, map_location='cpu')
        model_config = self.init_model_config(sample_method, steps, model_state_dict)
        model, diffusion = create_model_and_diffusion(**model_config) # Load models

        model=model.requires_grad_(False).eval().to(self.device)
        model.load_state_dict(model_state_dict, strict=False)
        del model_state_dict
        #model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        #model.requires_grad_(clip_guidance).eval().to(self.device)

        model.convert_to_dtype(model_config['use_fp16'])

        
        return model, diffusion, model_config


    def load_ldm(self, kl_path): # vae
        ldm = torch.load(kl_path, map_location=self.device)
        # TODO: verify that LDM can be frozen while using clip guidance. 
        # It does not throw an error, but need to verify that the output doesn't change.
        #ldm.requires_grad_(self.args.clip_guidance); #set_requires_grad(ldm, self.args.clip_guidance)
        ldm.freeze() 
        ldm.eval()

        return ldm

    def load_clip(self): 
        clip_model, clip_preprocess = clip.load('ViT-L/14', device=self.device, jit=False)
        clip_model = clip_model.eval().requires_grad_(False)

        return clip_model, clip_preprocess

    def load_bert(self, bert_path):
        bert = BERTEmbedder(1280, 32).to(self.device)
        bert = bert.half().eval().requires_grad_(False)
        bert.load_state_dict(torch.load(bert_path, map_location=self.device))

        return bert

    @contextmanager
    def ephemeral_bert(self, bert_path, text, negative=''):
        with torch.inference_mode():
            bert = BERTEmbedder(1280, 32).to(self.device)
            bert = bert.half().eval().requires_grad_(False)
            bert.load_state_dict(torch.load(bert_path, map_location=self.device), strict=False)
            #set_requires_grad(bert, False)
            
            #text_emb = bert.encode([self.args.text]).to(device=self.device, dtype=torch.float).expand(self.batch_size,-1,-1)
            #text_blank = bert.encode([self.args.negative]).to(device=self.device, dtype=torch.float).expand(self.batch_size,-1,-1)

            text_emb = bert.encode([text]*self.batch_size).to(device=self.device, dtype=torch.float)
            text_blank = bert.encode([negative]*self.batch_size).to(device=self.device, dtype=torch.float)

            try:
                yield text_emb, text_blank
            finally:
                del bert
                gc.collect()
                torch.cuda.empty_cache()

    
    def encode_text(self, text, negative=''):
        with self.ephemeral_bert(self._bert_path, text, negative) as bert_output:
            text_emb, text_blank = bert_output

        #text_emb = self.bert.encode([text]*self.batch_size).to(device=self.device, dtype=torch.float)
        #text_blank = self.bert.encode([negative]*self.batch_size).to(device=self.device, dtype=torch.float)

        toktext = clip.tokenize([text], truncate=True).expand(self.batch_size, -1).to(self.device)
        text_clip_blank = clip.tokenize([negative], truncate=True).expand(self.batch_size, -1).to(self.device)

        # clip context
        text_emb_clip = self.clip_model.encode_text(toktext)
        text_emb_clip_blank = self.clip_model.encode_text(text_clip_blank)

        return text_emb, text_blank, text_emb_clip, text_emb_clip_blank#, text_emb_norm

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
        if h.size(0) == 1:
            init = h.expand(2*self.batch_size, -1, -1, -1)
        else:
            osize=2*max(1,self.batch_size//h.size(0))
            init = h.tile(osize,1,1,1) # (2*BS, 4, H/8, W/8)
        
            if init.shape[0] < 2*self.batch_size:
                # Case when number of grid_idx samples does not evenly divide 2*batch_size
                # For now, just repeat encoding until 2*batch_size is reached
                # TODO: assess the impact of this on image diversity
                diff = 2*self.batch_size-init.shape[0]

                init = torch.cat([init, init[-diff:]], dim=0)
            

        return init



    def decode_sample(self, sample):
        if isinstance(sample, dict):
            sample = sample['pred_xstart'][:self.batch_size]
        
        images = sample / self._LDM_SCALE_FACTOR
        
        out=self.ldm.decode(images).add(1).div(2).clamp(0, 1) # shape: (batch_size, 3, height=256, width=256)

        return out


    def clf_free_sampling(self, x_t, ts, **kwargs):
        # Create a classifier-free guidance sampling function
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


    def sample_batches(self, num_batches, init_image_embed, skip_ts, model_kwargs):
        # TODO: the clip model can be unloaded in a similar fashion to Bert. 
        # after the initial image is encoded and the text is encoded.
        # In fact, the same is true for the LDM.
        # The only dependency for generation is the diffusion model and the sampler.
        # BUT, the LDM is needed for saving the samples afterwards, so may not make sense to unload it.
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

            batch_samples.append(sample['pred_xstart'][:self.batch_size].detach())
            # Cut at batch_size because data is duplicated afterwards (?)

           
        bsample = torch.cat(batch_samples, dim=0).detach_()
        
        return bsample


    #@functools.cache
    def update_model_kwargs(self, text: str, negative: str='', image_embed=None):
        #print('CLIPEMD, IMG_COND:',self.model_config['clip_embed_dim'], self.model_config['image_condition'])
        if self.model_config['image_condition'] and image_embed is None:
             # using inpaint model but no image is provided
            image_embed = torch.zeros(self.batch_size*2, 4, self.H//8, self.W//8, device=self.device)
        if (text, negative) != self._cache.get('texts', None):
            
            text_emb, text_blank, text_emb_clip, text_emb_clip_blank = utils.timed(self.encode_text, text=text, negative=negative)
            self._text_emb_clip = text_emb_clip

            self._model_kwargs = {
                "context": torch.cat([text_emb, text_blank], dim=0).float(),
                "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float() if self.model_config['clip_embed_dim'] else None,
                "image_embed": image_embed #(generated in GUI editting mode, unsupported for now)
            }

            self._cache['texts'] = (text, negative)

    def process_inputs(self, init_image: (Image.Image|str),  text: str):
        if text =='' and init_image != '':
            text = Path(init_image).stem.replace('_', ' ')
        
        if isinstance(init_image, str):
            init_image = Image.open(init_image).convert('RGB')
            
        
        if isinstance(init_image, Image.Image):
            init_image = tops.ungrid(init_image, h_out=256, w_out=256)
            init_image = init_image.to(self.device, dtype=torch.float).div(255.).clamp(0,1)

        return text, init_image


    def gen_samples(self, init_image:(Image.Image|str), text: str,  negative: str='', num_batches: int=1, grid_idx:(int|list)=None, skip_rate=0.5,  outdir: str=None, seed=-1,) -> torch.Tensor:
        if seed > 0:
            torch.manual_seed(seed)


        text, init_image = self.process_inputs(init_image, text)

        with torch.no_grad(): # inference mode breaks clip guidance
            init_image_embed = self.encode_image_grid(init_image, grid_idx=grid_idx)
            self.update_model_kwargs(text, negative)
            
        
        with torch.inference_mode(self._inference_safe):
            bsample = self.sample_batches(num_batches, init_image_embed, skip_ts=int(self.steps*skip_rate), model_kwargs = self._model_kwargs) 
        

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



            
