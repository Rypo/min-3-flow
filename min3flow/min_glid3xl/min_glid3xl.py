import os
import gc

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


def _rel_model_root(model_file=None):
    pretrained_dir = os.path.join(os.path.dirname(__file__), 'pretrained')
    rel_model_path = os.path.relpath(pretrained_dir, os.getcwd())
    if model_file is None:
        return rel_model_path
    return os.path.join(rel_model_path,model_file)


class Glid3XL:
    def __init__(self, guidance_scale=3.0, batch_size=16, steps=100, sample_method='plms', imout_size=(256,256), 
                 model_path=_rel_model_root('finetuned.pt'), 
                 kl_path=_rel_model_root('kl-f8.pt'), 
                 bert_path=_rel_model_root('bert.pt'), seed=-1) -> None:
        
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.steps = steps
        #self.skip_rate = skip_rate
        self.sample_method = sample_method
        self.H, self.W = int(imout_size[0]), int(imout_size[1])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seed = seed

        self._model_path = model_path
        self._kl_path = kl_path
        self._bert_path = bert_path

        self.cond_fn = None
        self.cur_t = None
        self._LDM_SCALE_FACTOR = 0.18215

        self._cache = {}
        
        self._setup_init(model_path=self._model_path, kl_path=self._kl_path, sample_method=sample_method, steps=steps)


    def _setup_init(self, model_path, kl_path, sample_method, steps):
        self.download_weights()
        self.model, self.diffusion, self.model_config = utils.timed(self.load_models, model_path=model_path, sample_method=sample_method, steps=steps, clip_guidance=False)
        self.clip_model, self.clip_preprocess = utils.timed(self.load_clip)
        self.ldm = utils.timed(self.load_ldm, kl_path=kl_path)

        # Brittle and hacky way to set the sample function, fix
        self.sample_fn = self.diffusion.plms_sample_loop_progressive if sample_method=='plms' else  self.diffusion.ddim_sample_loop_progressive 


    def download_weights(self):
        base_url = 'https://dall-3.com/models/glid-3-xl/'
        for wpath in [self._model_path, self._kl_path, self._bert_path]:
            wpath = Path(wpath)
            if not wpath.exists():
                print("downloading {} to {}".format(wpath.name, str(wpath)))
                r = requests.get(base_url + wpath.name)
                with open(wpath, 'wb') as f: 
                    f.write(r.content)
            #else:
            #    print("found {}".format(wpath.name))
    


    @property
    def _models_list(self):
        return [self.model, self.clip_model, self.ldm]
    def _to(self, device):
        for model in self._models_list:
            model.to(device)
        

    def init_model_config(self, sample_method, steps):
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
            'clip_embed_dim': 768, #768 if 'clip_proj.weight' in model_state_dict else None,
            'image_condition': False, #True if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8 else False,
            'super_res_condition': False, #True if 'external_block.0.0.weight' in model_state_dict else False,
        }
        
        if sample_method=='ddpm':
            model_params['timestep_respacing'] = 1000
        if sample_method=='ddim':
            model_params['timestep_respacing'] = 'ddim'+ (str(steps) if steps else '50')
        elif steps:
            model_params['timestep_respacing'] = str(steps)

        model_config = model_and_diffusion_defaults()
        model_config.update(model_params)

        #if self.args.cpu:
        if self.device.type=='cpu':
            model_config['use_fp16'] = False

        return model_config


    def load_models(self, model_path, sample_method, steps, clip_guidance=False):
        model_config = self.init_model_config(sample_method, steps)
        model, diffusion = create_model_and_diffusion(**model_config) # Load models

        #model_state_dict = torch.load(model_path, map_location='cpu')
        model.requires_grad_(clip_guidance).eval().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        #model.requires_grad_(clip_guidance).eval().to(self.device)


        if model_config['use_fp16']:
            model.convert_to_fp16()
        else:
            model.convert_to_fp32()

        
        return model, diffusion, model_config


    def load_ldm(self, kl_path):
        # vae
        ldm = torch.load(kl_path, map_location=self.device)#"cpu")
        # TODO: verify that LDM can be frozen while using clip guidance. 
        # It does not throw an error, but need to verify that the output doesn't change.
        ldm.freeze()
        ldm.eval()
        #ldm.to(self.device)
        #ldm.requires_grad_(self.args.clip_guidance)
        #set_requires_grad(ldm, self.args.clip_guidance)

        return ldm

    def load_clip(self):
        # clip
        clip_model, clip_preprocess = clip.load('ViT-L/14', device=self.device, jit=False)
        clip_model.eval().requires_grad_(False)

        return clip_model, clip_preprocess

    def load_bert(self, bert_path):
        bert = BERTEmbedder(1280, 32).to(self.device)
        bert.half().eval()
        bert.load_state_dict(torch.load(bert_path, map_location=self.device))#"cuda:0"))

    @contextmanager
    def load_unload_bert(self, bert_path, text, negative=''):
        with torch.inference_mode():
            bert = BERTEmbedder(1280, 32).to(self.device)
            bert.half().eval()
            bert.load_state_dict(torch.load(bert_path, map_location=self.device))#"cuda:0"))
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
        with self.load_unload_bert(self._bert_path, text, negative) as bert_output:
           text_emb, text_blank = bert_output


        toktext = clip.tokenize([text]*self.batch_size, truncate=True).to(self.device)
        text_clip_blank = clip.tokenize([negative]*self.batch_size, truncate=True).to(self.device)

        # clip context
        text_emb_clip = self.clip_model.encode_text(toktext)
        text_emb_clip_blank = self.clip_model.encode_text(text_clip_blank)

        
        text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)

        return text_emb, text_blank, text_emb_clip, text_emb_clip_blank, text_emb_norm

    #@torch.inference_mode()
    def encode_image_grid(self, init_img, text_emb_norm, grid_idx=None):
        if isinstance(init_img, str):
            init_img = Image.open(init_img).convert('RGB') #print(init.width, init.height)
        # assume grid for now, will break if 512x512 or similar. TODO: add argument later.
        images = utils.ungrid(init_img, h_out=256, w_out=256)
        if grid_idx is None:
            ssims = self.clip_scores(images, text_emb_norm=text_emb_norm, sort=True)
            #print(ssims)
            s_images = images[ssims.indices]
            init = s_images[:s_images.shape[0]//2].to(self.device, dtype=torch.float).div(255.).clamp(0,1)
        else:
            s_image = images[grid_idx]
            init = s_image.to(self.device, dtype=torch.float).div(255.).unsqueeze(0).clamp(0,1)
            #init = init.resize((self.W, self.H), Image.Resampling.LANCZOS)
            #init = TF.to_tensor(init).to(self.device).unsqueeze(0).clamp(0,1)

        #init = init.to(self.device, dtype=torch.float).div(255.).unsqueeze(0).clamp(0,1)
        #h = self.ldm.encode(init * 2 - 1).sample() *  self.LDM_SCALE_FACTOR
        h = self.ldm.encode(init.mul(2).sub(1)).sample() * self._LDM_SCALE_FACTOR
        #init = torch.cat(self.batch_size*2*[h], dim=0)
        #init = torch.tile(h, (32, 1, 1, 1))
        #init = torch.repeat_interleave(h, 2*(self.batch_size//h.size(0)), dim=0, output_size=2*self.batch_size) # (2*BS, 4, H/8, W/8)
        
        #osize=max(2*self.batch_size,2*max(1,self.batch_size//h.size(0)))
        osize=2*max(1,self.batch_size//h.size(0))
        init = h.tile(osize,1,1,1) # (2*BS, 4, H/8, W/8)
        #h.expand(32, -1, -1, -1)

        return init

    #@torch.inference_mode()
    def clip_scores(self, img_batch, text_emb_norm, sort=False):
        imgs_proc = torch.stack([self.clip_preprocess(TF.to_pil_image(img)) for img in img_batch], dim=0)
        image_embs = self.clip_model.encode_image(imgs_proc.to(self.device))
        #image_emb = self.clip_model.encode_image(self.clip_preprocess(out).unsqueeze(0).to(self.device))
        image_emb_norm = image_embs / image_embs.norm(dim=-1, keepdim=True)
        #print(image_embs.shape, self.text_emb_norm.shape)
        sims = F.cosine_similarity(image_emb_norm, text_emb_norm, dim=-1)
        if sort:
            return torch.sort(sims, descending=True)
        return sims

    def decode_sample(self, sample):
        if isinstance(sample, dict):
            sample = sample['pred_xstart'][:self.batch_size]
        
        images = sample / self._LDM_SCALE_FACTOR
        
        out=self.ldm.decode(images).add(1).div(2).clamp(0, 1) # shape: (batch_size, 3, height=256, width=256

        return out

    #@torch.inference_mode()
    def save_sample(self, sample, batch_idx, outpath):
        out=self.decode_sample(sample)

        vutils.save_image(out, outpath, nrow=int(np.math.sqrt(self.batch_size*max(1,batch_idx))), padding=0)
        

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
                self.clf_free_sampling, #model_fn,
                (self.batch_size*2, 4, int(self.H/8), int(self.W/8)),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                cond_fn=self.cond_fn,
                device=self.device,
                progress=True,
                init_image=init_image_embed,
                skip_timesteps=skip_ts #self.skip_ts #int(args.steps*args.skip_rate)#args.skip_timesteps,
            )

            for j, sample in enumerate(samples):
                self.cur_t -= 1
                #if j % 5 == 0 and j != self.diffusion.num_timesteps - 1:
                    #self.save_sample(sample, i, fname)
                    #save_sample(i, sample)

            batch_samples.append(sample['pred_xstart'][:self.batch_size].detach())
            # Cut at batch_size because data is duplicated afterwards (?)

           
        bsample = torch.cat(batch_samples, dim=0).detach_()
        #print(bsample.shape)
        return bsample

    def pre_sampling(self, text: str, init_image: str, negative: str='', grid_idx: int=0):
        with torch.no_grad():
            if text =='' and init_image != '':
                text = Path(init_image).stem.replace('_', ' ')

            if self._cache.get('text',False) != text and self._cache.get('negative',False) != negative:
                text_emb, text_blank, text_emb_clip, text_emb_clip_blank, text_emb_norm = utils.timed(self.encode_text, text=text, negative=negative)
                self._text_emb_clip = text_emb_clip
                self._cache['text'] = text
                self._cache['negative'] = negative

                self._text_emb_norm = text_emb_norm
        

                self._model_kwargs = {
                    "context": torch.cat([text_emb, text_blank], dim=0).float(),
                    "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float(), # if self.model_config['clip_embed_dim'] else None,
                    "image_embed": None #image_embed (generated in GUI editting mode, unsupported for now)
                }

            init_image_embed = self.encode_image_grid(init_image, self._text_emb_norm, grid_idx=grid_idx)
            return init_image_embed, self._model_kwargs

    @torch.inference_mode()
    def gen_samples(self, text: str, init_image: str, negative: str='', num_batches: int=1, grid_idx: int=None, skip_rate=0.5, outdir: str=None,):
        if self.seed > 0:
            torch.manual_seed(self.seed)

        skip_ts = int(self.steps*skip_rate)

        init_image_embed, model_kwargs = self.pre_sampling(text, init_image, negative, grid_idx)
        
        bsample = self.sample_batches(num_batches, init_image_embed, skip_ts, model_kwargs) #torch.cat(batch_samples, dim=0).detach_()
        
        #self.model.to('cpu'); torch.cuda.empty_cache()
        if outdir is not None:
            fname = Path(init_image).name if init_image else 'GEN__'+text.replace(' ', '_')+'.png'
            outpath = os.path.join(outdir,  f'{num_batches:05d}_{fname}')
            self.save_sample(bsample, num_batches, outpath)
        else:
            imout = self.decode_sample(bsample)
            return utils.to_pil(imout, nrow = int(np.math.sqrt(self.batch_size*max(1,num_batches))))




class Glid3XLClip(Glid3XL):
    # May be cleaner to use a single class for both Glid3XL and Glid3XLClip. 
    # However, seperation allows for future optimization over non-clip model default. 
    # The non-clip guided model could temporarily unload clip and ldm while generating samples, and reload for final output.
    # The clip guided model needs to keep both models loaded for generating samples.
    def __init__(self, clip_guidance_scale=500, cutn=16, guidance_scale=3.0, batch_size=1, steps=100, sample_method='plms', imout_size=(256,256), 
                 model_path='pretrained/finetuned.pt', kl_path='pretrained/kl-f8.pt', bert_path='pretrained/bert.pt', seed=-1) -> None:

        super().__init__(guidance_scale=guidance_scale, batch_size=batch_size, steps=steps, sample_method=sample_method, imout_size=imout_size, 
                 model_path=model_path, kl_path=kl_path, bert_path=bert_path, seed=seed)

        self.cond_fn = self.conditioning_fn


        self.clip_guidance_scale = clip_guidance_scale
        self.cutn = cutn
    
        N_PX = self.clip_model.visual.input_resolution # 224
        self.cutnorm = T.Compose([utils.MakeCutouts(N_PX, self.cutn), T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    
    def conditioning_fn(self, x, t, context=None, clip_embed=None, image_embed=None):
        
        kw = {
            'context': context[:self.batch_size],
            'clip_embed': clip_embed[:self.batch_size], #if self.model_config['clip_embed_dim'] else None,
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
            x_in = out['pred_xstart'] * fac + x * (1 - fac)

            x_in /= self._LDM_SCALE_FACTOR

            x_img = self.ldm.decode(x_in)

            clip_in = self.cutnorm(x_img.add(1).div(2))
            clip_embeds = self.clip_model.encode_image(clip_in).float()
            dists = utils.spherical_dist_loss(clip_embeds.unsqueeze(1), self._text_emb_clip.unsqueeze(0))
            dists = dists.view([self.cutn, n, -1])

            losses = dists.sum(2).mean(0)

            loss = losses.sum() * self.clip_guidance_scale

            return -torch.autograd.grad(loss, x)[0]



    def gen_samples(self, text: str, init_image: str, negative: str='', num_batches: int=1, grid_idx: int=None, skip_rate=0.5, outdir: str=None,):
        if self.seed > 0:
            torch.manual_seed(self.seed)
        skip_ts = int(self.steps*skip_rate)

        init_image_embed, model_kwargs = self.pre_sampling(text, init_image, negative, grid_idx)
        
        
        self.model.requires_grad_(True)
        bsample = self.sample_batches(num_batches, init_image_embed, skip_ts, model_kwargs) #torch.cat(batch_samples, dim=0).detach_()

        if outdir is not None:
            fname = Path(init_image).name if init_image else 'GEN__'+text.replace(' ', '_')+'.png'
            outpath = os.path.join(outdir,  f'{num_batches:05d}_{fname}')
            self.save_sample(bsample, num_batches, outpath)
        else:
            imout = self.decode_sample(bsample)
            return utils.to_pil(imout, num_batches)
    
