import io
import os
import time
import requests
from pathlib import Path
from contextlib import contextmanager

import PIL
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import functional as TF
import torchvision.utils as vutils

from einops import rearrange



class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

#@torch.jit.script
def spherical_dist_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2., dim=-1)
    y = F.normalize(y, p=2., dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])



def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def to_pil(tensor, nrow):
    imgrid = vutils.make_grid(tensor.detach_().cpu().clamp(0,1).squeeze(0), nrow=nrow, padding=0, pad_value=1)
    return TF.to_pil_image(imgrid)

def grid_from_images(images: torch.FloatTensor) -> Image.Image:
    grid_size = int(np.math.sqrt(images.shape[0]))
    images = images.reshape([grid_size] * 2 + list(images.shape[1:]))
    image = images.flatten(1, 2).transpose(0, 1).flatten(1, 2)
    image = Image.fromarray(image.detach().to('cpu').numpy())
    return image

def ungrid(imgrid: Image.Image, h_out=256, w_out=256):
    
    tgrid = torch.from_numpy(np.array(imgrid))
    imbatch = rearrange(tgrid, '(b1 h) (b2 w) c -> (b1 b2) c h w ', h=h_out, w=w_out)
    return imbatch

def mkgrid(imbatch, nrow=4):
    imgrid = TF.to_pil_image(rearrange(imbatch, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=nrow))

    return imgrid

def timed(fn, *args, **kwargs):
    start = time.perf_counter() # print(msg, end=' ')
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    print(f'{fn.__name__} ({end - start:.2f}s)') # print('({:.2f}s)'.format(time.perf_counter()-t0))
    return result



def prepend_clip_score(filename, similarity):
    final_filename = filename.split('_') #f'output/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.png'
    final_filename.insert(1, f'{similarity.item():0.3f}')
    final_filename = '_'.join(final_filename)
    #final_filename = f'output/{args.prefix}_{i * args.batch_size + k:05}_{similarity.item():0.3f}.png'
    os.rename(filename, final_filename)
    
    npy_filename = filename.replace('output/','output_npy/').replace('.png','.npy')
    npy_final = final_filename.replace('output/','output_npy/').replace('.png','.npy') 
    #f'output_npy/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.npy'
    #npy_final = f'output_npy/{args.prefix}_{i * args.batch_size + k:05}_{similarity.item():0.3f}.npy'
    os.rename(npy_filename, npy_final)





    # def save_sample(i, sample, clip_score=False):
    #     for k, image in enumerate(sample['pred_xstart'][:args.batch_size]):
    #         image /= 0.18215
    #         im = image.unsqueeze(0)
    #         out = ldm.decode(im)

    #         npy_filename = f'output_npy/{args.prefix}{i * args.batch_size + k:05}.npy'
    #         with open(npy_filename, 'wb') as outfile:
    #             np.save(outfile, image.detach().cpu().numpy())

    #         out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))

    #         filename = f'output/{args.prefix}{i * args.batch_size + k:05}.png'
    #         out.save(filename)

    #         if clip_score:
    #             image_emb = clip_model.encode_image(clip_preprocess(out).unsqueeze(0).to(device))
    #             image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)

    #             similarity = F.cosine_similarity(image_emb_norm, text_emb_norm, dim=-1)

    #             final_filename = f'output/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.png'
    #             #final_filename = f'output/{args.prefix}_{i * args.batch_size + k:05}_{similarity.item():0.3f}.png'
    #             os.rename(filename, final_filename)

    #             npy_final = f'output_npy/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.npy'
    #             #npy_final = f'output_npy/{args.prefix}_{i * args.batch_size + k:05}_{similarity.item():0.3f}.npy'
    #             os.rename(npy_filename, npy_final)






# def get_kwargs(self):
#     image_embed = None

#     # image context
#     #if self.args.edit:
#     #    image_embed = edit_mode(self.ldm, self.args)
#     #elif self.model_config['image_condition']:
#         # using inpaint model but no image is provided
#     #    image_embed = torch.zeros(self.args.batch_size*2, 4, self.args.height//8, self.args.width//8, device=self.device)

#     kwargs = {
#         "context": torch.cat([self.text_emb, self.text_blank], dim=0).float(),
#         "clip_embed": torch.cat([self.text_emb_clip, self.text_emb_clip_blank], dim=0).float() if self.model_config['clip_embed_dim'] else None,
#         "image_embed": image_embed
#     }
#     return kwargs



# @torch.inference_mode()
# def load_encode_bert(self):
#     bert = BERTEmbedder(1280, 32, device=self.device)
#     bert.half().eval()
#     #sd = 
#     bert.load_state_dict(torch.load(self.args.bert_path, map_location=self.device))
#     #del sd
#     #bert.to(self.device)
#     #bert.half().eval()
#     utils.set_requires_grad(bert, False)
    
    
#     text_emb = bert.encode([self.args.text]*self.batch_size).to(device=self.device, dtype=torch.float)
#     text_blank = bert.encode([self.args.negative]*self.batch_size).to(device=self.device, dtype=torch.float)
#     del bert
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     return text_emb, text_blank


# def save_npimage(self, i, k, image):
#     npy_filename = f'output_npy/{self.args.prefix}{i * self.batch_size + k:05}.npy'
#     with open(npy_filename, 'wb') as outfile:
#         np.save(outfile, image.detach().cpu().numpy())

# #@torch.inference_mode()
# def clip_score(self, out):

#     #image_emb = self.clip_model.encode_image(self.clip_proct(out).to(self.device))
#     image_emb = self.clip_model.encode_image(self.clip_preprocess(out).unsqueeze(0).to(self.device))
#     image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)
#     similarity = F.cosine_similarity(image_emb_norm, self.text_emb_norm, dim=-1)
#     return similarity