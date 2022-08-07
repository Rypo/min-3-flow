import io
import os
import time
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T



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


#@torch.inference_mode()
# def clip_scores(self, img_batch, text_emb_norm, sort=False):
#     imgs_proc = torch.stack([self.clip_preprocess(TF.to_pil_image(img)) for img in img_batch], dim=0)
#     image_embs = self.clip_model.encode_image(imgs_proc.to(self.device))
#     #image_emb = self.clip_model.encode_image(self.clip_preprocess(out).unsqueeze(0).to(self.device))
#     image_emb_norm = image_embs / image_embs.norm(dim=-1, keepdim=True)
#     #print(image_embs.shape, self.text_emb_norm.shape)
#     sims = F.cosine_similarity(image_emb_norm, text_emb_norm, dim=-1)
#     if sort:
#         return torch.sort(sims, descending=True)
#     return sims

# def clip_sort(self, img_batch, text_embs):
#     '''Sort image batch by cosine similarity to CLIP text embeddings.
    
#     Args:
#         img_batch (tensor): uint8 tensor of shape (BS, C, H, W)
#         text_embs (tensor): text embeddingings produced by clip_model

#     Returns:
#         tensor: sorted image batch of shape (BS, C, H, W)
#     '''
    
#     # annoyingly, without first converting to PIL, the outputs differ enough to throw off rankings.
#     imgs_proc = torch.stack([self.clip_preprocess(TF.to_pil_image(img)) for img in img_batch], dim=0)
#     image_embs = self.clip_model.encode_image(imgs_proc.to(self.device))
#     sims = F.cosine_similarity(image_embs, text_embs, dim=-1)
#     ssims = torch.sort(sims, descending=True)

#     return img_batch[ssims.indices]

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _clip_preprocess(n_px):
    return T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _clip_preprocess_tensor(n_px):
    '''Attempt to mirror the preprocessing of the clip_preprocess function on a tensor of shape [..., H, W].
    Unfortunately, does not replicate PIL behavior, thus making the CLIP scores inaccurate.
    '''
    return T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(n_px),
        T.Lambda(lambda x: x.to(torch.float).div(255.)),
        #T.Lambda(lambda x: torch.as_tensor(x, dtype=torch.float).div(255)),
        #_convert_image_to_rgb,
        #T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


