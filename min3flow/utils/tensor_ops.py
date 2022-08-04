import numpy as np
from PIL import Image


import torch
import torchvision.utils as vutils
from torchvision.transforms import functional as TF

from einops import rearrange

def to_pil(tensor, nrow):
    imgrid = vutils.make_grid(tensor.detach_().cpu().clamp(0,1).squeeze(0), nrow=nrow, padding=0, pad_value=1)
    return TF.to_pil_image(imgrid)

def grid_from_images(images: torch.FloatTensor) -> Image.Image:
    grid_size = int(np.math.sqrt(images.shape[0]))
    images = images.reshape([grid_size] * 2 + list(images.shape[1:]))
    image = images.flatten(1, 2).transpose(0, 1).flatten(1, 2)
    image = Image.fromarray(image.detach().to('cpu').numpy())
    return image

def ungrid(imgrid: Image.Image, h_out=256, w_out=256, channel_first=True):
    
    tgrid = torch.from_numpy(np.array(imgrid))
    dord='c h w' if channel_first else 'h w c'
    imbatch = rearrange(tgrid, f'(b1 h) (b2 w) c -> (b1 b2) {dord} ', h=h_out, w=w_out)
    return imbatch

def mkgrid(imbatch, nrow=4):
    imgrid = TF.to_pil_image(rearrange(imbatch, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=nrow))

    return imgrid