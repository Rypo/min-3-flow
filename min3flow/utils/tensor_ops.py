import numpy as np
from PIL import Image


import torch
import torchvision.utils as vutils
from torchvision.transforms import functional as TF

from einops import rearrange

def to_pil(tensor, nrow):
    imgrid = vutils.make_grid(tensor.detach_().cpu().clamp(0,1).squeeze(0), nrow=nrow, padding=0, pad_value=1)
    return TF.to_pil_image(imgrid)

def to_image(tensor, n_rows=None):
    if tensor.ndim == 4:
        if n_rows is None:
            n_rows = round(tensor.shape[0]**0.5)
        tensor = vutils.make_grid(tensor, nrow=n_rows, padding=0, normalize=False)
    
    # will fail if (H,W,C) from min_dalle
    image = TF.to_pil_image(tensor)
    return image

def grid_from_images(images: torch.FloatTensor) -> Image.Image:
    grid_size = int(np.math.sqrt(images.shape[0]))
    images = images.reshape([grid_size] * 2 + list(images.shape[1:]))
    image = images.flatten(1, 2).transpose(0, 1).flatten(1, 2)
    image = Image.fromarray(image.detach().to('cpu').numpy())
    return image

def ungrid(imgrid: (torch.FloatTensor|Image.Image), hw_out:(int|tuple)=256, channel_first=True) -> torch.FloatTensor:
    if isinstance(hw_out, int):
        hw_out = (hw_out, hw_out)

    if isinstance(imgrid, Image.Image):
        imgrid = TF.to_tensor(imgrid)

    dord='c h w' if channel_first else 'h w c'
    imbatch = rearrange(imgrid, f'c (b1 h) (b2 w) -> (b1 b2) {dord} ', h=hw_out[0], w=hw_out[1])
    return imbatch

def mkgrid(imbatch, nrow=4) -> Image.Image:
    imgrid = TF.to_pil_image(rearrange(imbatch, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=nrow))

    return imgrid