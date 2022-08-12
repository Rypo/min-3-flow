'''Much of the refactoring code inspired by or directly taken from: https://github.com/Lin-Sinorodin/SwinIR_wrapper/'''

import os
from pathlib import Path

#import cv2
from PIL import Image
import numpy as np
import torch
import requests
from tqdm.auto import tqdm

from .models.network_swinir import SwinIR as net
from . import utils as util
from ..utils.io import download_weights

#MODULE_DIR = Path(__file__).resolve().parent
#ROOT_DIR = MODULE_DIR.parent
_WEIGHT_DOWNLOAD_URL = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'
_DEFAULT_WEIGHT_ROOT = "~/.cache/min3flow/swinir"


def define_model(task, scale, large_model, training_patch_size, noise=None, jpeg=None, weight_root=None):
    # 001 classical image sr
    if task == 'classical_sr':
        model = net(upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6]*6, embed_dim=180, num_heads=[6]*6,
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif task == 'lightweight_sr':
        model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6]*4, embed_dim=60, num_heads=[6]*4,
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif task == 'real_sr':
        if not large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6]*6, embed_dim=180, num_heads=[6]*6,
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6]*9, embed_dim=240, num_heads=[8]*9,
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6]*6, embed_dim=180, num_heads=[6]*6,
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6]*6, embed_dim=180, num_heads=[6]*6,
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6]*6, embed_dim=180, num_heads=[6]*6,
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    if weight_root is None:
        weight_root = os.path.expanduser(_DEFAULT_WEIGHT_ROOT)
    
    weight_name = util.construct_weightname(task, scale, large_model, training_patch_size, noise, jpeg)
    model_path = os.path.join(weight_root, weight_name)
    model_path = download_weights(model_path, _WEIGHT_DOWNLOAD_URL.format(weight_name))
    pretrained_model = torch.load(model_path)
    #model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.load_state_dict(pretrained_model.get(param_key_g, pretrained_model), strict=True)

    return model




class SwinIR:
    def __init__(self, task='real_sr', scale=4, large_model=True, training_patch_size=None, noise=None, jpeg=None, weight_root=None, device=None) -> None:
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = define_model(task, scale, large_model, training_patch_size, noise, jpeg, weight_root).eval().to(self.device)

        self.task = task
        self.scale = scale
        self.large_model = large_model
        self.training_patch_size = training_patch_size
        self.noise = noise
        self.jpeg = jpeg
        
        # minimal extract from setup() fuction
        self.border = scale if task in ['classical_sr', 'lightweight_sr'] else 0
        self.window_size = 7 if task in ['jpeg_car'] else 8

    @classmethod
    def for_denoising(cls, noise=15, color=True, model_dir=None):
        task='color_dn' if color else 'gray_dn'
        return cls(task, scale=1, large_model=False, training_patch_size=None, noise=noise, jpeg=None, model_dir=model_dir)

    @classmethod
    def for_jpegcar(cls, jpeg=40, model_dir=None):
        return cls(task='jpeg_car', scale=1, large_model=False, training_patch_size=None, noise=None, jpeg=jpeg, model_dir=model_dir)
    
    @classmethod
    def for_sr(cls, task='real_sr', scale=4, large_model=True, training_patch_size=None,  model_dir=None):
        return cls(task=task, scale=scale, large_model=large_model, training_patch_size=training_patch_size, model_dir=model_dir)



    @property
    def _models_list(self):
        return [self.model]
    def _to(self, device):
        for model in self._models_list:
            model.to(device)


    def _arr2tensor(self, img: np.array) -> torch.tensor:
        """cv2 format - np.array HWC-BGR -> model format torch.tensor NCHW-RGB. (from the official repo)"""
        
        img = img.astype(np.float32) / 255.  # image to HWC-RGB, float32
        #img = np.transpose(img if img.shape[2] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img = np.transpose(img, (2, 0, 1))  # HWC-RGB to CHW-RGB
        img = torch.from_numpy(img).unsqueeze(0).to(self.device, dtype=torch.float)  # CHW-RGB to NCHW-RGB
        return img

    def _window_pad_img(self, img: torch.tensor, window_size=8) -> torch.tensor:
        """pad input image to be a multiple of window_size (pretrained with window_size=8). (from the official repo)"""
        if img.ndim == 3:
            img = img.unsqueeze(0)
        _, _, h_old, w_old = img.size()
        h_new = (h_old // window_size + 1) * window_size
        w_new = (w_old // window_size + 1) * window_size
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_new, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_new]
        return img


    def preprocess_img(self, img_lq):
        if not isinstance(img_lq, torch.Tensor):
            if isinstance(img_lq, Image.Image):
                img_lq = np.array(img_lq)
            img_lq = self._arr2tensor(img_lq)
        h_old, w_old = img_lq.size()[-2:]
        img_lq = self._window_pad_img(img_lq, self.window_size)

        return img_lq, h_old, w_old 


    def postprocess_output(self, output: torch.tensor, h_old: int, w_old: int) -> torch.tensor:
        output = output[..., :h_old * self.scale, :w_old * self.scale]
        output = output.detach().squeeze().float().clamp_(0, 1)
        
        return output

    def to_numpy(self, output):
        output = output.cpu().numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return output
        
    @torch.inference_mode()
    def incremental_upscale_patchwise(self, img_lq: np.array, slice_dim=256, slice_overlap=0, outpath=None, keep_pbar=False) -> np.array:
        """Apply super resolution on smaller patches and return full image. 
        Preprocesses and transfers to GPU 1 patch at a time. Slower than upscale_patchwise, but more memory efficient."""
        if isinstance(img_lq, str):
            img_lq = Image.open(img_lq).convert('RGB')
            #img_lq = cv2.imread(img_lq, cv2.IMREAD_COLOR)
        scale = self.scale
        h, w, c = img_lq.shape
        img_hq = np.zeros((h * scale, w * scale, c))

        slice_step = slice_dim - slice_overlap
        num_patches = int(np.ceil(h / slice_step) * np.ceil(w / slice_step))
        with tqdm(total=num_patches, unit='patch', desc='Performing SR on patches', leave=keep_pbar) as pbar:
            for h_slice in range(0, h, slice_step):
                for w_slice in range(0, w, slice_step):
                    h_max = min(h_slice + slice_dim, h)
                    w_max = min(w_slice + slice_dim, w)
                    pbar.set_postfix(Status=f'[{h_slice:4d}-{h_max:4d}, {w_slice:4d}-{w_max:4d}]')

                    # apply super resolution on slice
                    img_slice = img_lq[h_slice:h_max, w_slice:w_max]
                    img_slice_hq = self.upscale(img_slice)

                    # update full image
                    img_hq[h_slice * scale:h_max * scale, w_slice * scale:w_max * scale] = img_slice_hq
                    pbar.update(1)

            pbar.set_postfix(Status='Done')
        
        if outpath is not None:
            util.save_image(np.uint8(img_hq), outpath)
            print('Saved to:', outpath)
        else:
            return np.uint8(img_hq)
        #return np.uint8(img_hq)

    def extract_patches(self, img_lq: np.array, slice_dim=256, slice_overlap=0) -> np.array:
        """Extract patches from input image"""
        h, w, c = img_lq.shape
        
        slice_step = slice_dim - slice_overlap
        
        patches = []
        slice_inds = []
        for h_slice in range(0, h, slice_step):
            for w_slice in range(0, w, slice_step):
                h_max = min(h_slice + slice_dim, h)
                w_max = min(w_slice + slice_dim, w)

                # extract slice
                patches.append(img_lq[h_slice:h_max, w_slice:w_max])
                slice_inds.append([h_slice,h_max, w_slice,w_max])
        
        return np.stack(patches, 0), np.array(slice_inds)


    @torch.inference_mode()
    def upscale_patchwise(self, img_lq: np.array, slice_dim=256, slice_overlap=0, outpath=None) -> np.array:
        """Apply super resolution on smaller patches and return full image.
        Preprocesses and transfers to all patches to GPU at once. Faster than incremental_upscale_patchwise, but less memory efficient.
        """
        if isinstance(img_lq, str):
            img_lq = Image.open(img_lq).convert('RGB')

        if isinstance(img_lq, Image.Image):
            img_lq = np.array(img_lq)
        
        h, w, c = img_lq.shape[-3:]
        #img_hq = np.zeros((h * scale, w * scale, c))
        patbatch,patinds = self.extract_patches(img_lq, slice_dim, slice_overlap)
        patches = torch.as_tensor(patbatch, dtype=torch.float).permute(0,3,1,2).div(255.) # B,H,W,C-RGB -> B, C-RGB, H, W
        
        
        padded_patches = self._window_pad_img(patches).to(self.device)
        scaled_inds = torch.from_numpy(patinds*self.scale).to(self.device)


        img_hq = torch.zeros(1, c, (h*self.scale), (w*self.scale), device=self.device)
        
        
        for i, (hs,he,ws,we) in enumerate(tqdm(scaled_inds)):
            img_hq[...,hs:he,ws:we] = self.model(padded_patches[[i]])[...,:h,:w]

        img_out = self.postprocess_output(img_hq,h,w)
        if outpath is None:
            return img_out
        # else
        img_out = self.to_numpy(img_out)
        util.save_image(img_out, outpath)
        print('Saved to:', outpath)
        
            


    @torch.inference_mode()
    def upscale(self, img, outpath=None):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        img, h_old, w_old = self.preprocess_img(img)
        output = self.model(img)
        output = self.postprocess_output(output, h_old, w_old)

        if outpath is None:
            return output
                    
        # else
        img_out = self.to_numpy(output)
        util.save_image(img_out, outpath)
        print('Saved to:', outpath)

    @torch.inference_mode()
    def upscale_prebatched(self, img:torch.FloatTensor, outpath=None):
        
        imgs, h_old, w_old = self.preprocess_img(img)
        
        # output = torch.stack([self.model(img.unsqueeze(0))[...,:h_old,:w_old] for img in tqdm(imgs)],0)
        output = torch.stack([self.model(img.unsqueeze(0)) for img in tqdm(imgs)],0)
        #output = self.upscale_patchwise(img, tile_size, tile_overlap)
        output = self.postprocess_output(output, h_old, w_old)

        if outpath is None:
            return output
        # else
        img_out = self.to_numpy(output)
        util.save_image(img_out, outpath)
        print('Saved to:', outpath)

