'''Much of the refactoring code inspired by or directly taken from: https://github.com/Lin-Sinorodin/SwinIR_wrapper/'''

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import requests
from tqdm.auto import tqdm

from .models.network_swinir import SwinIR as net
from . import utils as util

#MODULE_DIR = Path(__file__).resolve().parent
#ROOT_DIR = MODULE_DIR.parent

def _rel_model_root(model_file=None):
    pretrained_dir = os.path.join(os.path.dirname(__file__), 'pretrained')
    rel_model_path = os.path.relpath(pretrained_dir, os.getcwd())
    if model_file is None:
        return rel_model_path
    return os.path.join(rel_model_path, model_file)



def download_weights(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(model_path))
    r = requests.get(url, allow_redirects=True)
    print(f'downloading model {model_path}')
    with open(model_path, 'wb') as f:
        f.write(r.content)


def define_model(task, scale, large_model, training_patch_size, noise=None, jpeg=None, model_dir=_rel_model_root()):
    weight_name = util.construct_weightname(task, scale, large_model, training_patch_size, noise, jpeg)
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

    model_path = os.path.join(model_dir, weight_name)
    if not os.path.exists(model_path):
        download_weights(model_path)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model




class SwinIR:
    def __init__(self, task='real_sr', scale=4, large_model=True, training_patch_size=None, noise=None, jpeg=None, model_dir=_rel_model_root()) -> None:
        #self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = define_model(task, scale, large_model, training_patch_size, noise, jpeg, model_dir).eval().to(self.device)

        self.task = task
        self.scale = scale
        self.large_model = large_model
        self.training_patch_size = training_patch_size
        self.noise = noise
        self.jpeg = jpeg
        
        # minimal extract from setup() fuction
        self.border = scale if task in ['classical_sr', 'lightweight_sr'] else 0
        self.window_size = 7 if task in ['jpeg_car'] else 8

    @property
    def _models_list(self):
        return [self.model]
    def _to(self, device):
        for model in self._models_list:
            model.to(device)


    def _arr2tensor(self, img: np.array) -> torch.tensor:
        """cv2 format - np.array HWC-BGR -> model format torch.tensor NCHW-RGB. (from the official repo)"""
        
        img = img.astype(np.float32) / 255.  # image to HWC-BGR, float32
        img = np.transpose(img if img.shape[2] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img = torch.from_numpy(img).unsqueeze(0).to(self.device, dtype=torch.float)  # CHW-RGB to NCHW-RGB
        return img

    def _window_pad_img(self, img: torch.tensor, window_size=8) -> torch.tensor:
        """pad input image to be a multiple of window_size (pretrained with window_size=8). (from the official repo)"""
        _, _, h_old, w_old = img.size()
        h_new = (h_old // window_size + 1) * window_size
        w_new = (w_old // window_size + 1) * window_size
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_new, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_new]
        return img


    def preprocess_img(self, img_lq):
        img_lq = self._arr2tensor(img_lq)
        h_old, w_old = img_lq.size()[-2:]
        img_lq = self._window_pad_img(img_lq, self.window_size)

        return img_lq, h_old, w_old 

    def postprocess_output(self, output: torch.tensor, h_old: int, w_old: int):
        output = output[..., :h_old * self.scale, :w_old * self.scale]
        output = output.detach().squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return output
        
    @torch.inference_mode()
    def incremental_upscale_patchwise(self, img_lq: np.array, slice_dim=256, slice_overlap=0, outpath=None, keep_pbar=False) -> np.array:
        """Apply super resolution on smaller patches and return full image. 
        
        Preprocesses and transfers to GPU 1 patch at a time. Slower than upscale_patchwise, but more memory efficient."""
        if isinstance(img_lq, str):
            img_lq = cv2.imread(img_lq, cv2.IMREAD_COLOR)
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
        scale = self.scale
        h, w, c = img_lq.shape
        
        slice_step = slice_dim - slice_overlap
        
        patches = []
        slice_inds = []
        for h_slice in range(0, h, slice_step):
            for w_slice in range(0, w, slice_step):
                h_max = min(h_slice + slice_dim, h)
                w_max = min(w_slice + slice_dim, w)
                #print(f'[{h_slice:4d}:{h_max:4d}, {w_slice:4d}:{w_max:4d}]')

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
            img_lq = cv2.imread(img_lq, cv2.IMREAD_COLOR)
        scale = self.scale
        h, w, c = img_lq.shape
        #img_hq = np.zeros((h * scale, w * scale, c))
        patbatch,patinds = self.extract_patches(img_lq, slice_dim, slice_overlap)
        patches = (torch.from_numpy(patbatch[...,[2,1,0]].transpose(0,3,1,2)).float()/255.) # B,H,W,C-BGR -> B, C-RGB, H, W
        
        padded_patches = self._window_pad_img(patches).to(self.device)
        scaled_inds = torch.from_numpy(patinds*4).to(self.device)

        h_out = h * scale
        w_out = w * scale
        img_hq = torch.zeros(1, c, h_out, w_out, device=self.device)
        
        #torch.utils.data
        for i, (hs,he,ws,we) in enumerate(tqdm(scaled_inds)):
            img_hq[...,hs:he,ws:we] = self.model(padded_patches[[i]])[...,:h,:w]

        img_out = self.postprocess_output(img_hq,h,w)
        if outpath is not None:
            util.save_image(img_out, outpath)
            print('Saved to:', outpath)
        else:
            return img_out


    @torch.inference_mode()
    def upscale(self, img, outpath=None):
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
        img, h_old, w_old = self.preprocess_img(img)
        output = self.model(img)
        output = self.postprocess_output(output, h_old, w_old)

        if outpath is not None:
            util.save_image(output, outpath)
            print('Saved to:', outpath)
        else:
            return output

    @torch.inference_mode()
    def upscale_tiled(self, img, tile_size=256, tile_overlap=0, outpath=None):
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
        
        img, h_old, w_old = self.preprocess_img(img)
        output = self.upscale_patchwise(img, tile_size, tile_overlap)
        output = self.postprocess_output(output, h_old, w_old)

        if outpath is not None:
            util.save_image(output, outpath)
            print('Saved to:', outpath)
        else:
            return output

