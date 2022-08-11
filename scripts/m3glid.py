import os
import gc
import time
import argparse
from pathlib import Path


from PIL import Image
import numpy as np

import torch
from min3flow.min_glid3xl import Glid3XL, Glid3XLClip



def get_parser():
    # argument parsing
    parser = argparse.ArgumentParser(description="Glid-3 XL", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--diffusion_weight', type=str, default = 'finetune.pt', help='path to the diffusion model')
    parser.add_argument('--kl_weight', type=str, default = 'kl-f8.pt', help='path to the LDM first stage model')
    parser.add_argument('--bert_weight', type=str, default = 'bert.pt', help='path to the bert model')
    parser.add_argument('--weight_root', type=str, default = None, help='path to directory containing the above models. If None, use user .cache')


    parser.add_argument('--outdir', type=str, default = 'output/1_diffuse/', help='path to image output directory')

    parser.add_argument('--text', type = str, required = False, default = '', help='your text prompt')
    parser.add_argument('--negative', type = str, required = False, default = '', help='negative text prompt')
    parser.add_argument('--init_image', type=str, required = False, default = None, help='init image to use')

    parser.add_argument('--grid_idx', type=int, required = False, default = None, help='index of image in grid to iterate on')

    parser.add_argument('--num_batches', '-nb', type = int, default = 1, required = False, help='number of batches')
    parser.add_argument('--batch_size', '-bs', type = int, default = 1, required = False, help='batch size')

    parser.add_argument('--width', type = int, default = 256, required = False, help='image size of output (multiple of 8)')
    parser.add_argument('--height', type = int, default = 256, required = False, help='image size of output (multiple of 8)')

    parser.add_argument('--seed', type = int, default=-1, required = False, help='random seed')

    parser.add_argument('--guidance_scale', '-gs', type = float, default = 3.0, required = False, help='classifier-free guidance scale')

    parser.add_argument('--steps', type = int, default = 100, required = False, help='number of diffusion steps')
    parser.add_argument('--skip_rate', type = float, default = 0.5, required = False, help='percent of diffusion steps to skip')

    #parser.add_argument('--cpu', dest='cpu', action='store_true')

    parser.add_argument('--sample_method', type=str, default='plms', choices=['plms', 'ddim', 'ddpm'], required = False, help='sampling method')

    # Clip Guidance arguments

    parser.add_argument('--clip_guidance','-cg', dest='clip_guidance', action='store_true')

    parser.add_argument('--clip_guidance_scale', '-cgs', type = float, default = 500, required = False, # default = 150
                        help='Controls how much the image should look like the prompt') # may need to use lower value for ddim

    parser.add_argument('--cutn', type = int, default = 16, required = False, help='Number of cuts')

    return parser


#@torch.inference_mode()
def do_run():
    args = get_parser().parse_args()
    #if args.edit and not args.mask:
    #    from gui import Draw, QApplication
    print(args)
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    
    if not args.clip_guidance:
        gxl = Glid3XL(
            guidance_scale=args.guidance_scale, 
            batch_size=args.batch_size, 
            steps=args.steps, 
            #skip_rate=args.skip_rate, 
            sample_method=args.sample_method,
            imout_size=(args.height,args.width), 
            diffusion_weight=args.diffusion_weight, #'pretrained/finetune.pt', 
            kl_weight=args.kl_weight,       #'pretrained/kl-f8.pt', 
            bert_weight=args.bert_weight,   #'pretrained/bert.pt', 
            weight_root=args.weight_root,
        )
    else:
        gxl = Glid3XLClip(
            clip_guidance_scale=args.clip_guidance_scale,
            cutn=args.cutn,
            guidance_scale=args.guidance_scale, 
            batch_size=args.batch_size, 
            steps=args.steps, 
            #skip_rate=args.skip_rate, 
            sample_method=args.sample_method,
            imout_size=(args.height,args.width), 
            diffusion_weight=args.diffusion_weight, #'pretrained/finetune.pt', 
            kl_weight=args.kl_weight,       #'pretrained/kl-f8.pt', 
            bert_weight=args.bert_weight,   #'pretrained/bert.pt', 
            weight_root=args.weight_root,
        )

    print('Using device:', gxl.device)
    
    gxl.gen_samples(
        init_image=args.init_image, 
        text=args.text, 
        
        negative=args.negative, 
        num_batches=args.num_batches,
        grid_idx=args.grid_idx,
        skip_rate=args.skip_rate, 
        outdir=args.outdir,
        seed=args.seed,
    )



if __name__ == '__main__':
    
    #import ..min3flow
    #import min3flow#.min_glid3xl import Glid3XL, Glid3XLClip
    #torch.set_float32_matmul_precision("highest")
    
    #torch.set_float32_matmul_precision("high")
    #torch.set_float32_matmul_precision("medium")
    #torch.backends.cudnn.allow_tf32 = True
    #torch.backends.cudnn.benchmark = True
    gc.collect()
    do_run()

