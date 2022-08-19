
import argparse
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

#from .models.network_swinir import SwinIR as net
#import min_swinir.utils as util
from min3flow.min_swinir import SwinIR


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('init_img', type=str, help='input image path')
    
    parser.add_argument('--task', type=str, default='real_sr', choices = ['classical_sr', 'lightweight_sr', 'real_sr', 'gray_dn', 'color_dn', 'jpeg_car'], help='The task to be performed')
    parser.add_argument('--scale', type=int, default=4, choices=[1, 2, 3, 4, 8], help='scale factor if performing upsamping. (1 for --task {"gray_dn" | "color_dn" | "jpeg_car"})') # 1 for dn and jpeg car
    parser.add_argument('--tile', type=int, default=None, help='Tile size. If None, entire image is passed to model.')
    parser.add_argument('--tile_overlap', type=int, default=0, metavar='N_PIX', help='Overlapping of different tiles') #32
    parser.add_argument('--output_dir', type=str, default='output/2_upsample/', metavar='DIR', help='output directory for upsampled images')
    parser.add_argument('--model_dir', type=str, default=None, metavar='DIR', help='path to pretrained models')

    taskarg_group = parser.add_mutually_exclusive_group(required=False)
    taskarg_group.add_argument('--small_model', action='store_true', help='Ignored unless --task "real_sr". Use smaller model, trained on less data.')
    taskarg_group.add_argument('--noise', type=int, default=15, choices=[15, 25, 50], help='Ignored unless --task {"gray_dn" | "color_dn"}. noise strength level') # : 15, 25, 50
    taskarg_group.add_argument('--jpeg', type=int, default=40, choices=[10,20,30,40], help='Ignored unless --task "jpeg_car". jpeg artifact level') # scale factor: 10, 20, 30, 40
    taskarg_group.add_argument('--training_patch_size', type=int, default=None, 
                        help='Ignored unless --task "classical_sr". If using classical_sr, a value of 48 or 64 can be used to differentiate between the two models.')
    return parser




def main():
    args = get_parser().parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model

    sir = SwinIR(
        task=args.task, 
        scale=args.scale, 
        large_model=(not args.small_model), 
        training_patch_size=args.training_patch_size, 
        noise=args.noise, 
        jpeg=args.jpeg,
        model_dir=args.model_dir, 
    )
    filename = Path(args.init_img).name
    outpath = Path(args.output_dir).joinpath(filename).as_posix()
    
    if args.tile is None:
        sir.upscale(args.init_img, outpath=outpath)
    else:
        #sir.upscale_tiled(args.init_img, args.tile, args.tile_overlap, outpath=outpath)
        #sir.upscale_using_patches(args.init_img, slice_dim = args.tile, slice_overlap= args.tile_overlap, outpath=outpath, keep_pbar = False)
        sir.upscale_patchwise(args.init_img, slice_dim = args.tile, slice_overlap= args.tile_overlap, outpath=outpath,)
    



if __name__ == '__main__':
    main()
