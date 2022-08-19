import string
import unicodedata
import argparse
#from operator import mod
import os
from PIL import Image
import torch

from min3flow.min_dalle import MinDalle, MinDalleExt



FILESAFECHRS=frozenset(string.ascii_letters+string.digits+'-_.')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_variant', type=str, default='mega', choices=['mini', 'mega', 'mega_full', 'mega_beta'])
    parser.add_argument('-d','--dtype', type=str, default='f32', choices=['f32', 'f16', 'bf16'])
    parser.add_argument('--text', type=str, default='Dali painting of WALL E')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--grid-size', type=int, default=1)
    parser.add_argument('--image-path', type=str, default='output/0_generate/')
    parser.add_argument('--models-root', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=256)
    parser.add_argument('-sf','--supercondition_factor', type=int, default=32)
    return parser


def ascii_from_image(image: Image.Image, size: int) -> str:
    rgb_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in rgb_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)


def to_ascii(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode()

def fn_safe(text):
    '''Replace all non-filename-safe characters with underscore.'''
    #return ''.join([c if c in FILESAFECHRS else f'-ORD{ord(c)}-' for c in text.replace(' ','_')])
    return ''.join([c if c in FILESAFECHRS else '_' for c in to_ascii(text).replace(' ','_') ])


def save_image(image: Image.Image, path: str, text: str = None):
    if os.path.isdir(path):
        out_name = (fn_safe(text) if text is not None else 'generated')+'.png'
        path = os.path.join(path, out_name)
    elif not path.endswith('.png'):
        path += '.png'
    print("saving image to", path)
    image.save(path)
    return image

@torch.inference_mode()
def generate_image(
    model_variant: str,
    dtype: torch.dtype,
    text: str,
    seed: int,
    grid_size: int,
    top_k: int,
    supercondition_factor: int,
    image_path: str,
    models_root: str,
    #row_count: int,
):
    model = MinDalleExt(
        model_variant=model_variant, 
        dtype=dtype,
        models_root=models_root,
        is_reusable=False,
        is_verbose=True
    )
    
    image = model.generate_image(
        text, 
        seed, 
        grid_size, 
        temperature=1.0,
        top_k = top_k,
        supercondition_factor=supercondition_factor,
        is_verbose=True
    )

    save_image(image, image_path, text)
    print(ascii_from_image(image, size=128))
    
    

if __name__ == '__main__':
    #torch.set_float32_matmul_precision("highest")
    
    #torch.set_float32_matmul_precision("high")
    #torch.set_float32_matmul_precision("medium")
    #torch.backends.cudnn.allow_tf32 = True
    #44s for torch 1.12 defaults
    #41s for torch matmul_precision=medium + cudnn.allow_tf32=True
    args = get_parser().parse_args()
    print(args)
    dtype_map={'f32':torch.float32, 'f16':torch.float16, 'bf16':torch.bfloat16}
    generate_image(
        model_variant=args.model_variant,
        dtype=dtype_map[args.dtype],
        text=args.text,
        seed=args.seed,
        grid_size=args.grid_size,
        top_k=args.top_k,
        supercondition_factor=args.supercondition_factor,
        image_path=args.image_path,
        models_root=args.models_root,
        #row_count=args.row_count
    )