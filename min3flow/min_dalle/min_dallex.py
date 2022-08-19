import os
from pathlib import Path
from PIL import Image

import numpy as np
import torch

#from .text_tokenizer import TextTokenizer
from .models import DalleBartEncoder, DalleBartDecoder, VQGanDetokenizer
from .min_dalle import MinDalle

def freeze(model, set_eval=False):
    for param in model.parameters():
        param.requires_grad_(False)
    if set_eval:
        model = model.eval()
    return model


torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True # Default: False
torch.backends.cuda.matmul.allow_tf32 = True # Default: False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

MIN_DALLE_REPO = 'https://huggingface.co/kuprel/min-dalle/resolve/main/'
IMAGE_TOKEN_COUNT = 256

_WEIGHT_DOWNLOAD_URL = 'https://huggingface.co/kuprel/min-dalle/resolve/main/{}'
_DEFAULT_WEIGHT_ROOT = "~/.cache/min3flow/min_dalle"

class MinDalleExt(MinDalle):
    def __init__(
        self,
        models_root: str = None,
        dtype: torch.dtype = torch.float32,
        device: str = None,
        model_variant: str = 'mega', 
        is_reusable: bool = True,
        is_verbose = True
    ):
        is_mega = (model_variant != 'mini')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.dtype = dtype
        self.is_mega = is_mega
        self.is_reusable = is_reusable
        self.is_verbose = is_verbose

        self.text_token_count = 64
        self.layer_count = 24 if is_mega else 12
        self.attention_head_count = 32 if is_mega else 16
        self.embed_count = 2048 if is_mega else 1024
        self.glu_embed_count = 4096 if is_mega else 2730
        self.text_vocab_count = 50272 if is_mega else 50264
        self.image_vocab_count = 16415 if is_mega else 16384

        
        if models_root is None: 
            models_root = os.path.expanduser(_DEFAULT_WEIGHT_ROOT)

        if self.is_verbose: print("initializing MinDalleExt")
        #model_name = 'dalle_bart_{}'.format('mega' if is_mega else 'mini')
        model_name = 'dalle_bart_{}'.format(model_variant)
        dalle_path = os.path.join(models_root, model_name)
        vqgan_path = os.path.join(models_root, 'vqgan')
        if not os.path.exists(dalle_path): os.makedirs(dalle_path)
        if not os.path.exists(vqgan_path): os.makedirs(vqgan_path)
        self.vocab_path = os.path.join(dalle_path, 'vocab.json')
        self.merges_path = os.path.join(dalle_path, 'merges.txt')
        self.encoder_params_path = os.path.join(dalle_path, 'encoder.pt')
        self.decoder_params_path = os.path.join(dalle_path, 'decoder.pt')
        self.detoker_params_path = os.path.join(vqgan_path, 'detoker.pt')

        self.init_tokenizer()
        if is_reusable:
            self.init_encoder()
            self.init_decoder()
            self.init_detokenizer()


    def init_encoder(self):

        is_downloaded = os.path.exists(self.encoder_params_path)
        if not is_downloaded: self.download_encoder()
        if self.is_verbose: print("initializing DalleBartEncoder")
        self.encoder = DalleBartEncoder(
            layer_count = self.layer_count,
            embed_count = self.embed_count,
            attention_head_count = self.attention_head_count,
            text_vocab_count = self.text_vocab_count,
            text_token_count = self.text_token_count,
            glu_embed_count = self.glu_embed_count,
            device = self.device,
        ).eval()#.to(dtype=self.dtype,)

        self.encoder = freeze(self.encoder).to(device=self.device)#.to(device=self.device, dtype=self.dtype)#, non_blocking=True)
        self.encoder.load_state_dict(torch.load(self.encoder_params_path, map_location=self.device), strict=False)
        self.encoder = self.encoder.to(dtype=self.dtype,)
        


    def init_decoder(self):
        
        is_downloaded = os.path.exists(self.decoder_params_path)
        if not is_downloaded: self.download_decoder()
        if self.is_verbose: print("initializing DalleBartDecoder")
        self.decoder = DalleBartDecoder(
            image_vocab_count = self.image_vocab_count,
            embed_count = self.embed_count,
            attention_head_count = self.attention_head_count,
            glu_embed_count = self.glu_embed_count,
            layer_count = self.layer_count,
            device = self.device,
        ).eval()#.to(dtype=self.dtype,)

        self.decoder = freeze(self.decoder).to(device=self.device)#.to(device=self.device, dtype=self.dtype)#, non_blocking=True)
        self.decoder.load_state_dict(torch.load(self.decoder_params_path, map_location=self.device), strict=False)
        self.decoder = self.decoder.to(dtype=self.dtype,)
        


    def init_detokenizer(self):

        is_downloaded = os.path.exists(self.detoker_params_path)
        if not is_downloaded: self.download_detokenizer()
        if self.is_verbose: print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer().eval()#.to(dtype=self.dtype,)

        self.detokenizer = freeze(self.detokenizer).to(device=self.device)#.to(device=self.device, dtype=self.dtype)#.to(self.device)#, non_blocking=True)
        self.detokenizer.load_state_dict(torch.load(self.detoker_params_path, map_location=self.device), strict=False)
        self.detokenizer = self.detokenizer.to(dtype=self.dtype,)