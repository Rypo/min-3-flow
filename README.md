# Min-3-Flow

Min-3-Flow is a 3-stage text-to-image generation pipeline. Its structure is modeled after [dalle-flow](https://github.com/jina-ai/dalle-flow/) while its design principles are inspired by [min-dalle](https://github.com/kuprel/min-dalle). It forgos the the client-server architecture in favor of modularity and configurabilty. 

## Install
```sh
git clone https://github.com/Rypo/min-3-flow.git
cd min-3-flow
conda create -n min3flow mamba # mamba recommended, not required. Optionally, replace 'mamba' with 'conda'
conda activate min3flow

mamba install jupyter notebook
mamba install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge

 # (Glid3XL requirements)
mamba install -c conda-forge transformers einops

# CLIP requirements
mamba install ftfy regex
pip install git+https://github.com/openai/CLIP.git

# SwinIR requirements
pip install timm

# ldm requirements
pip install pytorch-lightning # pip package typically more up to date than conda-forge
mamba install -c conda-forge omegaconf

# order is important, taming-transforms install before latent-diffusion
git clone https://github.com/CompVis/latent-diffusion.git && cd latent-diffusion
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e .

cd ..
# install min3flow (Not strictly required unless using scripts)
pip install -e. 
```

## Basic Usage

### 1. Generate an initial set of images from a text prompt
```py
from min3flow.min3flow import Min3Flow
mflw = Min3Flow(global_seed=42)

prompt = 'Dali painting of a glider in infrared'
grid_size = 4 # Will create a grid of (4,4) images. Reduce this if running into OOM errors.
image = mflw.generate(prompt, grid_size=grid_size)
mflw.show_grid(image)
```

### 2. Select your favorite(s) and refine with diffusion
```py
favs = 5 # Pick a few you like: favs=[0,2,4,8] or pass them all: favs=None
img_diff = mflw.diffuse(image, grid_idx=favs)
mflw.show_grid(img_diff)
```  

### 3. Upsample the images to 1024x1024
```py
img_up = mflw.upscale(img_diff)
mflw.show_grid(img_up)
```

## Min-3-Flow vs DALL·E Flow
At a high level, both packages do the same thing in a similar way. 
1. Generate an image from a text prompt using DALL·E-Mega weights
2. Diffusion refinement with GLID-3-XL
3. Upsample the 256x256 output images to 1024x1024 with SwinIR

A few thousand feet lower and you'll note that:
1. Min-3-Flow uses [min-dalle](https://github.com/kuprel/min-dalle) instead of [dalle-mini](https://github.com/borisdayma/dalle-mini) for text-to-image generation. This means the entire pipeline is based on PyTorch, i.e. no flax dependency. 
2. The diffusion library, [GLID-3-XL](https://github.com/Jack000/glid-3-xl) has been **heavily** refactored and extented. It now functions as standalone module, not just a command line script and supports additional [ldm-finetune](https://github.com/LAION-AI/ldm-finetune) weights.
3. Similar to the Glid3XL treatment, [SwinIR](https://github.com/JingyunLiang/SwinIR) is no-longer commandline bound. (Kudos to [SwinIR_wrapper](https://github.com/Lin-Sinorodin/SwinIR_wrapper/) for the inspiration)


## TODO

- [ ] Min-Dalle
  - [ ] Add optional dependencies for Extended model support
- [ ] Glid3XL 
  - [ ] Further reduce codebase
    - [ ] Clean and optimize guided_diffusion or replace functionality with existing libraries
  - [ ] Reintroduce masking and autoedit capablities
    - [x] Add support for inpaint weight and LAION variants (ongo, erlich, puck)
    - [ ] Clean and add mask generation GUI
    - [ ] Clean and add autoedit functionality
  - [ ] Modify clip guided conditioning function to allow batch sizes greater than 1
  - [ ] Modify to allow direct weight path without requiring a models_roots
- [ ] SwinIR
  - [ ] Test if non-SR tasks are functional and useful, if not remove
- [ ] General
  - [x] Standardize all generation outputs as tensors, convert to Image.Image in Min3Flow class
  - [ ] Update documentation for new weight path scheme
  - [ ] environment.yml and/or requirements.txt
  - [ ] Google Colab notebook demo
    - [ ] python 3.7.3 compatibility

---
## Q/A 
<details>
<summary>How to pronounce Min-3-Flow?</summary>
  I'm partial to "min-ee-flow" but "min-three-flow" is fair game. 
  
  My intention with the l337 style "E" was to sound less like some sort of Minecraft auto clicker (cf. MineFlow). 
</details>
<details>
<summary>Why reinvent the wheel?</summary>
  
  1. I found the client-server paradigm to be somewhat limiting in terms of parameter tuning. There are a lot more knobs that can be tuned than are allowed in DALL·E Flow. In persuit of this tunability, I ended up adding more functionality than existed with any of the base packages alone, 

  2. I couldn't get DocArray to install on my machine. So, why spend an hour debugging when you can spend a month building your own!
</details>