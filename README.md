# Min-3-Flow

Min-3-Flow is a 3-stage text to image generation pipeline. Its structure is modeled after [dalle-flow](https://github.com/jina-ai/dalle-flow/) while its design principles are inspired by [min-dalle](https://github.com/kuprel/min-dalle). It forgos the the client-server architecture in favor of modularity and configurabilty. 

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
pip install timm --no-deps

# ldm requirements
pip install pytorch-lightning # pip package typically more up to date than conda-forge
mamba install -c conda-forge omegaconf

# order is important, taming-transforms install before latent-diffusion
git clone https://github.com/CompVis/latent-diffusion.git && cd latent-diffusion
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e .

cd ..
# install min3flow
pip install -e. # Not strictly required unless using scripts
```

## TODO

- [ ] Min-Dalle
  - [ ] Add optional dependencies for Extended model support
- [ ] Glid3XL 
  - [ ] Further reduce codebase
    - [ ] Clean and optimize guided_diffusion or replace functionality with existing libraries
  - [ ] Reintroduce masking and autoedit capablities
  - [ ] Modify clip guided conditioning function to allow batch sizes greater than 1
- [ ] SwinIR
  - [ ] Test if non-SR tasks are functional and useful, if not remove
- [ ] General
  - [ ] Standardize all generation outputs as tensors, convert to Image.Image in Min3Flow class
  - [ ] Weight options regarding function purity vs convinence trade-off
  - [ ] Update documentation for new weight path scheme
