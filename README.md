# Min-3-Flow

Min-3-Flow is a 3-stage text to image generation pipeline. Its structure is modeled after [dalle-flow](https://github.com/jina-ai/dalle-flow/) while its design principles are inspired by [min-dalle](https://github.com/kuprel/min-dalle). It forgos the the client-server architecture in favor of modularity and configurabilty. 

## Install

pip install -e.

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
