# Min-3-Flow

Min-3-Flow is a 3-stage text-to-image generation pipeline. Its structure is modeled after [dalle-flow](https://github.com/jina-ai/dalle-flow/) but forgos the the client-server architecture in favor of modularity and configurabilty. The underlying packages have all been stripped down and optimized for inference, taking design inspiration from [min-dalle](https://github.com/kuprel/min-dalle).


## Min-3-Flow vs DALL·E Flow
At a high level, both packages do the same thing in a similar way. 
1. Generate an image from a text prompt using DALL·E-Mega weights
2. Diffusion refinement with GLID-3-XL
3. Upsample the 256x256 output images to 1024x1024 with SwinIR

A few thousand feet lower and you'll note that:
1. Min-3-Flow uses [min-dalle](https://github.com/kuprel/min-dalle) instead of [dalle-mini](https://github.com/borisdayma/dalle-mini) for text-to-image generation. This means the pipeline is **entirely PyTorch based**, i.e. no flax dependency. 
2. The diffusion library, [GLID-3-XL](https://github.com/Jack000/glid-3-xl) has been **heavily** refactored and extented. It now functions as standalone module, not just a command line script and supports additional [ldm-finetune](https://github.com/LAION-AI/ldm-finetune) weights.
3. Similar to the Glid3XL treatment, [SwinIR](https://github.com/JingyunLiang/SwinIR) is no-longer commandline bound. (Kudos to [SwinIR_wrapper](https://github.com/Lin-Sinorodin/SwinIR_wrapper/) for the inspiration)

## Install
### Conda/Mamba
```sh
git clone https://github.com/Rypo/min-3-flow.git
cd min-3-flow

conda create -n min3flow matplotlib jupyter notebook
conda activate min3flow

conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge

# (Glid3XL requirements)
conda install -c conda-forge transformers einops

# CLIP requirements
conda install ftfy regex
pip install git+https://github.com/openai/CLIP.git

# SwinIR requirements
pip install timm

# ldm requirements
conda install -c conda-forge omegaconf
pip install pytorch-lightning # pip package typically more up to date than conda-forge

# order is important, taming-transforms install before latent-diffusion
git clone https://github.com/CompVis/latent-diffusion.git && cd latent-diffusion
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e .

cd ..
# install min3flow
pip install -e. 
```
### Pip
```sh
pip install matplotlib jupyter notebook
pip install torch torchvision

# (Glid3XL requirements)
pip install transformers==4.3.1 einops

# CLIP requirements
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
# SwinIR requirements
pip install timm
# ldm requirements
pip install pytorch-lightning omegaconf 


git clone https://github.com/Rypo/min-3-flow.git && cd min-3-flow
git clone https://github.com/CompVis/latent-diffusion.git && cd latent-diffusion

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

# install latent-diffusion
pip install -e .

cd ..
# install min3flow
pip install -e . 
```
May need to add the following lines to the top of notebooks/scripts if `ldm` is not found
```py
import sys
sys.path.append('latent-diffusion')
sys.path.append('latent-diffusion/src/taming-transformers/')
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
grid_idx = [0,2,3] #  or pass them all: grid_idx=None
img_diff = mflw.diffuse(prompt, image[grid_idx])
mflw.show_grid(img_diff)
```  

### 3. Select and upsample the images to 1024x1024
```py
grid_idx = [3,4,13,15] #  or pass them all: grid_idx=None
img_up = mflw.upscale(img_diff[grid_idx])
mflw.show_grid(img_up, plot_index=False)
```

## Results Gallery
<img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_realistic_photo_of_a_muddy_dog.png?raw=True" width="32%" alt="a realistic photo of a muddy dog" title="a realistic photo of a muddy dog"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/A_scientist_comparing_apples_and_oranges,_by_Norman_Rockwell.png?raw=True" width="32%" alt="A scientist comparing apples and oranges, by Norman Rockwell" title="A scientist comparing apples and oranges, by Norman Rockwell"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/an_oil_painting_portrait_of_the_regal_Burger_King_posing_with_a_Whopper.png?raw=True" width="32%" alt="an oil painting portrait of the regal Burger King posing with a Whopper" title="an oil painting portrait of the regal Burger King posing with a Whopper"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/Eternal_clock_powered_by_a_human_cranium,_artstation.png?raw=True" width="32%" alt="Eternal clock powered by a human cranium, artstation" title="Eternal clock powered by a human cranium, artstation"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/another_planet_amazing_landscape.png?raw=True" width="32%" alt="another planet amazing landscape" title="another planet amazing landscape"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/The_Decline_and_Fall_of_the_Roman_Empire_board_game_kickstarter.png?raw=True" width="32%" alt="The Decline and Fall of the Roman Empire board game kickstarter" title="The Decline and Fall of the Roman Empire board game kickstarter"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/A_raccoon_astronaut_with_the_cosmos_reflecting_on_the_glass_of_his_helmet_dreaming_of_the_stars,_digital_art.png?raw=True" width="32%" alt="A raccoon astronaut with the cosmos reflecting on the glass of his helmet dreaming of the stars, digital art" title="A raccoon astronaut with the cosmos reflecting on the glass of his helmet dreaming of the stars, digital art"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/A_photograph_of_an_apple_that_is_a_disco_ball,_85_mm_lens,_studio_lighting.png?raw=True" width="32%" alt="A photograph of an apple that is a disco ball, 85 mm lens, studio lighting" title="A photograph of an apple that is a disco ball, 85 mm lens, studio lighting"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_cubism_painting_Donald_trump_happy_cyberpunk.png?raw=True" width="32%" alt="a cubism painting Donald trump happy cyberpunk" title="a cubism painting Donald trump happy cyberpunk"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/oil_painting_of_a_hamster_drinking_tea_outside.png?raw=True" width="32%" alt="oil painting of a hamster drinking tea outside" title="oil painting of a hamster drinking tea outside"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/Colossus_of_Rhodes_by_Max_Ernst.png?raw=True" width="32%" alt="Colossus of Rhodes by Max Ernst" title="Colossus of Rhodes by Max Ernst"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/landscape_with_great_castle_in_middle_of_forest.png?raw=True" width="32%" alt="landscape with great castle in middle of forest" title="landscape with great castle in middle of forest"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/an_medieval_oil_painting_of_Kanye_west_feels_satisfied_while_playing_chess_in_the_style_of_Expressionism.png?raw=True" width="32%" alt="an medieval oil painting of Kanye west feels satisfied while playing chess in the style of Expressionism" title="an medieval oil painting of Kanye west feels satisfied while playing chess in the style of Expressionism"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/An_oil_pastel_painting_of_an_annoyed_cat_in_a_spaceship.png?raw=True" width="32%" alt="An oil pastel painting of an annoyed cat in a spaceship" title="An oil pastel painting of an annoyed cat in a spaceship"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/dinosaurs_at_the_brink_of_a_nuclear_disaster.png?raw=True" width="32%" alt="dinosaurs at the brink of a nuclear disaster" title="dinosaurs at the brink of a nuclear disaster"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/fantasy_landscape_with_medieval_city.png?raw=True" width="32%" alt="fantasy landscape with medieval city" title="fantasy landscape with medieval city"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/GPU_chip_in_the_form_of_an_avocado,_digital_art.png?raw=True" width="32%" alt="GPU chip in the form of an avocado, digital art" title="GPU chip in the form of an avocado, digital art"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_giant_rubber_duck_in_the_ocean.png?raw=True" width="32%" alt="a giant rubber duck in the ocean" title="a giant rubber duck in the ocean"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/Paddington_bear_as_austrian_emperor_in_antique_black_&_white_photography.png?raw=True" width="32%" alt="Paddington bear as austrian emperor in antique black & white photography" title="Paddington bear as austrian emperor in antique black & white photography"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_rainy_night_with_a_superhero_perched_above_a_city,_in_the_style_of_a_comic_book.png?raw=True" width="32%" alt="a rainy night with a superhero perched above a city, in the style of a comic book" title="a rainy night with a superhero perched above a city, in the style of a comic book"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/A_synthwave_style_sunset_above_the_reflecting_water_of_the_sea,_digital_art.png?raw=True" width="32%" alt="A synthwave style sunset above the reflecting water of the sea, digital art" title="A synthwave style sunset above the reflecting water of the sea, digital art"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/an_oil_painting_of_ocean_beach_front_in_the_style_of_Titian.png?raw=True" width="32%" alt="an oil painting of ocean beach front in the style of Titian" title="an oil painting of ocean beach front in the style of Titian"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/an_oil_painting_of_Klingon_general_in_the_style_of_Rubens.png?raw=True" width="32%" alt="an oil painting of Klingon general in the style of Rubens" title="an oil painting of Klingon general in the style of Rubens"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/city,_top_view,_cyberpunk,_digital_realistic_art.png?raw=True" width="32%" alt="city, top view, cyberpunk, digital realistic art" title="city, top view, cyberpunk, digital realistic art"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/an_oil_painting_of_a_medieval_cyborg_automaton_made_of_magic_parts_and_old_steampunk_mechanics.png?raw=True" width="32%" alt="an oil painting of a medieval cyborg automaton made of magic parts and old steampunk mechanics" title="an oil painting of a medieval cyborg automaton made of magic parts and old steampunk mechanics"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_watercolour_painting_of_a_top_view_of_a_pirate_ship_sailing_on_the_clouds.png?raw=True" width="32%" alt="a watercolour painting of a top view of a pirate ship sailing on the clouds" title="a watercolour painting of a top view of a pirate ship sailing on the clouds"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_knight_made_of_beautiful_flowers_and_fruits_by_Rachel_ruysch_in_the_style_of_Syd_brak.png?raw=True" width="32%" alt="a knight made of beautiful flowers and fruits by Rachel ruysch in the style of Syd brak" title="a knight made of beautiful flowers and fruits by Rachel ruysch in the style of Syd brak"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_3D_render_of_a_rainbow_colored_hot_air_balloon_flying_above_a_reflective_lake.png?raw=True" width="32%" alt="a 3D render of a rainbow colored hot air balloon flying above a reflective lake" title="a 3D render of a rainbow colored hot air balloon flying above a reflective lake"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_teddy_bear_on_a_skateboard_in_Times_Square_.png?raw=True" width="32%" alt="a teddy bear on a skateboard in Times Square " title="a teddy bear on a skateboard in Times Square "><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/cozy_bedroom_at_night.png?raw=True" width="32%" alt="cozy bedroom at night" title="cozy bedroom at night"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/an_oil_painting_of_monkey_using_computer.png?raw=True" width="32%" alt="an oil painting of monkey using computer" title="an oil painting of monkey using computer"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/the_diagram_of_a_search_machine_invented_by_Leonardo_da_Vinci.png?raw=True" width="32%" alt="the diagram of a search machine invented by Leonardo da Vinci" title="the diagram of a search machine invented by Leonardo da Vinci"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/A_stained_glass_window_of_toucans_in_outer_space.png?raw=True" width="32%" alt="A stained glass window of toucans in outer space" title="A stained glass window of toucans in outer space"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_campfire_in_the_woods_at_night_with_the_milky-way_galaxy_in_the_sky.png?raw=True" width="32%" alt="a campfire in the woods at night with the milky-way galaxy in the sky" title="a campfire in the woods at night with the milky-way galaxy in the sky"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/Bionic_killer_robot_made_of_AI_scarab_beetles.png?raw=True" width="32%" alt="Bionic killer robot made of AI scarab beetles" title="Bionic killer robot made of AI scarab beetles"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/The_Hanging_Gardens_of_Babylon_in_the_middle_of_a_city,_in_the_style_of_Dalí.png?raw=True" width="32%" alt="The Hanging Gardens of Babylon in the middle of a city, in the style of Dalí" title="The Hanging Gardens of Babylon in the middle of a city, in the style of Dalí"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/painting_oil_of_Izhevsk.png?raw=True" width="32%" alt="painting oil of Izhevsk" title="painting oil of Izhevsk"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_hyper_realistic_photo_of_a_marshmallow_office_chair.png?raw=True" width="32%" alt="a hyper realistic photo of a marshmallow office chair" title="a hyper realistic photo of a marshmallow office chair"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/fantasy_landscape_with_city.png?raw=True" width="32%" alt="fantasy landscape with city" title="fantasy landscape with city"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/ocean_beach_front_view_in_Van_Gogh_style.png?raw=True" width="32%" alt="ocean beach front view in Van Gogh style" title="ocean beach front view in Van Gogh style"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/An_oil_painting_of_a_family_reunited_inside_of_an_airport,_digital_art.png?raw=True" width="32%" alt="An oil painting of a family reunited inside of an airport, digital art" title="An oil painting of a family reunited inside of an airport, digital art"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/antique_photo_of_a_knight_riding_a_T-Rex.png?raw=True" width="32%" alt="antique photo of a knight riding a T-Rex" title="antique photo of a knight riding a T-Rex"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_top_view_of_a_pirate_ship_sailing_on_the_clouds.png?raw=True" width="32%" alt="a top view of a pirate ship sailing on the clouds" title="a top view of a pirate ship sailing on the clouds"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/an_oil_painting_of_a_humanoid_robot_playing_chess_in_the_style_of_Matisse.png?raw=True" width="32%" alt="an oil painting of a humanoid robot playing chess in the style of Matisse" title="an oil painting of a humanoid robot playing chess in the style of Matisse"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_cubism_painting_of_a_cat_dressed_as_French_emperor_Napoleon.png?raw=True" width="32%" alt="a cubism painting of a cat dressed as French emperor Napoleon" title="a cubism painting of a cat dressed as French emperor Napoleon"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/a_husky_dog_wearing_a_hat_with_sunglasses.png?raw=True" width="32%" alt="a husky dog wearing a hat with sunglasses" title="a husky dog wearing a hat with sunglasses"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/A_mystical_castle_appears_between_the_clouds_in_the_style_of_Vincent_di_Fate.png?raw=True" width="32%" alt="A mystical castle appears between the clouds in the style of Vincent di Fate" title="A mystical castle appears between the clouds in the style of Vincent di Fate"><img src="https://github.com/Rypo/min-3-flow/blob/gallery/.github/gallery/golden_gucci_airpods_realistic_photo.png?raw=True" width="32%" alt="golden gucci airpods realistic photo" title="golden gucci airpods realistic photo">

### 🍒Picking Procedure

For each prompt, a batch of 16 images was generated with 7 different configuration (A-G below). The same global seed (42) was used across all prompts and configurations. 

* A,B,C are images generated with Glid3XL alone (no initial image) and correspond to 3 different diffusion weights (finetune.pt, inpaint.pt, and ongo.pt). 

* D is images generated by creating an initial image with MinDalle(dtype=float32, supercondition factor=32) and pass that image along with the prompt to Glid3XL(classifier guidance=5.0, steps=200, skip rate=0.5)
  
* E,F,G are images generated with MinDalle alone using float16+supercondition factor 16, float32+super conditionfactor 16, float32+supercondition factor 32
  
```
Before upsampling (1+ per prompt) 
[(A, 7), (B, 4), (C, 8), (D, 25), (E, 16), (F, 13), (G, 24)]

After upsampling (1 per prompt) 
[(A, 4), (B, 3), (C, 5), (D, 11), (E, 11), (F, 4), (G, 10)]

A: 'glid3xl-cg5-finetune-200step-0.0skip'
B: 'glid3xl-cg5-inpaint-200step-0.0skip'
C: 'glid3xl-cg5-ongo-200step-0.0skip'
D: 'mindalle-f32-sf32 -> glid3xl-cg5-inpaint-200step-0.5skip'
E: 'mindalle-f16-sf16'
F: 'mindalle-f32-sf16'
G: 'mindalle-f32-sf32'
```
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
  - [ ] Allow batch sizes greater than 1 in clip guidance function
  - [x] Allow direct weight path without requiring a models_roots
  - [ ] Option to generate images from scratch (i.e. not to pass dalle output as diffusion input)
- [ ] SwinIR
  - [ ] Test if non-SR tasks are functional and useful, if not remove
- [ ] General
  - [x] Standardize all generation outputs as tensors, convert to Image.Image in Min3Flow class
  - [ ] Update documentation for new weight path scheme
  - [ ] environment.yml and/or requirements.txt
  - [ ] Google Colab notebook demo
    - [ ] python 3.7.3 compatibility
  - [ ] Add VRAM usage estimates

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