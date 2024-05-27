# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:28:38 2024

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

applies airgan to all the cubes in a folder 
"""

import os
os.sys.path.insert(0, 'E:\\dev\\packages')
from proUtils import utils

import numpy as np
from skimage import img_as_ubyte

import methods

from keras.models import load_model

from tqdm import tqdm


# data dir
data_dir = "E:\\Data\\air_bubbles\\3d\\with_air\\"

# where all the cubes are
sample_dir = os.path.join(data_dir, 'test')
all_cubes = os.listdir(sample_dir)

# save dir 
save_dir = os.path.join(data_dir, 'air_removed_test')
if not os.path.exists(save_dir):
            os.makedirs(save_dir)

# model to be used 
model = load_model('E:\\projects\\AirGANs\\air3d_202401031030\\src2tar_air3d_after_179999.h5')


for cube in tqdm(all_cubes):
    cube_path = os.path.join(sample_dir, cube)
    patch = utils.load_roi(cube_path, check_blank=False)

    patch_inShape = ((patch-127.5)/127.5).astype(np.float32)
    air_rem_patch = methods.apply_model_to_patch(model, patch_inShape)
    air_rem_patch = (air_rem_patch+1)/2.0
    try:
        air_rem_patch = img_as_ubyte(air_rem_patch)
    except:
        air_rem_patch = np.clip(air_rem_patch, -1, 1)
        air_rem_patch = img_as_ubyte(air_rem_patch)
    
    fName = os.path.join(save_dir, cube)
    utils.save_vol_as_slices(air_rem_patch, fName)
    
        
        










