# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:35:19 2024

@author: mrahm

This script calculates the ratio of tissue to volume and writes to file or 
creates new file to write the value to 
"""


import os 
os.sys.path.insert(0, 'E:\\dev\\packages')

import numpy as np
from scipy import ndimage as nd
import json

from tqdm import tqdm

from proUtils import utils


# data dir
data_dir = "E:\\Data\\air_bubbles\\3d\\with_air\\"


# folder where rois are and where to save the results(folders are relative to the data_dir)
roi_folder = 'air_removed_test'
# where all the cubes are
sample_dir = os.path.join(data_dir, roi_folder)

result_folder = 'air_removed_test_porosity'

# save dir 
save_dir = os.path.join(data_dir, result_folder)
if not os.path.exists(save_dir):
            os.makedirs(save_dir)

# threshold for binarization 
bin_th = 55
# threshold for distance transform to create seed for segmentation 
dt_th = 10

file_sufix = f'_dth{dt_th}.json'

all_cubes = os.listdir(sample_dir)

for cube in tqdm(all_cubes):
    cube_path = os.path.join(sample_dir, cube)
    vol = utils.load_roi(cube_path, check_blank=False)
    vol = nd.median_filter(vol, size=2)
    th_vol = vol < bin_th
    th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
    th_vol = nd.binary_closing(th_vol, np.ones((2,2,2)))
    
    # Count the number of pore voxels (True values in th_vol)
    num_pore_voxels = np.sum(th_vol)
    
    # Calculate the total number of voxels
    total_voxels = np.prod(th_vol.shape)
    
    # Calculate porosity
    porosity = num_pore_voxels / total_voxels
    
    fileName = f'{cube}_bth{bin_th}{file_sufix}'
    
    filePath = os.path.join(data_dir, result_folder, fileName)
    
    # Read existing JSON file or create a new dictionary
    if os.path.exists(filePath):
        with open(filePath, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {}
    
    # Append the porosity value
    if "porosity" in data:
        pass
    else:
        data["porosity"] = [porosity]
    
    jsonString = json.dumps(data)
    jsonFile = open(filePath, 'w')
    jsonFile.write(jsonString)
    jsonFile.close()
    
    
    
    
    
    
    
    
    


