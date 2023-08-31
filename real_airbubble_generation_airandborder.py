# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:36:09 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Using threshold to capture both air and border then morphological operations to
create a mask that represents the airbubbles.
"""

import os 
from tkinter import Tcl
import numpy as np
import matplotlib.pyplot as plt 
import tifffile

from PIL import Image

from tqdm import tqdm

from scipy import ndimage as nd
from skimage.measure import label

# Using porespy 
import porespy as ps
import scipy.ndimage as spim
ps.visualization.set_mpl_style()
np.random.seed(1)

data_dir = 'E:\\Data\\air_bubbles'

w_real_air_dir = os.path.join(data_dir, 'with_air_real', 'test')
wo_air_dir = os.path.join(data_dir, 'without_air', 'test')

w_gen_air_dir = os.path.join(data_dir, 'extracted_real_air', 'test2')

list_realAir_slices = Tcl().call('lsort', '-dict', os.listdir(w_real_air_dir))

for i, aSlice in enumerate(tqdm(list_realAir_slices)):
    #iterating through one image at a time
    img = Image.open(os.path.join(w_real_air_dir, aSlice))
    img = np.array(img)
    #saving the image to make a pair
#     file_name = f'slice_{i}.tif'
    file_name = aSlice
#     tifffile.imsave(os.path.join(wo_air_dir, file_name), img)
    
    # putting threshold in high and low range to capture airbubbles and white borders
    th_img_w_air = img < 1
    th_img_border = img > 150
    
    #adding both mask to create airbubbles with border mask
    th_img = th_img_w_air + th_img_border
    #morphological operation to close the gap between border and airbubbles
    th_img = nd.binary_closing(th_img, np.ones((3,3)))
    th_img = nd.binary_fill_holes(th_img, np.ones((3,3)))
    #adding extra pixel around white borders to capture blured borders
    th_img = nd.binary_dilation(th_img, np.ones((9,9)))
    # creating labels for regionsprops
    markers = label(th_img)
    #getting the properties of each of the labels 
    props = ps.metrics.regionprops_3D(markers)
    #taking regions with area size greater than 1000
    filtered_regions = [prop for prop in props if prop.area >= 1000]
    # extracting the coordinates of the selected regions
    coordinates = [tuple(coord) for region in filtered_regions for coord in region.coords]
    
    img_wo_air = Image.open(os.path.join(wo_air_dir, aSlice))
    img_wo_air = np.array(img_wo_air)
    img_w_gen_air = np.copy(img_wo_air)
    
    for coord in coordinates:
        row, col = coord
        img_w_gen_air[row, col] = img[row, col]
    
    # saving the imgae with airbubbles into the directory    
    tifffile.imsave(os.path.join(w_gen_air_dir, file_name), img_w_gen_air)
    
    