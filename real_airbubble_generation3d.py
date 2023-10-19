# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:17:26 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Using threshold to capture only air then morphological operations to
create a mask that represents the airbubbles. then copy the pixel to a
non airbubble sample to make a pair 
"""



import os 
os.sys.path.insert(0, 'E:\\dev\\packages')

import numpy as np 
from tkinter import Tcl
from PIL import Image

from tqdm import tqdm

from scipy import ndimage as nd
from skimage.measure import label

import porespy as ps
ps.visualization.set_mpl_style()
np.random.seed(1)


from proUtils import utils


root_dir = 'E:\\Data\\air_bubbles\\3d\\'

with_air_dir = os.path.join(root_dir, 'with_air')
without_air_dir = os.path.join(root_dir, 'without_air')

extracted_air_dir = os.path.join(root_dir, 'extracted_air')

vols_without_air = os.listdir(without_air_dir) 
vols_with_air = os.listdir(with_air_dir)

for i, aVol in enumerate(tqdm(vols_with_air)):
    vol_w_air = []
    slices_w_air = os.listdir(os.path.join(with_air_dir, aVol))
    slices_w_air = Tcl().call('lsort', '-dict', slices_w_air)
    for i, aslice in enumerate(slices_w_air):
        imarray = Image.open(os.path.join(with_air_dir, aVol, aslice))
        imarray = np.array(imarray)
        vol_w_air.append(imarray)
    vol_w_air = np.array(vol_w_air)
    th_vol_w_air = vol_w_air < 1
    th_vol = nd.binary_opening(th_vol_w_air, np.ones((9,9,9)))
    th_vol = nd.binary_dilation(th_vol, np.ones((18,18,18)))
    markers = label(th_vol)
    props = ps.metrics.regionprops_3D(markers)
    filtered_regions = [prop for prop in props if prop.area >= 10000]
    coordinates = [tuple(coord) for region in filtered_regions for coord in region.coords]
    
    vol_wo_air = []
    slices_wo_air = os.listdir(os.path.join(without_air_dir, aVol))
    slices_wo_air = Tcl().call('lsort', '-dict', slices_wo_air)
    for i, aslice in enumerate(slices_wo_air):
        imarray = Image.open(os.path.join(without_air_dir, aVol, aslice))
        imarray = np.array(imarray)
        vol_wo_air.append(imarray)
    vol_wo_air = np.array(vol_wo_air)
    air_bubble_copied_vol = np.copy(vol_wo_air)
    for coord in coordinates:
        row, col, depth = coord
        air_bubble_copied_vol[row, col, depth] = vol_w_air[row, col, depth]
        
    folderName = os.path.join(extracted_air_dir, aVol)
    utils.save_vol_as_slices(air_bubble_copied_vol, folderName)