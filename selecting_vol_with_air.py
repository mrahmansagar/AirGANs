# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:50:00 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Seperating rois with and without air bubbles
"""

import os 
import numpy as np

from tkinter import Tcl 
from PIL import Image 

from tqdm import tqdm
from pathlib import Path
import shutil


from scipy import ndimage as nd
from skimage.measure import label

# Using porespy 
import porespy as ps
ps.visualization.set_mpl_style()
np.random.seed(1)

root_dir = 'E:\\Data\\sam_data\\new\\'

samples = os.listdir(root_dir)

for sample in samples:
    data_dir = os.path.join(root_dir, sample)
    with_air_dir = Path(os.path.join(data_dir, 'with_air'))
    without_air_dir = Path(os.path.join(data_dir, 'withour_air'))

    if not with_air_dir.exists():
        with_air_dir.mkdir()

    if not without_air_dir.exists():
        without_air_dir.mkdir()
    
    roi_path = os.path.join(data_dir, 'roi')
    rois = os.listdir(roi_path)
    
    for roi in tqdm(rois):
        try:
    #         print('processing....', os.path.join(roi_path, roi))

            tiffs = os.listdir(os.path.join(roi_path, roi))
            slices = Tcl().call('lsort', '-dict', tiffs)
            vol = np.empty(shape=(300, 300, 300), dtype=np.uint8)
            # Temporary list to hold blank slices
            blank_slices = []

            for i, fname in enumerate(slices):
                im = Image.open(os.path.join(roi_path, roi, fname))
                imarray = np.array(im)

                if np.all(imarray == 0):
                    blank_slices.append(imarray)
                else:
                    vol[i - len(blank_slices), :, :] = imarray
        #     interactive_visualize(vol)
            th_vol = vol < 1
            th_vol = nd.binary_opening(th_vol, np.ones((9,9,9)))
            th_vol = nd.binary_dilation(th_vol, np.ones((12,12,12)))
        #         interactive_visualize(th_vol)
            markers = label(th_vol)
            props = ps.metrics.regionprops_3D(markers)
            filtered_regions = [prop for prop in props if prop.area >= 10000]

            src_path = os.path.join(roi_path, roi)

            if len(filtered_regions) == 0:
                des_path = Path(os.path.join(without_air_dir, roi))
            else:
                des_path = Path(os.path.join(with_air_dir, roi))

            if not des_path.exists():
                des_path.mkdir()

            for afile in os.listdir(src_path):
                shutil.copy(os.path.join(src_path, afile), des_path)    


        except:
            print('skipping....', os.path.join(roi_path, roi))
            pass