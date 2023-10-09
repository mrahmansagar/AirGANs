# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:01:00 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Data loading/pre-processing and using GANs packgae training a pix2pix network 
using 3D data.
"""


import os 
import sys
sys.path.insert(0, 'E:\\dev\\packages')

import numpy as np
from tqdm import tqdm

from tkinter import Tcl
from keras.utils import load_img, img_to_array

from GANs.pix2pix import models

data_dir = 'E:\\Data\\air_bubbles\\3d'
src_vol_dir = os.path.join(data_dir, 'extracted_air', 'train')
tar_vol_dir = os.path.join(data_dir, 'without_air', 'train')

# load images/data in shape for training data generation 
def load_volumes_in_shape(vol_dir, **kwargs):
    """
    Load images from a directory and convert them to a NumPy array with specified shape.

    Args:
        img_dir (str): Path to the directory containing the images.
        **kwargs: Additional keyword arguments to be passed to `load_img()` function.

    Returns:
        numpy.ndarray: Array containing the loaded images in the specified shape.
    """
    vol_data_in_shape = []
    list_of_volumes = Tcl().call('lsort', '-dict', os.listdir(vol_dir))

    for v in tqdm(list_of_volumes):
        vol = []
        list_of_images = Tcl().call('lsort', '-dict', os.listdir(os.path.join(vol_dir, v)))
        for im in list_of_images:
            im = load_img(os.path.join(vol_dir, v, im), **kwargs)
            imarray = img_to_array(im)#, dtype='uint8')
            imarray = (imarray- 127.5) / 127.5
            vol.append(imarray)
        vol = np.asarray(vol)
        vol_data_in_shape.append(vol)
    
    vol_data_in_shape = np.asarray(vol_data_in_shape)    
    print('Loaded', vol_data_in_shape.shape, 'volumes')
    
    return vol_data_in_shape

src_vol = load_volumes_in_shape(src_vol_dir, color_mode='grayscale')
tar_vol = load_volumes_in_shape(tar_vol_dir, color_mode='grayscale')

# define image shape
vol_shape = src_vol.shape[1:]
# define the models
disc = models.build_discriminator(vol_shape)
gene = models.build_generator(vol_shape)
# define the composite model
p2p_model = models.build_pix2pix(gene, disc, vol_shape)
models.train_pix2pix(gene, disc, p2p_model, src_vol, tar_vol, epochs=100, summary_interval=10, name='air3d/src2tar_air3d')


