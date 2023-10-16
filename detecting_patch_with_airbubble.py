# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:53:45 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import os 
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image 

from scipy import ndimage as nd
from skimage import img_as_ubyte


os.sys.path.insert(0, 'E:\\dev\\packages')
from proUtils import utils

import methods

data_dir = 'F:\\MD_1264_A6_1_Z3.3mm_corr_phrt\\slices'
slice_no = 'slice_0850.tif'

new_air = 0.00009706
new_paraffin = 0.0004179

new_diff = new_paraffin - new_air

old_air = 0.0002605
old_paraffin = 0.0009999

old_diff = old_paraffin - old_air


# Reading all the slices in a Volume 
im = Image.open(os.path.join(data_dir, slice_no))
imarray = np.array(im)
# imarray = (imarray - new_air) / new_diff * old_diff + old_air
imarray = np.clip(imarray, 0.0005, 0.003)
imarray = utils.norm8bit(imarray)

rotated_image, _, lowdist = utils.tilt_correction(imarray, edge_th=0.1, ang_vari=4, plot=True)
_, _, highdist = utils.tilt_correction(imarray, edge_th=250)

croped_imarray = rotated_image[int(abs(lowdist[0])+100):int(abs(highdist[0])), :]
# croped_imarray = croped_imarray[:, 500:3300]
plt.figure(figsize=(12,6))
plt.imshow(croped_imarray, cmap='gray')

air_imarray = croped_imarray < 1
plt.figure(figsize=(12,6))
plt.imshow(air_imarray, cmap='gray')
air_imarray = nd.binary_fill_holes(air_imarray, np.ones((10, 10)))
plt.figure(figsize=(12,6))
plt.imshow(air_imarray, cmap='gray')

air_imarray = nd.binary_opening(air_imarray, np.ones((10, 10)))
plt.figure(figsize=(12,6))
plt.imshow(air_imarray, cmap='gray')


object_params, labels = methods.calc_binObject_params(air_imarray, maxObj_height=512, maxObj_width=512, plot=True)

patch_results = methods.detect_patch_with_object(croped_imarray, labels)


croped_imarray, patch_results = methods.validate_patches(rotated_image, lowdist, 100, highdist, patch_results)

# loading the model which is trained for 30 epochs
from keras.models import load_model
model = load_model('models/src2tar_with_real_air_after_165000.h5')

removed_air = np.copy(croped_imarray)

for params in patch_results:
    
    if params['num_objects_in_patch'] > 0:
        x1, y1 = params['patch_coords'][0]
        x2, y2 = params['patch_coords'][1]

        
        patch = croped_imarray[y1:y2, x1:x2]
#         patch = nd.zoom(patch, zoom=0.5)
        patch_inShape = np.expand_dims((patch-127.5)/127.5, axis=(0, -1))
        air_rem_patch = np.squeeze(model.predict(patch_inShape, verbose=0))
        air_rem_patch = (air_rem_patch+1)/2.0
        air_rem_patch = img_as_ubyte(air_rem_patch)
        removed_air[y1:y2, x1:x2] = air_rem_patch
#         removed_air[y1:y2, x1:x2] = nd.zoom(air_rem_patch, zoom=2)


plt.figure(figsize=(16,8))       
plt.imshow(croped_imarray, cmap='gray')
plt.show()
plt.figure(figsize=(16,8))       
plt.imshow(removed_air, cmap='gray')
plt.show() 