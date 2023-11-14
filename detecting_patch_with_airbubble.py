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

data_dir = 'F:\\MD_1264_A3_1_Z9.9mm\\slices'
slice_no = 'slice_0003.tif'


# to scale the data to the original data intensity 
new_air = 0.00009706
new_paraffin = 0.0004179

new_diff = new_paraffin - new_air

old_air = 0.0002605
old_paraffin = 0.0009999

old_diff = old_paraffin - old_air


# Reading the slice 
im = Image.open(os.path.join(data_dir, slice_no))
imarray = np.array(im)
# imarray = (imarray - new_air) / new_diff * old_diff + old_air
imarray = np.clip(imarray, 0.0005, 0.003)
imarray = utils.norm8bit(imarray)


# rotation correction and find the start and end point of the patches 
rotated_image, _, lowdist = utils.tilt_correction(imarray, edge_th=0.1, ang_vari=4, plot=True)
_, _, highdist = utils.tilt_correction(imarray, edge_th=250)

start_point = int(abs(lowdist[0]))
end_point = int(abs(highdist[0]))
h_offset = 150

# with the calculated range getting the section of the image 
croped_imarray = rotated_image[start_point+h_offset:end_point, :]

plt.figure(figsize=(12,6))
plt.imshow(croped_imarray, cmap='gray')
plt.show()

# finding the airbubbles in the slice with thresholding and some binary morphology
air_imarray = croped_imarray < 1
# plt.figure(figsize=(12,6))
# plt.imshow(air_imarray, cmap='gray')
# plt.show()
air_imarray = nd.binary_fill_holes(air_imarray, np.ones((10, 10)))
# plt.figure(figsize=(12,6))
# plt.imshow(air_imarray, cmap='gray')
# plt.show()
air_imarray = nd.binary_opening(air_imarray, np.ones((10, 10)))
# plt.figure(figsize=(12,6))
# plt.imshow(air_imarray, cmap='gray')
# plt.show()


# setting the patch size
patch_size = 256
# calculation of airbubble parameters with options to exclude large airbubbles/edges 
object_params, labels = methods.calc_binObject_params(air_imarray, maxObj_height=512, maxObj_width=512, plot=True)
# object_params are not used here for the further steps but can be usefull for 
# other calculations

# get the patches with airbubbles which are identified 
patch_results = methods.detect_patch_with_object(croped_imarray, labels, patch_size=patch_size, step_size=patch_size)

# validate the patch_result for the edge patches.
# adjust imarray with padding for edge patches  
adjusted_imarray, patch_results = methods.validate_patches(rotated_image, croped_imarray, start_point, h_offset, end_point, patch_results, patch_size=patch_size)

# loading the model which is trained for 30 epochs
from keras.models import load_model
model = load_model('models/src2tar_with_real_air_after_165000.h5')

# making a copy of adjusted image 
removed_air = np.copy(adjusted_imarray)

# iterating over the patches and applying the Generator model 
for params in patch_results:
    
    if params['num_objects_in_patch'] > 0:
        x1, y1 = params['patch_coords'][0]
        x2, y2 = params['patch_coords'][1]

        patch = adjusted_imarray[y1:y2, x1:x2]
        
        patch = ((patch-127.5)/127.5).astype(np.float32)
        
        air_rem_patch = methods.apply_model_to_patch(model, patch)
        
        air_rem_patch = (air_rem_patch+1)/2.0
        air_rem_patch = img_as_ubyte(air_rem_patch)
        
        removed_air[y1:y2, x1:x2] = air_rem_patch

# getting the shape of non padded image 
rows, cols = croped_imarray.shape


# showing results with slice with airbubbles
plt.figure(figsize=(16,8))       
plt.imshow(croped_imarray, cmap='gray')
plt.show()
# showing airbubble removed slice with original size
plt.figure(figsize=(16,8))       
plt.imshow(removed_air[:rows, :cols], cmap='gray')
plt.show() 