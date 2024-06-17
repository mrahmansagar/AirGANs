# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:26:38 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import os 
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image 
from scipy import ndimage as nd
from skimage import img_as_ubyte
import tifffile

from keras.models import load_model

from tqdm import tqdm

os.sys.path.insert(0, 'E:\\dev\\packages')
from proUtils import utils


import methods

data_dir = 'F:\\MD_1264_A10_Z6.6mm\\'
slice_dir = os.path.join(data_dir, 'slices')


# setting the patch size
patch_size = (256, 256, 256)

# all the slices in the directory 
slices = os.listdir(slice_dir)


# save dir 
save_dir = os.path.join(data_dir, 'air3d_removed_slices_v2')
if not os.path.exists(save_dir):
            os.makedirs(save_dir)

# new_air = 0.00009706
# new_paraffin = 0.0004179

# new_diff = new_paraffin - new_air

# old_air = 0.0002605
# old_paraffin = 0.0009999

# old_diff = old_paraffin - old_air


"""
Here a sample slice is selected to find the start and end point. 
for 3D the slices are not rotated 
"""
sample_slice = 'slice_0346.tif'

# Reading the slice 
im = Image.open(os.path.join(slice_dir, sample_slice))
imarray = np.array(im)
# imarray = (imarray - new_air) / new_diff * old_diff + old_air
imarray = np.clip(imarray, 0.0005, 0.003)
imarray = utils.norm8bit(imarray)

# rotation correction and find the start and end point of the patches 
rotated_image, rotation, lowdist = utils.tilt_correction(imarray, edge_th=0.1, ang_vari=4, plot=True)
_, _, highdist = utils.tilt_correction(imarray, edge_th=250)

start_point = int(abs(lowdist[0]))
end_point = int(abs(highdist[0]))
h_offset = 150

# with the calculated range getting the section of the image 
croped_imarray = rotated_image[start_point+h_offset:end_point, :]
plt.figure(figsize=(12,6))
plt.imshow(croped_imarray, cmap='gray')
plt.show()


def chunk_list(input_list, chunk_size):
    """
    Generate chunks of a specified size from a given list.

    Parameters:
    - input_list: The input list to be chunked.
    - chunk_size: The size of each chunk.

    Returns:
    A generator that produces chunks of the specified size.
    """
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]


model = load_model('E:\\projects\\AirGANs\\air3d_202401031030\\src2tar_air3d_after_179999.h5')

"""
Selecting chunks according to the size of the patch. e.g. if the patch size is 
(256,256,256) then the in a chunk is 256 slices.

while loading the slices in a chunk checking if a slice with nan values exists
then replace it with previous slice so in the end there is no blanck slice in
chunk.

Then each slice in the chunk is used to identify if air with thresholding and 
if the size of an object is bigger than 512 then they are filled with background
(here 0). This is to companset round edges in the slice. 

Then the  

"""
# number of slices to process at a time. normally same as the size of the patch depth 
chunk_depth = patch_size[0]

for step, chunk in enumerate(chunk_list(slices, chunk_size=chunk_depth)):
    print('Processing...', step+1, 'of', len(list(chunk_list(slices, chunk_size=chunk_depth))))
    vol_chunk = np.empty(shape=(len(chunk), *imarray.shape), dtype=imarray.dtype)
    prev_imarray = None
    for i, fname in enumerate(tqdm(chunk, desc='Loading slices....')):
        im = Image.open(os.path.join(slice_dir, fname))
        imarray = np.array(im)
    #     imarray = (imarray - new_air) / new_diff * old_diff + old_air
        # imarray = np.clip(imarray, 0.0005, 0.003)
        nan_indices = np.isnan(imarray)
        if np.any(nan_indices) and prev_imarray is not None:
            vol_chunk[i, :, :] = prev_imarray
        else:
            # imarray = utils.norm8bit(imarray)
            vol_chunk[i, :, :] = imarray
            prev_imarray = imarray
    
    cropped_vol_chunk = vol_chunk[:,start_point+h_offset:end_point, :]
    
    # Identifing airs in the chunk 
    air_vol_chunk = np.empty_like(cropped_vol_chunk)
    for i in tqdm(range(0, len(cropped_vol_chunk)), desc='Identifying air...'):
        aslice = cropped_vol_chunk[i]
        th_slice = aslice < 1
        th_slice = nd.binary_fill_holes(th_slice, np.ones((10,10)))
        th_slice = nd.binary_opening(th_slice, np.ones((10,10)))
        th_labels, num_features = nd.label(th_slice)
        
        if num_features > 1:
            for lab in range(0, num_features + 1):
                # Find coordinates of the labeled component
                coords = np.column_stack(np.where(th_labels == lab))
                # Calculate the bounding box coordinates
                y1, x1 = coords.min(axis=0)
                y2, x2 = coords.max(axis=0) 

                # Calculate width and height
                width = x2 - x1 + 1 
                height = y2 - y1 + 1

                if width < 512 and height < 512:
                    # Do something with the parameters if needed
                    pass
                else:
                    # Set the value to 0 in the corresponding label coordinates in the th_slice
                    th_slice[th_labels == lab] = 0

            # You can assign the modified th_slice to air_vol if needed
            air_vol_chunk[i] = th_slice
        else:
            # You can assign the modified th_slice to air_vol if needed
            air_vol_chunk[i] = th_slice      
    
   
    labels, _ = nd.label(cropped_vol_chunk)
    
    depth, height, width = air_vol_chunk.shape
    d_patch, h_patch, w_patch = patch_size
    
    air_rem_chunk = np.copy(cropped_vol_chunk)
    for d in range(0, depth, d_patch):
        for h in tqdm(range(0, height, h_patch), desc='Removing Air...'):
            for w in range(0, width, w_patch):
                d_end = min(d + d_patch, depth)
                h_end = min(h + h_patch, height)
                w_end = min(w + w_patch, width)
    
                patch_lab = labels[d:d_end, h:h_end, w:w_end]    
                # Find unique labels within the patch
                unique_labels = np.unique(patch_lab)
                # Count the number of unique labels (excluding background label)
                num_objects_in_patch = len(unique_labels) - 1
                
                if num_objects_in_patch > 0:
                    patch = cropped_vol_chunk[d:d_end, h:h_end, w:w_end]
                    patch_inShape = ((patch-127.5)/127.5).astype(np.float32)
                    air_rem_patch = methods.apply_model_to_patch(model, patch_inShape)
                    air_rem_patch = (air_rem_patch+1)/2.0
                    try:
                        air_rem_patch = img_as_ubyte(air_rem_patch)
                    except:
                        air_rem_patch = np.clip(air_rem_patch, -1, 1)
                        air_rem_patch = img_as_ubyte(air_rem_patch)
                        
                    air_rem_chunk[d:d_end, h:h_end, w:w_end] = air_rem_patch
    # saving back to the original volume to retain the original shape 
    full_air_rem_chunk = np.copy(vol_chunk)
    full_air_rem_chunk[:,start_point+h_offset:end_point, :] = air_rem_chunk
    
    print('Saving slices....')
    for i, f in enumerate(chunk):
        fName = os.path.join(save_dir, f) 
        tifffile.imwrite(fName, full_air_rem_chunk[i])

print('completed.')