# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:22:45 2023

@author: mrahm
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def norm8bit(v, minVal=None, maxVal=None):
    """
    NORM8BIT function takes an array and normalized it before converting it into 
    a 8 bit unsigned integer and returns it.

    Parameters
    ----------
    v : numpy.ndarray
        Array of N dimension.
    minVal : number 
        Any value that needs to be used as min value for normalization. If no
        value is provided then it uses min value of the given array. The default is None.
    maxVal : number 
        Any value that needs to be used as max value for normalization. If no
        value is provided then it uses max value of the given array. The default is None.

    Returns
    -------
    numpy.ndarray (uint8)
        Numpy Array of same dimension as input with data type as unsigned integer 8 bit

    """
    if minVal == None:
        minVal = v.min()
    
    if maxVal == None:
        maxVal = v.max()
      
    maxVal -= minVal
      
    v = ((v - minVal)/maxVal) * 255
    
    return v.astype(np.uint8)

data_dir = 'F:\\MD_1264_A6_1_Z6.6mm_corr_phrt\\slices'

new_air = 0.00009706
new_paraffin = 0.0004179

new_diff = new_paraffin - new_air

old_air = 0.0002605
old_paraffin = 0.0009999

old_diff = old_paraffin - old_air

# Reading all the slices in a Volume 
im = Image.open(os.path.join(data_dir, 'slice_0052.tif'))
imarray = np.array(im)
# imarray = (imarray - new_air) / new_diff * old_diff + old_air
imarray = np.clip(imarray, 0.0005, 0.003)
imarray = norm8bit(imarray)

from scipy import ndimage as nd
from skimage.filters import prewitt
from skimage.transform import hough_line, hough_line_peaks
from skimage import img_as_ubyte

def tilt_correction(imarray, edge_th=1, plot=False):
    th_im = imarray > edge_th
    th_im = nd.binary_fill_holes(th_im, np.ones((20,20)))
    edges = img_as_ubyte(prewitt(th_im))
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    hspace, theta, distance = hough_line(edges, tested_angles)
    h, q, d = hough_line_peaks(hspace, theta, distance)
    
    angle_list=[]
    for _, angle, dist in zip(*hough_line_peaks(hspace, theta, distance)):
        angle_list.append(angle*180/np.pi)
#     angle_list = np.array(angle_list)
    
    angle_variation = np.zeros(shape=(len(angle_list),len(angle_list)))
    for i in range(len(angle_list)):
        for j in range(len(angle_list)):
            angle_variation[i, j] = abs(angle_list[j] - angle_list[i])
    
    if angle_variation.max() > 2:
        print("Could not find exact straight line for tilt correction")
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(edges, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(np.log(1 + hspace),
                     extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), distance[-1], distance[0]],
                     cmap='gray', aspect=1/1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(edges, cmap='gray')

        origin = np.array((0, edges.shape[1]))

        for _, angle, dist in zip(*hough_line_peaks(hspace, theta, distance)):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[2].plot(origin, (y0, y1), '-r')
        ax[2].set_xlim(origin)
        ax[2].set_ylim((edges.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')
        
        plt.tight_layout()
        plt.show()
        
        return None, angle_variation
    
    else:
        rotation = np.array([a for a in angle_list]).mean()
        if rotation < 0 :
            rotated_imarray = nd.rotate(imarray, angle=90+rotation, reshape=False)
        else:
            rotated_imarray = nd.rotate(imarray, angle=180+90+rotation, reshape=False)
        
        if plot:
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(imarray, cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(rotated_imarray, cmap='gray')
            plt.show()
        
        return rotated_imarray, rotation, d

rotated_image, _, lowdist = tilt_correction(imarray, edge_th=0.1, plot=True)
_, _, highdist = tilt_correction(imarray, edge_th=250)
croped_imarray = rotated_image[int(abs(lowdist[0])+100):int(abs(highdist[0])), :]
plt.imshow(croped_imarray, cmap='gray')

# loading the model which is trained for 30 epochs
from keras.models import load_model
model = load_model('models/src2tar_with_real_air_after_165000.h5')

patch_size = 512
padding_size = 2
removed_air = np.empty_like(croped_imarray)#,shape=croped_imarray.shape, dtype=np.uint8)
for row in list(range(0, croped_imarray.shape[0]-patch_size, patch_size)):
    for col in list(range(0, croped_imarray.shape[1]-patch_size, patch_size)):
        patch = croped_imarray[row:row+patch_size, col:col+patch_size]
        patch = nd.zoom(patch, zoom=0.5)
        patch_inShape = np.expand_dims((patch-127.5)/127.5, axis=(0, -1))
        try:
            air_rem_patch = np.squeeze(model.predict(patch_inShape, verbose=0))
            air_rem_patch = (air_rem_patch+1)/2.0
            air_rem_patch = img_as_ubyte(air_rem_patch)
            removed_air[row:row+patch_size, col:col+patch_size] = nd.zoom(air_rem_patch, zoom=2)
            break
        except:
            pass
        

plt.figure(figsize=(16,8))       
plt.imshow(croped_imarray, cmap='gray')
plt.show()
plt.figure(figsize=(16,8))       
plt.imshow(removed_air, cmap='gray')