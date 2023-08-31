# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:19:45 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

"""

import os
from tkinter import Tcl
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from PIL import Image 
from scipy import ndimage
import tifffile

from tqdm import tqdm

data_dir = 'E:\\Data\\air_bubbles'

w_air_dir = os.path.join(data_dir, 'with_air', 'test')
wo_air_dir = os.path.join(data_dir, 'without_air', 'test')

listofslices = os.listdir(wo_air_dir)

# minimum number of circle
min_n_circle = 1
max_n_circle = 10

min_circle_radius = 10
max_circle_radius = 50

img_size = 256 


for i, aSlice in enumerate(tqdm(listofslices)):
    #iterating through one image at a time
    img = Image.open(os.path.join(wo_air_dir, aSlice))
    #resizing the image
#     img = img.resize((img_size, img_size))
    img = np.array(img)
    #saving the image to make a pair
#     file_name = f'slice_{i}.tif'
    file_name = aSlice
#     tifffile.imsave(os.path.join(wo_air_dir, file_name), img)
    
    #=================Airbubble================
    #randomly selecting number of circles between 1 to 10 for each slice
    n_circles = np.random.randint(min_n_circle, max_n_circle)
    # randomly choosing the centroids
    coordinates = np.random.randint(0, img_size, size=(n_circles, 2))
    # randomly defining the circle size for each circle 
    circle_radii = np.random.randint(min_circle_radius, max_circle_radius, size=n_circles)
    
    # inscribing a circle with values between 200-255 in the image which will become the border of the circle
    for c in range(n_circles):
        y, x = coordinates[c]
        radius = circle_radii[c]

        circle = Circle((x, y), radius, fill=False)

        for y in range(img_size):
            for x in range(img_size):
                if circle.contains_point((x, y)):
                    img[y, x] = np.random.randint(200, 255) 

    
    
    # Create a smoothed version of the image
    smoothed_img = ndimage.gaussian_filter(img, sigma=3)
    
    # Inscribing smaller circles on top of previous circles with values 0
    for c in range(n_circles):
        y, x = coordinates[c]
        radius = circle_radii[c] - np.random.randint(7,10) #border size

        circle = Circle((x, y), radius, fill=False)

        for y in range(img_size):
            for x in range(img_size):
                if circle.contains_point((x, y)):
                    smoothed_img[y, x] = 0
    
    
    
    # copy these circles with border into the original image. 
    source_img = np.copy(smoothed_img)
    target_img = np.copy(img)


    # Inscribe the circles in the image
    for c in range(n_circles):
        # Define the circle parameters
        center_y, center_x =  coordinates[c]
        circle_radius = circle_radii[c] + 10 # taking extra pixels to copy the smooth border of the circle(s)

        # Create a Circle patch for visualization
        circle_patch = Circle((center_x, center_y), circle_radius, fill=False)


        # Iterate through the pixels in the circle and copy values from source to target
        for y in range(img_size):
            for x in range(img_size):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= circle_radius:
                    target_img[y, x] = source_img[y, x]
    
    
    # saving the imgae with airbubbles into the directory    
    tifffile.imsave(os.path.join(w_air_dir, file_name), target_img)