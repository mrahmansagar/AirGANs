# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:50:06 2024

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')

import numpy as np
from scipy import ndimage as nd

import pandas as pd

from proUtils import utils
import matplotlib.pyplot as plt

# data dir
data_dir = "E:\\Data\\air_bubbles\\3d\\with_air\\"

csv_file = os.path.join(data_dir, 'air_volume_slices_with_air.csv')

air_vol_dir = os.path.join(data_dir, 'test')
air_removed_vol_dir = os.path.join(data_dir, 'air_removed_test')

csv_df = pd.read_csv(csv_file)


def calculate_porosity(vol, th_value=50):
    vol = nd.median_filter(vol, size=2)
    th_vol = vol < th_value
    # th_vol = nd.binary_fill_holes(th_vol, np.ones((3,3,3)))
    # th_vol = nd.binary_closing(th_vol, np.ones((2,2,2)))
    
    # Count the number of pore voxels (True values in th_vol)
    num_pore_voxels = np.sum(th_vol)
    
    # Calculate the total number of voxels
    total_voxels = np.prod(th_vol.shape)
    
    # Calculate porosity
    return num_pore_voxels / total_voxels
    
    

# Initialize an empty list to store the porosity values
air_porosity = []
air_removed_porosity = []

for index, row in csv_df.iterrows():
    if not pd.isna(row[1]):
        air_vol_path = os.path.join(air_vol_dir, row[0])
        air_vol = utils.load_roi(air_vol_path, file_range=[ int(row[1]), int(row[2])] )
        
        porosity = calculate_porosity(air_vol)
        air_porosity.append(porosity)
        
        air_removed_vol_path = os.path.join(air_removed_vol_dir, row[0])
        air_removed_vol = utils.load_roi(air_removed_vol_path, file_range=[ int(row[1]), int(row[2])] )
        
        porosity = calculate_porosity(air_removed_vol)
        air_removed_porosity.append(porosity)
    
    else:
        air_porosity.append('')
        air_removed_porosity.append('')
        
    
csv_df['air_porosity'] = air_porosity
csv_df['air_removed_porosity'] = air_removed_porosity

csv_df = csv_df.dropna()





# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(csv_df['volume'], csv_df['air_porosity'], 'o', label='Air Porosity')
plt.plot(csv_df['volume'], csv_df['air_removed_porosity'], '*' label='Air Removed Porosity')

plt.xlabel('Volume')
plt.xticks(rotation=90)
plt.ylabel('Porosity')
plt.title('Air Porosity and Air Removed Porosity vs Volume')
plt.legend()
plt.grid(True)

plt.show()    
    
    