# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:15:10 2024

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Analysis of pore features with and without air and removed air cubes  

"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import numpy as np 
import matplotlib.pyplot as plt 

import pandas as pd 
import json
from glob import glob 



# Directory where the scans are stored with results 
root_dir = "E:\\Data\\air_bubbles\\3d"
# porespy output with cubes without air bubbles
with_air_dir = os.path.join(root_dir, 'with_air\\test_porosity')
# porespy output with cubes where air bubbles were copied from other cubes
# extracted_air_dir = os.path.join(root_dir, 'extracted_air\\test\\porespy')
# porespy outpur with cubes with removed air bubbles using AirGAN
removed_air_dir = os.path.join(root_dir, 'with_air\\air_removed_test_porosity')

files_dict = {}

# listing all the porespy files in the respective folders
porespy_files_with_air = glob(with_air_dir + '\*.json')
# porespy_files_extracted_air = glob(extracted_air_dir + '\*.json')
porespy_files_removed_air = glob(removed_air_dir + '\*.json') 

# Assigning lists of file paths to dictionary keys
files_dict['with_air'] = porespy_files_with_air
# files_dict['extracted_air'] = porespy_files_extracted_air
files_dict['removed_air'] = porespy_files_removed_air


# feature to be analyzed 
pore_feature = 'porosity'

feature_dic = {}

for group, file_list in files_dict.items():
    
    feature = []
    for file in file_list:
        f = open(file)
        data = json.load(f)
        f.close()
        
        # do what we want to plot for exampe, blob volume
        extract_feature = data[pore_feature]
        
        if len(extract_feature) > 0:
            # parameter += extract_feature
            feature.append(np.mean(extract_feature))
            # parameter.append(utils.ratio_above_threshold(extract_feature, 100000))

    feature_dic[group] = feature
        
    
# Create a figure and axis
fig, ax = plt.subplots(figsize=(20, 10))

# Plot each group's data
for group, data in feature_dic.items():
    ax.plot(data, 'o', label=group)

# Customize the plot
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Data from Different Groups')
ax.legend()
ax.set_xticks(list(range(0, 94)))
ax.set_xticklabels(list(range(0, 94)), rotation=90)
# Show the plot
plt.show()