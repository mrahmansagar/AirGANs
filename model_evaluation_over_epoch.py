# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:25:15 2024

@author: mrahm
"""



import os
import numpy as np
import matplotlib.pyplot as plt

os.sys.path.insert(0, 'E:\\dev\\packages')
from proUtils import utils


from skimage import img_as_ubyte

from PIL import Image

from glob import glob

from keras.models import load_model
from keras.utils import load_img, img_to_array


model_dir = 'E:\\dev\\GANs\\air4\\'

data_dir = 'E:\\Data\\air_bubbles\\with_air\\test\\'

test_samples = glob(data_dir+'*')

# selected_sample = np.random.choice(test_samples)
selected_sample = 'E:\\Data\\air_bubbles\\with_air_real\\test\\slice_5554.tif'

print('selected sample', selected_sample)

all_trained_model = glob(model_dir + '*.h5')


img = load_img(selected_sample, color_mode='grayscale')
plt.imshow(img, cmap='gray')
plt.show()
imarray = img_to_array(img)
imarray_scaled = (imarray - 127.5) / 127.5
imarray_in_shape = np.expand_dims(imarray_scaled, axis=0)


all_gen_images = []

for aModel in all_trained_model:
    model = load_model(aModel)
    
    gen_imarray = model.predict(imarray_in_shape)
    
    gen_imarray = (gen_imarray + 1) / 2.0 
    
    gen_imarray_PIL = Image.fromarray(img_as_ubyte(np.squeeze(gen_imarray)))

    all_gen_images.append(gen_imarray_PIL)
    
    # plt.imshow(gen_imarray[0], cmap='gray')
    # plt.show()

all_gen_images[0].save('air4_gen_real_slice_5554.gif', save_all=True, append_images=all_gen_images[1:], duration=500, loop=0)




with_air_path = 'E:\\Data\\air_bubbles\\with_air\\train\\'
without_air_path = 'E:\\Data\\air_bubbles\\without_air\\train\\'
all_files = os.listdir(with_air_path)
num_of_slice = 30
selected_files = np.random.choice(all_files, num_of_slice)
selected_img = []
for afile in selected_files:
    img = Image.open(os.path.join(without_air_path, afile))
    selected_img.append(img)

selected_img[0].save('outputs/without_air.gif', save_all=True, append_images=selected_img[1:], duration=500, loop=0)
