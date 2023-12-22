# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:14:10 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Training a cycleGAN with 2d images of with and without air
"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
import matplotlib.pyplot as plt
import numpy as np

from GANs import utils
from GANs.cycleGAN import models

air_img_dir = 'E:\\Data\\air_bubbles\\with_air_real\\train\\'
wo_air_img_dir = 'E:\\Data\\air_bubbles\\without_air\\train\\'


data_Air = utils.load_images_in_shape(air_img_dir, target_size=(256,256), color_mode = "grayscale")
data_woAir = utils.load_images_in_shape(wo_air_img_dir, target_size=(256,256), color_mode = "grayscale")

idxA = np.random.randint(0, len(data_Air), 5)
idxB = np.random.randint(0, len(data_woAir), 5)

plt.figure(figsize=(10,5))
for i, idx in enumerate(idxA):
    plt.subplot(1, 5, 1 + i)
    plt.axis('off')
    plt.imshow(data_Air[idx].astype('uint8'), cmap='gray')
plt.show()
plt.figure(figsize=(10,5))
for i, idx in enumerate(idxB):
    plt.subplot(1, 5, 1 + i)
    plt.axis('off')
    plt.imshow(data_woAir[idx].astype('uint8'), cmap='gray')
plt.show()


data_Air = utils.scale_data(data_Air)
data_woAir = utils.scale_data(data_woAir)


image_shape = data_Air.shape[1:]
gen_Air2woAir = models.build_generator(input_shape=image_shape)
gen_woAir2Air = models.build_generator(input_shape=image_shape)

disc_Air = models.build_discriminator(input_shape=image_shape)
disc_woAir = models.build_discriminator(input_shape=image_shape)

cgan_Air2woAir = models.build_cycleGAN(gen_Air2woAir, disc_woAir, gen_woAir2Air,input_shape=image_shape)
cgan_woAir2Air = models.build_cycleGAN(gen_woAir2Air, disc_Air, gen_Air2woAir,input_shape=image_shape)
models.train_cycleGAN(disc_Air, disc_woAir, gen_Air2woAir, gen_woAir2Air, cgan_Air2woAir, cgan_woAir2Air, data_Air, data_woAir,epochs=100, summary_interval=10)
