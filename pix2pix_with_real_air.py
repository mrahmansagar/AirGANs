# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:50:42 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Data loading/pre-processing and using GANs packgae training a pix2pix network
"""

import os
import sys
sys.path.insert(0, 'E:\\dev\\packages')

import numpy as np
import matplotlib.pyplot as plt

from GANs import utils 
from GANs.pix2pix import models 

data_dir = 'E:\\Data\\air_bubbles'
src_img_dir = os.path.join(data_dir, 'extracted_real_air\\train2')
tar_img_dir = os.path.join(data_dir, 'without_air\\train')


src_img = utils.load_images_in_shape(src_img_dir, color_mode = 'grayscale')
tar_img = utils.load_images_in_shape(tar_img_dir, color_mode = 'grayscale')

print('loaded', src_img.shape, 'source images')
print('loaded', tar_img.shape, 'target images')

n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(src_img[i].astype('uint8'), cmap='gray')
    
# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(tar_img[i].astype('uint8'), cmap='gray')
plt.show()

# scalling the data between -1 to 1
src_img = (src_img - 127.5) / 127.5
tar_img = (tar_img - 127.5) / 127.5

# define image shape
img_shape = src_img.shape[1:]
# define the models
disc = models.build_discriminator(img_shape)
gene = models.build_generator(img_shape)
# define the composite model
p2p_model = models.build_pix2pix(gene, disc, img_shape)
models.train_pix2pix(gene, disc, p2p_model, src_img, tar_img, epochs=30, summary_interval=1, name='models/src2tar_with_real_air')

# loading the model which is trained for 30 epochs
from keras.models import load_model
model = load_model('models/src2tar_with_real_air_after_165000.h5')


test_src_img_dir = os.path.join(data_dir, 'extracted_real_air\\test2')
test_tar_img_dir = os.path.join(data_dir, 'without_air\\test')

test_src_img = utils.load_images_in_shape(test_src_img_dir, color_mode = 'grayscale')
test_tar_img = utils.load_images_in_shape(test_tar_img_dir, color_mode = 'grayscale')

def plot_src_gen_tar(src, tar, gen_model, sample_size=5):
    idx = np.random.randint(0, len(src), sample_size)
    
    sel_src = src[idx]
    scaled_src = (sel_src - 127.5) / 127.5
    sel_tar = tar[idx]
    gen = gen_model.predict(scaled_src)
    gen = (gen + 1) / 2.0
    
    
    fig, axes = plt.subplots(3, sample_size, figsize=(10, 7))
    
    for i in range(sample_size):
        axes[0, i].imshow(sel_src[i], cmap='gray')
        axes[0, i].set_title('Source Image')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(gen[i], cmap='gray')
        axes[1, i].set_title('Generated Image')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(sel_tar[i], cmap='gray')
        axes[2, i].set_title('Target Image')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
plot_src_gen_tar(test_src_img, test_tar_img, model)


real_air_dir = 'E:\\Data\\air_bubbles\\with_air_real\\test'
real_src_img = utils.load_images_in_shape(real_air_dir, color_mode='grayscale')

def plot_src_gen(src, gen_model, sample_size=5):
    idx = np.random.randint(0, len(src), sample_size)
    
    sel_src = src[idx]
    scaled_src = (sel_src - 127.5) / 127.5
    gen = gen_model.predict(scaled_src)
    gen = (gen + 1) / 2.0
    
    
    fig, axes = plt.subplots(2, sample_size, figsize=(10, 5))
    
    for i in range(sample_size):
        axes[0, i].imshow(sel_src[i], cmap='gray')
        axes[0, i].set_title('Source Image')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(gen[i], cmap='gray')
        axes[1, i].set_title('Generated Image')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()
    
plot_src_gen(real_src_img, model, 5)
