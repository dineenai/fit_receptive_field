# Example of finding the gabor receptive field of a given neuron model
import numpy as np
import matplotlib.pyplot as plt

# from fitgabor import GaborGenerator,DOGGenerator,trainer_fn, trainer_fn2
from fit_receptive_field import GaborGenerator,DOGGenerator,trainer_fn, trainer_fn2

# from fitgabor.utils import dog_fn,gabor_fn
from fit_receptive_field.utils import dog_fn, gabor_fn
import matplotlib.image as img  

import cv2

import torch
from torch import nn
from torch.nn import functional as F

im=np.load('/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g0_60e_e60_np_arr/supRN50_conv1_21_g0_60e_e60_filter_np_arr_7.npy')
im=im[1,:,:]
im=im-np.mean(im)
im=im/(np.std(im))

# /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g0_60e_e60_np_arr/supRN50_conv1_21_g0_60e_e60_filter_np_arr_7.npy
# fit_receptive_field

class Neuron(nn.Module):
    def __init__(self, rf):
        super().__init__()
        h, w = rf.shape
        self.rf = torch.tensor(rf.reshape(1, 1, h, w).astype(np.float32))
        
    def forward(self, x):
        return F.elu((x * self.rf).sum()) + 1

# TO DO: 

fig, (ax) = plt.subplots(1, 1, figsize=(6, 3), dpi=100)

ax.imshow(im)


ax.set(xticks=[], yticks=[]);
neuron = Neuron(im)



